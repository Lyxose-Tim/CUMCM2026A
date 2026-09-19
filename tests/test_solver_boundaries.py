"""Solver regressions: exact time boundaries and invalid accepted states."""
from types import SimpleNamespace

import numpy as np
import pytest

from drymodel import config as cfgmod, runners
from drymodel import solver_be as BE, solver_bdf as BDF


@pytest.fixture
def cfg():
    return cfgmod.load_config()


def _be_run(cfg, *, t_end, dt, record_times=None, breakpoints=()):
    op, _ = runners.build_fixed_operator(cfg, "q1", 20, interface="integral")
    seen = []
    result = BE.integrate_be(
        op, runners.initial_state(cfg, 20), t_end, dt, C0_ref=cfg.C0,
        picard=cfg.picard, retry=cfg.retry, record_times=record_times,
        breakpoints=breakpoints, flux_recorder=lambda t, ds, f: seen.append((t, ds)))
    return result, seen


@pytest.mark.parametrize("t_end", [2.4, 2.6])
def test_be_stops_exactly_at_nonintegral_endpoint(cfg, t_end):
    result, steps = _be_run(cfg, t_end=t_end, dt=1.0)
    assert result.ok
    assert steps[-1][0] == t_end
    assert result.t[-1] == t_end
    assert sum(ds for _, ds in steps) == pytest.approx(t_end)
    assert max(t for t, _ in steps) <= t_end


def test_be_preserves_fractional_samples_and_breakpoints(cfg):
    samples = [0.0, 0.25, 0.5, 0.75, 1.0]
    result, steps = _be_run(cfg, t_end=1.0, dt=1.0,
                           record_times=samples, breakpoints=(0.6,))
    assert result.ok
    np.testing.assert_array_equal(result.t, samples)
    assert any(t == 0.6 for t, _ in steps)
    assert all(not (t-ds < 0.6 < t) for t, ds in steps)


def test_be_decimal_samples_do_not_create_roundoff_microsteps(cfg):
    """真实 Q1：0.3 与 arange 的 0.30000000000000004 不能各推进一次。"""
    samples = [0., .3, .6, 1.]
    result, steps = _be_run(cfg, t_end=1., dt=.1, record_times=samples)
    assert result.ok
    np.testing.assert_array_equal(result.t, samples)
    assert result.total_steps == 10
    assert all(ds > .09 for _, ds in steps[1:])
    assert sum(ds for _, ds in steps) == pytest.approx(1.)


@pytest.mark.parametrize("t_end", [.3, np.nextafter(.3, np.inf),
                                   np.nextafter(np.nextafter(.3, np.inf), np.inf)])
def test_be_roundoff_near_endpoint_keeps_exact_endpoint(cfg, t_end):
    result, steps = _be_run(cfg, t_end=t_end, dt=.1)
    assert result.ok
    assert result.total_steps == 3
    assert result.t[-1] == t_end
    assert steps[-1][0] == t_end
    assert all(ds > .09 for _, ds in steps[1:])
    assert max(t for t, _ in steps) <= t_end


@pytest.mark.parametrize("point", [np.nextafter(.3, 0.), .3, np.nextafter(.3, np.inf)])
def test_be_roundoff_breakpoint_yields_to_explicit_sample(cfg, point):
    samples = [0., .3, .6, 1.]
    result, steps = _be_run(cfg, t_end=1., dt=.1, record_times=samples,
                            breakpoints=(point,))
    assert result.ok and result.total_steps == 10
    np.testing.assert_array_equal(result.t, samples)
    assert sum(t == .3 for t, _ in steps) == 1


def test_be_decimal_breakpoint_replaces_nearby_default_grid(cfg):
    result, steps = _be_run(cfg, t_end=1., dt=.1, breakpoints=(.3,))
    assert result.ok and result.total_steps == 10
    assert .3 in result.t
    assert sum(t == .3 for t, _ in steps) == 1
    assert all(ds > .09 for _, ds in steps[1:])


@pytest.mark.parametrize("point_source", ["sample", "breakpoint", "endpoint"])
@pytest.mark.parametrize("gap", [1e-6, 1e-10])
def test_be_distinguishable_short_intervals_are_not_merged(cfg, monkeypatch, point_source, gap):
    """调度回归：真实短间隔保留；替身单步不声称验证微步的 PDE 精度。"""
    def constant_step(op, y, t, dt, **kwargs):
        return y.copy(), True, 1, float(np.min(y[:op.N+1]))

    monkeypatch.setattr(BE, "step_be", constant_step)
    start, end = .3, .3 + gap
    kwargs = {"t_end": 1., "dt": .1, "record_times": [0., start, 1.]}
    if point_source == "sample":
        kwargs["record_times"].append(end)
    elif point_source == "breakpoint":
        kwargs["breakpoints"] = (end,)
    else:
        kwargs["t_end"] = end
        kwargs["record_times"] = [0., start, end]
    result, steps = _be_run(cfg, **kwargs)
    assert result.ok
    ends = [t for t, _ in steps]
    index = ends.index(start)
    assert ends[index + 1] == end
    assert steps[index + 1][1] == end - start
    np.testing.assert_array_equal(result.t, sorted(kwargs["record_times"]))


@pytest.mark.parametrize("t_end,dt", [(-1., 1.), (1., 0.), (1., -1.), (np.nan, 1.)])
def test_be_rejects_invalid_time_parameters(cfg, t_end, dt):
    with pytest.raises(ValueError):
        _be_run(cfg, t_end=t_end, dt=dt)


def _bdf_op(augmented=False):
    # Four physical entries; the optional cumulative flux may equal zero.
    return SimpleNamespace(N=1, augmented=augmented, rhs=lambda t, y: np.zeros_like(y))


def _fake_solution(y):
    return SimpleNamespace(success=True, y=y, t=np.array([0., 1.]),
                           t_events=None, message="", sol=lambda t: y[:, -1])


@pytest.mark.parametrize("index,value", [(0, 0.), (1, -0.1), (2, 0.), (3, -1.), (2, np.nan)])
def test_bdf_rejects_invalid_accepted_physical_states(cfg, monkeypatch, index, value):
    y0 = np.array([1., 1., 300., 300.])
    states = np.column_stack([y0, y0])
    states[index, 1] = value
    monkeypatch.setattr(BDF, "solve_ivp", lambda *a, **kw: _fake_solution(states))
    result = BDF.integrate_bdf(_bdf_op(), y0, 0., 1., cfg.bdf, breakpoints=())
    assert not result.ok
    assert "接受态" in result.message
    with pytest.raises(ValueError):
        result.eval([1.])


def test_bdf_zero_auxiliary_flux_is_valid(cfg):
    result = BDF.integrate_bdf(_bdf_op(True), np.array([1., 1., 300., 300., 0.]),
                               0., 1., cfg.bdf, breakpoints=())
    assert result.ok
    assert result.y_end[-1] == 0.


@pytest.mark.parametrize("t0,t_end", [(1., 1.), (2., 1.), (np.nan, 1.), (0., np.inf)])
def test_bdf_rejects_invalid_interval(cfg, t0, t_end):
    with pytest.raises(ValueError):
        BDF.integrate_bdf(_bdf_op(), np.array([1., 1., 300., 300.]),
                          t0, t_end, cfg.bdf, breakpoints=())


def test_bdf_eval_rejects_nonfinite_and_outside_interval(cfg):
    result = BDF.integrate_bdf(_bdf_op(), np.array([1., 1., 300., 300.]),
                               0., 1., cfg.bdf, breakpoints=())
    for t in (np.nan, np.inf, -1e-8, 1.+1e-8):
        with pytest.raises(ValueError):
            result.eval(t)


def test_bdf_deduplicates_breakpoints_and_honors_air_data_end(cfg, monkeypatch):
    calls = []
    actual = BDF.solve_ivp
    def capture(fun, span, y0, **kwargs):
        calls.append((span, kwargs["max_step"]))
        return actual(fun, span, y0, **kwargs)
    monkeypatch.setattr(BDF, "solve_ivp", capture)
    result = BDF.integrate_bdf(_bdf_op(), np.array([1., 1., 300., 300.]),
                               0., 3., cfg.bdf, breakpoints=(1., 1.), air_data_end=2.)
    assert result.ok
    assert [span for span, _ in calls] == [(0., 1.), (1., 2.), (2., 3.)]
    assert [step for _, step in calls] == [cfg.bdf["max_step_data_s"],
                                         cfg.bdf["max_step_data_s"],
                                         cfg.bdf["max_step_after_s"]]
