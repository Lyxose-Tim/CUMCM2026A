"""Candidate tables must share the event trajectory and fail closed."""
from types import SimpleNamespace

import numpy as np
import pytest

from drymodel import runners, production, config
from drymodel.criterion import Detection
from drymodel.grid import RadialGrid, RefGrid


@pytest.fixture(scope="module")
def cfg():
    return config.load_config()


def detection(sample=30060.0):
    return Detection(30000.0, 30000.0, sample, None, float("nan"),
                     float("nan"), 0, True, 0.15)


@pytest.mark.parametrize("question", ["q23", "q4"])
def test_candidate_reuses_event_and_continuation(cfg, monkeypatch, question):
    n = 21
    event, continuation = object(), object()
    grid = RadialGrid(20, cfg.R0) if question == "q23" else RefGrid(20)
    op = SimpleNamespace(N=20, grid=grid)
    calls = []

    def evaluate(times):
        times = np.asarray(times)
        c = np.repeat((0.15 - (times-30000.0)*1e-6)[:, None], n, axis=1)
        return np.hstack([c, np.full_like(c, 320.0)])

    full = SimpleNamespace(t_end=30600.0, eval=evaluate)
    monkeypatch.setattr(runners, "q23_detect", lambda *a, **k:
                        (detection(), event, continuation, op))

    def merge(a, b):
        assert a is event and b is continuation
        calls.append(True)
        return full

    monkeypatch.setattr(runners.SBDF, "merge_bdf", merge)

    def forbidden(*a, **k):
        pytest.fail("candidate must not integrate a second trajectory")

    monkeypatch.setattr(runners, "_integrate_full", forbidden)
    result = getattr(runners, question + "_candidate")(cfg, N=20, save=False)
    assert calls == [True]
    assert result["t_sample_s"] == 30060.0
    table = np.asarray(result["table5" if question == "q23" else "table6"])
    assert table[-1, 1] == pytest.approx(0.15)


@pytest.mark.parametrize("question", ["q23", "q4"])
def test_candidate_rejects_missing_minute_sample(cfg, monkeypatch, question):
    monkeypatch.setattr(runners, "q23_detect", lambda *a, **k:
                        (detection(None), None, None, SimpleNamespace(N=20)))
    with pytest.raises(RuntimeError, match="60 s"):
        getattr(runners, question + "_candidate")(cfg, N=20, save=False)


def test_detection_rejects_failed_continuation(cfg, monkeypatch):
    event = SimpleNamespace(ok=True, t_cross=30000.0,
                            y_cross=np.r_[np.full(21, 0.15), np.full(21, 320.0)])
    monkeypatch.setattr(runners.SBDF, "integrate_bdf", lambda *a, **k: event)
    monkeypatch.setattr(runners.SBDF, "continue_bdf", lambda *a, **k:
                        SimpleNamespace(ok=False, message="intentional failure"))
    with pytest.raises(RuntimeError, match="事件后续算失败"):
        runners.q23_detect(cfg, N=20)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -0.1])
def test_production_rejects_invalid_final_cmax(bad):
    with pytest.raises(RuntimeError, match="严格合格末行"):
        production.gate_production("Q23", detection(), bad, 0.15)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.16])
def test_production_checks_post_max_not_only_flag(bad):
    det = detection()
    det.post_max_cmax = bad
    with pytest.raises(RuntimeError, match="续算回穿"):
        production.gate_production("Q4", det, 0.149, 0.15)
