"""Numerical equivalence checks for the round-2 integral-interface optimization.

No timing assertions: machine performance is recorded by the benchmark script.
Independent adaptive quadrature checks the appendix formulas; an old-method
trajectory comparison catches accumulated changes in the coupled equations.
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from drymodel import config, props as P, runners
from drymodel.solver_bdf import integrate_bdf


def _integral_before(Ci, Cj, Tb, Dfun, npts=8):
    Ci, Cj, Tb = [np.asarray(value, dtype=float) for value in (Ci, Cj, Tb)]
    xi, wt = P._gauss01(npts)
    total = np.zeros(np.broadcast(Ci, Cj, Tb).shape)
    for x, w in zip(xi, wt):
        total = total + w * Dfun((1-x)*Ci + x*Cj, Tb)
    return total


@pytest.mark.parametrize("prop", [P.PropsQ1(), P.PropsQ23(), P.PropsQ4()])
@pytest.mark.parametrize("npts", [8, 16])
def test_integral_matches_independent_adaptive_quadrature(prop, npts):
    ci = np.array([.05, .8, 2.4])
    cj = np.array([.052, .82, 2.55])
    tb = np.array([301.15, 313.15, 323.15])
    expected = []
    for a, b, temp in zip(ci, cj, tb):
        # Do not reuse production D or Gauss nodes in the reference calculation.
        gamma = getattr(prop, "D_gamma", 0.)
        value, _ = quad(lambda x: math.exp(-prop.D_beta / ((1-x)*a + x*b)
                                           - gamma/temp), 0., 1., epsabs=1e-14, epsrel=1e-13)
        expected.append(prop.D_pref * value)
    np.testing.assert_allclose(P.D_face_integral(ci, cj, tb, prop.D, npts),
                               expected, rtol=5e-13, atol=0.)


@pytest.mark.parametrize("prop", [P.PropsQ1(), P.PropsQ23(), P.PropsQ4()])
@pytest.mark.parametrize("npts", [8, 16])
def test_integral_preserves_multiaxis_broadcast_and_scalar(prop, npts):
    ci = np.array([.05, .8, 2.4])[:, None]
    cj = np.array([.07, .4, 1., 2.55])[None, :]
    tb = np.linspace(301.15, 323.15, 24).reshape(2, 3, 4)
    result = P.D_face_integral(ci, cj, tb, prop.D, npts)
    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(result, _integral_before(ci, cj, tb, prop.D, npts),
                               rtol=3e-15, atol=0.)
    scalar = P.D_face_integral(.5, .8, 310., prop.D, npts)
    assert np.shape(scalar) == ()
    np.testing.assert_allclose(scalar, _integral_before(.5, .8, 310., prop.D, npts),
                               rtol=3e-15, atol=0.)
    # Q1 ignores T, but its output must still include dimensions introduced by T.
    temp_only_shape = P.D_face_integral(.5, .8, tb, prop.D, npts)
    assert temp_only_shape.shape == tb.shape
    np.testing.assert_allclose(temp_only_shape, _integral_before(.5, .8, tb, prop.D, npts),
                               rtol=3e-15, atol=0.)


@pytest.mark.parametrize("prop", [P.PropsQ1(), P.PropsQ23(), P.PropsQ4()])
@pytest.mark.parametrize("npts", [8, 16])
def test_integral_preserves_zero_guard_small_C_and_equal_limit(prop, npts):
    c = np.array([-1., 0., 1e-300, 1e-4, .05, .8, 2.55])
    temp = np.full_like(c, 313.15)
    with np.errstate(all="raise"):
        result = P.D_face_integral(c, c, temp, prop.D, npts)
    assert np.array_equal(result[:4], np.zeros(4))
    assert np.all(result[4:] > 0)
    np.testing.assert_allclose(result, prop.D(c, temp), rtol=3e-15, atol=0.)
    close = P.D_face_integral(c[4:], np.nextafter(c[4:], np.inf), temp[4:], prop.D, npts)
    np.testing.assert_allclose(close, result[4:], rtol=5e-15, atol=0.)
    assert P.D_face_integral(np.empty((0, 1)), np.empty((0, 3)), 310., prop.D, npts).shape == (0, 3)


@pytest.mark.parametrize("question", ["q1", "q23", "q4"])
def test_old_and_new_quadrature_give_same_coupled_trajectory(question, monkeypatch):
    cfg = config.load_config()
    n = 40
    if question == "q4":
        op, _, _ = runners.build_ref_operator(cfg, question, n, integral_npts=16)
    else:
        op, _ = runners.build_fixed_operator(cfg, question, n, integral_npts=16)
    y0 = runners.initial_state(cfg, n)
    controls = dict(cfg.bdf, rtol=1e-10, atol_C=1e-12, atol_T_K=1e-10)
    current = integrate_bdf(op, y0, 0., 300., controls, breakpoints=(150.,))
    with monkeypatch.context() as context:
        context.setattr(P, "D_face_integral", _integral_before)
        previous = integrate_bdf(op, y0, 0., 300., controls, breakpoints=(150.,))
    assert current.ok and previous.ok
    # Include segment endpoints, repetitions, arbitrary order and a scalar sample.
    times = np.array([300., 10., 150., 0., 75., 150., 210.])
    before, after = previous.eval(times), current.eval(times)
    np.testing.assert_allclose(after[:, :n+1], before[:, :n+1], rtol=0., atol=1e-12)
    np.testing.assert_allclose(after[:, n+1:], before[:, n+1:], rtol=0., atol=1e-10)
    np.testing.assert_array_equal(current.eval(150.)[0], after[2])
    assert current.eval([]).shape == (0, 2*(n+1))
    assert np.all(after > 0.)
