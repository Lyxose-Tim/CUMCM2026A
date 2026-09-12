"""operators 单元测试：离散守恒恒等式、稳态、参考坐标一致性。"""
import numpy as np
import pytest

from drymodel import config as cfgmod, data_io
from drymodel.grid import RadialGrid, RefGrid
from drymodel.operators import FVMOperator, jac_sparsity


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_discrete_mass_balance_fixed(cfg):
    """ΣV_i Ċ_i = -R0·h_m·(C_N - C_env)（面通量精确抵消）。"""
    N = 100
    op, env = _fixed(cfg, "q23", N)
    rng = np.random.default_rng(0)
    C = 2.0 + 0.4 * rng.standard_normal(N + 1)
    T = 301.15 + 5.0 * rng.standard_normal(N + 1)
    t = 500.0
    dydt = op.rhs(t, op.merge(C, T))
    dC = dydt[:N + 1]
    lhs = np.dot(op.V, dC)
    rhs_expected = -cfg.R0 * cfg.hm * (C[-1] - env.C_env(t))
    assert lhs == pytest.approx(rhs_expected, rel=1e-12, abs=1e-18)


def test_discrete_heat_balance_fixed(cfg):
    """Σ b_i V_i Ṫ_i = -R0·h·(T_N - T_air)。"""
    N = 100
    op, env = _fixed(cfg, "q23", N)
    rng = np.random.default_rng(1)
    C = 2.0 + 0.4 * rng.standard_normal(N + 1)
    T = 301.15 + 5.0 * rng.standard_normal(N + 1)
    t = 500.0
    dydt = op.rhs(t, op.merge(C, T))
    dT = dydt[N + 1:]
    b = op.props.b(C)
    lhs = np.dot(b * op.V, dT)
    rhs_expected = -cfg.R0 * cfg.h * (T[-1] - env.T_air_K(t))
    assert lhs == pytest.approx(rhs_expected, rel=1e-12, abs=1e-15)


def test_equilibrium_zero_rhs(cfg):
    """C≡C_env、T≡T_air 时右端项为 0（无源稳态）。"""
    N = 60
    op, env = _fixed(cfg, "q23", N)
    t = 500.0
    C = np.full(N + 1, float(env.C_env(t)))
    T = np.full(N + 1, float(env.T_air_K(t)))
    dydt = op.rhs(t, op.merge(C, T))
    assert np.max(np.abs(dydt)) < 1e-20


def test_reference_mass_balance(cfg):
    """参考坐标：ΣV_i ċ_i = -(1/R)·h_m·(c_N - C_env)。"""
    N = 80
    grid = RefGrid(N)
    props = cfg.props("q4")
    env = data_io.make_env_functions(cfg, "base")
    radius = data_io.make_radius_function(cfg)
    op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                     interface="harmonic", radius_fn=radius)
    rng = np.random.default_rng(2)
    c = 2.0 + 0.3 * rng.standard_normal(N + 1)
    T = 301.15 + 5.0 * rng.standard_normal(N + 1)
    t = 3600.0
    dydt = op.rhs(t, op.merge(c, T))
    dc = dydt[:N + 1]
    lhs = np.dot(op.V, dc)
    R = float(radius.R(t))
    rhs_expected = -(cfg.hm / R) * (c[-1] - env.C_env(t))
    assert lhs == pytest.approx(rhs_expected, rel=1e-11, abs=1e-15)


def test_jac_sparsity_shape(cfg):
    S = jac_sparsity(50)
    assert S.shape == (102, 102)
    # 每行非零数：块三对角(≤3) + 交叉块(≤3) = ≤6
    assert S.getnnz() <= 6 * 102


def _fixed(cfg, question, N):
    grid = RadialGrid(N, cfg.R0)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, "base")
    op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                     interface="harmonic")
    return op, env
