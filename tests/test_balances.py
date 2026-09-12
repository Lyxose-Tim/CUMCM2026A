"""V-4a/b/c 收支与能量残差单元测试（对照探针 E19）。"""
import pytest

from drymodel import config as cfgmod, runners, verify as V


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_v4a_discrete_balance(cfg):
    op, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 200)
    r = V.mass_balances_be(op, y0, 1800.0, 1.0, cfg)
    assert r["rel_v4a"] < 1e-12          # 离散代数收支精确


def test_v4b_flux_integral_matches_theory(cfg):
    """V-4b 与 (Δt/2)(f_0-f_N) 逐位一致，且 O(Δt) 下降。"""
    op, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 200)
    r1 = V.mass_balances_be(op, y0, 1800.0, 1.0, cfg)
    assert r1["diff_v4b"] == pytest.approx(r1["theory_v4b"], rel=1e-9)
    assert r1["rel_v4b"] == pytest.approx(1.64264e-4, rel=1e-3)
    op2, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    r2 = V.mass_balances_be(op2, y0, 1800.0, 0.25, cfg)
    # 一阶：dt 减 4 倍，相对差约减 4 倍
    assert r1["rel_v4b"] / r2["rel_v4b"] == pytest.approx(4.0, rel=0.05)


def test_v4c_energy_residual_small(cfg):
    op, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral", decoupled=False)
    y0 = runners.initial_state(cfg, 200)
    re = V.energy_residual_be(op, y0, 100.0, 1.0, cfg)
    assert re["RE_W_per_m"] < 1e-6 and re["rel"] < 1e-8
