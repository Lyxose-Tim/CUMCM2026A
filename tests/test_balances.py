"""V-4a/b/c 收支与能量残差单元测试（对照探针 E19）。"""
import numpy as np
import pytest

from drymodel import config as cfgmod, runners, verify as V
from drymodel import solver_be as SBE


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


def test_v4c_energy_residual_uses_real_half_step(cfg):
    op, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral")
    y0 = runners.initial_state(cfg, 200)
    re = V.energy_residual_be(op, y0, 100.0, 0.5, cfg)
    assert re["status"] == "pass"
    assert re["actual_dt_s"] == pytest.approx(0.5)
    assert abs(re["RE_W_per_m"]) < 1e-6
    assert re["rel"] < 1e-8


def test_v4a_uses_actual_accepted_substep_weights(cfg, monkeypatch):
    op, _ = runners.build_fixed_operator(cfg, "q1", 40, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 40)
    original = SBE.step_be

    def force_first_base_step_to_split(op_, y_old, t_old, dt, **kwargs):
        if t_old == 0.0 and dt == 1.0:
            C, _ = op_.split(y_old)
            return y_old.copy(), False, 1, float(np.min(C))
        return original(op_, y_old, t_old, dt, **kwargs)

    monkeypatch.setattr(SBE, "step_be", force_first_base_step_to_split)
    result = V.mass_balances_be(op, y0, 10.0, 1.0, cfg)
    assert result["v4a"]["status"] == "pass"
    assert result["rel_v4a"] < 1e-12
    assert result["solver"].accepted_substeps == 11
    assert result["solver"].diagnostics["accepted_step_sizes"][:2] == [0.5, 0.5]


def test_v4a_propagates_total_solver_failure(cfg, monkeypatch):
    op, _ = runners.build_fixed_operator(cfg, "q1", 40, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 40)

    def always_fail(op_, y_old, t_old, dt, **kwargs):
        C, _ = op_.split(y_old)
        return y_old.copy(), False, 1, float(np.min(C))

    monkeypatch.setattr(SBE, "step_be", always_fail)
    result = V.mass_balances_be(op, y0, 10.0, 1.0, cfg)
    assert result["status"] == "fail"
    assert result["v4a"]["status"] == "fail"
    assert result["v4b"]["status"] == "fail"
    assert np.isnan(result["rel_v4a"])
    assert result["solver"].t_end == 0.0


def test_v4a_moving_domain_uses_radius_factor(cfg):
    op, _, _ = runners.build_ref_operator(cfg, "q4", 40, interface="integral")
    y0 = runners.initial_state(cfg, 40)
    result = V.mass_balances_be(op, y0, 10.0, 1.0, cfg)
    assert result["v4a"]["status"] == "pass"
    assert result["rel_v4a"] < 1e-12


def test_v4a_bdf_augmented_fixed(cfg):
    """V-4a（BDF 增广态）：C̄(t)−C̄(0)+I(t)≈0（固定域）。"""
    r = V.mass_balance_bdf(cfg, question="q23", N=200, interface="integral", t_end_s=3600.0)
    assert r["max_abs_resid"] < 1e-10


def test_v4a_bdf_augmented_moving(cfg):
    """V-4a（BDF 增广态）：Q4 动域 R(t) 含 1/R 因子。"""
    r = V.mass_balance_bdf(cfg, N=200, interface="integral", t_end_s=3600.0, moving=True)
    assert r["max_abs_resid"] < 1e-10


def test_augmented_jac_sparsity_dim():
    """增广态雅可比稀疏模式维数 2n+1，辅助行依赖表面 C。"""
    from drymodel.operators import jac_sparsity
    N = 30
    S = jac_sparsity(N, augmented=True)
    assert S.shape == (2 * (N + 1) + 1, 2 * (N + 1) + 1)
    n = N + 1
    assert S[2 * n, N] == 1          # İ 依赖表面 C 节点
    assert S[2 * n, n] == 0          # 不依赖 T
