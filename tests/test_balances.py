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
    """V-4b 与 Σ Δt_k(f_{k-1}-f_k)/2 逐位一致，且 O(Δt) 下降（对照 E19）。"""
    op, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 200)
    r1 = V.mass_balances_be(op, y0, 1800.0, 1.0, cfg)
    assert r1["diff_v4b"] == pytest.approx(r1["theory_v4b"], rel=1e-9)
    assert r1["rel_v4b"] == pytest.approx(1.64264e-4, rel=1e-3)
    op2, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    r2 = V.mass_balances_be(op2, y0, 1800.0, 0.25, cfg)
    # 一阶：dt 减 4 倍，相对差约减 4 倍
    assert r1["rel_v4b"] / r2["rel_v4b"] == pytest.approx(4.0, rel=0.05)


def test_v4a_uses_real_substep_cumflux(cfg):
    """V-4a 用真实子步累计通量：ΔC̄ = −I_BE，且 n_flux_pts>基础步数（若发生减步）。"""
    op, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y0 = runners.initial_state(cfg, 200)
    r = V.mass_balances_be(op, y0, 1800.0, 1.0, cfg)
    assert abs(r["dCbar"] + r["I_BE"]) / abs(r["dCbar"]) < 1e-12
    assert r["n_flux_pts"] >= 1801           # 含初始点


def test_v4b_bdf_independent_quadrature(cfg):
    """V-4b(BDF)：独立连续自适应求积与 −ΔC̄ 一致，达 10×rtol 判据。"""
    r = V.flux_integral_bdf(cfg, question="q23", N=200, t_end_s=10800.0)
    assert r["ok"] and r["rel"] <= r["tol_10x_rtol"]


def test_v4c_fine_step_no_key_collision(cfg):
    """V-4c 细步 dt=0.25 不因 round(t) 键碰撞出错，残差小。"""
    op, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral")
    y0 = runners.initial_state(cfg, 200)
    re = V.energy_residual_be(op, y0, 100.0, 0.25, cfg)
    assert re["dt_actual"] == pytest.approx(0.25) and re["rel"] < 1e-8


def test_v5_envelope_fixed_and_moving(cfg):
    """V-5：接受解落在历史包络内（固定域 Q23 + Q4 动域）。"""
    ev = V.envelope_check(cfg, question="q23", N=200, t_end_s=8000.0)
    assert ev["ok"] and ev["worst_C_excess"]["exceed"] == 0.0
    ev4 = V.envelope_check(cfg, N=200, moving=True, t_end_s=3600.0)
    assert ev4["ok"]


def test_v4c_energy_residual_small(cfg):
    op, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral", decoupled=False)
    y0 = runners.initial_state(cfg, 200)
    re = V.energy_residual_be(op, y0, 100.0, 1.0, cfg)
    assert re["RE_W_per_m"] < 1e-6 and re["rel"] < 1e-8


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
