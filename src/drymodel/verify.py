"""verify.py —— 验证工具（V-1/V-2 起步；其余 V-3~V-13 在后续阶段补齐）。

对齐《A题_建模方案.md》§8.1、§8.5：
- `analytic_cylinder_robin`：圆柱第三类边界瞬态导热/扩散级数解（≥80 项）。
- `hi_precision_reference`：紧容差 BDF（rtol 1e-11、atol 1e-13）积分同一半离散 ODE，
  **称高精度参考解，非真正半离散精确解**（审计 A-F06）。
- V-1（热）/V-2（质）：空间阶（半离散 vs 级数解）与时间阶（BE Δt 序列 vs 高精度参考）分离。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import jv

from dataclasses import dataclass as _dataclass

from .grid import RadialGrid, RefGrid
from .operators import FVMOperator, jac_sparsity
from . import solver_be
from . import solver_bdf as _SBDF


# --------------------------------------------------------------------------
# 常物性 / 常环境（仅供解析验证）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstProps:
    rho_v: float
    cp_v: float
    k_v: float
    D_v: float

    def rho(self, C):
        C = np.asarray(C, float); return np.full_like(C, self.rho_v)

    def cp(self, C):
        C = np.asarray(C, float); return np.full_like(C, self.cp_v)

    def k(self, C):
        C = np.asarray(C, float); return np.full_like(C, self.k_v)

    def b(self, C):
        return self.rho(C) * self.cp(C)

    def D(self, C, T=None):
        C = np.asarray(C, float); return np.full_like(C, self.D_v)


@dataclass(frozen=True)
class ConstEnv:
    T_air_K_v: float
    C_env_v: float

    def T_air_K(self, t):
        t = np.asarray(t, float); out = np.full_like(t, self.T_air_K_v)
        return out if out.shape else float(out)

    def C_env(self, t):
        t = np.asarray(t, float); out = np.full_like(t, self.C_env_v)
        return out if out.shape else float(out)


# --------------------------------------------------------------------------
# 解析解
# --------------------------------------------------------------------------
def eigenvalues_robin(Bi: float, nterms: int = 80) -> np.ndarray:
    """求 λ J1(λ) = Bi J0(λ) 的前 nterms 个正根。"""
    def f(lam):
        return lam * jv(1, lam) - Bi * jv(0, lam)

    lam_max = nterms * np.pi + 10.0
    xs = np.arange(1e-6, lam_max, 0.005)
    fx = f(xs)
    roots = []
    sign = np.sign(fx)
    idx = np.where(np.diff(sign) != 0)[0]
    for i in idx:
        a, b = xs[i], xs[i + 1]
        try:
            r = brentq(f, a, b, xtol=1e-13, rtol=1e-14, maxiter=200)
            roots.append(r)
        except Exception:
            pass
        if len(roots) >= nterms:
            break
    return np.array(roots[:nterms])


def analytic_cylinder_robin(eta, Fo, Bi: float, nterms: int = 80) -> np.ndarray:
    """无量纲 θ(η, Fo)：θ=Σ c_n J0(λ_n η) exp(-λ_n² Fo)，c_n=2Bi/[(λ_n²+Bi²)J0(λ_n)]。"""
    eta = np.atleast_1d(np.asarray(eta, float))
    lam = eigenvalues_robin(Bi, nterms)
    J0lam = jv(0, lam)
    cn = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * J0lam)          # 长度 nterms
    # θ(η) = Σ cn J0(λ η) exp(-λ² Fo)
    J0 = jv(0, np.outer(eta, lam))                          # (n_eta, nterms)
    decay = np.exp(-(lam ** 2) * Fo)                        # 标量 Fo
    theta = (J0 * (cn * decay)).sum(axis=1)
    return theta


# --------------------------------------------------------------------------
# 高精度参考解（紧容差 BDF 积分半离散 ODE）
# --------------------------------------------------------------------------
def hi_precision_reference(op, y0, t_eval, rtol=1e-11, atol=1e-13):
    """紧容差 BDF 积分半离散 ODE，返回在 t_eval 处的状态（形状 (len(t_eval), 2(N+1))）。"""
    t_eval = np.atleast_1d(np.asarray(t_eval, float))
    n = op.N + 1
    atol_vec = np.full(2 * n, atol)
    S = jac_sparsity(op.N)
    sol = solve_ivp(op.rhs, (0.0, float(t_eval[-1])), y0, method="BDF",
                    rtol=rtol, atol=atol_vec, jac_sparsity=S, dense_output=True,
                    max_step=np.inf)
    if not sol.success:
        raise RuntimeError(f"高精度参考解失败：{sol.message}")
    return np.array([sol.sol(t) for t in t_eval])


# --------------------------------------------------------------------------
# V-1（热）/ V-2（质）
# --------------------------------------------------------------------------
def _make_const_op(N, R0, props, env, h, hm):
    grid = RadialGrid(N, R0)
    return FVMOperator(grid, props, env, h=h, hm=hm, R0=R0,
                       interface="harmonic", decoupled=True)


def run_V1_heat(R0=0.02, T0_C=28.0, Tair_C=50.0, h=25.0,
                Ns=(200, 400, 800), t_probe=100.0, nterms=80):
    """V-1 空间阶：常物性附录 2 热问题，T_air≡50°C。返回各 N 表面点误差 (°C)。"""
    rho, cp, k = 820.0, 2600.0, 0.36
    alpha = k / (rho * cp)
    Bi = h * R0 / k
    T0, Tair = T0_C + 273.15, Tair_C + 273.15
    props = ConstProps(rho, cp, k, D_v=7e-9 * np.exp(-0.89 / 2.55))
    env = ConstEnv(Tair, 0.05)
    Fo = alpha * t_probe / R0 ** 2

    results = {}
    for N in Ns:
        op = _make_const_op(N, R0, props, env, h, hm=8e-7)
        y0 = np.concatenate([np.full(N + 1, 2.55), np.full(N + 1, T0)])
        ref = hi_precision_reference(op, y0, [t_probe])[0]
        T_ref = ref[N + 1:]                      # 半离散（时间近精确）
        # 解析解
        eta = op.grid.r / R0
        theta = analytic_cylinder_robin(eta, Fo, Bi, nterms)
        T_ana = Tair + (T0 - Tair) * theta
        err = np.abs(T_ref - T_ana)
        results[N] = {"surface_err_C": float(err[-1]),
                      "max_err_C": float(np.max(err))}
    return {"Bi": Bi, "alpha": alpha, "Fo": Fo, "t_probe": t_probe, "by_N": results}


def run_V2_mass(R0=0.02, C0=2.55, Cenv=0.05, hm=8e-7,
                Ns=(200, 400, 800), t_probe=100.0, nterms=80):
    """V-2 空间阶：D≡D(C0) 常数、C_env≡0.05。返回各 N 表面点误差 (kg/kg)。"""
    D0 = 7e-9 * np.exp(-0.89 / C0)               # 附录 2 D(C0)
    Bi_m = hm * R0 / D0
    props = ConstProps(820.0, 2600.0, 0.36, D_v=D0)
    env = ConstEnv(50.0 + 273.15, Cenv)
    Fo = D0 * t_probe / R0 ** 2

    results = {}
    for N in Ns:
        op = _make_const_op(N, R0, props, env, h=25.0, hm=hm)
        y0 = np.concatenate([np.full(N + 1, C0), np.full(N + 1, 50.0 + 273.15)])
        ref = hi_precision_reference(op, y0, [t_probe])[0]
        C_ref = ref[:N + 1]
        eta = op.grid.r / R0
        theta = analytic_cylinder_robin(eta, Fo, Bi_m, nterms)
        C_ana = Cenv + (C0 - Cenv) * theta
        err = np.abs(C_ref - C_ana)
        results[N] = {"surface_err": float(err[-1]), "max_err": float(np.max(err))}
    return {"Bi_m": Bi_m, "D0": D0, "Fo": Fo, "t_probe": t_probe, "by_N": results}


def order_from_errors(errs):
    """由相邻网格误差估计收敛阶（假设网格每次加倍）。"""
    errs = list(errs)
    orders = []
    for i in range(len(errs) - 1):
        if errs[i + 1] > 0:
            orders.append(float(np.log2(errs[i] / errs[i + 1])))
    return orders


# --------------------------------------------------------------------------
# V-4a/b：离散代数收支 与 独立连续通量积分（BE，固定域）
# --------------------------------------------------------------------------
def mass_balances_be(op, y0, t_end, dt, cfg):
    """V-4a：C̄_N-C̄_0 = -Δt Σ f_n（f 取步末）；V-4b：梯形积分 vs ΔC̄。

    f(t) = (2 h_m / R0)(C_N - C_env)。返回残差与理论差 (Δt/2)(f_0-f_N)。
    """
    recs = []

    def rec(t, C, T):
        f = (2.0 * op.hm / op.R0) * (C[-1] - float(op.env.C_env(t)))
        recs.append((float(t), op.grid.cbar(C), float(f)))

    solver_be.integrate_be(op, y0, t_end, dt, C0_ref=cfg.C0, picard=cfg.picard,
                           retry=cfg.retry, record_times={int(t_end)}, scalar_recorder=rec)
    ts = np.array([r[0] for r in recs])
    cbar = np.array([r[1] for r in recs])
    f = np.array([r[2] for r in recs])

    dCbar = cbar[-1] - cbar[0]
    f_steps = f[1:]                                    # f_n（步末，n=1..Nsteps）
    balance_rhs = -dt * np.sum(f_steps)                # V-4a
    res_v4a = abs(dCbar - balance_rhs)
    rel_v4a = res_v4a / max(abs(dCbar), 1e-300)

    I_trap = dt * (0.5 * f[0] + np.sum(f[1:-1]) + 0.5 * f[-1])
    diff_v4b = I_trap - (-dCbar)
    theory_v4b = 0.5 * dt * (f[0] - f[-1])             # (Δt/2)(f_0-f_N)
    rel_v4b = abs(diff_v4b) / max(abs(dCbar), 1e-300)

    return {
        "dCbar": float(dCbar), "rel_v4a": float(rel_v4a),
        "rel_v4b": float(rel_v4b), "diff_v4b": float(diff_v4b),
        "theory_v4b": float(theory_v4b), "f0": float(f[0]), "fN": float(f[-1]),
    }


def energy_residual_be(op, y0, t_probe, dt, cfg):
    """V-4c（W/m 口径）：R_E^(ℓ)=2π∫_0^R b(C)∂_tT r dr − 2πR h[T_air−T_s]（固定域）。

    用 BE 差商近似 ∂_tT；返回绝对/相对残差与参考尺度。
    """
    recorded = {}

    def rec(t, C, T):
        recorded[round(t)] = (C.copy(), T.copy())

    solver_be.integrate_be(op, y0, t_probe, dt, C0_ref=cfg.C0, picard=cfg.picard,
                           retry=cfg.retry,
                           record_times={int(t_probe), int(t_probe) - int(dt)},
                           scalar_recorder=rec)
    tp = int(round(t_probe))
    C1, T1 = recorded[tp]
    C0a, T0a = recorded[tp - int(dt)]
    dTdt = (T1 - T0a) / dt

    R0 = op.R0
    b = op.props.b(C1)
    # 2π ∫ b ∂_tT r dr ≈ 2π Σ V_i b_i dTdt_i（V_i 已含 r 度量，单位长度弧度积分 → ×2π）
    integral = 2.0 * np.pi * np.sum(op.V * b * dTdt)
    Tair = float(op.env.T_air_K(t_probe))
    surface = 2.0 * np.pi * R0 * op.h * (Tair - T1[-1])
    RE = integral - surface
    ref_scale = 2.0 * np.pi * R0 * op.h * max(abs(Tair - (cfg.T0_K)), 1.0)
    return {"RE_W_per_m": float(RE), "rel": float(abs(RE) / max(ref_scale, 1e-300)),
            "ref_scale": float(ref_scale)}


# --------------------------------------------------------------------------
# V-10：静态极限（Q4 动域求解器 R≡R0 vs 固定域求解器）
# --------------------------------------------------------------------------
@_dataclass(frozen=True)
class _ConstRadius:
    R0: float

    def R(self, t):
        t = np.asarray(t, float)
        out = np.full_like(t, self.R0)
        return out if out.shape else float(out)

    def is_extrapolated(self, t):
        return False


def static_limit_q4(cfg, *, N=200, interface="integral", t_probe=3600.0):
    """R≡R0、附录 4、同一初边值：动域求解器 vs 固定域求解器逐点比较（V-10）。"""
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4")
    R0 = cfg.R0

    # 动域（参考坐标），R≡R0
    op_mov = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=R0,
                         interface=interface, integral_npts=8,
                         radius_fn=_ConstRadius(R0))
    # 固定域
    op_fix = FVMOperator(RadialGrid(N, R0), props, env, h=cfg.h, hm=cfg.hm, R0=R0,
                         interface=interface, integral_npts=8)

    y0 = np.concatenate([np.full(N + 1, cfg.C0), np.full(N + 1, cfg.T0_K)])
    rm = _SBDF.integrate_bdf(op_mov, y0, 0.0, t_probe, cfg.bdf, breakpoints=(14400.0,))
    rf = _SBDF.integrate_bdf(op_fix, y0, 0.0, t_probe, cfg.bdf, breakpoints=(14400.0,))
    ym = rm.eval([t_probe])[0]
    yf = rf.eval([t_probe])[0]
    denom = np.maximum(np.abs(yf), 1e-12)
    rel = float(np.max(np.abs(ym - yf) / denom))
    absdiff = float(np.max(np.abs(ym - yf)))
    return {"rel_max": rel, "abs_max": absdiff, "t_probe": t_probe, "N": N}
