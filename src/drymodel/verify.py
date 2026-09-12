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
    atol_vec = np.full(op.n_state, atol)
    S = jac_sparsity(op.N, getattr(op, "augmented", False))
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
    """BE 收支（**离散代数恒等式**，非独立时间精度证明；时间精度由 V-3 独立核验）。

    V-4a：C̄_N−C̄_0 = −I_BE，I_BE=Σ Δt_sub·f(step-end)（真实子步权重，W4）。
    V-4b：独立**实际步点**梯形求积 I_trap vs I_BE，变步长理论差 Σ Δt_k(f_{k-1}−f_k)/2。
    f(t)=(2 h_m/R)(C_N−C_env)。求解失败显式抛错。
    """
    cbar0 = op.grid.cbar(op.split(y0)[0])
    flux_pts = []                          # [(t, dt_sub, f), ...]（含初始点 dt=0）

    def frec(t, dt_sub, f):
        flux_pts.append((float(t), float(dt_sub), float(f)))

    res = solver_be.integrate_be(op, y0, t_end, dt, C0_ref=cfg.C0, picard=cfg.picard,
                                 retry=cfg.retry, record_times={int(t_end)},
                                 flux_recorder=frec)
    if not res.ok:
        raise RuntimeError(f"V-4 BE 积分失败：{res.message}")

    cbarN = op.grid.cbar(res.C[-1])
    dCbar = cbarN - cbar0
    I_BE = res.cum_flux                    # = Σ Δt_sub·f(step-end)

    # V-4a：ΔC̄ = −I_BE（离散代数恒等式）
    rel_v4a = abs(dCbar + I_BE) / max(abs(dCbar), 1e-300)

    # V-4b：实际步点梯形（右端点=BE），变步长
    tarr = np.array([p[0] for p in flux_pts])
    dtarr = np.array([p[1] for p in flux_pts])          # dtarr[0]=0（初始点）
    farr = np.array([p[2] for p in flux_pts])
    I_trap = np.sum(dtarr[1:] * 0.5 * (farr[:-1] + farr[1:]))
    theory_v4b = np.sum(dtarr[1:] * 0.5 * (farr[:-1] - farr[1:]))   # Σ Δt_k(f_{k-1}-f_k)/2
    diff_v4b = I_trap - I_BE
    rel_v4b = abs(diff_v4b) / max(abs(dCbar), 1e-300)

    return {
        "dCbar": float(dCbar), "I_BE": float(I_BE), "rel_v4a": float(rel_v4a),
        "rel_v4b": float(rel_v4b), "diff_v4b": float(diff_v4b),
        "theory_v4b": float(theory_v4b), "f0": float(farr[0]), "fN": float(farr[-1]),
        "n_flux_pts": int(len(flux_pts)), "total_halvings": int(res.total_halvings),
    }


def mass_balance_bdf(cfg, *, question="q23", N=200, interface="integral",
                     t_end_s=3600.0, moving=False):
    """V-4a（BDF）：用累计通量增广态 I 检验归一化收支 C̄(t)−C̄(0)+I(t)≈0。

    y=[C..,T..,I]，I(0)=0，İ=(2 h_m/R)(C_N−C_env)。固定域 R=R0；moving=True 用 Q4 动域 R(t)。
    """
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4") if moving else cfg.props(question)
    if moving:
        grid = RefGrid(N)
        radius = data_io.make_radius_function(cfg)
        op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, integral_npts=8, radius_fn=radius, augmented=True)
    else:
        grid = RadialGrid(N, cfg.R0)
        op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, integral_npts=8, augmented=True)
    n = N + 1
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K), [0.0]])
    res = _SBDF.integrate_bdf(op, y0, 0.0, t_end_s, cfg.bdf, breakpoints=(14400.0,))
    if not res.ok:
        raise RuntimeError(res.message)
    cbar0 = op.grid.cbar(y0[:n])
    worst = 0.0
    for t in np.linspace(0.0, t_end_s, 13)[1:]:
        y = res.eval([t])[0]
        resid = op.grid.cbar(y[:n]) - cbar0 + y[-1]      # 归一化收支，应 ~0
        worst = max(worst, abs(resid))
    return {"max_abs_resid": float(worst), "rel": float(worst / max(abs(cbar0), 1e-300)),
            "cbar0": float(cbar0), "moving": moving, "N": N,
            "note": "同一 RHS 的离散代数恒等式（C̄−C̄0+I=0），非独立时间精度证明；时间精度由 V-3 独立核验"}


def flux_integral_bdf(cfg, *, question="q23", N=200, interface="integral",
                      t_end_s=3600.0, moving=False):
    """V-4b（BDF）：**独立连续通量自适应求积**（非同一 RHS 恒等式）。

    对连续解独立求积 ∫f dt（f=(2h_m/R)(C_N−C_env)），scipy.integrate.quad 自适应，
    在断点处分段（points），与 −ΔC̄ 比较，判据 10×rtol（不静默放宽）；rel_refine 为
    进一步收紧 epsrel 后的复核差。与 mass_balance_bdf（代数恒等式）互补。
    """
    from scipy.integrate import quad
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4") if moving else cfg.props(question)
    if moving:
        radius = data_io.make_radius_function(cfg)
        op = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, radius_fn=radius)
    else:
        op = FVMOperator(RadialGrid(N, cfg.R0), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface)
    n = N + 1
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
    res = _SBDF.integrate_bdf(op, y0, 0.0, t_end_s, cfg.bdf, breakpoints=(14400.0,))
    if not res.ok:
        raise RuntimeError(res.message)
    cbar0 = op.grid.cbar(y0[:n])
    dCbar = op.grid.cbar(res.eval([t_end_s])[0][:n]) - cbar0

    def fval(t):
        return op.flux_cbar(t, res.eval([t])[0][:n])

    def I_adaptive(epsrel):
        import warnings
        from scipy.integrate import IntegrationWarning
        # 断点 14400 s 在区间内则分段积分（早期陡变段单独自适应）
        bpts = [14400.0] if 0.0 < 14400.0 < t_end_s else []
        edges = [0.0] + bpts + [t_end_s]
        tot = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            for a, b in zip(edges[:-1], edges[1:]):
                val, _ = quad(fval, a, b, epsabs=1e-12, epsrel=epsrel, limit=200)
                tot += val
        return tot

    Iq = I_adaptive(1e-9)
    Iq2 = I_adaptive(1e-11)                       # 收紧 epsrel 复核（提高求积精度）
    rel = abs(Iq - (-dCbar)) / max(abs(dCbar), 1e-300)
    rel_refine = abs(Iq2 - Iq) / max(abs(dCbar), 1e-300)
    tol = 10.0 * float(cfg.bdf["rtol"])           # 10×rtol
    return {"rel": float(rel), "rel_refine": float(rel_refine), "tol_10x_rtol": float(tol),
            "ok": bool(rel <= tol), "I_quad": float(Iq), "dCbar": float(dCbar),
            "moving": moving, "N": N,
            "note": "独立连续通量自适应求积（非同一 RHS 恒等式）；判据 10×rtol"}


def energy_residual_be(op, y0, t_probe, dt, cfg):
    """V-4c（W/m 口径）：R_E^(ℓ)=2π∫_0^R b(C)∂_tT r dr − 2πR h[T_air−T_s]（固定域）。

    用 BE 实际相邻两步差商近似 ∂_tT（细步安全，键用实际 t）。**检离散代数平衡，
    非独立时间精度**（时间精度由 V-3 独立核验）。
    """
    steps = []

    def rec(t, C, T):
        steps.append((float(t), C.copy(), T.copy()))

    res = solver_be.integrate_be(op, y0, t_probe, dt, C0_ref=cfg.C0, picard=cfg.picard,
                                 retry=cfg.retry, record_times={int(t_probe)},
                                 scalar_recorder=rec)
    if not res.ok:
        raise RuntimeError(f"V-4c BE 积分失败：{res.message}")
    # 实际相邻两步（bracket t_probe）
    t1, C1, T1 = steps[-1]
    t0, C0a, T0a = steps[-2]
    dt_actual = t1 - t0
    dTdt = (T1 - T0a) / dt_actual

    R0 = op.R0
    b = op.props.b(C1)
    integral = 2.0 * np.pi * np.sum(op.V * b * dTdt)   # 2π Σ V_i b_i dTdt_i（W/m）
    Tair = float(op.env.T_air_K(t1))
    surface = 2.0 * np.pi * R0 * op.h * (Tair - T1[-1])
    RE = integral - surface
    ref_scale = 2.0 * np.pi * R0 * op.h * max(abs(Tair - cfg.T0_K), 1.0)   # W/m（无 L）
    return {"RE_W_per_m": float(RE), "rel": float(abs(RE) / max(ref_scale, 1e-300)),
            "ref_scale": float(ref_scale), "t_probe": float(t1), "dt_actual": float(dt_actual),
            "note": "离散代数平衡（W/m），非独立时间精度证明"}


def envelope_check(cfg, *, question="q23", N=200, interface="integral",
                   t_end_s=None, moving=False, n_sample=25):
    """V-5：接受解/重构值须落在历史包络内（H17/H18）。

    C_lo(t)=min(C0, inf_{s≤t}C_env), C_hi(t)=max(C0, sup C_env)；T 同理用 T0/T_air。
    容差取 numerics.envelope_tol；越界超容差→报告最大越界及其时空位置（不裁剪）。
    """
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4") if moving else cfg.props(question)
    if moving:
        radius = data_io.make_radius_function(cfg)
        op = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, radius_fn=radius)
    else:
        op = FVMOperator(RadialGrid(N, cfg.R0), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface)
    n = N + 1
    tol_C = float(cfg.envelope_tol["C"])
    tol_T = float(cfg.envelope_tol["T_K"])
    if t_end_s is None:
        t_end_s = 20000.0 if not moving else 3600.0
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
    res = _SBDF.integrate_bdf(op, y0, 0.0, t_end_s, cfg.bdf, breakpoints=(14400.0,))
    if not res.ok:
        raise RuntimeError(res.message)

    ts = np.linspace(0.0, t_end_s, n_sample)
    # 历史包络（在密集网格上取 inf/sup 的边界历史）
    dense = np.linspace(0.0, t_end_s, 4001)
    Cenv_d = np.array([env.C_env(s) for s in dense])
    Tair_d = np.array([env.T_air_K(s) for s in dense])
    worst_C = {"exceed": 0.0, "t": None, "node": None}
    worst_T = {"exceed": 0.0, "t": None, "node": None}
    for t in ts:
        mask = dense <= t + 1e-9
        C_lo = min(cfg.C0, float(np.min(Cenv_d[mask])))
        C_hi = max(cfg.C0, float(np.max(Cenv_d[mask])))
        T_lo = min(cfg.T0_K, float(np.min(Tair_d[mask])))
        T_hi = max(cfg.T0_K, float(np.max(Tair_d[mask])))
        y = res.eval([t])[0]
        C, T = y[:n], y[n:2 * n]
        exC = np.maximum(C_lo - C, C - C_hi)          # >0 表示越界量
        exT = np.maximum(T_lo - T, T - T_hi)
        iC = int(np.argmax(exC)); iT = int(np.argmax(exT))
        if exC[iC] - tol_C > worst_C["exceed"]:
            worst_C = {"exceed": float(exC[iC] - tol_C), "t": float(t), "node": iC}
        if exT[iT] - tol_T > worst_T["exceed"]:
            worst_T = {"exceed": float(exT[iT] - tol_T), "t": float(t), "node": iT}
    ok = worst_C["exceed"] <= 0.0 and worst_T["exceed"] <= 0.0
    return {"ok": bool(ok), "worst_C_excess": worst_C, "worst_T_excess": worst_T,
            "tol_C": tol_C, "tol_T_K": tol_T, "moving": moving, "N": N, "t_end_s": t_end_s}


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


# --------------------------------------------------------------------------
# V-3：按问题 / 输出对象分别实施的网格与时间收敛（W3）
# --------------------------------------------------------------------------
# 证据范围分列：
#  - 空间对照：半离散/高精度参考解（时间近精确）在网格 N 序列上的差 → 空间离散误差；
#  - 时间对照：固定网格、收紧时间容差/步长 → 时间离散误差。
#  二者不可互替：时长收敛（t*）不替代表点/固定厘米位置收敛（尤其 Q4 r=1.2 cm）。

def _fixed_op(cfg, question, N, interface="integral"):
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    return FVMOperator(RadialGrid(N, cfg.R0), cfg.props(question), env,
                       h=cfg.h, hm=cfg.hm, R0=cfg.R0, interface=interface,
                       decoupled=(question == "q1"))


def _ref_op(cfg, N, interface="integral"):
    from . import data_io
    env = data_io.make_env_functions(cfg, "base")
    radius = data_io.make_radius_function(cfg)
    return FVMOperator(RefGrid(N), cfg.props("q4"), env, h=cfg.h, hm=cfg.hm,
                       R0=cfg.R0, interface=interface, radius_fn=radius), radius


def v3_early_surface(cfg, *, question="q1", Ns=(200, 400, 800), interface="integral",
                     probe_times=(1.0, 10.0, 100.0)):
    """空间对照：result1/2 早期逐秒行表面 C（半离散/高精度参考）随 N 的相邻差。"""
    rows = {}
    prev = {}
    for N in Ns:
        op = _fixed_op(cfg, question, N, interface)
        y0 = np.concatenate([np.full(N + 1, cfg.C0), np.full(N + 1, cfg.T0_K)])
        ref = hi_precision_reference(op, y0, list(probe_times))
        surf = {tp: float(ref[k][:N + 1][-1]) for k, tp in enumerate(probe_times)}
        diffs = {tp: (None if N == Ns[0] else abs(surf[tp] - prev[tp])) for tp in probe_times}
        rows[N] = {"C_surf": surf, "d_vs_prev": diffs}
        prev = surf
    max_diff = max((d for N in Ns[1:] for d in rows[N]["d_vs_prev"].values()), default=0.0)
    return {"kind": "spatial", "question": question, "interface": interface,
            "probe_times": list(probe_times), "by_N": rows, "max_adjacent_diff": float(max_diff)}


def v3_q4_profile(cfg, *, Ns=(200, 400, 800), interface="integral", t_probe_h=24.0):
    """空间对照：Q4 全部固定厘米位置（0.1–1.9 cm + 表面）晚期剖面随 N 收敛。

    **单列 r=1.2 cm 回归点**（独立于 t*，不以时长收敛替代）。返回各位置相邻差与最大差·位置。
    """
    from . import postprocess as PP
    t_probe = t_probe_h * 3600.0
    cols = [round(0.1 * j, 4) for j in range(1, 20)]     # 0.1..1.9
    prof = {}
    surf = {}
    for N in Ns:
        op, radius = _ref_op(cfg, N, interface)
        n = N + 1
        y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
        res = _SBDF.integrate_bdf(op, y0, 0.0, t_probe, cfg.bdf, breakpoints=(14400.0,))
        if not res.ok:
            raise RuntimeError(res.message)
        c = res.eval([t_probe])[0][:n]
        R_t = float(radius.R(t_probe))
        row = PP.sample_q4_row(c, op.grid, R_t, cols)
        prof[N] = {cols[j]: row[j] for j in range(len(cols))}
        surf[N] = float(c[-1])
    # 相邻差（仅域内位置）
    worst = {"diff": 0.0, "r_cm": None, "N_pair": None}
    for j, rc in enumerate(cols):
        for a, b in zip(Ns[:-1], Ns[1:]):
            va, vb = prof[a][rc], prof[b][rc]
            if va is not None and vb is not None:
                d = abs(va - vb)
                if d > worst["diff"]:
                    worst = {"diff": float(d), "r_cm": rc, "N_pair": (a, b)}
    # r=1.2 cm 回归点单列
    reg = {N: prof[N].get(1.2) for N in Ns}
    reg_diffs = [abs(prof[b][1.2] - prof[a][1.2])
                 for a, b in zip(Ns[:-1], Ns[1:])
                 if prof[a].get(1.2) is not None and prof[b].get(1.2) is not None]
    return {"kind": "spatial", "t_probe_h": t_probe_h, "cols_cm": cols,
            "profile_by_N": prof, "surface_by_N": surf,
            "worst": worst, "r1_2cm_by_N": reg,
            "r1_2cm_max_adjacent_diff": float(max(reg_diffs, default=0.0))}


def v3_tstar_grid(cfg, *, question="q23", Ns=(200, 400, 800), interface="integral",
                  t_cap_h=200.0):
    """空间对照：t*（h）随 N（阈值穿越，网格收敛）。moving 由 question 推断。"""
    from . import runners
    moving = (question == "q4")
    out = {}
    for N in Ns:
        det, *_ = runners.q23_detect(cfg, N=N, interface=interface, question=question,
                                     moving=moving, t_cap_h=t_cap_h)
        out[N] = det.t_cross / 3600.0
    diffs = [abs(out[b] - out[a]) for a, b in zip(Ns[:-1], Ns[1:])]
    return {"kind": "spatial", "question": question, "by_N": out,
            "max_adjacent_diff_h": float(max(diffs, default=0.0))}


def v3_tstar_time(cfg, *, question="q23", N=400, interface="integral",
                  rtols=(1e-8, 1e-10), t_cap_h=200.0):
    """时间对照：固定 N，收紧 BDF rtol（时间/稠密输出精度）→ t*（h）差。"""
    from . import runners
    base_bdf = dict(cfg.bdf)
    out = {}
    for rt in rtols:
        bdf = dict(base_bdf); bdf["rtol"] = rt
        # 临时替换 cfg.bdf：用 q23_detect 但注入 bdf 通过 monkey——改为直接积分
        import numpy as _np
        moving = (question == "q4")
        if moving:
            op, radius = _ref_op(cfg, N, interface)
        else:
            op = _fixed_op(cfg, question, N, interface)
        n = N + 1
        y0 = _np.concatenate([_np.full(n, cfg.C0), _np.full(n, cfg.T0_K)])
        res = _SBDF.integrate_bdf(op, y0, 0.0, t_cap_h * 3600.0, bdf,
                                  breakpoints=(14400.0,), threshold=cfg.threshold)
        if res.t_cross is None:
            raise RuntimeError(f"t* 未在 {t_cap_h} h 内命中（rtol={rt}）")
        out[rt] = res.t_cross / 3600.0
    diffs = [abs(out[b] - out[a]) for a, b in zip(rtols[:-1], rtols[1:])]
    return {"kind": "temporal", "question": question, "N": N, "by_rtol": out,
            "max_adjacent_diff_h": float(max(diffs, default=0.0))}


def v3_flux_mean(cfg, *, question="q23", Ns=(200, 400, 800), interface="integral",
                 t_probe_s=10800.0):
    """空间对照：表面通量 f 与平均含水率 C̄ 在 t_probe 随 N 收敛（通量/平均量对象）。"""
    moving = (question == "q4")
    out = {}
    for N in Ns:
        if moving:
            op, _ = _ref_op(cfg, N, interface)
        else:
            op = _fixed_op(cfg, question, N, interface)
        n = N + 1
        y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
        res = _SBDF.integrate_bdf(op, y0, 0.0, t_probe_s, cfg.bdf, breakpoints=(14400.0,))
        if not res.ok:
            raise RuntimeError(res.message)
        y = res.eval([t_probe_s])[0]
        C = y[:n]
        out[N] = {"flux": float(op.flux_cbar(t_probe_s, C)), "cbar": float(op.grid.cbar(C))}
    fdiffs = [abs(out[b]["flux"] - out[a]["flux"]) for a, b in zip(Ns[:-1], Ns[1:])]
    cdiffs = [abs(out[b]["cbar"] - out[a]["cbar"]) for a, b in zip(Ns[:-1], Ns[1:])]
    return {"kind": "spatial", "question": question, "t_probe_s": t_probe_s, "by_N": out,
            "max_flux_diff": float(max(fdiffs, default=0.0)),
            "max_cbar_diff": float(max(cdiffs, default=0.0))}
