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

from .grid import RadialGrid
from .operators import FVMOperator, jac_sparsity
from . import solver_be


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
