"""verify.py —— 验证工具（V-1/V-2 起步；其余 V-3~V-13 在后续阶段补齐）。

对齐《A题_建模方案.md》§8.1、§8.5：
- `analytic_cylinder_robin`：圆柱第三类边界瞬态导热/扩散级数解（≥80 项）。
- `hi_precision_reference`：紧容差 BDF（rtol 1e-11、atol 1e-13）积分同一半离散 ODE，
  **称高精度参考解，非真正半离散精确解**（审计 A-F06）。
- V-1（热）/V-2（质）：空间阶（半离散 vs 级数解）与时间阶（BE Δt 序列 vs 高精度参考）分离。
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import brentq
from scipy.special import jv

from dataclasses import dataclass

from .grid import RadialGrid, RefGrid
from .operators import FVMOperator, jac_sparsity
from . import data_io
from . import solver_be
from . import solver_bdf as _SBDF


VALID_STATUSES = {"pass", "fail", "partial", "not_run"}


def check_record(check_id, question, status, *, metric, limit, config, coverage,
                 worst_location=None, message=""):
    """构造统一、可序列化的验证证据记录。"""
    if status not in VALID_STATUSES:
        raise ValueError(f"非法验证状态 {status!r}")
    return {
        "check_id": str(check_id),
        "question": str(question),
        "status": status,
        "metric": metric,
        "limit": limit,
        "config": config,
        "coverage": coverage,
        "worst_location": worst_location,
        "message": str(message),
    }


def _config_snapshot(op, *, fallback=None):
    run = getattr(op, "run_config", None)
    if run is not None:
        return {"digest": run.digest, **run.snapshot()}
    return {} if fallback is None else dict(fallback)


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


def analytic_records(cfg, *, Ns=(200, 400, 800), temporal_N=400,
                     temporal_dts=(1.0, 0.5, 0.25), t_probe=100.0):
    """V-1/V-2 的空间与时间证据分栏，状态由实测收敛阶生成。"""
    v1 = run_V1_heat(Ns=Ns, t_probe=t_probe)
    v2 = run_V2_mass(Ns=Ns, t_probe=t_probe)

    def spatial_record(check_id, question, data, error_key):
        errors = [data["by_N"][N][error_key] for N in Ns]
        orders = order_from_errors(errors)
        status = "pass" if orders and all(1.7 <= order <= 2.3 for order in orders) else "fail"
        return check_record(
            check_id, question, status,
            metric={"errors": dict(zip(map(str, Ns), errors)), "orders": orders},
            limit={"order_min": 1.7, "order_max": 2.3},
            config={"Ns": list(Ns), "t_probe_s": float(t_probe),
                    "time_reference": "tight BDF semi-discrete"},
            coverage={"kind": "analytic_spatial", "field": question,
                      "nodes": list(Ns), "probe_time_s": float(t_probe)},
        )

    spatial_heat = spatial_record("V-1-spatial", "q1-heat", v1, "max_err_C")
    spatial_mass = spatial_record("V-2-spatial", "q1-mass", v2, "max_err")

    rho, cp, k = 820.0, 2600.0, 0.36
    D0 = 7e-9 * np.exp(-0.89 / cfg.C0)
    props = ConstProps(rho, cp, k, D_v=D0)
    env = ConstEnv(323.15, 0.05)
    op = _make_const_op(temporal_N, cfg.R0, props, env, cfg.h, cfg.hm)
    n = temporal_N + 1
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
    ref = hi_precision_reference(op, y0, [t_probe])[0]
    heat_errors = []
    mass_errors = []
    runs = []
    for dt in temporal_dts:
        result = solver_be.integrate_be(
            op, y0, t_probe, dt, C0_ref=cfg.C0, picard=cfg.picard,
            retry=cfg.retry, record_times={t_probe},
        )
        result.require_reached(t_probe, label=f"V-1/V-2 BE dt={dt:g}s")
        heat_errors.append(float(np.max(np.abs(result.T[-1] - ref[n:2 * n]))))
        mass_errors.append(float(np.max(np.abs(result.C[-1] - ref[:n]))))
        runs.append({"dt_s": float(dt), "accepted_substeps": result.accepted_substeps})

    def temporal_record(check_id, question, errors):
        orders = [float(np.log(errors[i] / errors[i + 1])
                           / np.log(float(temporal_dts[i]) / float(temporal_dts[i + 1])))
                  for i in range(len(errors) - 1)]
        status = "pass" if orders and all(0.8 <= order <= 1.2 for order in orders) else "fail"
        return check_record(
            check_id, question, status,
            metric={"errors": dict(zip(map(str, temporal_dts), errors)), "orders": orders},
            limit={"order_min": 0.8, "order_max": 1.2},
            config={"N": temporal_N, "dt_s": list(temporal_dts),
                    "picard": dict(cfg.picard), "retry": dict(cfg.retry)},
            coverage={"kind": "BE_temporal", "probe_time_s": float(t_probe),
                      "runs": runs},
        )

    temporal_heat = temporal_record("V-1-temporal", "q1-heat", heat_errors)
    temporal_mass = temporal_record("V-2-temporal", "q1-mass", mass_errors)
    return {
        "v1": v1, "v2": v2,
        "records": [spatial_heat, spatial_mass, temporal_heat, temporal_mass],
    }


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
    C_init, _ = op.split(np.asarray(y0, dtype=float))
    recs = [(0.0, op.grid.cbar(C_init), float(op.flux_cbar(0.0, C_init)))]

    def rec_step(step):
        C, _ = op.split(step.y1)
        recs.append((float(step.t1), op.grid.cbar(C),
                     float(op.flux_cbar(step.t1, C))))

    run = getattr(op, "run_config", None)
    picard = cfg.picard if run is None else run.picard_options
    retry = cfg.retry if run is None else run.retry_options
    result = solver_be.integrate_be(
        op, y0, t_end, dt, C0_ref=cfg.C0, picard=picard,
        retry=retry, record_times={float(t_end)}, step_recorder=rec_step,
    )
    config_snapshot = _config_snapshot(
        op, fallback={"scheme": "backward_euler", "dt_s": float(dt), "N": op.N,
                      "interface": op.interface, "integral_npts": op.integral_npts},
    )
    coverage = {
        "t_start_s": 0.0,
        "t_end_requested_s": float(t_end),
        "t_end_actual_s": float(result.t_end),
        "accepted_substeps": int(result.accepted_substeps),
        "moving_domain": bool(op.is_reference),
    }
    if not result.ok or abs(result.t_end - float(t_end)) > 1e-9 * max(1.0, abs(t_end)):
        message = result.message or "BE 未到请求终点"
        failed = check_record(
            "V-4a", "q4" if op.is_reference else "fixed", "fail",
            metric={"relative_residual": None},
            limit={"relative_residual_max": float(cfg.raw["acceptance"]["discrete_balance_rel"])},
            config=config_snapshot, coverage=coverage, message=message,
        )
        failed_b = check_record(
            "V-4b", "q4" if op.is_reference else "fixed", "fail",
            metric={"relative_difference": None},
            limit={"relative_difference_max": float(
                cfg.raw["acceptance"]["flux_integral"]["be"]["tol_rel"]
            )},
            config=config_snapshot, coverage=coverage, message=message,
        )
        return {
            "status": "fail", "message": message, "solver": result,
            "v4a": failed, "v4b": failed_b,
            "dCbar": float("nan"), "rel_v4a": float("nan"),
            "rel_v4b": float("nan"), "diff_v4b": float("nan"),
            "theory_v4b": float("nan"), "f0": recs[0][2], "fN": float("nan"),
        }

    ts = np.array([r[0] for r in recs], dtype=float)
    cbar = np.array([r[1] for r in recs], dtype=float)
    f = np.array([r[2] for r in recs], dtype=float)
    dts = np.diff(ts)
    if len(dts) == 0 or np.any(dts <= 0) or not np.all(np.isfinite(f)):
        raise RuntimeError("BE 接受子步账本为空、非递增或含非有限通量")

    dCbar = cbar[-1] - cbar[0]
    f_steps = f[1:]                                    # f_n（步末，n=1..Nsteps）
    balance_rhs = -np.sum(dts * f_steps)               # V-4a：真实子步权重
    res_v4a = abs(dCbar - balance_rhs)
    rel_v4a = res_v4a / max(abs(dCbar), 1e-300)

    I_trap = np.sum(0.5 * dts * (f[:-1] + f[1:]))
    diff_v4b = I_trap - (-dCbar)
    theory_v4b = np.sum(0.5 * dts * (f[:-1] - f[1:]))
    rel_v4b = abs(diff_v4b) / max(abs(dCbar), 1e-300)

    limit_a = float(cfg.raw["acceptance"]["discrete_balance_rel"])
    limit_b = float(cfg.raw["acceptance"]["flux_integral"]["be"]["tol_rel"])
    v4a = check_record(
        "V-4a", "q4" if op.is_reference else "fixed",
        "pass" if rel_v4a <= limit_a else "fail",
        metric={"relative_residual": float(rel_v4a)},
        limit={"relative_residual_max": limit_a}, config=config_snapshot,
        coverage=coverage,
    )
    v4b = check_record(
        "V-4b", "q4" if op.is_reference else "fixed",
        "pass" if rel_v4b <= limit_b else "fail",
        metric={"relative_difference": float(rel_v4b)},
        limit={"relative_difference_max": limit_b}, config=config_snapshot,
        coverage=coverage,
    )

    return {
        "status": "pass" if v4a["status"] == v4b["status"] == "pass" else "fail",
        "solver": result, "v4a": v4a, "v4b": v4b,
        "dCbar": float(dCbar), "rel_v4a": float(rel_v4a),
        "rel_v4b": float(rel_v4b), "diff_v4b": float(diff_v4b),
        "theory_v4b": float(theory_v4b), "f0": float(f[0]), "fN": float(f[-1]),
    }


def mass_balance_bdf(cfg, *, question="q23", N=200, interface="integral",
                     t_end_s=3600.0, moving=False):
    """V-4a（BDF）：用累计通量增广态 I 检验归一化收支 C̄(t)−C̄(0)+I(t)≈0。

    y=[C..,T..,I]，I(0)=0，İ=(2 h_m/R)(C_N−C_env)。固定域 R=R0；moving=True 用 Q4 动域 R(t)。
    """
    from . import runners
    if moving:
        op, _, _ = runners.build_ref_operator(
            cfg, "q4", N, interface=interface, augmented=True,
            purpose="V-4a-BDF-short",
        )
    else:
        op, _ = runners.build_fixed_operator(
            cfg, question, N, interface=interface, augmented=True,
            purpose="V-4a-BDF-short",
        )
    n = N + 1
    y0 = runners.initial_state(cfg, N, augmented=True)
    run = op.run_config
    res = _SBDF.integrate_bdf(
        op, y0, 0.0, t_end_s, run.bdf,
        breakpoints=run.breakpoints_s if run.restart_at_breakpoints else (),
        air_data_end=run.air_data_end_s,
    )
    res.require_reached(t_end_s, label="V-4a BDF")
    cbar0 = op.grid.cbar(y0[:n])
    worst = 0.0
    for t in np.linspace(0.0, t_end_s, 13)[1:]:
        y = res.eval([t])[0]
        resid = op.grid.cbar(y[:n]) - cbar0 + y[-1]      # 归一化收支，应 ~0
        worst = max(worst, abs(resid))
    rel = float(worst / max(abs(cbar0), 1e-300))
    limit = float(cfg.raw["acceptance"]["discrete_balance_rel"])
    record = check_record(
        "V-4a-BDF", "q4" if moving else question,
        "pass" if rel <= limit else "fail",
        metric={"max_abs_residual": float(worst), "relative_residual": rel},
        limit={"relative_residual_max": limit},
        config={"digest": op.run_config.digest, **op.run_config.snapshot()},
        coverage={"t_start_s": 0.0, "t_end_s": float(t_end_s), "samples": 12,
                  "moving_domain": bool(moving)},
    )
    return {"max_abs_resid": float(worst), "rel": rel, "cbar0": float(cbar0),
            "moving": moving, "N": N, "status": record["status"], "record": record}


def energy_residual_be(op, y0, t_probe, dt, cfg):
    """V-4c（W/m 口径）：R_E^(ℓ)=2π∫_0^R b(C)∂_tT r dr − 2πR h[T_air−T_s]（固定域）。

    用 BE 差商近似 ∂_tT；返回绝对/相对残差与参考尺度。
    """
    final_step = None

    def rec_step(step):
        nonlocal final_step
        if abs(step.t1 - float(t_probe)) <= 1e-10 * max(1.0, abs(t_probe)):
            final_step = step

    run = getattr(op, "run_config", None)
    picard = cfg.picard if run is None else run.picard_options
    retry = cfg.retry if run is None else run.retry_options
    result = solver_be.integrate_be(
        op, y0, t_probe, dt, C0_ref=cfg.C0, picard=picard,
        retry=retry, record_times={float(t_probe)}, step_recorder=rec_step,
    )
    config_snapshot = _config_snapshot(
        op, fallback={"scheme": "backward_euler", "dt_s": float(dt), "N": op.N,
                      "interface": op.interface, "integral_npts": op.integral_npts},
    )
    coverage = {
        "t_probe_s": float(t_probe), "t_end_actual_s": float(result.t_end),
        "accepted_substeps": int(result.accepted_substeps),
        "moving_domain": bool(op.is_reference),
    }
    if not result.ok or final_step is None:
        message = result.message or "BE 未产生结束于热残差探针时刻的接受子步"
        record = check_record(
            "V-4c", "q4" if op.is_reference else "fixed", "fail",
            metric={"RE_W_per_m": None, "relative_residual": None},
            limit={
                "abs_W_per_m_max": float(cfg.raw["acceptance"]["energy_residual"]["abs_W_per_m"]),
                "relative_residual_max": float(cfg.raw["acceptance"]["energy_residual"]["rel"]),
            },
            config=config_snapshot, coverage=coverage, message=message,
        )
        return {"status": "fail", "record": record, "RE_W_per_m": float("nan"),
                "rel": float("nan"), "ref_scale": float("nan"), "solver": result}

    C1, T1 = op.split(final_step.y1)
    _, T0a = op.split(final_step.y0)
    actual_dt = final_step.dt
    dTdt = (T1 - T0a) / actual_dt

    R0 = float(op.radius_fn.R(t_probe)) if op.is_reference else op.R0
    b = op.props.b(C1)
    # 2π ∫ b ∂_tT r dr ≈ 2π Σ V_i b_i dTdt_i（V_i 已含 r 度量，单位长度弧度积分 → ×2π）
    volume_scale = R0 ** 2 if op.is_reference else 1.0
    integral = 2.0 * np.pi * volume_scale * np.sum(op.V * b * dTdt)
    Tair = float(op.env.T_air_K(t_probe))
    surface = 2.0 * np.pi * R0 * op.h * (Tair - T1[-1])
    RE = integral - surface
    ref_scale = 2.0 * np.pi * R0 * op.h * max(abs(Tair - (cfg.T0_K)), 1.0)
    rel = float(abs(RE) / max(ref_scale, 1e-300))
    abs_limit = float(cfg.raw["acceptance"]["energy_residual"]["abs_W_per_m"])
    rel_limit = float(cfg.raw["acceptance"]["energy_residual"]["rel"])
    status = "pass" if abs(RE) <= abs_limit and rel <= rel_limit else "fail"
    coverage["actual_dt_s"] = float(actual_dt)
    record = check_record(
        "V-4c", "q4" if op.is_reference else "fixed", status,
        metric={"RE_W_per_m": float(RE), "relative_residual": rel},
        limit={"abs_W_per_m_max": abs_limit, "relative_residual_max": rel_limit},
        config=config_snapshot, coverage=coverage,
    )
    return {"status": status, "record": record, "RE_W_per_m": float(RE),
            "rel": rel, "ref_scale": float(ref_scale), "solver": result,
            "actual_dt_s": float(actual_dt)}


# --------------------------------------------------------------------------
# V-3/V-5/V-6/V-11：候选轨迹输出对象的分块对照
# --------------------------------------------------------------------------
def _chunks(values, size=512):
    values = np.asarray(values, dtype=float)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _max_with_location(diff, times, positions):
    if diff.size == 0:
        return 0.0, None
    flat = int(np.argmax(diff))
    i, j = np.unravel_index(flat, diff.shape)
    return float(diff[i, j]), {"time_s": float(times[i]), "position": positions[j]}


def compare_fixed_trajectories(cfg, candidate, reference, *, question, check_id,
                               comparison, times_s, cols_cm=None):
    """分块比较固定域轨迹的全部指定输出行、表点、平均量和通量。"""
    if cols_cm is None:
        cols_cm = [round(0.1 * j, 4) for j in range(21)]
    times = np.unique(np.asarray(times_s, dtype=float))
    if times.size == 0:
        raise ValueError("固定域轨迹对照必须给至少一个时刻")
    ca, cb = candidate["trajectory"], reference["trajectory"]
    Na, Nb = int(candidate["N"]), int(reference["N"])
    ga, gb = RadialGrid(Na, cfg.R0), RadialGrid(Nb, cfg.R0)
    ia = np.array([ga.output_index(r) for r in cols_cm])
    ib = np.array([gb.output_index(r) for r in cols_cm])
    na, nb = Na + 1, Nb + 1
    env = data_io.make_env_functions(cfg, "base")

    maxima = {"dC": 0.0, "dT_C": 0.0, "dCbar": 0.0,
              "dflux": 0.0, "dCmax": 0.0}
    locations = {key: None for key in maxima}
    for chunk in _chunks(times):
        ya = ca.eval(chunk)
        yb = cb.eval(chunk)
        Ca, Cb = ya[:, :na], yb[:, :nb]
        Ta, Tb = ya[:, na:2 * na], yb[:, nb:2 * nb]
        dC = np.abs(Ca[:, ia] - Cb[:, ib])
        dT = np.abs(Ta[:, ia] - Tb[:, ib])
        value, loc = _max_with_location(dC, chunk, list(cols_cm))
        if value > maxima["dC"]:
            maxima["dC"], locations["dC"] = value, loc
        value, loc = _max_with_location(dT, chunk, list(cols_cm))
        if value > maxima["dT_C"]:
            maxima["dT_C"], locations["dT_C"] = value, loc

        cbar_a = Ca @ ga.cbar_weight
        cbar_b = Cb @ gb.cbar_weight
        diff = np.abs(cbar_a - cbar_b)
        idx = int(np.argmax(diff))
        if float(diff[idx]) > maxima["dCbar"]:
            maxima["dCbar"] = float(diff[idx])
            locations["dCbar"] = {"time_s": float(chunk[idx]), "position": "volume_average"}

        cmax_diff = np.abs(np.max(Ca, axis=1) - np.max(Cb, axis=1))
        idx = int(np.argmax(cmax_diff))
        if float(cmax_diff[idx]) > maxima["dCmax"]:
            maxima["dCmax"] = float(cmax_diff[idx])
            locations["dCmax"] = {"time_s": float(chunk[idx]), "position": "full_field_max"}

        Cenv = np.asarray(env.C_env(chunk), dtype=float)
        flux_a = 2.0 * cfg.hm / cfg.R0 * (Ca[:, -1] - Cenv)
        flux_b = 2.0 * cfg.hm / cfg.R0 * (Cb[:, -1] - Cenv)
        flux_diff = np.abs(flux_a - flux_b)
        idx = int(np.argmax(flux_diff))
        if float(flux_diff[idx]) > maxima["dflux"]:
            maxima["dflux"] = float(flux_diff[idx])
            locations["dflux"] = {"time_s": float(chunk[idx]), "position": "surface"}

    dC_limit = float(cfg.raw["acceptance"]["table_points"]["dC"])
    dT_limit = float(cfg.raw["acceptance"]["table_points"]["dT_degC"])
    flux_limit = 2.0 * cfg.hm / cfg.R0 * dC_limit
    tstar_diff = None
    if "t_star_h" in candidate and "t_star_h" in reference:
        tstar_diff = abs(float(candidate["t_star_h"]) - float(reference["t_star_h"]))
    limits = {
        "dC_max": dC_limit, "dT_C_max": dT_limit, "dCbar_max": dC_limit,
        "dCmax_max": dC_limit, "dflux_max": flux_limit,
        "dt_star_h_max": float(cfg.raw["acceptance"]["t_star_h"]),
    }
    passed = (
        maxima["dC"] <= limits["dC_max"]
        and maxima["dT_C"] <= limits["dT_C_max"]
        and maxima["dCbar"] <= limits["dCbar_max"]
        and maxima["dCmax"] <= limits["dCmax_max"]
        and maxima["dflux"] <= limits["dflux_max"]
        and (tstar_diff is None or tstar_diff <= limits["dt_star_h_max"])
    )
    metrics = {**maxima, "dt_star_h": tstar_diff}
    return check_record(
        check_id, question, "pass" if passed else "fail", metric=metrics,
        limit=limits,
        config={"candidate": candidate["run_config"], "reference": reference["run_config"]},
        coverage={
            "comparison": comparison, "rows": int(times.size),
            "time_start_s": float(times[0]), "time_end_s": float(times[-1]),
            "positions_cm": list(cols_cm), "variables": ["C", "T", "Cbar", "Cmax", "flux"],
        },
        worst_location=locations,
    )


def _sample_ref_fields(C, T, grid, radius_m, cols_cm):
    x = np.asarray(cols_cm, dtype=float) / (100.0 * float(radius_m))
    inside = x <= 1.0 + 1e-12
    values_C = np.full(len(cols_cm) + 1, np.nan)
    values_T = np.full(len(cols_cm) + 1, np.nan)
    if np.any(inside):
        xi = np.minimum(x[inside], 1.0)
        values_C[:-1][inside] = np.interp(xi, grid.x, C)
        values_T[:-1][inside] = np.interp(xi, grid.x, T)
    values_C[-1] = C[-1]
    values_T[-1] = T[-1]
    return values_C, values_T, np.r_[inside, True]


def compare_moving_trajectories(cfg, candidate, reference, *, check_id,
                                comparison, times_s, cols_cm=None):
    """分块比较 Q4 固定厘米列、表面列、平均量、通量与阈值。"""
    if cols_cm is None:
        cols_cm = [round(0.1 * j, 4) for j in range(20)]
    times = np.unique(np.asarray(times_s, dtype=float))
    if times.size == 0:
        raise ValueError("Q4 轨迹对照必须给至少一个时刻")
    ca, cb = candidate["trajectory"], reference["trajectory"]
    Na, Nb = int(candidate["N"]), int(reference["N"])
    ga, gb = RefGrid(Na), RefGrid(Nb)
    na, nb = Na + 1, Nb + 1
    radius = data_io.make_radius_function(cfg)
    env = data_io.make_env_functions(cfg, "base")
    positions = [*cols_cm, "surface"]
    maxima = {"dC": 0.0, "dT_C": 0.0, "dCbar": 0.0,
              "dflux": 0.0, "dCmax": 0.0, "dC_r1p2cm": 0.0,
              "mask_mismatches": 0}
    locations = {key: None for key in maxima}
    compared_cells = 0

    for chunk in _chunks(times, size=256):
        ya = ca.eval(chunk)
        yb = cb.eval(chunk)
        Ca, Cb = ya[:, :na], yb[:, :nb]
        Ta, Tb = ya[:, na:2 * na], yb[:, nb:2 * nb]
        sampled_Ca, sampled_Cb, sampled_Ta, sampled_Tb, masks = [], [], [], [], []
        for i, t in enumerate(chunk):
            R = float(radius.R(t))
            c_a, t_a, m_a = _sample_ref_fields(Ca[i], Ta[i], ga, R, cols_cm)
            c_b, t_b, m_b = _sample_ref_fields(Cb[i], Tb[i], gb, R, cols_cm)
            mismatch = int(np.count_nonzero(m_a != m_b))
            maxima["mask_mismatches"] += mismatch
            sampled_Ca.append(c_a); sampled_Cb.append(c_b)
            sampled_Ta.append(t_a); sampled_Tb.append(t_b); masks.append(m_a & m_b)
        sampled_Ca = np.asarray(sampled_Ca)
        sampled_Cb = np.asarray(sampled_Cb)
        sampled_Ta = np.asarray(sampled_Ta)
        sampled_Tb = np.asarray(sampled_Tb)
        masks = np.asarray(masks, dtype=bool)
        compared_cells += int(np.count_nonzero(masks))
        dC = np.where(masks, np.abs(sampled_Ca - sampled_Cb), -1.0)
        dT = np.where(masks, np.abs(sampled_Ta - sampled_Tb), -1.0)
        value, loc = _max_with_location(dC, chunk, positions)
        if value > maxima["dC"]:
            maxima["dC"], locations["dC"] = value, loc
        value, loc = _max_with_location(dT, chunk, positions)
        if value > maxima["dT_C"]:
            maxima["dT_C"], locations["dT_C"] = value, loc

        if 1.2 in cols_cm:
            j = list(cols_cm).index(1.2)
            valid = masks[:, j]
            if np.any(valid):
                dr = np.where(valid, np.abs(sampled_Ca[:, j] - sampled_Cb[:, j]), -1.0)
                idx = int(np.argmax(dr))
                if float(dr[idx]) > maxima["dC_r1p2cm"]:
                    maxima["dC_r1p2cm"] = float(dr[idx])
                    locations["dC_r1p2cm"] = {"time_s": float(chunk[idx]), "position": 1.2}

        cbar_a = Ca @ ga.cbar_weight
        cbar_b = Cb @ gb.cbar_weight
        diff = np.abs(cbar_a - cbar_b)
        idx = int(np.argmax(diff))
        if float(diff[idx]) > maxima["dCbar"]:
            maxima["dCbar"] = float(diff[idx])
            locations["dCbar"] = {"time_s": float(chunk[idx]), "position": "volume_average"}
        cmax_diff = np.abs(np.max(Ca, axis=1) - np.max(Cb, axis=1))
        idx = int(np.argmax(cmax_diff))
        if float(cmax_diff[idx]) > maxima["dCmax"]:
            maxima["dCmax"] = float(cmax_diff[idx])
            locations["dCmax"] = {"time_s": float(chunk[idx]), "position": "full_field_max"}
        R = np.asarray(radius.R(chunk), dtype=float)
        Cenv = np.asarray(env.C_env(chunk), dtype=float)
        flux_a = 2.0 * cfg.hm / R * (Ca[:, -1] - Cenv)
        flux_b = 2.0 * cfg.hm / R * (Cb[:, -1] - Cenv)
        flux_diff = np.abs(flux_a - flux_b)
        idx = int(np.argmax(flux_diff))
        if float(flux_diff[idx]) > maxima["dflux"]:
            maxima["dflux"] = float(flux_diff[idx])
            locations["dflux"] = {"time_s": float(chunk[idx]), "position": "surface"}

    dC_limit = float(cfg.raw["acceptance"]["table_points"]["dC"])
    dT_limit = float(cfg.raw["acceptance"]["table_points"]["dT_degC"])
    min_radius = float(np.min(radius.R(times)))
    flux_limit = 2.0 * cfg.hm / min_radius * dC_limit
    tstar_diff = abs(float(candidate["t_star_h"]) - float(reference["t_star_h"]))
    limits = {
        "dC_max": dC_limit, "dT_C_max": dT_limit, "dCbar_max": dC_limit,
        "dCmax_max": dC_limit, "dC_r1p2cm_max": dC_limit,
        "dflux_max": flux_limit,
        "dt_star_h_max": float(cfg.raw["acceptance"]["t_star_h"]),
        "mask_mismatches_max": 0,
    }
    passed = (
        maxima["dC"] <= limits["dC_max"]
        and maxima["dT_C"] <= limits["dT_C_max"]
        and maxima["dCbar"] <= limits["dCbar_max"]
        and maxima["dCmax"] <= limits["dCmax_max"]
        and maxima["dC_r1p2cm"] <= limits["dC_r1p2cm_max"]
        and maxima["dflux"] <= limits["dflux_max"]
        and tstar_diff <= limits["dt_star_h_max"]
        and maxima["mask_mismatches"] == 0
    )
    return check_record(
        check_id, "q4", "pass" if passed else "fail",
        metric={**maxima, "dt_star_h": tstar_diff}, limit=limits,
        config={"candidate": candidate["run_config"], "reference": reference["run_config"]},
        coverage={
            "comparison": comparison, "rows": int(times.size),
            "compared_cells": compared_cells,
            "time_start_s": float(times[0]), "time_end_s": float(times[-1]),
            "positions_cm": positions,
            "variables": ["C", "T", "Cbar", "Cmax", "flux", "domain_mask"],
        },
        worst_location=locations,
    )


def _independent_flux_quadrature(trajectory, op, t_end, step_s):
    times = np.arange(
        float(trajectory.t_start), float(t_end), float(step_s)
    )
    if times.size == 0 or times[-1] != float(t_end):
        times = np.r_[times, float(t_end)]
    flux_parts = []
    for chunk in _chunks(times, size=1024):
        Y = trajectory.eval(chunk)
        n = op.N + 1
        Csurf = Y[:, n - 1]
        R = (np.asarray(op.radius_fn.R(chunk), dtype=float)
             if op.is_reference else np.full(len(chunk), op.R0))
        Cenv = np.asarray(op.env.C_env(chunk), dtype=float)
        flux = 2.0 * op.hm / R * (Csurf - Cenv)
        flux_parts.append(np.asarray(flux, dtype=float))
    flux = np.concatenate(flux_parts)
    return float(simpson(flux, x=times)), int(len(times))


def bdf_mass_balance_record(cfg, candidate, *, question, time_reference,
                            quadrature_step_s=15.0, quadrature_levels=7):
    """V-4a/b：增广收支、独立通量求积加密与独立 BDF 时间加密。"""
    trajectory = candidate["trajectory"]
    op = candidate["operator"]
    n = op.N + 1
    ref_trajectory = time_reference["trajectory"]
    if (not op.augmented or trajectory.state_size != 2 * n + 1
            or ref_trajectory.state_size != 2 * n + 1):
        return check_record(
            "V-4-BDF", question, "fail", metric={"reason": "missing_augmented_I"},
            limit={}, config=candidate["run_config"],
            coverage={"t_end_s": float(trajectory.t_end)},
            message="候选 BDF 轨迹未启用累计通量状态 I",
        )
    sample_times = np.linspace(trajectory.t_start, trajectory.t_end, 25)
    Y = trajectory.eval(sample_times)
    cbar0 = op.grid.cbar(Y[0, :n])
    residuals = np.array([op.grid.cbar(row[:n]) - cbar0 + row[-1] for row in Y])
    max_abs = float(np.max(np.abs(residuals)))
    rel_aug = max_abs / max(abs(cbar0), 1e-300)
    limit_aug = float(cfg.raw["acceptance"]["discrete_balance_rel"])
    limit_quad = (
        float(cfg.raw["acceptance"]["flux_integral"]["bdf"]["tol_rel_factor_of_rtol"])
        * float(candidate["run_config"]["rtol"])
    )

    common_end = min(float(trajectory.t_end), float(ref_trajectory.t_end))
    y_start = trajectory.eval([trajectory.t_start])[0]
    y_common = trajectory.eval([common_end])[0]
    y_ref_start = ref_trajectory.eval([ref_trajectory.t_start])[0]
    y_ref_common = ref_trajectory.eval([common_end])[0]
    I_end = float(y_common[-1] - y_start[-1])
    I_ref = float(y_ref_common[-1] - y_ref_start[-1])
    rel_time_ref = abs(I_end - I_ref) / max(abs(I_ref), 1e-300)

    refinements = []
    step = float(quadrature_step_s)
    previous = None
    rel_quad_refine = None
    for _ in range(int(quadrature_levels)):
        independent, count = _independent_flux_quadrature(
            trajectory, op, common_end, step,
        )
        rel_quad = abs(independent - I_end) / max(abs(I_end), 1e-300)
        rel_quad_refine = (
            None if previous is None
            else abs(independent - previous) / max(abs(independent), 1e-300)
        )
        refinements.append({
            "step_s": step,
            "samples": count,
            "integral": independent,
            "relative_to_augmented_I": rel_quad,
            "relative_to_previous_level": rel_quad_refine,
        })
        if (rel_quad <= limit_quad and rel_quad_refine is not None
                and rel_quad_refine <= limit_quad):
            break
        previous = independent
        step /= 2.0

    final_quad = refinements[-1]
    rel_quad = float(final_quad["relative_to_augmented_I"])
    refine_ok = (
        final_quad["relative_to_previous_level"] is not None
        and float(final_quad["relative_to_previous_level"]) <= limit_quad
    )
    status = "pass" if (
        rel_aug <= limit_aug
        and rel_quad <= limit_quad
        and refine_ok
        and rel_time_ref <= limit_quad
    ) else "fail"
    return check_record(
        "V-4-BDF", question, status,
        metric={
            "max_abs_augmented_residual": max_abs,
            "relative_augmented_residual": rel_aug,
            "independent_flux_integral": float(final_quad["integral"]),
            "augmented_I_change": I_end,
            "relative_independent_quadrature_difference": rel_quad,
            "relative_quadrature_refinement_difference": (
                None if final_quad["relative_to_previous_level"] is None
                else float(final_quad["relative_to_previous_level"])
            ),
            "time_refined_augmented_I_change": I_ref,
            "relative_time_refined_I_difference": rel_time_ref,
            "quadrature_refinements": refinements,
        },
        limit={
            "relative_augmented_residual_max": limit_aug,
            "relative_independent_quadrature_difference_max": limit_quad,
            "relative_quadrature_refinement_difference_max": limit_quad,
            "relative_time_refined_I_difference_max": limit_quad,
        },
        config={
            "candidate": candidate["run_config"],
            "time_reference": time_reference["run_config"],
        },
        coverage={
            "t_start_s": float(trajectory.t_start), "t_end_s": common_end,
            "candidate_trajectory_end_s": float(trajectory.t_end),
            "time_reference_trajectory_end_s": float(ref_trajectory.t_end),
            "augmented_samples": int(len(sample_times)),
            "independent_quadrature_initial_step_s": float(quadrature_step_s),
            "independent_quadrature_final_step_s": float(final_quad["step_s"]),
            "independent_quadrature_samples": int(final_quad["samples"]),
            "independent_quadrature_levels": int(len(refinements)),
            "moving_domain": bool(op.is_reference),
        },
        worst_location={"augmented_residual_time_s": float(sample_times[np.argmax(np.abs(residuals))])},
    )


def envelope_record(cfg, candidate, *, question, sample_times_s):
    """V-5：检查接受状态和指定重构/输出时刻的有限性及最大值包络。"""
    trajectory = candidate["trajectory"]
    op = candidate["operator"]
    n = op.N + 1
    env = op.env
    C_lower = min(cfg.C0, float(np.min(env.C_nodes)), float(env.C_const))
    C_upper = max(cfg.C0, float(np.max(env.C_nodes)), float(env.C_const))
    T_lower = min(cfg.T0_K, float(np.min(env.T_nodes_K)), float(env.T_const_K))
    T_upper = max(cfg.T0_K, float(np.max(env.T_nodes_K)), float(env.T_const_K))
    tol_C = float(cfg.envelope_tol["C"])
    tol_T = float(cfg.envelope_tol["T_K"])

    extrema = {"C_min": np.inf, "C_max": -np.inf, "T_min_K": np.inf, "T_max_K": -np.inf}
    accepted_states = 0

    def consume(Y):
        nonlocal accepted_states
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        C = Y[:n]
        T = Y[n:2 * n]
        accepted_states += C.shape[1]
        extrema["C_min"] = min(extrema["C_min"], float(np.min(C)))
        extrema["C_max"] = max(extrema["C_max"], float(np.max(C)))
        extrema["T_min_K"] = min(extrema["T_min_K"], float(np.min(T)))
        extrema["T_max_K"] = max(extrema["T_max_K"], float(np.max(T)))

    for segment in trajectory.segments:
        consume(segment.y)
    dense_count = 0
    for chunk in _chunks(np.unique(np.asarray(sample_times_s, dtype=float)), size=512):
        Y = trajectory.eval(chunk).T
        consume(Y)
        dense_count += len(chunk)

    finite = all(np.isfinite(value) for value in extrema.values())
    passed = (
        finite
        and extrema["C_min"] >= C_lower - tol_C
        and extrema["C_max"] <= C_upper + tol_C
        and extrema["T_min_K"] >= T_lower - tol_T
        and extrema["T_max_K"] <= T_upper + tol_T
    )
    return check_record(
        "V-5", question, "pass" if passed else "fail", metric=extrema,
        limit={
            "C_min": C_lower - tol_C, "C_max": C_upper + tol_C,
            "T_min_K": T_lower - tol_T, "T_max_K": T_upper + tol_T,
            "finite_required": True,
        },
        config=candidate["run_config"],
        coverage={
            "accepted_solver_states": accepted_states - dense_count,
            "dense_or_reconstructed_times": dense_count,
            "t_start_s": float(trajectory.t_start), "t_end_s": float(trajectory.t_end),
        },
    )


def event_accuracy_record(cfg, candidate, space_reference, time_reference, *, question):
    """V-6：以实际空间/时间加密差定义 delta 与 dt_star。"""
    dt_space = abs(float(candidate["t_star_h"]) - float(space_reference["t_star_h"]))
    dt_time = abs(float(candidate["t_star_h"]) - float(time_reference["t_star_h"]))
    dt_star = max(dt_space, dt_time)
    def cmax_at(payload, t):
        n_local = int(payload["N"]) + 1
        return float(np.max(payload["trajectory"].eval([float(t)])[0, :n_local]))

    delta_values = []
    for label, reference in (("space", space_reference), ("time", time_reference)):
        probes = [
            float(candidate["t_star_h"]) * 3600.0,
            float(reference["t_star_h"]) * 3600.0,
            float(candidate["t_sample_s"]),
        ]
        common_end = min(
            float(candidate["trajectory"].t_end),
            float(reference["trajectory"].t_end),
        )
        for probe in probes:
            if probe <= common_end + 1e-9:
                delta_values.append({
                    "reference": label,
                    "time_s": probe,
                    "difference": abs(cmax_at(candidate, probe) - cmax_at(reference, probe)),
                })
    worst_delta = max(delta_values, key=lambda row: row["difference"])
    delta = float(worst_delta["difference"])
    sample_t = float(candidate["t_sample_s"])
    n = candidate["N"] + 1
    sample_cmax = float(np.max(candidate["trajectory"].eval([sample_t])[0, :n]))
    strict_sample = np.isclose(sample_t % 60.0, 0.0, atol=1e-9) and sample_cmax < cfg.threshold
    passed = (
        dt_star <= float(cfg.raw["acceptance"]["t_star_h"])
        and delta <= float(cfg.raw["acceptance"]["table_points"]["dC"])
        and bool(candidate["post_ok"])
        and strict_sample
    )
    candidate["delta"] = delta
    candidate["dt_star_h"] = dt_star
    return check_record(
        "V-6", question, "pass" if passed else "fail",
        metric={
            "delta_C": delta, "dt_star_h": dt_star,
            "dt_star_space_h": dt_space, "dt_star_time_h": dt_time,
            "sample_time_s": sample_t, "sample_cmax": sample_cmax,
            "post_max_cmax": float(candidate["post_max_cmax"]),
            "delta_probes": delta_values,
        },
        limit={
            "delta_C_max": float(cfg.raw["acceptance"]["table_points"]["dC"]),
            "dt_star_h_max": float(cfg.raw["acceptance"]["t_star_h"]),
            "sample_cmax_strict": float(cfg.threshold),
            "post_cmax_max": float(cfg.threshold + 1e-9),
        },
        config={
            "candidate": candidate["run_config"],
            "space_reference": space_reference["run_config"],
            "time_reference": time_reference["run_config"],
        },
        coverage={
            "event": "continuous Cmax downward root",
            "strict_sample_grid_s": 60,
            "post_event_check_s": float(cfg.post_margin_s),
        },
        worst_location={
            "argmax_node": int(candidate["argmax_node"]),
            "delta_C": worst_delta,
        },
    )


def moving_geometry_record(cfg, candidate):
    """V-13：独立核对 Q4 的 R(t)、1/R²、1/R 与外推标记。"""
    trajectory = candidate["trajectory"]
    op = candidate["operator"]
    if not op.is_reference:
        return check_record(
            "V-13", "q4", "fail", metric={"reason": "not_reference_domain"},
            limit={}, config=candidate["run_config"],
            coverage={"moving_domain": False},
            message="Q4 候选未使用参考坐标运动域算子",
        )

    radius = data_io.make_radius_function(cfg)
    source_t, source_r_cm = data_io.load_attachment2(cfg.radius_file())
    source_r_m = np.asarray(source_r_cm, dtype=float) / 100.0
    probe_times = np.unique(np.r_[
        0.0,
        np.asarray(cfg.breakpoints_s, dtype=float),
        float(trajectory.t_end),
        float(source_t[-1]),
    ])
    probe_times = probe_times[(probe_times >= 0.0) & np.isfinite(probe_times)]
    probe_radius = np.asarray(radius.R(probe_times), dtype=float)

    diff_scale_err = 0.0
    surf_c_err = 0.0
    surf_t_err = 0.0
    flux_err = 0.0
    state_probe_times = probe_times[probe_times <= trajectory.t_end]
    for t in state_probe_times:
        R = float(radius.R(t))
        diff_scale_err = max(diff_scale_err, abs(op._diff_scale(t) - 1.0 / R**2))
        surf_c, surf_t = op._surf_coeff(t)
        surf_c_err = max(surf_c_err, abs(surf_c - op.hm / R))
        surf_t_err = max(surf_t_err, abs(surf_t - op.h / R))
        C = trajectory.eval([t])[0, :op.N + 1]
        independent_flux = 2.0 * op.hm / R * (C[-1] - float(op.env.C_env(t)))
        flux_err = max(flux_err, abs(op.flux_cbar(t, C) - independent_flux))

    source_finite_positive = bool(
        np.all(np.isfinite(source_r_m)) and np.all(source_r_m > 0.0)
    )
    source_nonincreasing = bool(np.all(np.diff(source_r_m) <= 1e-12))
    source_match = float(np.max(np.abs(np.asarray(radius.R(source_t)) - source_r_m)))
    after_t = float(source_t[-1] + 1.0)
    extrapolation_marked = bool(radius.is_extrapolated(after_t))
    extrapolation_constant = abs(float(radius.R(after_t)) - source_r_m[-1])

    table_markers_match = True
    for row in candidate.get("radius_small", []):
        t = float(row[0]) * 3600.0
        table_markers_match = table_markers_match and (
            bool(row[2]) == bool(radius.is_extrapolated(t))
        )

    atol = 1e-12
    passed = (
        source_finite_positive and source_nonincreasing
        and source_match <= atol
        and diff_scale_err <= atol
        and surf_c_err <= atol
        and surf_t_err <= atol
        and flux_err <= atol
        and extrapolation_marked
        and extrapolation_constant <= atol
        and table_markers_match
    )
    return check_record(
        "V-13", "q4", "pass" if passed else "fail",
        metric={
            "source_radius_match_max_m": source_match,
            "diffusion_scale_abs_error": diff_scale_err,
            "surface_mass_coeff_abs_error": surf_c_err,
            "surface_heat_coeff_abs_error": surf_t_err,
            "flux_factor_abs_error": flux_err,
            "source_finite_positive": source_finite_positive,
            "source_nonincreasing": source_nonincreasing,
            "extrapolation_marked": extrapolation_marked,
            "extrapolation_constant_abs_error_m": extrapolation_constant,
            "table_extrapolation_markers_match": bool(table_markers_match),
        },
        limit={
            "absolute_error_max": atol,
            "source_finite_positive_required": True,
            "source_nonincreasing_required": True,
            "extrapolation_mark_required": True,
            "table_marker_match_required": True,
        },
        config=candidate["run_config"],
        coverage={
            "moving_domain": True,
            "source_radius_rows": int(len(source_t)),
            "operator_probe_times_s": [float(value) for value in state_probe_times],
            "source_last_time_s": float(source_t[-1]),
            "extrapolation_probe_time_s": after_t,
        },
    )


# --------------------------------------------------------------------------
# V-10：静态极限（Q4 动域求解器 R≡R0 vs 固定域求解器）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
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
    run = cfg.resolve_run("q4", N=N, interface=interface, purpose="V-10-static-limit")

    # 动域（参考坐标），R≡R0
    op_mov = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=R0,
                         interface=run.interface, integral_npts=run.integral_npts,
                         radius_fn=_ConstRadius(R0))
    # 固定域
    op_fix = FVMOperator(RadialGrid(N, R0), props, env, h=cfg.h, hm=cfg.hm, R0=R0,
                         interface=run.interface, integral_npts=run.integral_npts)
    op_mov.run_config = run
    op_fix.run_config = run

    y0 = np.concatenate([np.full(N + 1, cfg.C0), np.full(N + 1, cfg.T0_K)])
    breakpoints = run.breakpoints_s if run.restart_at_breakpoints else ()
    rm = _SBDF.integrate_bdf(
        op_mov, y0, 0.0, t_probe, run.bdf, breakpoints=breakpoints,
        air_data_end=run.air_data_end_s,
    )
    rf = _SBDF.integrate_bdf(
        op_fix, y0, 0.0, t_probe, run.bdf, breakpoints=breakpoints,
        air_data_end=run.air_data_end_s,
    )
    rm.require_reached(t_probe, label="V-10 动域静态极限")
    rf.require_reached(t_probe, label="V-10 固定域静态极限")
    ym = rm.eval([t_probe])[0]
    yf = rf.eval([t_probe])[0]
    denom = np.maximum(np.abs(yf), 1e-12)
    rel = float(np.max(np.abs(ym - yf) / denom))
    absdiff = float(np.max(np.abs(ym - yf)))
    limit = float(cfg.raw["acceptance"]["static_limit_rel"])
    record = check_record(
        "V-10", "q4", "pass" if rel <= limit else "fail",
        metric={"relative_max": rel, "absolute_max": absdiff},
        limit={"relative_max": limit},
        config={"digest": run.digest, **run.snapshot()},
        coverage={"t_probe_s": float(t_probe), "nodes": int(N + 1),
                  "comparison": "R(t)=R0 reference-domain vs fixed-domain"},
    )
    return {"rel_max": rel, "abs_max": absdiff, "t_probe": t_probe, "N": N,
            "status": record["status"], "record": record}
