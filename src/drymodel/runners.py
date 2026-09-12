"""runners.py —— 各问候选求解器（Q1/Q2Q3/Q4）与共用装配。

**候选阶段**：以推荐候选配置（integral 8 点界面 + BDF）计算，缓存候选数据与标量诊断；
正式 result1–4 的官方导出受 D12 授权门控（见 run_all.py）。判据与验证一律用未舍入值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import criterion as CR
from . import data_io
from . import postprocess as PP
from .grid import RadialGrid, RefGrid
from .operators import FVMOperator
from . import solver_bdf as SBDF

CACHE = cfgmod.PROJECT_ROOT / "cache" / "candidates"
CACHE.mkdir(parents=True, exist_ok=True)

RESULT_COLS_CM = [round(0.1 * j, 4) for j in range(21)]   # 0.0, 0.1, ..., 2.0


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------
def _resolve_run(cfg, question, N, *, purpose, interface, integral_npts, scenario,
                 h_mult, hm_mult, augmented, bdf_overrides=None, run_config=None):
    if run_config is not None:
        if run_config.question != question or run_config.N != N:
            raise ValueError(
                f"运行配置与装配请求不一致：{run_config.question}/N={run_config.N} "
                f"!= {question}/N={N}"
            )
        return run_config
    return cfg.resolve_run(
        question, purpose=purpose, N=N, interface=interface,
        integral_npts=integral_npts, scenario=scenario,
        h_mult=h_mult, hm_mult=hm_mult, augmented=augmented,
        bdf_overrides=bdf_overrides,
    )


def build_fixed_operator(cfg, question, N, *, interface=None, integral_npts=None,
                         scenario="base", decoupled=False, h_mult=1.0, hm_mult=1.0,
                         augmented=False, purpose="candidate", bdf_overrides=None,
                         run_config=None):
    run = _resolve_run(
        cfg, question, N, purpose=purpose, interface=interface,
        integral_npts=integral_npts, scenario=scenario, h_mult=h_mult,
        hm_mult=hm_mult, augmented=augmented, bdf_overrides=bdf_overrides,
        run_config=run_config,
    )
    grid = RadialGrid(run.N, cfg.R0)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, run.scenario)
    op = FVMOperator(
        grid, props, env, h=cfg.h * run.h_mult, hm=cfg.hm * run.hm_mult, R0=cfg.R0,
        interface=run.interface, integral_npts=run.integral_npts,
        decoupled=decoupled, augmented=run.augmented,
    )
    op.run_config = run
    return op, env


def build_ref_operator(cfg, question, N, *, interface=None, integral_npts=None,
                       scenario="base", h_mult=1.0, hm_mult=1.0, augmented=False,
                       purpose="candidate", bdf_overrides=None, run_config=None):
    run = _resolve_run(
        cfg, question, N, purpose=purpose, interface=interface,
        integral_npts=integral_npts, scenario=scenario, h_mult=h_mult,
        hm_mult=hm_mult, augmented=augmented, bdf_overrides=bdf_overrides,
        run_config=run_config,
    )
    grid = RefGrid(run.N)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, run.scenario)
    radius = data_io.make_radius_function(cfg)
    op = FVMOperator(
        grid, props, env, h=cfg.h * run.h_mult, hm=cfg.hm * run.hm_mult, R0=cfg.R0,
        interface=run.interface, integral_npts=run.integral_npts, radius_fn=radius,
        decoupled=False, augmented=run.augmented,
    )
    op.run_config = run
    return op, env, radius


def initial_state(cfg, N, *, augmented=False):
    parts = [np.full(N + 1, cfg.C0), np.full(N + 1, cfg.T0_K)]
    if augmented:
        parts.append(np.array([0.0]))
    return np.concatenate(parts)


def _run_evidence(run):
    """Return the complete effective configuration with its stable digest."""
    return {"digest": run.digest, **run.snapshot()}


def _run_breakpoints(run):
    """Return the configured restart points only when restart is enabled."""
    return run.breakpoints_s if run.restart_at_breakpoints else ()


# --------------------------------------------------------------------------
# Q1 候选：0–1800 s，附录 2，热质解耦
# --------------------------------------------------------------------------
def run_q1(cfg, *, N=800, interface=None, integral_npts=None, t_end=1800.0,
           save=True, bdf_overrides=None, purpose="candidate"):
    """Q1 候选求解 + result1 网格采样 + 表 1/2 + V-3 早期行收敛。"""
    op, env = build_fixed_operator(
        cfg, "q1", N, interface=interface, integral_npts=integral_npts,
        decoupled=True, purpose=purpose, bdf_overrides=bdf_overrides,
    )
    run = op.run_config
    y0 = initial_state(cfg, N)
    res = SBDF.integrate_bdf(
        op, y0, 0.0, t_end, run.bdf,
        breakpoints=_run_breakpoints(run), threshold=None,
        air_data_end=run.air_data_end_s,
    )
    res.require_reached(t_end, label="Q1 BDF")

    # result1 采样：t=1..1800（整数秒），21 个固定位置（落在节点）
    ts = np.arange(1, int(t_end) + 1)
    node_idx = [op.grid.output_index(rc) for rc in RESULT_COLS_CM]
    Y = res.eval(ts)                                  # (1800, 2(N+1))
    n = N + 1
    C_grid = Y[:, node_idx]                           # (1800, 21)
    T_grid = Y[:, np.array(node_idx) + n] - 273.15    # °C

    # 表 1/2：7 个时刻（PATCH-08，明确非"100…1800"）× r=0,0.5,1,1.5,2 cm
    table_ts = np.array([100, 300, 600, 900, 1200, 1500, 1800])
    table_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    table_idx = np.array([op.grid.output_index(rc) for rc in table_cm])
    Yt = res.eval(table_ts)
    table_T = Yt[:, table_idx + n] - 273.15       # T 在第二块
    table_C = Yt[:, table_idx]

    out = {
        "N": N, "interface": run.interface, "integral_npts": run.integral_npts,
        "run_config": _run_evidence(run), "config_digest": run.digest,
        "trajectory": res, "operator": op, "t_end": t_end,
        "cols_cm": np.array(RESULT_COLS_CM),
        "result1_t": ts, "result1_C": C_grid, "result1_T": T_grid,
        "table_ts": table_ts, "table_cm": np.array(table_cm),
        "table_T_C": table_T, "table_C": table_C,
    }
    if save:
        np.savez_compressed(
            CACHE / "q1_candidate.npz",
            **{k: v for k, v in out.items()
               if isinstance(v, np.ndarray) or isinstance(v, (str, int, float, bool))},
        )
    return out


def v3_early_rows_q1(cfg, *, Ns=(200, 400, 800, 1600, 3200), interface="integral",
                     probe_times=(1.0, 10.0, 100.0)):
    """V-3 早期行：各 N 的半离散（高精度参考）表面 C，报告相邻差与阶。"""
    from . import verify as V
    rows = {}
    prev = None
    for N in Ns:
        op, _ = build_fixed_operator(cfg, "q1", N, interface=interface, decoupled=True)
        y0 = initial_state(cfg, N)
        ref = V.hi_precision_reference(op, y0, list(probe_times))
        Csurf = [float(ref[k][:N + 1][-1]) for k in range(len(probe_times))]
        d1 = None if prev is None else abs(Csurf[0] - prev)
        prev = Csurf[0]
        rows[N] = {"C_surf": Csurf, "d_surf_1s_vs_prev": d1}
    return {"probe_times": list(probe_times), "interface": interface, "by_N": rows}


# --------------------------------------------------------------------------
# Q2/Q3 候选：整个烘干过程（附录 3），达标判据 t*
# --------------------------------------------------------------------------
def _cmax_fn_from(res, n):
    def f(t):
        return CR.cmax(res.eval([t])[0][:n])
    return f


def q23_detect(cfg, *, N=400, interface=None, integral_npts=None, scenario="base",
               t_cap_h=90.0, post_s=None, question="q23", moving=None,
               h_mult=1.0, hm_mult=1.0, bdf_overrides=None,
               purpose="candidate", augmented=True):
    """达标检测：积分至 C_max 下穿 0.15，续算 post_s 检查回穿。

    - question 决定物性组（q23=附录3；q4=附录4）。
    - moving=None 时由 question=='q4' 推断动域；显式给 moving=False 可在固定 R0 上
      运行附录 4 物性（S10 情形②，分离几何效应）。
    - h_mult/hm_mult：灵敏度倍率（S1/S2），只在 scenario 层改，不改基础题给值。
    返回 (Detection, res, cont, op)。
    """
    thr = cfg.threshold
    post_s = float(cfg.post_margin_s if post_s is None else post_s)
    if moving is None:
        moving = (question == "q4")
    if moving:
        op, env, _ = build_ref_operator(
            cfg, "q4", N, interface=interface, integral_npts=integral_npts,
            scenario=scenario, h_mult=h_mult, hm_mult=hm_mult,
            augmented=augmented, purpose=purpose, bdf_overrides=bdf_overrides,
        )
    else:
        op, env = build_fixed_operator(
            cfg, question, N, interface=interface, integral_npts=integral_npts,
            scenario=scenario, decoupled=False, h_mult=h_mult, hm_mult=hm_mult,
            augmented=augmented, purpose=purpose, bdf_overrides=bdf_overrides,
        )
    run = op.run_config
    n = N + 1
    y0 = initial_state(cfg, N, augmented=run.augmented)
    res = SBDF.integrate_bdf(
        op, y0, 0.0, t_cap_h * 3600.0, run.bdf,
        breakpoints=_run_breakpoints(run), threshold=thr,
        air_data_end=run.air_data_end_s,
    )
    if not res.ok:
        raise RuntimeError(res.message)
    if res.t_cross is None:
        raise RuntimeError(f"{question} 未在 {t_cap_h} h 内达标（N={N}, {run.interface}）")

    t_cross = res.t_cross
    argmax_node = int(np.argmax(res.y_cross[:n]))
    cont = SBDF.continue_bdf(
        op, res.y_cross, t_cross, post_s, run.bdf,
        air_data_end=run.air_data_end_s,
    )
    cont.require_reached(t_cross + post_s, label=f"{question} 事件后续算")
    trajectory = res.concat(cont)
    cmax_cont = _cmax_fn_from(trajectory, n)

    tt = np.linspace(t_cross, t_cross + post_s, 61)
    post_max = max(cmax_cont(t) for t in tt)
    post_ok = post_max <= thr + 1e-9

    t_sample = CR.first_sample_below(cmax_cont, t_cross, 60.0, thr, t_cross + post_s)
    if t_sample is None:
        raise RuntimeError(
            f"{question} 在事件后 {post_s:g}s 内未找到严格合格的 60s 网格采样点"
        )

    det = CR.Detection(
        t_cross=t_cross, t_star=t_cross, t_sample=t_sample, t_safe=None,
        delta=None, dt_star=None,
        argmax_node=argmax_node, post_ok=post_ok, post_max_cmax=post_max,
        threshold=thr,
    )
    return det, trajectory, cont, op


def q23_interface_grid_study(cfg, *, Ns=(200, 400, 800), interfaces=("harmonic", "integral"),
                             t_cap_h=90.0):
    """V-11/F01：harmonic vs integral × N 的 t*（h）收敛研究（对照 E17 探针）。"""
    table = {}
    for interface in interfaces:
        table[interface] = {}
        for N in Ns:
            det, *_ = q23_detect(cfg, N=N, interface=interface, t_cap_h=t_cap_h)
            table[interface][N] = det.t_cross / 3600.0
    return table


# --------------------------------------------------------------------------
# S10（§8.7）：附录 4 固定半径对照，分离物性与几何效应
# --------------------------------------------------------------------------
def s10_fixed_radius_study(cfg, *, Ns=(200, 400), interface="integral", t_cap_h=200.0):
    """三情形 t*（h）：① 附录3/R0  ② 附录4/R0  ③ 附录4/R(t)（对照 E24 探针）。"""
    out = {"case1_app3_R0": {}, "case2_app4_R0": {}, "case3_app4_Rt": {}}
    for N in Ns:
        d1, *_ = q23_detect(cfg, N=N, interface=interface, question="q23",
                            moving=False, t_cap_h=t_cap_h)
        d2, *_ = q23_detect(cfg, N=N, interface=interface, question="q4",
                            moving=False, t_cap_h=t_cap_h)
        d3, *_ = q23_detect(cfg, N=N, interface=interface, question="q4",
                            moving=True, t_cap_h=t_cap_h)
        out["case1_app3_R0"][N] = d1.t_cross / 3600.0
        out["case2_app4_R0"][N] = d2.t_cross / 3600.0
        out["case3_app4_Rt"][N] = d3.t_cross / 3600.0
    return out


def _regular_plus_terminal(t_cross, *, step_h=6.0):
    """常规整 ``step_h`` 小时行加 t* 末行；根恰落常规行时只保留一次。"""
    step_s = float(step_h) * 3600.0
    regular = list(np.arange(step_s, float(t_cross) + 1e-9, step_s))
    tol = 1e-9 * max(1.0, abs(float(t_cross)))
    if regular and abs(regular[-1] - float(t_cross)) <= tol:
        regular[-1] = float(t_cross)
        return regular
    return [*regular, float(t_cross)]


def q23_candidate(cfg, *, N=800, interface=None, integral_npts=None,
                  t_cap_h=90.0, save=True, bdf_overrides=None,
                  purpose="candidate"):
    """Q2/Q3 候选：t*、t_sample、t_end_1s、表 3/4/5、result3 候选数据（60 s）。"""
    det, trajectory, _, op = q23_detect(
        cfg, N=N, interface=interface, integral_npts=integral_npts,
        t_cap_h=t_cap_h, bdf_overrides=bdf_overrides, purpose=purpose,
    )
    run = op.run_config
    n = N + 1
    thr = cfg.threshold
    t_cross = det.t_cross

    def cmax_at(t):
        return CR.cmax(trajectory.eval([t])[0][:n])

    # t_end_1s：首个整数秒使 C_max<0.15（由连续解求值）
    t_end_1s = CR.first_sample_below(
        cmax_at, t_cross, 1.0, thr, trajectory.t_end,
    )
    if t_end_1s is None:
        raise RuntimeError("Q23 未在同源续算轨迹中找到严格合格的整数秒")
    t_end_1s = int(t_end_1s)

    # 完整轨迹到 t_sample（供表/采样）
    t_sample = det.t_sample

    node_idx = np.array([op.grid.output_index(rc) for rc in RESULT_COLS_CM])

    # result3 候选：60..t_sample（60 s）
    ts3 = np.arange(60, int(t_sample) + 1, 60)
    Y3 = trajectory.eval(ts3)
    C3 = Y3[:, node_idx]

    # 表 3/4：t=0.5..3.0 h（温度/水分），r=0,0.5,1,1.5,2 cm
    tbl_ts = np.arange(0.5, 3.01, 0.5) * 3600.0
    tbl_idx = np.array([op.grid.output_index(rc) for rc in (0.0, 0.5, 1.0, 1.5, 2.0)])
    Yt = trajectory.eval(tbl_ts)
    tbl_T = Yt[:, tbl_idx + n] - 273.15
    tbl_C = Yt[:, tbl_idx]

    # 表 5：6,12,... h（≤t*）+ 末行 t*，五个题定固定半径列；Cmax 单独诊断。
    table5_cm = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    table5_idx = np.array([op.grid.output_index(rc) for rc in table5_cm])
    tbl5_t = _regular_plus_terminal(t_cross)
    tbl5 = []
    tbl5_cmax = []
    for tt in tbl5_t:
        y = trajectory.eval([tt])[0]
        C = y[:n]
        tbl5.append((tt / 3600.0, *[float(C[i]) for i in table5_idx]))
        tbl5_cmax.append(CR.cmax(C))

    out = {
        "N": N, "interface": run.interface, "integral_npts": run.integral_npts,
        "run_config": _run_evidence(run), "config_digest": run.digest,
        "trajectory": trajectory, "operator": op,
        "t_star_h": t_cross / 3600.0, "t_sample_s": t_sample,
        "t_end_1s": t_end_1s, "post_ok": det.post_ok, "post_max_cmax": det.post_max_cmax,
        "argmax_node": det.argmax_node,
        "result3_t": ts3, "result3_C": C3, "cols_cm": np.array(RESULT_COLS_CM),
        "table34_ts_h": tbl_ts / 3600.0, "table3_T_C": tbl_T, "table4_C": tbl_C,
        "table5_columns": np.array(["time_h", "r=0cm", "r=0.5cm", "r=1cm",
                                    "r=1.5cm", "r=2cm"]),
        "table5": np.array(tbl5), "table5_cmax": np.array(tbl5_cmax),
    }
    if save:
        np.savez_compressed(CACHE / "q23_candidate.npz",
                            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        np.savez_compressed(CACHE / "q23_scalars.npz",
                            t_star_h=out["t_star_h"], t_sample_s=out["t_sample_s"],
                            t_end_1s=out["t_end_1s"], post_max_cmax=out["post_max_cmax"])
    return out


def q4_candidate(cfg, *, N=800, interface=None, integral_npts=None,
                 t_cap_h=200.0, save=True, bdf_overrides=None,
                 purpose="candidate"):
    """Q4 候选：t*_4、result4 候选（60 s，域外留空）、表 6、半径小表。"""
    det, trajectory, _, op = q23_detect(
        cfg, N=N, interface=interface, integral_npts=integral_npts,
        question="q4", t_cap_h=t_cap_h, bdf_overrides=bdf_overrides,
        purpose=purpose,
    )
    run = op.run_config
    n = N + 1
    t_cross = det.t_cross
    t_sample = det.t_sample
    radius = data_io.make_radius_function(cfg)

    cols20 = [round(0.1 * j, 4) for j in range(20)]     # 0.0..1.9

    # result4 候选：60..t_sample（60 s）；20 列 + 表面；域外 None
    ts4 = np.arange(60, int(t_sample) + 1, 60)
    result4_rows = []
    radii_cm = []
    for t in ts4:
        c = trajectory.eval([float(t)])[0][:n]
        R_t = float(radius.R(float(t)))
        row = PP.sample_q4_row(c, op.grid, R_t, cols20)
        row.append(float(c[-1]))                        # 药材表面列
        result4_rows.append(row)
        radii_cm.append(R_t * 100.0)

    # 表 6：cols 0,0.5,1.0,表面；行 6,12,...(≤t*) + 末行 t*
    tbl6_t = _regular_plus_terminal(t_cross)
    tbl6 = []
    radius_small = []
    for tt in tbl6_t:
        c = trajectory.eval([tt])[0][:n]
        R_t = float(radius.R(tt))
        r0 = PP.sample_q4_row(c, op.grid, R_t, [0.0, 0.5, 1.0])
        tbl6.append((tt / 3600.0, r0[0], r0[1], r0[2], float(c[-1])))
        radius_small.append((tt / 3600.0, R_t * 100.0, radius.is_extrapolated(tt)))

    out = {
        "N": N, "interface": run.interface, "integral_npts": run.integral_npts,
        "run_config": _run_evidence(run), "config_digest": run.digest,
        "trajectory": trajectory, "operator": op,
        "t_star_h": t_cross / 3600.0, "t_sample_s": t_sample,
        "post_ok": det.post_ok, "post_max_cmax": det.post_max_cmax,
        "argmax_node": det.argmax_node,
        "result4_t": ts4, "result4_rows": result4_rows, "result4_radii_cm": radii_cm,
        "cols20_cm": cols20,
        "table6": tbl6, "radius_small": radius_small,
    }
    if save:
        np.savez_compressed(CACHE / "q4_scalars.npz",
                            t_star_h=out["t_star_h"], t_sample_s=out["t_sample_s"],
                            post_max_cmax=out["post_max_cmax"])
    return out
