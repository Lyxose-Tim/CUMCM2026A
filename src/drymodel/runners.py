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
def build_fixed_operator(cfg, question, N, *, interface="integral", scenario="base",
                         decoupled=False, h_mult=1.0, hm_mult=1.0):
    grid = RadialGrid(N, cfg.R0)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, scenario)
    op = FVMOperator(grid, props, env, h=cfg.h * h_mult, hm=cfg.hm * hm_mult, R0=cfg.R0,
                     interface=interface, integral_npts=8, decoupled=decoupled)
    return op, env


def build_ref_operator(cfg, question, N, *, interface="integral", scenario="base",
                       h_mult=1.0, hm_mult=1.0):
    grid = RefGrid(N)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, scenario)
    radius = data_io.make_radius_function(cfg)
    op = FVMOperator(grid, props, env, h=cfg.h * h_mult, hm=cfg.hm * hm_mult, R0=cfg.R0,
                     interface=interface, integral_npts=8, radius_fn=radius,
                     decoupled=False)
    return op, env, radius


def initial_state(cfg, N):
    return np.concatenate([np.full(N + 1, cfg.C0), np.full(N + 1, cfg.T0_K)])


# --------------------------------------------------------------------------
# Q1 候选：0–1800 s，附录 2，热质解耦
# --------------------------------------------------------------------------
def run_q1(cfg, *, N=1600, interface="integral", t_end=1800.0, save=True):
    """Q1 候选求解 + result1 网格采样 + 表 1/2 + V-3 早期行收敛。"""
    op, env = build_fixed_operator(cfg, "q1", N, interface=interface, decoupled=True)
    y0 = initial_state(cfg, N)
    res = SBDF.integrate_bdf(op, y0, 0.0, t_end, cfg.bdf,
                             breakpoints=(), threshold=None, air_data_end=14400.0)
    if not res.ok:
        raise RuntimeError(f"Q1 BDF 失败：{res.message}")

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
        "N": N, "interface": interface, "t_end": t_end,
        "cols_cm": np.array(RESULT_COLS_CM),
        "result1_t": ts, "result1_C": C_grid, "result1_T": T_grid,
        "table_ts": table_ts, "table_cm": np.array(table_cm),
        "table_T_C": table_T, "table_C": table_C,
    }
    if save:
        np.savez_compressed(CACHE / "q1_candidate.npz", **out)
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


def q23_detect(cfg, *, N=400, interface="integral", scenario="base",
               t_cap_h=90.0, post_s=None, question="q23", moving=None,
               h_mult=1.0, hm_mult=1.0):
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
        op, env, _ = build_ref_operator(cfg, "q4", N, interface=interface,
                                        scenario=scenario, h_mult=h_mult, hm_mult=hm_mult)
    else:
        op, env = build_fixed_operator(cfg, question, N, interface=interface,
                                       scenario=scenario, decoupled=False,
                                       h_mult=h_mult, hm_mult=hm_mult)
    n = N + 1
    y0 = initial_state(cfg, N)
    res = SBDF.integrate_bdf(op, y0, 0.0, t_cap_h * 3600.0, cfg.bdf,
                             breakpoints=(14400.0,), threshold=thr)
    if not res.ok:
        raise RuntimeError(res.message)
    if res.t_cross is None:
        raise RuntimeError(f"{question} 未在 {t_cap_h} h 内达标（N={N}, {interface}）")

    t_cross = res.t_cross
    argmax_node = int(np.argmax(res.y_cross[:n]))
    cont = SBDF.continue_bdf(op, res.y_cross, t_cross, post_s, cfg.bdf)
    cmax_cont = _cmax_fn_from(cont, n)

    tt = np.linspace(t_cross, t_cross + post_s, 61)
    post_max = max(cmax_cont(t) for t in tt)
    post_ok = post_max <= thr + 1e-9

    t_sample = CR.first_sample_below(cmax_cont, t_cross, 60.0, thr, t_cross + post_s)

    det = CR.Detection(
        t_cross=t_cross, t_star=t_cross, t_sample=t_sample, t_safe=None,
        delta=float("nan"), dt_star=float("nan"),
        argmax_node=argmax_node, post_ok=post_ok, post_max_cmax=post_max,
        threshold=thr,
    )
    return det, res, cont, op


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


# --------------------------------------------------------------------------
# 完整轨迹（无事件）与采样辅助
# --------------------------------------------------------------------------
def _integrate_full(op, y0, t_end, cfg):
    return SBDF.integrate_bdf(op, y0, 0.0, t_end, cfg.bdf, breakpoints=(14400.0,),
                              threshold=None)


def q23_candidate(cfg, *, N=800, interface="integral", t_cap_h=90.0, save=True):
    """Q2/Q3 候选：t*、t_sample、t_end_1s、表 3/4/5、result3 候选数据（60 s）。"""
    det, res, cont, op = q23_detect(cfg, N=N, interface=interface, t_cap_h=t_cap_h)
    n = N + 1
    thr = cfg.threshold
    t_cross = det.t_cross

    def cmax_at(t):
        y = res.eval([t])[0] if t <= t_cross + 1e-6 else cont.eval([t])[0]
        return CR.cmax(y[:n])

    # t_end_1s：首个整数秒使 C_max<0.15（由连续解求值）
    t_end_1s = int(np.ceil(t_cross))
    while cmax_at(float(t_end_1s)) >= thr:
        t_end_1s += 1

    # 完整轨迹到 t_sample（供表/采样）
    t_sample = det.t_sample if det.t_sample is not None else t_end_1s
    full = _integrate_full(op, initial_state(cfg, N), t_sample + 120.0, cfg)

    node_idx = np.array([op.grid.output_index(rc) for rc in RESULT_COLS_CM])

    # result3 候选：60..t_sample（60 s）
    ts3 = np.arange(60, int(t_sample) + 1, 60)
    Y3 = full.eval(ts3)
    C3 = Y3[:, node_idx]

    # 表 3/4：t=0.5..3.0 h（温度/水分），r=0,0.5,1,1.5,2 cm
    tbl_ts = np.arange(0.5, 3.01, 0.5) * 3600.0
    tbl_idx = np.array([op.grid.output_index(rc) for rc in (0.0, 0.5, 1.0, 1.5, 2.0)])
    Yt = full.eval(tbl_ts)
    tbl_T = Yt[:, tbl_idx + n] - 273.15
    tbl_C = Yt[:, tbl_idx]

    # 表 5：6,12,... h（≤t*）+ 末行 t*（水分中心/表面 + C_max）
    hours = np.arange(6, t_cross / 3600.0 + 1e-9, 6)
    tbl5_t = list(hours * 3600.0) + [t_cross]
    tbl5 = []
    for tt in tbl5_t:
        y = full.eval([tt])[0] if tt <= t_sample + 120 else res.eval([tt])[0]
        C = y[:n]
        tbl5.append((tt / 3600.0, float(C[0]), float(C[-1]), CR.cmax(C)))

    out = {
        "N": N, "interface": interface,
        "t_star_h": t_cross / 3600.0, "t_sample_s": t_sample,
        "t_end_1s": t_end_1s, "post_ok": det.post_ok, "post_max_cmax": det.post_max_cmax,
        "argmax_node": det.argmax_node,
        "result3_t": ts3, "result3_C": C3, "cols_cm": np.array(RESULT_COLS_CM),
        "table34_ts_h": tbl_ts / 3600.0, "table3_T_C": tbl_T, "table4_C": tbl_C,
        "table5": np.array(tbl5),
    }
    if save:
        np.savez_compressed(CACHE / "q23_candidate.npz",
                            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        np.savez_compressed(CACHE / "q23_scalars.npz",
                            t_star_h=out["t_star_h"], t_sample_s=out["t_sample_s"],
                            t_end_1s=out["t_end_1s"], post_max_cmax=out["post_max_cmax"])
    return out


def q4_candidate(cfg, *, N=800, interface="integral", t_cap_h=200.0, save=True):
    """Q4 候选：t*_4、result4 候选（60 s，域外留空）、表 6、半径小表。"""
    det, res, cont, op = q23_detect(cfg, N=N, interface=interface, question="q4",
                                    t_cap_h=t_cap_h)
    n = N + 1
    t_cross = det.t_cross
    t_sample = det.t_sample if det.t_sample is not None else int(np.ceil(t_cross))
    radius = data_io.make_radius_function(cfg)
    full = _integrate_full(op, initial_state(cfg, N), t_sample + 120.0, cfg)

    cols20 = [round(0.1 * j, 4) for j in range(20)]     # 0.0..1.9

    # result4 候选：60..t_sample（60 s）；20 列 + 表面；域外 None
    ts4 = np.arange(60, int(t_sample) + 1, 60)
    result4_rows = []
    radii_cm = []
    for t in ts4:
        c = full.eval([float(t)])[0][:n]
        R_t = float(radius.R(float(t)))
        row = PP.sample_q4_row(c, op.grid, R_t, cols20)
        row.append(float(c[-1]))                        # 药材表面列
        result4_rows.append(row)
        radii_cm.append(R_t * 100.0)

    # 表 6：cols 0,0.5,1.0,表面；行 6,12,...(≤t*) + 末行 t*
    hours = np.arange(6, t_cross / 3600.0 + 1e-9, 6)
    tbl6_t = list(hours * 3600.0) + [t_cross]
    tbl6 = []
    radius_small = []
    for tt in tbl6_t:
        c = full.eval([tt])[0][:n] if tt <= t_sample + 120 else res.eval([tt])[0][:n]
        R_t = float(radius.R(tt))
        r0 = PP.sample_q4_row(c, op.grid, R_t, [0.0, 0.5, 1.0])
        tbl6.append((tt / 3600.0, r0[0], r0[1], r0[2], float(c[-1])))
        radius_small.append((tt / 3600.0, R_t * 100.0, radius.is_extrapolated(tt)))

    out = {
        "N": N, "interface": interface,
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
