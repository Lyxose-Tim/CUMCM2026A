"""runners.py —— 各问候选求解器（Q1/Q2Q3/Q4）与共用装配。

**候选阶段**：以推荐候选配置（integral 8 点界面 + BDF）计算，缓存候选数据与标量诊断；
正式 result1–4 的官方导出受 D12 授权门控（见 run_all.py）。判据与验证一律用未舍入值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import data_io
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
                         decoupled=False):
    grid = RadialGrid(N, cfg.R0)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, scenario)
    op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                     interface=interface, integral_npts=8, decoupled=decoupled)
    return op, env


def build_ref_operator(cfg, question, N, *, interface="integral", scenario="base"):
    grid = RefGrid(N)
    props = cfg.props(question)
    env = data_io.make_env_functions(cfg, scenario)
    radius = data_io.make_radius_function(cfg)
    op = FVMOperator(grid, props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
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

    # 表 1/2：t=100..1800（每 100 s），r=0,0.5,1,1.5,2 cm
    table_ts = np.arange(100, 1801, 100)
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
