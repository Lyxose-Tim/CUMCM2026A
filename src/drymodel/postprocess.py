"""postprocess.py —— 输出采样与域掩码（D09、D11-O、§6.3）。

- `inside(r_j, R, tol)`：r_j ≤ R(1+tol) 判域内（含相等）；掩码与几何**共用同一函数**。
- Q4：每行 x_j=r_j/R(t)（未舍入插值 R）；x_j<1 节点线性插值，x_j=1（容差内）取 c_N，
  与「药材表面」列同值；域外单元格留空（None）；域内重构失败抛错。
- 判据与验证用未舍入值；仅写盘时 round(·,4)。
"""
from __future__ import annotations

import numpy as np


def inside(r_j: float, R: float, tol: float = 1e-12) -> bool:
    """r_j ≤ R(1+tol) 则在域内（r_j, R 同单位，通常 m）。"""
    return float(r_j) <= float(R) * (1.0 + tol)


def round4(x):
    """四位小数（None 透传）。"""
    if x is None:
        return None
    return float(np.round(x, 4))


def sample_fixed_row(C_nodes, node_idx):
    """固定域：按节点下标取值（Q1–Q3，节点即输出）。"""
    return [float(C_nodes[i]) for i in node_idx]


def sample_q4_row(c_nodes, refgrid, R_t, cols_cm, *, tol=1e-12):
    """Q4 单个时刻一行：cols_cm=[0.0,...,1.9]（20 列）；域外 None。返回列表。

    表面列（「药材表面」）单独由 c_nodes[-1] 给出（调用方拼接）。
    """
    x_nodes = refgrid.x
    row = []
    for r_cm in cols_cm:
        r_m = r_cm / 100.0
        if not inside(r_m, R_t, tol):
            row.append(None)                       # 域外留空
            continue
        xj = r_m / R_t
        if xj >= 1.0 - tol:
            val = float(c_nodes[-1])               # 表面
        else:
            val = float(np.interp(xj, x_nodes, c_nodes))
        if not np.isfinite(val):
            raise ValueError(f"Q4 域内重构失败 r={r_cm} cm, R={R_t} m")
        row.append(val)
    return row


def last_valid_time_for_column(r_cm, radius_fn, t_grid_s):
    """某固定厘米列在给定时间网格上最后一个域内时刻（诊断用）。"""
    r_m = r_cm / 100.0
    last = None
    for t in t_grid_s:
        if inside(r_m, float(radius_fn.R(t))):
            last = float(t)
    return last
