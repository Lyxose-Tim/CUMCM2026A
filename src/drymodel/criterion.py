"""criterion.py —— 达标判据（C_max < 0.15），F06 规则（§6.2、H14）。

- **重构场求根**：在与表格相同的未舍入分段线性重构场上求根
  g(a)=max_i[(1-a)C_i^n + a C_i^{n+1}]-0.15（**不先取端点极值再插值**）。
- BDF 用同一连续解（稠密输出）对 max_i C_i(t)-0.15 求根。
- **不回穿**依据比较原理（常数 0.15 为无源主线上解）：续算 600 s 只作数值异常检查
  （根后 C_max ≤ 0.15+1e-9）。
- t_sample：首个 60 s 网格点使实测（未舍入）C_max<0.15；t_safe：首个 C_max<0.15-δ（数值裕量）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


def cmax(C) -> float:
    """全域未舍入极值 = 节点最大值（分段线性重构）。"""
    return float(np.max(C))


def event_root_reconstructed(t_n, C_n, t_np1, C_np1, thr=0.15) -> float | None:
    """BE 单步：在重构场上求 g(a)=max_i[(1-a)C_i^n+a C_i^{n+1}]-thr=0 的根。

    返回 t_cross=t_n+a*·Δt；若步内不跨阈值返回 None。
    """
    C_n = np.asarray(C_n, float)
    C_np1 = np.asarray(C_np1, float)
    g0 = np.max(C_n) - thr
    g1 = np.max(C_np1) - thr
    if g0 <= 0:                       # 起点已达标（本步不产生首次向下穿越）
        return None
    if g1 > 0:                        # 终点仍未达标
        return None

    def g(a):
        return float(np.max((1.0 - a) * C_n + a * C_np1)) - thr

    a = brentq(g, 0.0, 1.0, xtol=1e-14, rtol=1e-14, maxiter=200)
    return t_n + a * (t_np1 - t_n)


def locate_cross_dense(cmax_fn, t_a, t_b, thr=0.15) -> float:
    """连续解（稠密输出）上求根：G(t)=max_i C_i(t)-thr=0，t∈[t_a,t_b]。"""
    def G(t):
        return cmax_fn(t) - thr
    return brentq(G, t_a, t_b, xtol=1e-9, rtol=1e-13, maxiter=200)


def first_sample_below(cmax_fn, t_start, step, thr=0.15, t_max=None) -> float | None:
    """首个 `step` 网格点 k·step ≥ t_start 使实测 cmax_fn < thr。"""
    k = int(np.ceil(t_start / step))
    t = k * step
    if t_max is None:
        t_max = t_start + 100 * step
    while t <= t_max + 1e-9:
        if cmax_fn(t) < thr:
            return float(t)
        t += step
    return None


@dataclass
class Detection:
    t_cross: float                    # 首次向下穿越（=严格达标集合下确界，即 t*）
    t_star: float
    t_sample: float | None            # 首个实测 <thr 的 60 s 点
    t_safe: float | None              # 首个 <thr-δ 的时刻（数值裕量）
    delta: float | None               # 实测 C 对照差；未运行时 None，禁止用 NaN/门槛代替
    dt_star: float | None             # 实测 t* 对照差；未运行时 None
    argmax_node: int                  # 穿越时刻极值所在节点
    post_ok: bool                     # 续算 600 s 未回穿
    post_max_cmax: float              # 续算段最大 cmax
    threshold: float = 0.15
