"""solver_bdf.py —— BDF 时间积分（长时生产/情景候选，§6.0.4）。

- scipy.integrate.solve_ivp(method="BDF")，rtol/atol、jac_sparsity（2×2 块三对角+交叉块）。
- 断点（14400 s）分段重启；max_step：空气数据段 60 s，其后 600 s。
- 稠密输出（dense_output）在采样时刻求值；**稠密输出精度须另以更严容差/更小 max_step
  独立重算核验**（见 verify.py），不以「有无 t_eval 得同值」冒充精度。
- 事件：g(t,y)=max_i C_i(t)-0.15（direction=-1）在连续解上求根（=重构场根）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .operators import jac_sparsity


@dataclass
class BDFResult:
    segments: list = field(default_factory=list)   # 每段 solve_ivp 结果（含 dense_output）
    t_end: float = 0.0
    y_end: np.ndarray = None
    t_cross: float | None = None                   # 事件（Cmax 下穿 0.15）时刻
    y_cross: np.ndarray | None = None
    ok: bool = True
    message: str = ""

    def eval(self, t):
        """在已积分区间内用稠密输出求值（分段）。t 可为标量或数组。"""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((t.size, self.segments[0].y.shape[0]))
        for k, ti in enumerate(t):
            seg = self._find_segment(ti)
            out[k] = seg.sol(ti)
        return out

    def _find_segment(self, ti):
        for seg in self.segments:
            if seg.t[0] - 1e-9 <= ti <= seg.t[-1] + 1e-9:
                return seg
        # 落在末段之后：用末段外插（不推荐；调用方应保证在区间内）
        return self.segments[-1]


def _atol_vector(op, bdf):
    n = op.N + 1
    parts = [np.full(n, float(bdf["atol_C"])), np.full(n, float(bdf["atol_T_K"]))]
    if getattr(op, "augmented", False):
        parts.append(np.array([float(bdf.get("atol_I", bdf["atol_C"]))]))  # 累计通量 I 绝对容差
    return np.concatenate(parts)


def _cmax_event(op, threshold):
    n = op.N + 1

    def ev(t, y):
        return float(np.max(y[:n])) - threshold
    ev.direction = -1.0
    ev.terminal = True
    return ev


def integrate_bdf(op, y0, t0, t_end, bdf, *, breakpoints=(14400.0,),
                  threshold=None, air_data_end=14400.0):
    """从 t0 积分到 t_end，断点分段重启。若给 threshold 则设终端事件（Cmax 下穿）。"""
    rtol = float(bdf["rtol"])
    atol = _atol_vector(op, bdf)
    max_step_data = float(bdf["max_step_data_s"])
    max_step_after = float(bdf["max_step_after_s"])
    S = jac_sparsity(op.N, getattr(op, "augmented", False))

    # 段边界：t0、区间内断点、t_end
    bpts = sorted(b for b in breakpoints if t0 < b < t_end)
    edges = [t0] + bpts + [t_end]

    segments = []
    y = np.asarray(y0, dtype=np.float64).copy()
    t_cross = None
    y_cross = None
    events = _cmax_event(op, threshold) if threshold is not None else None

    for a, b in zip(edges[:-1], edges[1:]):
        max_step = max_step_data if b <= air_data_end + 1e-9 else max_step_after
        sol = solve_ivp(op.rhs, (a, b), y, method="BDF", rtol=rtol, atol=atol,
                        jac_sparsity=S, dense_output=True, max_step=max_step,
                        events=events)
        if not sol.success:
            return BDFResult(segments=segments, t_end=a, y_end=y, ok=False,
                             message=f"BDF 段 [{a},{b}] 失败：{sol.message}")
        segments.append(sol)
        y = sol.y[:, -1].copy()
        # 事件命中（terminal）→ 记录并停止
        if events is not None and sol.t_events is not None and len(sol.t_events[0]) > 0:
            t_cross = float(sol.t_events[0][0])
            y_cross = sol.y_events[0][0].copy()
            return BDFResult(segments=segments, t_end=sol.t[-1], y_end=y,
                             t_cross=t_cross, y_cross=y_cross, ok=True)

    return BDFResult(segments=segments, t_end=t_end, y_end=y,
                     t_cross=t_cross, y_cross=y_cross, ok=True)


def continue_bdf(op, y0, t0, extra_s, bdf, *, air_data_end=14400.0):
    """从 t0 续算 extra_s（用于事件后 600 s 异常检查）。返回 BDFResult（无事件）。"""
    return integrate_bdf(op, y0, t0, t0 + extra_s, bdf, breakpoints=(), threshold=None,
                         air_data_end=air_data_end)
