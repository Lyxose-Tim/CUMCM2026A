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

    @property
    def t_start(self) -> float:
        if not self.segments:
            raise ValueError("BDF 轨迹为空，没有可求值起点")
        return float(self.segments[0].t[0])

    @property
    def state_size(self) -> int:
        if not self.segments:
            raise ValueError("BDF 轨迹为空，没有状态维数")
        return int(self.segments[0].y.shape[0])

    def eval(self, t):
        """在已积分区间内用稠密输出求值（分段）。t 可为标量或数组。"""
        if not self.ok:
            raise RuntimeError(f"不能对失败的 BDF 轨迹求值：{self.message}")
        if not self.segments:
            raise ValueError("BDF 轨迹为空，不能求值")
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if not np.all(np.isfinite(t)):
            raise ValueError("BDF 查询时刻必须全部有限")
        out = np.empty((t.size, self.state_size))
        assigned = np.zeros(t.size, dtype=bool)
        for seg in self.segments:
            mask = (~assigned) & (t >= seg.t[0] - 1e-9) & (t <= seg.t[-1] + 1e-9)
            if np.any(mask):
                out[mask] = np.asarray(seg.sol(t[mask]), dtype=float).T
                assigned[mask] = True
        if not np.all(assigned):
            bad = float(t[np.flatnonzero(~assigned)[0]])
            self._find_segment(bad)  # 统一抛出带合法区间的异常
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("BDF 稠密输出含非有限状态")
        return out

    def _find_segment(self, ti):
        for seg in self.segments:
            if seg.t[0] - 1e-9 <= ti <= seg.t[-1] + 1e-9:
                return seg
        raise ValueError(
            f"查询时刻 {float(ti):.12g}s 超出已积分区间 "
            f"[{self.t_start:.12g}, {float(self.t_end):.12g}]s"
        )

    def require_reached(self, requested_end: float, *, label: str = "BDF") -> None:
        """确认成功且实际到达请求终点；事件轨迹应传真实事件终点。"""
        if not self.ok:
            raise RuntimeError(f"{label} 失败：{self.message}")
        tol = 1e-9 * max(1.0, abs(float(requested_end)))
        if abs(float(self.t_end) - float(requested_end)) > tol:
            raise RuntimeError(
                f"{label} 未到请求终点：实际 {self.t_end:.12g}s，"
                f"请求 {float(requested_end):.12g}s"
            )
        if self.y_end is None or not np.all(np.isfinite(self.y_end)):
            raise FloatingPointError(f"{label} 终点状态缺失或非有限")

    def concat(self, other: "BDFResult", *, state_atol: float = 1e-10) -> "BDFResult":
        """拼接连续成功轨迹；不重算、不重置增广态。"""
        if not self.ok or not other.ok:
            messages = "; ".join(x.message for x in (self, other) if not x.ok)
            raise RuntimeError(f"不能拼接失败轨迹：{messages}")
        if not self.segments or not other.segments:
            raise ValueError("不能拼接空 BDF 轨迹")
        scale = max(1.0, abs(float(self.t_end)), abs(float(other.t_start)))
        if abs(float(self.t_end) - float(other.t_start)) > 1e-9 * scale:
            raise ValueError(
                f"BDF 轨迹时间不连续：{self.t_end:.12g}s != {other.t_start:.12g}s"
            )
        if self.state_size != other.state_size:
            raise ValueError(
                f"BDF 轨迹状态维数不一致：{self.state_size} != {other.state_size}"
            )
        other_start = np.asarray(other.segments[0].y[:, 0], dtype=float)
        if self.y_end is None:
            raise ValueError("前一条 BDF 轨迹缺少终点状态")
        prior_end = np.asarray(self.y_end, dtype=float)
        if not np.allclose(prior_end, other_start, rtol=1e-9, atol=state_atol):
            gap = float(np.max(np.abs(prior_end - other_start)))
            raise ValueError(f"BDF 轨迹状态不连续，最大接缝差 {gap:.3e}")
        return BDFResult(
            segments=[*self.segments, *other.segments],
            t_end=float(other.t_end),
            y_end=np.asarray(other.y_end, dtype=float).copy(),
            t_cross=self.t_cross if self.t_cross is not None else other.t_cross,
            y_cross=(None if self.y_cross is None else np.asarray(self.y_cross).copy()),
            ok=True,
        )


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
    t0 = float(t0)
    t_end = float(t_end)
    if not (np.isfinite(t0) and np.isfinite(t_end) and t_end > t0):
        raise ValueError(f"BDF 积分区间须为有限递增区间，得 [{t0}, {t_end}]")
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
    if y.ndim != 1 or y.size != op.n_state or not np.all(np.isfinite(y)):
        raise ValueError(
            f"BDF 初态须为长度 {op.n_state} 的一维有限数组，得 shape={y.shape}"
        )
    t_cross = None
    y_cross = None
    events = _cmax_event(op, threshold) if threshold is not None else None

    for a, b in zip(edges[:-1], edges[1:]):
        max_step = max_step_data if b <= air_data_end + 1e-9 else max_step_after
        sol = solve_ivp(op.rhs, (a, b), y, method="BDF", rtol=rtol, atol=atol,
                        jac_sparsity=S, dense_output=True, max_step=max_step,
                        events=events)
        if not sol.success:
            actual_t = float(sol.t[-1]) if sol.t.size else float(a)
            actual_y = sol.y[:, -1].copy() if sol.y.size else y.copy()
            return BDFResult(segments=segments, t_end=actual_t, y_end=actual_y, ok=False,
                             message=f"BDF 段 [{a},{b}] 失败：{sol.message}")
        segments.append(sol)
        y = sol.y[:, -1].copy()
        if not np.all(np.isfinite(y)):
            return BDFResult(segments=segments, t_end=float(sol.t[-1]), y_end=y,
                             ok=False, message=f"BDF 段 [{a},{b}] 产生非有限状态")
        # 事件命中（terminal）→ 记录并停止
        if events is not None and sol.t_events is not None and len(sol.t_events[0]) > 0:
            t_cross = float(sol.t_events[0][0])
            y_cross = sol.y_events[0][0].copy()
            return BDFResult(segments=segments, t_end=sol.t[-1], y_end=y,
                             t_cross=t_cross, y_cross=y_cross, ok=True)
        tol = 1e-9 * max(1.0, abs(float(b)))
        if abs(float(sol.t[-1]) - float(b)) > tol:
            return BDFResult(segments=segments, t_end=float(sol.t[-1]), y_end=y,
                             ok=False,
                             message=f"BDF 段未到请求终点：{sol.t[-1]} != {b}")

    return BDFResult(segments=segments, t_end=t_end, y_end=y,
                     t_cross=t_cross, y_cross=y_cross, ok=True)


def continue_bdf(op, y0, t0, extra_s, bdf, *, air_data_end=14400.0):
    """从 t0 续算 extra_s（用于事件后 600 s 异常检查）。返回 BDFResult（无事件）。"""
    return integrate_bdf(op, y0, t0, t0 + extra_s, bdf, breakpoints=(), threshold=None,
                         air_data_end=air_data_end)
