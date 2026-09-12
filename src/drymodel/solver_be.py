"""solver_be.py —— 后向 Euler（BE）时间积分，验证基线（B06、§6.0.4）。

- 每步 Picard：用最新迭代值算 D/k/b → 解 C 三对角 → 解 T 三对角。
- 收敛：增量 ∞-范数 ||ΔC||≤1e-9、||ΔT||≤1e-7 K；且速率残差
  ||N_C||≤1e-10·C0/Δt、||N_T||≤1e-8 K/s。最多 30 次。
- 失败重试：未收敛或出现 C≤0 → Δt 减半（最小 1e-3 s，最多 10 次），
  子步终点重新对齐到整数秒；仍失败则终止并返回状态快照。
- 采样：每 1 s 步末状态即输出行（无插值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class BEResult:
    t: np.ndarray                       # 记录时刻
    C: np.ndarray                       # 形状 (len(t), N+1)
    T: np.ndarray
    ok: bool
    message: str = ""
    total_steps: int = 0
    total_halvings: int = 0
    accepted_substeps: int = 0
    t_end: float = 0.0
    y_end: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def require_reached(self, requested_end: float, *, label: str = "BE") -> None:
        if not self.ok:
            raise RuntimeError(f"{label} 失败：{self.message}")
        tol = 1e-10 * max(1.0, abs(float(requested_end)))
        if abs(self.t_end - float(requested_end)) > tol:
            raise RuntimeError(
                f"{label} 未到请求终点：实际 {self.t_end:.12g}s，"
                f"请求 {float(requested_end):.12g}s"
            )
        if self.y_end is None or not np.all(np.isfinite(self.y_end)):
            raise FloatingPointError(f"{label} 终点状态缺失或非有限")


@dataclass(frozen=True)
class AcceptedStep:
    """一个真正被接受的 BE 子步；失败尝试不会进入账本。"""

    t0: float
    t1: float
    y0: np.ndarray
    y1: np.ndarray

    @property
    def dt(self) -> float:
        return self.t1 - self.t0


def step_be(op, y_old, t_old, dt, *, C0_ref, picard):
    """单个 BE 步。返回 (y_new, converged, iters, min_C)。"""
    tol_dC = float(picard["tol_dC"])
    tol_dT = float(picard["tol_dT_K"])
    tol_res_C_rel = float(picard["tol_res_C_rel"])
    tol_res_T = float(picard["tol_res_T_K_per_s"])
    max_iter = int(picard["max_iter"])

    n = op.N + 1
    t_np1 = t_old + dt
    C_old, T_old = op.split(y_old)
    C_it = C_old.copy()
    T_it = T_old.copy()

    converged = False
    iters = 0
    for iters in range(1, max_iter + 1):
        abC, rhsC = op.assemble_C_banded(C_it, T_it, C_old, dt, t_np1)
        C_new = solve_banded((1, 1), abC, rhsC)
        abT, rhsT = op.assemble_T_banded(C_new, T_it, T_old, dt, t_np1)
        T_new = solve_banded((1, 1), abT, rhsT)

        dC = np.max(np.abs(C_new - C_it))
        dT = np.max(np.abs(T_new - T_it))
        C_it, T_it = C_new, T_new

        if dC <= tol_dC and dT <= tol_dT:
            # 速率残差校核
            y_new = op.merge(C_it, T_it)
            ydot = (y_new - y_old) / dt
            res = op.residual_rate(t_np1, y_new, ydot)
            rC = np.max(np.abs(res[:n]))
            rT = np.max(np.abs(res[n:]))
            if rC <= tol_res_C_rel * C0_ref / dt and rT <= tol_res_T:
                converged = True
                break

    y_new = op.merge(C_it, T_it)
    min_C = float(np.min(C_it))
    return y_new, converged, iters, min_C


def integrate_be(op, y0, t_end, dt, *, C0_ref, picard, retry,
                 record_times=None, breakpoints=(), scalar_recorder=None,
                 step_recorder=None):
    """从 t=0 积分到 t_end（含），基础步长 dt（通常 1 s）。

    record_times: 需保存整场状态的时刻集合（默认 = 每个整数秒到 t_end）。
    scalar_recorder(t, C, T): 可选，用于记录标量诊断（Cmax、Cbar、表面等）。
    """
    dt = float(dt)
    t_end = float(t_end)
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(t_end) and t_end >= 0):
        raise ValueError(f"BE 的 t_end/dt 非法：t_end={t_end!r}, dt={dt!r}")
    if getattr(op, "augmented", False):
        raise ValueError("BE 当前仅支持物理状态；累计通量增广态由 BDF 使用")
    if record_times is None:
        record_times = set(float(x) for x in np.arange(0.0, t_end + 0.5 * dt, dt)
                           if x <= t_end + 1e-12)
    else:
        record_times = set(float(x) for x in record_times)
    if not all(np.isfinite(x) and -1e-12 <= x <= t_end + 1e-12 for x in record_times):
        raise ValueError("BE record_times 必须位于积分区间 [0,t_end]")
    halvings_max = int(retry["halvings_max"])
    dt_min = float(retry["dt_min_s"])

    rec_t, rec_C, rec_T = [], [], []
    t = 0.0
    y = np.asarray(y0, dtype=np.float64).copy()
    if y.ndim != 1 or y.size != op.n_state or not np.all(np.isfinite(y)):
        raise ValueError(f"BE 初态须为长度 {op.n_state} 的一维有限数组")

    def should_record(value):
        return any(abs(value - target) <= 1e-10 * max(1.0, abs(target))
                   for target in record_times)

    # 记录初值
    if should_record(0.0):
        C0v, T0v = op.split(y)
        rec_t.append(0.0); rec_C.append(C0v.copy()); rec_T.append(T0v.copy())
    if scalar_recorder is not None:
        C0v, T0v = op.split(y)
        scalar_recorder(0.0, C0v, T0v)

    total_steps = 0
    total_halvings = 0
    accepted_substeps = 0
    step_sizes = []
    bpts = sorted(float(b) for b in breakpoints if 0.0 < float(b) < t_end)
    target_edges = [*bpts, t_end]
    edge_idx = 0
    while t < t_end - 1e-12:
        while edge_idx < len(target_edges) and target_edges[edge_idx] <= t + 1e-12:
            edge_idx += 1
        next_edge = target_edges[edge_idx] if edge_idx < len(target_edges) else t_end
        grid_dt = min(dt, next_edge - t, t_end - t)
        y_new, t_new, hv, ok, accepted = _advance_one_grid_step(
            op, y, t, grid_dt, C0_ref=C0_ref, picard=picard,
            halvings_max=halvings_max, dt_min=dt_min)
        total_steps += 1
        total_halvings += hv
        if not ok:
            return BEResult(np.array(rec_t), np.array(rec_C), np.array(rec_T),
                            ok=False, message=f"步进失败于 t={t:.6f}s",
                            total_steps=total_steps, total_halvings=total_halvings,
                            accepted_substeps=accepted_substeps, t_end=float(t),
                            y_end=y.copy(), diagnostics={"accepted_step_sizes": step_sizes})
        for accepted_step in accepted:
            accepted_substeps += 1
            step_sizes.append(float(accepted_step.dt))
            if step_recorder is not None:
                step_recorder(accepted_step)
            C_s, T_s = op.split(accepted_step.y1)
            if should_record(accepted_step.t1):
                rec_t.append(float(accepted_step.t1))
                rec_C.append(C_s.copy())
                rec_T.append(T_s.copy())
            if scalar_recorder is not None:
                scalar_recorder(accepted_step.t1, C_s, T_s)
        y, t = y_new, t_new

    return BEResult(np.array(rec_t), np.array(rec_C), np.array(rec_T),
                    ok=True, total_steps=total_steps, total_halvings=total_halvings,
                    accepted_substeps=accepted_substeps, t_end=float(t), y_end=y.copy(),
                    diagnostics={"accepted_step_sizes": step_sizes})


def _advance_one_grid_step(op, y, t, dt, *, C0_ref, picard, halvings_max, dt_min):
    """推进一个基础网格步（整数秒）；失败则内部减步并再对齐到步末整数秒。"""
    y_new, conv, _, min_C = step_be(op, y, t, dt, C0_ref=C0_ref, picard=picard)
    if conv and min_C > 0.0:
        accepted = [AcceptedStep(float(t), float(t + dt), y.copy(), y_new.copy())]
        return y_new, t + dt, 0, True, accepted

    # 减步重试：把 [t, t+dt] 细分为 2^k 子步，子步终点仍落在整数秒边界
    total_hv = 0
    for k in range(1, halvings_max + 1):
        sub_dt = dt / (2 ** k)
        if sub_dt < dt_min:
            break
        nsub = 2 ** k
        yy = y.copy()
        tt = t
        good = True
        accepted = []
        for _ in range(nsub):
            yy2, c2, _, mc2 = step_be(op, yy, tt, sub_dt, C0_ref=C0_ref, picard=picard)
            if not (c2 and mc2 > 0.0):
                good = False
                break
            accepted.append(AcceptedStep(float(tt), float(tt + sub_dt),
                                         yy.copy(), yy2.copy()))
            yy, tt = yy2, tt + sub_dt
        total_hv = k
        if good:
            return yy, t + dt, total_hv, True, accepted
    return y, t, total_hv, False, []
