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
    cum_flux: float = 0.0               # 真实子步累计通量 I_BE=Σ Δt_sub·f(step-end)（W4）
    diagnostics: dict = field(default_factory=dict)


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
                 flux_recorder=None):
    """从 t=0 积分到 t_end（含），基础步长 dt（通常 1 s）。

    record_times: 需保存整场状态的时刻集合（默认 = 每个整数秒到 t_end）。
    scalar_recorder(t, C, T): 可选，记录标量诊断。
    flux_recorder(t_end, dt_sub, f): 可选，每**实际(子)步**回调真实通量（W4）。
    返回 BEResult；累计通量 cum_flux 用各实际子步步长（非固定 1 s）。失败 ok=False 显式传播。
    """
    dt = float(dt)
    if record_times is None:
        record_times = set(int(round(x)) for x in np.arange(0, t_end + 1, dt))
    else:
        record_times = set(int(round(x)) for x in record_times)
    halvings_max = int(retry["halvings_max"])
    dt_min = float(retry["dt_min_s"])

    rec_t, rec_C, rec_T = [], [], []
    t = 0.0
    y = np.asarray(y0, dtype=np.float64).copy()
    cum_flux = 0.0

    # 记录初值
    if 0 in record_times:
        C0v, T0v = op.split(y)
        rec_t.append(0.0); rec_C.append(C0v.copy()); rec_T.append(T0v.copy())
    if scalar_recorder is not None:
        C0v, T0v = op.split(y)
        scalar_recorder(0.0, C0v, T0v)
    if flux_recorder is not None:
        C0v, _ = op.split(y)
        flux_recorder(0.0, 0.0, float(op.flux_cbar(0.0, C0v)))   # 初始通量点（dt_sub=0）

    total_steps = 0
    total_halvings = 0
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        y, t, hv, ok, substeps = _advance_one_grid_step(
            op, y, t, dt, C0_ref=C0_ref, picard=picard,
            halvings_max=halvings_max, dt_min=dt_min)
        total_steps += 1
        total_halvings += hv
        # 真实子步累计通量（W4：变步长权重）
        for (t_sub, dt_sub, y_sub) in substeps:
            C_sub, _ = op.split(y_sub)
            f_sub = float(op.flux_cbar(t_sub, C_sub))
            cum_flux += dt_sub * f_sub
            if flux_recorder is not None:
                flux_recorder(t_sub, dt_sub, f_sub)
        if not ok:
            return BEResult(np.array(rec_t), np.array(rec_C), np.array(rec_T),
                            ok=False, message=f"步进失败于 t={t:.6f}s",
                            total_steps=total_steps, total_halvings=total_halvings,
                            cum_flux=cum_flux)
        ti = int(round(t))
        if ti in record_times and abs(t - ti) < 1e-9:
            C_s, T_s = op.split(y)
            rec_t.append(float(ti)); rec_C.append(C_s.copy()); rec_T.append(T_s.copy())
        if scalar_recorder is not None:
            C_s, T_s = op.split(y)
            scalar_recorder(t, C_s, T_s)

    return BEResult(np.array(rec_t), np.array(rec_C), np.array(rec_T),
                    ok=True, total_steps=total_steps, total_halvings=total_halvings,
                    cum_flux=cum_flux)


def _advance_one_grid_step(op, y, t, dt, *, C0_ref, picard, halvings_max, dt_min):
    """推进一个基础网格步（整数秒）；失败则内部减步并再对齐到步末整数秒。

    返回 (y_new, t_new, halvings, ok, substeps)；substeps 为**实际接受的子步**列表
    [(t_end, dt_sub, y_end), ...]，供真实子步通量累计（W4）。
    """
    y_new, conv, _, min_C = step_be(op, y, t, dt, C0_ref=C0_ref, picard=picard)
    if conv and min_C > 0.0:
        return y_new, t + dt, 0, True, [(t + dt, dt, y_new)]

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
        substeps = []
        for _ in range(nsub):
            yy2, c2, _, mc2 = step_be(op, yy, tt, sub_dt, C0_ref=C0_ref, picard=picard)
            if not (c2 and mc2 > 0.0):
                good = False
                break
            yy, tt = yy2, tt + sub_dt
            substeps.append((tt, sub_dt, yy))
        total_hv = k
        if good:
            return yy, t + dt, total_hv, True, substeps
    return y, t, total_hv, False, []
