"""grid.py —— 节点型径向 FVM 网格（固定域）与参考坐标网格（Q4 动域）。

对齐《A题_建模方案.md》§6.0.2 / §6.0.3（B05；唯一执行规则：节点型）：
- 节点 r_i = i·Δr，i=0..N；面 r_{i+1/2}=(i+½)Δr；面积 A_{i+1/2}=r_{i+1/2}。
- 控制体：V_0=Δr²/8，V_i=r_iΔr（1≤i≤N-1），V_N=R0Δr/2−Δr²/8。
- 中心为节点 0，表面为节点 N；节点值即输出值；分段线性重构全域最大值=节点最大值。
- 守恒度量：ΣV_i = R0²/2，故 C̄ = (2/R0²)ΣV_iC_i。参考网格 ΣV_i = 1/2，C̄ = 2ΣV_ic_i。
"""
from __future__ import annotations

import numpy as np

_NODE_TOL = 1e-9  # 判定输出位置是否落在网格节点上的相对容差


class RadialGrid:
    """固定域 [0, R0] 的节点型径向网格。"""

    def __init__(self, N: int, R0: float):
        if not (isinstance(N, int) and N > 0):
            raise ValueError(f"N 须为正整数，得 {N!r}")
        self.N = N
        self.R0 = float(R0)
        self.dr = self.R0 / N

        self.r = np.arange(N + 1) * self.dr                 # 节点 r_0..r_N
        self.r_face = (np.arange(N) + 0.5) * self.dr        # 面 r_{1/2}..r_{N-1/2}
        self.A_face = self.r_face.copy()                    # 面积 = r_{i+1/2}

        V = np.empty(N + 1)
        V[0] = self.dr ** 2 / 8.0
        V[1:N] = self.r[1:N] * self.dr
        V[N] = self.R0 * self.dr / 2.0 - self.dr ** 2 / 8.0
        self.V = V

        # 守恒量归一化：C̄ = (2/R0²) Σ V_i C_i
        self.cbar_weight = 2.0 / self.R0 ** 2 * self.V

    def output_index(self, r_cm: float) -> int:
        """固定厘米位置 r_cm（如 0.0,0.1,...,2.0）对应的节点下标；不在节点上抛错。"""
        r_m = r_cm / 100.0
        idx_f = r_m / self.dr
        idx = int(round(idx_f))
        if idx < 0 or idx > self.N:
            raise ValueError(f"输出位置 {r_cm} cm 超出网格 [0, {self.R0 * 100} cm]")
        if abs(idx * self.dr - r_m) > _NODE_TOL * self.R0:
            raise ValueError(
                f"输出位置 {r_cm} cm 未落在网格节点上（N={self.N} 须为 20 的倍数）")
        return idx

    def cbar(self, C: np.ndarray) -> float:
        return float(np.dot(self.cbar_weight, C))


class RefGrid:
    """Q4 动域参考坐标 x=r/R(t) ∈ [0,1] 的节点型网格。"""

    def __init__(self, N: int):
        if not (isinstance(N, int) and N > 0):
            raise ValueError(f"N 须为正整数，得 {N!r}")
        self.N = N
        self.dx = 1.0 / N

        self.x = np.arange(N + 1) * self.dx
        self.x_face = (np.arange(N) + 0.5) * self.dx
        self.A_face = self.x_face.copy()

        V = np.empty(N + 1)
        V[0] = self.dx ** 2 / 8.0
        V[1:N] = self.x[1:N] * self.dx
        V[N] = self.dx / 2.0 - self.dx ** 2 / 8.0
        self.V = V

        # C̄ = 2 Σ V_i c_i
        self.cbar_weight = 2.0 * self.V

    def cbar(self, c: np.ndarray) -> float:
        return float(np.dot(self.cbar_weight, c))
