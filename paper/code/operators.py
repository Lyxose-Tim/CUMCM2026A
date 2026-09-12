"""operators.py —— 节点型径向 FVM 空间离散（散度形式；固定域与 Q4 参考坐标统一）。

对齐《A题_建模方案.md》§6.0.2 / §6.0.3：
- 状态排列 y = [C_0..C_N, T_0..T_N]（长度 2(N+1)）。
- 面通量（外向为正）F_{i+1/2} = -D_{i+1/2}(C_{i+1}-C_i)/Δ；界面系数 harmonic / integral。
- 表面第三类边界；中心零通量正则性。
- 固定域（Q1–Q3）：Δ=Δr，几何 R0，扩散无 1/R² 因子。
- 参考坐标（Q4）：Δ=Δx，扩散含 1/R(t)²，表面项含 1/R(t)，R 隐式步取步末值。
- **不得改写为 D∇²C**（保留散度形式，系数空间变化项在界面系数与守恒离散中体现）。

传输率（transmissibility）w_{i+1/2} = A_{i+1/2} D_{i+1/2} / Δ · scale，
其中固定域 scale=1，参考坐标 scale=1/R(t)²。
"""
from __future__ import annotations

import numpy as np

from . import props as props_mod


class FVMOperator:
    """一维径向 FVM 算子。固定域传 radius_fn=None；Q4 动域传 RadiusFunction。"""

    def __init__(self, grid, props, env, *, h: float, hm: float, R0: float,
                 interface: str = "harmonic", integral_npts: int = 8,
                 radius_fn=None, decoupled: bool = False, augmented: bool = False):
        self.grid = grid
        self.props = props
        self.env = env
        self.h = float(h)
        self.hm = float(hm)
        self.R0 = float(R0)
        self.interface = interface
        self.integral_npts = int(integral_npts)
        self.radius_fn = radius_fn
        self.is_reference = radius_fn is not None
        self.decoupled = bool(decoupled)
        # 累计通量辅助态 I（PATCH-05）：启用时状态为 y=[C..,T..,I]，İ=(2 h_m/R)(C_N−C_env)。
        self.augmented = bool(augmented)

        self.N = grid.N
        self.A_face = grid.A_face
        self.V = grid.V
        self.dspace = grid.dx if self.is_reference else grid.dr

    @property
    def n_state(self) -> int:
        """状态维数：物理 2(N+1)；启用增广态时 2(N+1)+1。"""
        return 2 * (self.N + 1) + (1 if self.augmented else 0)

    def flux_cbar(self, t, C) -> float:
        """归一化失水通量 f(t)=(2 h_m/R)(C_N−C_env)，满足 dC̄/dt=−f（固定域 R=R0，Q4 R(t)）。"""
        R = float(self.radius_fn.R(t)) if self.is_reference else self.R0
        return 2.0 * self.hm / R * (float(C[-1]) - float(self.env.C_env(t)))

    # ---- 时变几何因子 ----
    def _diff_scale(self, t: float) -> float:
        if self.is_reference:
            R = float(self.radius_fn.R(t))
            return 1.0 / (R * R)
        return 1.0

    def _surf_coeff(self, t: float) -> tuple[float, float]:
        """返回 (表面 C 系数, 表面 T 系数)。固定域=R0·h_m、R0·h；参考=h_m/R、h/R。"""
        if self.is_reference:
            R = float(self.radius_fn.R(t))
            return self.hm / R, self.h / R
        return self.R0 * self.hm, self.R0 * self.h

    # ---- 界面系数 ----
    def _Dface(self, C: np.ndarray, T: np.ndarray) -> np.ndarray:
        if self.interface == "integral":
            Tb = 0.5 * (T[:-1] + T[1:])
            return props_mod.D_face_integral(C[:-1], C[1:], Tb, self.props.D,
                                             npts=self.integral_npts)
        # harmonic
        Dn = self.props.D(C, T)
        return props_mod.D_face_harmonic(Dn[:-1], Dn[1:])

    def _kface(self, C: np.ndarray) -> np.ndarray:
        kn = self.props.k(C)
        return props_mod.k_face_harmonic(kn[:-1], kn[1:])

    # ---- 传输率 ----
    def w_C(self, C, T, t):
        return self.A_face * self._Dface(C, T) / self.dspace * self._diff_scale(t)

    def w_T(self, C, t):
        return self.A_face * self._kface(C) / self.dspace * self._diff_scale(t)

    # ---- 右端项（供 BDF/solve_ivp）----
    def split(self, y):
        """返回物理分量 (C, T)；增广态的 I 分量（若有）不在此返回。"""
        n = self.N + 1
        return y[:n], y[n:2 * n]

    def merge(self, C, T):
        return np.concatenate([C, T])

    def rhs(self, t, y):
        """dy/dt = [Ċ; Ṫ]（; İ 当增广态）。物理方程不依赖 I。"""
        C, T = self.split(y)
        N = self.N
        Cenv = float(self.env.C_env(t))
        Tair = float(self.env.T_air_K(t))
        sC, sT = self._surf_coeff(t)

        wC = self.w_C(C, T, t)
        wT = self.w_T(C, t)

        # C 方程
        dC = np.empty(N + 1)
        # 内部：w_{i-1/2}(C_{i-1}-C_i) + w_{i+1/2}(C_{i+1}-C_i)
        flux_in = wC * (C[:-1] - C[1:])          # 长度 N，face i+1/2 贡献给右侧节点 i+1
        # 对节点 i：来自左面 wC[i-1]*(C[i-1]-C[i]) 和右面 wC[i]*(C[i+1]-C[i])
        dC[1:N] = (wC[:N - 1] * (C[:N - 1] - C[1:N])
                   + wC[1:N] * (C[2:N + 1] - C[1:N])) / self.V[1:N]
        dC[0] = wC[0] * (C[1] - C[0]) / self.V[0]
        dC[N] = (wC[N - 1] * (C[N - 1] - C[N]) - sC * (C[N] - Cenv)) / self.V[N]

        # T 方程
        b = self.props.b(C)
        dT = np.empty(N + 1)
        dT[1:N] = (wT[:N - 1] * (T[:N - 1] - T[1:N])
                   + wT[1:N] * (T[2:N + 1] - T[1:N])) / (b[1:N] * self.V[1:N])
        dT[0] = wT[0] * (T[1] - T[0]) / (b[0] * self.V[0])
        dT[N] = (wT[N - 1] * (T[N - 1] - T[N]) - sT * (T[N] - Tair)) / (b[N] * self.V[N])

        _ = flux_in  # 保留可读性（未直接使用）
        if self.augmented:
            dI = self.flux_cbar(t, C)                 # İ = (2 h_m/R)(C_N − C_env)
            return np.concatenate([dC, dT, [dI]])
        return self.merge(dC, dT)

    # ---- 后向 Euler 隐式装配（供 solver_be；系数在 Picard 迭代中冻结）----
    def assemble_C_banded(self, C_iter, T_iter, C_old, dt, t_np1):
        """返回 (ab, rhs) 供 scipy.linalg.solve_banded 解 C^{n+1}。系数用当前迭代值。"""
        N = self.N
        Cenv = float(self.env.C_env(t_np1))
        sC, _ = self._surf_coeff(t_np1)
        w = self.w_C(C_iter, T_iter, t_np1)      # 长度 N
        diag = np.empty(N + 1)
        lower = np.zeros(N + 1)                   # lower[i] 乘 C_{i-1}
        upper = np.zeros(N + 1)                   # upper[i] 乘 C_{i+1}
        rhs = np.empty(N + 1)

        VdT = self.V / dt
        # 节点 0
        diag[0] = VdT[0] + w[0]
        upper[0] = -w[0]
        rhs[0] = VdT[0] * C_old[0]
        # 内部
        diag[1:N] = VdT[1:N] + w[:N - 1] + w[1:N]
        lower[1:N] = -w[:N - 1]
        upper[1:N] = -w[1:N]
        rhs[1:N] = VdT[1:N] * C_old[1:N]
        # 节点 N（表面 Robin）
        diag[N] = VdT[N] + w[N - 1] + sC
        lower[N] = -w[N - 1]
        rhs[N] = VdT[N] * C_old[N] + sC * Cenv

        ab = _to_banded(lower, diag, upper)
        return ab, rhs

    def assemble_T_banded(self, C_iter, T_iter, T_old, dt, t_np1):
        N = self.N
        Tair = float(self.env.T_air_K(t_np1))
        _, sT = self._surf_coeff(t_np1)
        w = self.w_T(C_iter, t_np1)
        b = self.props.b(C_iter)
        diag = np.empty(N + 1)
        lower = np.zeros(N + 1)
        upper = np.zeros(N + 1)
        rhs = np.empty(N + 1)

        bVdT = b * self.V / dt
        diag[0] = bVdT[0] + w[0]
        upper[0] = -w[0]
        rhs[0] = bVdT[0] * T_old[0]
        diag[1:N] = bVdT[1:N] + w[:N - 1] + w[1:N]
        lower[1:N] = -w[:N - 1]
        upper[1:N] = -w[1:N]
        rhs[1:N] = bVdT[1:N] * T_old[1:N]
        diag[N] = bVdT[N] + w[N - 1] + sT
        lower[N] = -w[N - 1]
        rhs[N] = bVdT[N] * T_old[N] + sT * Tair

        ab = _to_banded(lower, diag, upper)
        return ab, rhs

    def residual_rate(self, t, y, ydot):
        """速率形式残差 N = rhs(t,y) - ydot（单位 kg/kg/s、K/s）。"""
        return self.rhs(t, y) - ydot

    def cbar(self, C):
        return self.grid.cbar(C)

    def surface_flux_C(self, t, C):
        """归一化表面失水通量 -D∂_rC|_R = h_m(C_N - C_env)（固定域口径）。"""
        return self.hm * (C[-1] - float(self.env.C_env(t)))


def _to_banded(lower, diag, upper):
    """三对角 → scipy.linalg.solve_banded 的 (l=1,u=1) ab 矩阵。

    ab[0,1:] = upper[:-1]（超对角），ab[1,:] = diag，ab[2,:-1] = lower[1:]（次对角）。
    """
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return ab


def jac_sparsity(N: int, augmented: bool = False):
    """BDF 稀疏雅可比模式：2×2 块三对角 + 交叉块（T→D，C→b/k）。

    augmented=True 时扩到 2n+1：辅助行 İ 仅依赖表面 C 节点（列 N）；物理方程不依赖 I。
    """
    from scipy.sparse import lil_matrix
    n = N + 1
    m = 2 * n + (1 if augmented else 0)
    S = lil_matrix((m, m), dtype=np.int8)
    for i in range(n):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                S[i, j] = 1           # C-C 块
                S[i, n + j] = 1       # C-T 交叉块（T→D）
                S[n + i, n + j] = 1   # T-T 块
                S[n + i, j] = 1       # T-C 交叉块（C→b/k）
    if augmented:
        S[2 * n, N] = 1               # İ 依赖表面 C 节点（数组索引 N）
        S[2 * n, 2 * n] = 1           # 对角占位（∂İ/∂I=0，结构上包含无害）
    return S.tocsr()
