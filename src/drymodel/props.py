"""props.py —— 题给经验物性（附录 2/3/4）与界面系数。

设计要点（对齐《A题_建模方案.md》§6.0.0/§6.0.2、§9.3）：
- 附录 2/3/4 公式**原样采用，不重新拟合**；D 中的 T 为药材局部绝对温度 (K)。
- 防护（§6.0.4）：C<=0 → D:=0（不裁剪状态，接受解中出现由求解器判失败并计数）；
  指数 e^{-β/C} 的指数参数 <-700 取 0（避免下溢告警），不抬高 D 下限、不取绝对值。
- 界面系数两法：调和平均 `harmonic`（B11 默认）与积分界面 `integral`
  （8 点 Gauss–Legendre，16 点对照；近等值极限自动退化，无有效数字消去）。
- 不使用 eval：全部为具名函数实现，config 中的公式字符串仅作说明。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

# 防护常量（与 A题_config.yaml props.guards 一致）
EXP_UNDERFLOW_ARG = -700.0
C_FLOOR_FOR_EXP = 1.0e-300


def _exp_neg_over(beta: float, C: np.ndarray) -> np.ndarray:
    """安全计算 exp(-beta / C)，仅用于 C>0；C<=0 的位置返回 0（配合 D:=0）。

    指数参数 arg = -beta/C；arg < -700 时取 0（§6.0.4）。
    """
    C = np.asarray(C, dtype=np.float64)
    out = np.zeros_like(C)
    pos = C > 0.0
    if np.any(pos):
        arg = -beta / C[pos]
        keep = arg >= EXP_UNDERFLOW_ARG
        vals = np.zeros_like(arg)
        vals[keep] = np.exp(arg[keep])
        out[pos] = vals
    return out


def _exp_neg_over_T(gamma: float, T: np.ndarray) -> np.ndarray:
    """计算 exp(-gamma / T)，T 为绝对温度 (K)，恒正；指数 <-700 取 0。"""
    T = np.asarray(T, dtype=np.float64)
    arg = -gamma / T
    out = np.zeros_like(arg)
    keep = arg >= EXP_UNDERFLOW_ARG
    out[keep] = np.exp(arg[keep])
    return out


# --------------------------------------------------------------------------
# 附录 2（Q1）：常物性 + D 只依赖 C
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PropsQ1:
    rho_const: float = 820.0
    cp_const: float = 2600.0
    k_const: float = 0.36
    D_pref: float = 7.0e-9
    D_beta: float = 0.89

    def rho(self, C):
        C = np.asarray(C, dtype=np.float64)
        return np.full_like(C, self.rho_const)

    def cp(self, C):
        C = np.asarray(C, dtype=np.float64)
        return np.full_like(C, self.cp_const)

    def k(self, C):
        C = np.asarray(C, dtype=np.float64)
        return np.full_like(C, self.k_const)

    def b(self, C):
        """有效体积热容 b = rho * cp。"""
        return self.rho(C) * self.cp(C)

    def D(self, C, T=None):
        """D = 7e-9 * exp(-0.89/C)；不依赖 T（热质解耦）。"""
        return self.D_pref * _exp_neg_over(self.D_beta, C)


# --------------------------------------------------------------------------
# 附录 3（Q2/Q3）：变物性，D(C,T) 双向耦合
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PropsQ23:
    D_pref: float = 2.4e-3
    D_beta: float = 0.45
    D_gamma: float = 3850.0

    def rho(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 650.0 + 128.0 * C

    def cp(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 1450.0 + 2736.0 * C / (C + 1.0)

    def k(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 0.21 + 0.38 * C / (C + 1.0)

    def b(self, C):
        return self.rho(C) * self.cp(C)

    def D(self, C, T):
        """D = 2.4e-3 * exp(-0.45/C) * exp(-3850/T)，T 单位 K。"""
        return self.D_pref * _exp_neg_over(self.D_beta, C) * _exp_neg_over_T(self.D_gamma, T)


# --------------------------------------------------------------------------
# 附录 4（Q4）：变物性 + 动域
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PropsQ4:
    D_pref: float = 4.2e-4
    D_beta: float = 0.30
    D_gamma: float = 3850.0

    def rho(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 760.0 + 90.0 * C

    def cp(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 1850.0 + 2150.0 * C / (C + 1.0)

    def k(self, C):
        C = np.asarray(C, dtype=np.float64)
        return 0.12 + 0.20 * C / (C + 1.0)

    def b(self, C):
        return self.rho(C) * self.cp(C)

    def D(self, C, T):
        """D = 4.2e-4 * exp(-0.30/C) * exp(-3850/T)，T 单位 K。"""
        return self.D_pref * _exp_neg_over(self.D_beta, C) * _exp_neg_over_T(self.D_gamma, T)


PROPS_BY_QUESTION = {"q1": PropsQ1, "q23": PropsQ23, "q4": PropsQ4}


def make_props(question: str):
    key = question.lower()
    if key not in PROPS_BY_QUESTION:
        raise KeyError(f"未知物性组 {question!r}，应为 q1/q23/q4")
    return PROPS_BY_QUESTION[key]()


# --------------------------------------------------------------------------
# 界面系数
# --------------------------------------------------------------------------
def D_face_harmonic(Di, Dj):
    """调和平均界面系数 2 Di Dj / (Di + Dj)；任一为 0 则取 0。"""
    Di = np.asarray(Di, dtype=np.float64)
    Dj = np.asarray(Dj, dtype=np.float64)
    s = Di + Dj
    out = np.zeros_like(s)
    nz = s > 0.0
    out[nz] = 2.0 * Di[nz] * Dj[nz] / s[nz]
    return out


# 预计算 Gauss–Legendre 节点（[-1,1]），映射到 [0,1] 时权重减半。
_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss01(npts: int) -> tuple[np.ndarray, np.ndarray]:
    if npts not in _GAUSS_CACHE:
        u, w = leggauss(npts)          # 区间 [-1,1]
        xi = 0.5 * (u + 1.0)           # 映射到 [0,1]
        wt = 0.5 * w
        _GAUSS_CACHE[npts] = (xi, wt)
    return _GAUSS_CACHE[npts]


def D_face_integral(Ci, Cj, Tb, Dfun, npts: int = 8):
    """积分界面系数：D^I = ∫_0^1 D((1-ξ)Ci + ξ Cj, Tb) dξ（§6.0.2）。

    - `Dfun(C, T)`：物性对象的 D 方法（附录 2 忽略 T）。
    - `Tb`：界面平均温度 (K)，标量或与 Ci/Cj 同形数组。
    - 8 点 Gauss–Legendre 求值（16 点作对照）。求值形式无原函数相减，
      近等值极限 Cj→Ci 时自动退化为 D(Ci, Tb)，无有效数字消去。
    """
    Ci = np.asarray(Ci, dtype=np.float64)
    Cj = np.asarray(Cj, dtype=np.float64)
    Tb = np.asarray(Tb, dtype=np.float64)
    xi, wt = _gauss01(npts)
    acc = np.zeros(np.broadcast(Ci, Cj, Tb).shape, dtype=np.float64)
    for xk, wk in zip(xi, wt):
        Cmid = (1.0 - xk) * Ci + xk * Cj
        acc = acc + wk * Dfun(Cmid, Tb)
    return acc


def k_face_harmonic(ki, kj):
    """k 界面用调和平均（k 随 C 变化平缓；§6.0.2）。"""
    return D_face_harmonic(ki, kj)
