# -*- coding: utf-8 -*-
"""
analysis.py —— 论文第 6～12 节新增数值：两阶段识别、干燥曲线、湿球约束、工艺参数、收缩律与数值检验

用法：python analysis.py [--data 附件目录] [--out 导出目录] [--skip-heavy]
依赖：同目录 drying_model.py；结果写为 CSV。--skip-heavy 跳过 N=1600 的场点加密对照。
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, curve_fit
from scipy.sparse import lil_matrix
from scipy.special import j0, j1, jn_zeros

import drying_model as dm

P_ATM = 101325.0            # 烘房按标准大气压计 / Pa
N_SCAN = 400                # 情景扫描网格（与 N=800 的烘干时长相差约 1e-4 h）
T_CAP_SCAN = 400.0          # 情景扫描积分上限 / h
T_SCAN = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0)            # 恒温段温度情景 / °C
C_SCAN = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07)            # 恒温段含水浓度情景 / (kg/kg)


# ============================== 1 湿空气性质（ASHRAE Fundamentals 2001，第 6 章） ==============================
def p_ws(t_C):
    """饱和水蒸气压 / Pa（Hyland–Wexler 关联式，0～200 °C，式 (6)）。"""
    T = np.asarray(t_C, dtype=np.float64) + 273.15
    return np.exp(-5.8002206e3 / T + 1.3914993 - 4.8640239e-2 * T + 4.1764768e-5 * T ** 2
                  - 1.4452093e-8 * T ** 3 + 6.5459673 * np.log(T))


def rel_humidity(t_C, W, p=P_ATM):
    """由干球温度与含湿量 W（kg 水/kg 干空气）求相对湿度（式 (22)）。"""
    return p * W / (0.62198 + W) / p_ws(t_C)


def wet_bulb(t_C, W, p=P_ATM):
    """湿球温度 / °C：解式 (35) W = [(2501-2.381t*)W_s* - 1.006(t-t*)]/(2501+1.805t-4.186t*)。"""
    def f(tw):
        ps = p_ws(tw)
        Ws = 0.62198 * ps / (p - ps)
        return ((2501.0 - 2.381 * tw) * Ws - 1.006 * (t_C - tw)) / (2501.0 + 1.805 * t_C - 4.186 * tw) - W
    return float(brentq(f, -20.0, float(t_C), xtol=1e-10))


def latent_heat(t_C):
    """汽化潜热 h_fg ~ h_g - h_w = (2501+1.805t) - 4.186t = (2501 - 2.381t) kJ/kg（式 (32)(34)）-> J/kg。"""
    return (2501.0 - 2.381 * t_C) * 1.0e3


# ============================== 2 环境情景 ==============================
def env_variant(t, T_C, C, *, T_set=None, C_set=None, wetbulb=False):
    """在附件 1 基础上构造情景：
    T_set：升温段按 T_air-28 °C 等比例缩放，4 h 后取 T_set；
    C_set：含水浓度的增量按升温进度 (T_air-28)/(Tbar-28) 叠加，初值不变，4 h 后取 C_set；
    wetbulb：以湿球温度代替空气温度（供能量一致模型取 T_wb）。不加参数时与基准环境完全相同。"""
    win = (t >= dm.WINDOW[0]) & (t <= dm.WINDOW[1])
    Tm, Cm = float(np.mean(T_C[win])), float(np.mean(C[win]))
    T_nodes, C_nodes, T_c, C_c = T_C.copy(), C.copy(), Tm, Cm
    if T_set is not None:
        T_nodes = T_C[0] + (T_C - T_C[0]) * (T_set - T_C[0]) / (Tm - T_C[0])
        T_c = float(T_set)
    if C_set is not None:
        C_nodes = C + (C_set - Cm) * np.clip((T_C - T_C[0]) / (Tm - T_C[0]), 0.0, None)
        C_c = float(C_set)
    if wetbulb:
        T_nodes = np.array([wet_bulb(a, b) for a, b in zip(T_nodes, C_nodes)])
        T_c = wet_bulb(T_c, C_c)
    return dm.Env(t, T_nodes + dm.KELVIN, C_nodes, T_c + dm.KELVIN, C_c)


# ============================== 3 采样与统计工具 ==============================
def write_csv(path, header, rows, fmt="{:.6g}"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                             for v in r) + "\n")


def series(model, traj, ts):
    """时程：中心/表面温度与含水率、平均含水率、平均干燥速率、物料与烘房的最大温差。"""
    n = model.N + 1
    Y = traj.eval(ts)
    C, T = Y[:, :n], Y[:, n:2 * n]
    Tair = np.asarray(model.env.T_air_K(ts))
    cbar = np.array([model.cbar(c) for c in C])
    rate = np.array([model.drying_rate(t, c) for t, c in zip(ts, C)])
    dTmax = np.max(np.abs(T - Tair[:, None]), axis=1)
    return {"t": ts, "Tair": Tair - dm.KELVIN, "Tc": T[:, 0] - dm.KELVIN, "Ts": T[:, -1] - dm.KELVIN,
            "Cc": C[:, 0], "Cs": C[:, -1], "Cmax": C.max(axis=1), "Cbar": cbar, "rate": rate,
            "dTmax": dTmax, "Cenv": np.asarray(model.env.C_env(ts))}


def preheat_end(ts, dTmax, tol=0.5):
    """预热平衡结束时刻：此后物料各处与烘房温差始终不超过 tol（K）。"""
    bad = np.where(dTmax > tol)[0]
    if len(bad) == 0:
        return float(ts[0])
    i = bad[-1] + 1
    return float(ts[i]) if i < len(ts) else None


def tstar(model, cap_h=T_CAP_SCAN, rtol=dm.RTOL):
    _, t, _, _ = dm.solve_to_dry(model, cap_h, rtol=rtol)
    return t / 3600.0


def integrate_rate(model, traj, t_end, nodes):
    """int_0^t_end (-dCbar/dt) dt：以 BDF 步点与输入数据节点划分小区间，每段 8 点 Gauss 求积
    （稠密输出在步内是多项式，环境与半径在节点间线性，故求积误差只剩舍入）。"""
    n = model.N + 1
    knots = np.concatenate([seg.t for seg in traj.segments] + [np.asarray(nodes, float), [0.0, t_end]])
    knots = np.unique(knots[(knots >= 0.0) & (knots <= t_end)])
    a, b = knots[:-1], knots[1:]
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    tq = (mid[:, None] + half[:, None] * dm._U[None, :]).ravel()
    wq = (half[:, None] * dm._W[None, :]).ravel()
    Y = traj.eval(tq)
    return float(sum(w * model.drying_rate(t, y[:n]) for t, w, y in zip(tq, wq, Y)))


def degree_hours(env, t_end):
    """烘房相对初温的加热度时 int (T_air - T0)dt / (K·h)，作热损失的代理量。"""
    ts = np.linspace(0.0, t_end, 20001)
    return float(np.trapezoid(np.asarray(env.T_air_K(ts)) - dm.KELVIN - dm.T0_C, ts)) / 3600.0


def thin_layer_fits(t_h, MR):
    """薄层干燥经验模型拟合（Erbay–Icier 综述中的常用形式）。返回 [模型, R², RMSE, 参数…]。"""
    models = (("Henderson-Pabis", lambda t, a, k: a * np.exp(-k * t), [1.0, 0.05]),
              ("Page", lambda t, k, n: np.exp(-k * t ** n), [0.1, 0.8]),
              ("Two-term", lambda t, a, k0, b, k1: a * np.exp(-k0 * t) + b * np.exp(-k1 * t), [0.5, 0.5, 0.5, 0.05]))
    rows = []
    for name, f, p0 in models:
        p, _ = curve_fit(f, t_h, MR, p0=p0, maxfev=20000)
        res = MR - f(t_h, *p)
        r2 = 1.0 - float(np.sum(res ** 2)) / float(np.sum((MR - MR.mean()) ** 2))
        rows.append([name, r2, float(np.sqrt(np.mean(res ** 2)))] + [float(v) for v in p] + [None] * (4 - len(p)))
    return rows


# ============================== 4 常系数圆柱解析解（数值检验） ==============================
def bessel_roots(Bi, M=200):
    """lambda·J1(lambda) = Bi·J0(lambda) 的前 M 个正根；第 n 个根位于 J1 的第 n-1 个零点与 J0 的第 n 个零点之间。"""
    z0 = jn_zeros(0, M)
    z1 = np.concatenate([[0.0], jn_zeros(1, M - 1)])
    return np.array([brentq(lambda s: s * j1(s) - Bi * j0(s), a + 1e-12, b - 1e-12) for a, b in zip(z1, z0)])


def series_cylinder(Bi, Fo, xi, M=200):
    """常系数无限长圆柱、第三类边界、均匀初值的无量纲解（Crank 第 5 章；Carslaw–Jaeger 第 7 章）：
    theta = sum 2Bi·J0(lambda_n xi)·exp(-lambda_n² Fo) / [(lambda_n² + Bi²)·J0(lambda_n)]，theta = (u - u_inf)/(u_0 - u_inf)。"""
    lam = bessel_roots(Bi, M)
    A = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * j0(lam))
    return j0(np.outer(xi, lam)) @ (A * np.exp(-lam ** 2 * Fo))


class PropsConst(dm.PropsQ1):
    """附录 2 的常热物性，扩散系数取初始含水率处的常值（解析对照用）。"""
    D0 = 7.0e-9 * np.exp(-0.89 / dm.C0)

    def D(self, C, T=None):
        return np.full_like(np.asarray(C, dtype=np.float64), self.D0)


def analytic_check(N, T_inf_C=50.0, C_inf=0.05, t_end=86400.0):
    """恒定环境、常系数下与级数解对比：温度取 0～1800 s（每 10 s），含水率取 0～24 h（每 600 s）。"""
    t_nodes = np.array([0.0, dm.T_SWITCH])
    env = dm.Env(t_nodes, np.full(2, T_inf_C + dm.KELVIN), np.full(2, C_inf), T_inf_C + dm.KELVIN, C_inf)
    p = PropsConst()
    m = dm.Model(N, p, env)
    tr = dm.integrate(m, m.initial_state(), 0.0, t_end, breakpoints=())
    n = N + 1
    idx = np.array([dm.node_index(m, r) for r in dm.COLS21])
    xi = m.grid.x[idx] / dm.R0
    alpha = 0.36 / (820.0 * 2600.0)
    eT = eC = 0.0
    for t in np.arange(10.0, 1800.0 + 1, 10.0):
        exact = T_inf_C + (dm.T0_C - T_inf_C) * series_cylinder(dm.H * dm.R0 / 0.36, alpha * t / dm.R0 ** 2, xi)
        eT = max(eT, float(np.max(np.abs(tr.eval([t])[0][n + idx] - dm.KELVIN - exact))))
    for t in np.arange(600.0, t_end + 1, 600.0):
        exact = C_inf + (dm.C0 - C_inf) * series_cylinder(dm.HM * dm.R0 / p.D0, p.D0 * t / dm.R0 ** 2, xi)
        eC = max(eC, float(np.max(np.abs(tr.eval([t])[0][idx] - exact))))
    return eT, eC


# ============================== 5 能量一致的表面蒸发（恒速—降速干燥） ==============================
class EvapLimitedModel(dm.Model):
    """表面失水通量不超过对流供热所能蒸发的水量（恒速干燥段由传热控制）：
    phi = min(h_m(C_s - C_env), phi_c)，phi_c = h(T_air - T_wb) / (L_v(T_wb)·rho_s)；
    表面热平衡同时扣除蒸发吸热 L_v(T_s)·rho_s·phi。phi 按干基含水率计（m/s），rho_s 为干物质密度，
    固定域取初始值 rho(C0)/(1+C0)，问题四按干物质守恒取 rho_s0·(R0/R)²。"""
    def __init__(self, N, props, env, env_wb, *, radius=None, **kw):
        super().__init__(N, props, env, radius=radius, **kw)
        self.env_wb = env_wb
        self.rho_s0 = float(props.rho(dm.C0)) / (1.0 + dm.C0)

    def flux(self, t, Cs):
        """返回 (实际失水通量 phi, 当前半径 R, 干物质密度 rho_s)。"""
        R = float(self.radius.R(t)) if self.ref else self.R0
        rho_s = self.rho_s0 * (self.R0 / R) ** 2 if self.ref else self.rho_s0
        Ta, Tw = float(self.env.T_air_K(t)), float(self.env_wb.T_air_K(t))
        phi_c = self.h * max(Ta - Tw, 0.0) / (latent_heat(Tw - dm.KELVIN) * rho_s)
        return min(self.hm * (Cs - float(self.env.C_env(t))), phi_c), R, rho_s

    def rhs(self, t, y):
        d = super().rhs(t, y)
        n = self.N + 1
        Cs, Ts = float(y[n - 1]), float(y[2 * n - 1])
        phi, R, rho_s = self.flux(t, Cs)
        area, V = (1.0 / R if self.ref else self.R0), self.grid.V[-1]    # 与 Model.surf 的几何因子一致
        d[n - 1] += area * (self.hm * (Cs - float(self.env.C_env(t))) - phi) / V
        b_N = float(self.props.b(y[n - 1:n])[0])
        d[2 * n - 1] -= area * latent_heat(Ts - dm.KELVIN) * rho_s * phi / (b_N * V)
        return d

    def drying_rate(self, t, C):
        phi, R, _ = self.flux(t, float(C[-1]))
        return 2.0 * phi / R


# ============================== 6 问题四：半径—平均含水率收缩律 ==============================
class ShrinkModel(dm.Model):
    """参考坐标模型，半径由平均含水率决定：R = g(Cbar)，Cbar = C0 - I，I 为累计失水（增广状态）。"""
    def __init__(self, N, props, env, g):
        super().__init__(N, props, env, radius=dm.Radius(np.array([0.0, 1.0]), np.array([dm.R0, dm.R0])))
        self.g, self._R = g, dm.R0

    def surf(self, t):
        return self.hm / self._R, self.h / self._R

    def scale(self, t):
        return 1.0 / (self._R * self._R)

    def rhs(self, t, y):
        n = self.N + 1
        self._R = float(self.g(dm.C0 - y[2 * n]))
        d = super().rhs(t, y[:2 * n])
        dI = 2.0 * self.hm / self._R * (float(y[n - 1]) - float(self.env.C_env(t)))
        return np.concatenate([d, [dI]])


def solve_shrink(model, cap_h=T_CAP_SCAN):
    """增广状态下积分到全域最大含水率降至 0.15，返回烘干时长 / h。"""
    n = model.N + 1
    base = dm.sparsity(model.N).tolil()
    S = lil_matrix((2 * n + 1, 2 * n + 1), dtype=np.int8)
    S[:2 * n, :2 * n] = base
    S[:, 2 * n] = 1                      # 各方程经 R(I) 依赖累计失水
    S[2 * n, n - 1] = 1                  # 失水速率依赖表面含水率
    S = S.tocsr()
    atol = np.concatenate([np.full(n, dm.ATOL_C), np.full(n, dm.ATOL_T), [1e-11]])

    def event(t, y):
        return float(np.max(y[:n])) - dm.THRESH
    event.direction, event.terminal = -1.0, True
    y = np.concatenate([model.initial_state(), [0.0]])
    for a, b in ((0.0, dm.T_SWITCH), (dm.T_SWITCH, cap_h * 3600.0)):
        sol = solve_ivp(model.rhs, (a, b), y, method="BDF", rtol=dm.RTOL, atol=atol, jac_sparsity=S,
                        max_step=dm.MAXSTEP_DATA if b <= dm.T_SWITCH else dm.MAXSTEP_AFTER, events=event)
        if not sol.success:
            raise RuntimeError(sol.message)
        if len(sol.t_events[0]) > 0:
            return float(sol.t_events[0][0]) / 3600.0
        y = sol.y[:, -1].copy()
    raise RuntimeError("未达到烘干要求")


# ============================== 7 主流程 ==============================
def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here.parent.parent / "附件"))
    ap.add_argument("--out", default=str(here.parent.parent / "exports" / "redo"))
    ap.add_argument("--skip-heavy", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    os.makedirs(out, exist_ok=True)
    t1, T1, C1 = dm.load_env_data(a.data)
    env, radius = dm.make_env(t1, T1, C1), dm.load_radius(a.data)
    clock = time.time()

    def log(msg):
        print(f"[{time.time() - clock:7.1f}s] {msg}", flush=True)

    # ---- 5.1 烘房湿空气状态（附件 1 原始节点） ----
    W = C1                                   # 含水浓度按空气含湿量（kg 水/kg 干空气）理解
    rows = [[t1[i], T1[i], W[i], rel_humidity(T1[i], W[i]), wet_bulb(T1[i], W[i])] for i in range(len(t1))]
    win = (t1 >= dm.WINDOW[0]) & (t1 <= dm.WINDOW[1])
    Tm, Wm = float(np.mean(T1[win])), float(np.mean(W[win]))
    rows.append(["plateau", Tm, Wm, rel_humidity(Tm, Wm), wet_bulb(Tm, Wm)])
    write_csv(out / "env_psychro.csv", ["t_s", "T_air_C", "W", "RH", "T_wb_C"], rows)
    log(f"湿空气：RH {min(r[3] for r in rows[:-1]):.3f}–{max(r[3] for r in rows[:-1]):.3f}，"
        f"恒温段 T_wb={wet_bulb(Tm, Wm):.2f} °C")

    # ---- 5.2 问题一：预热平衡阶段 ----
    m1 = dm.Model(dm.N_FINAL["q1"], dm.PropsQ1(), env)
    tr1 = dm.integrate(m1, m1.initial_state(), 0.0, 1800.0, breakpoints=())
    s1 = series(m1, tr1, np.arange(0.0, 1800.0 + 1, 10.0))
    write_csv(out / "q1_series.csv", list(s1), list(zip(*s1.values())))
    idx = [int(round(r / 100.0 / m1.grid.d)) for r in np.arange(0, 2.0001, 0.025)]
    prof = []
    for tt in (100, 300, 600, 900, 1200, 1500, 1800):
        y = tr1.eval([tt])[0]
        n1 = m1.N + 1
        prof += [[tt, m1.grid.x[i] * 100, y[n1 + i] - dm.KELVIN, y[i]] for i in idx]
    write_csv(out / "q1_profiles.csv", ["t_s", "r_cm", "T_C", "C"], prof)
    y1800 = tr1.eval([1800.0])[0][:m1.N + 1]
    depth = (dm.R0 - m1.grid.x[np.argmax(y1800 < dm.C0 - 0.01)]) * 100
    alpha = 0.36 / (820.0 * 2600.0)
    q1 = [["Fo_1800", alpha * 1800 / dm.R0 ** 2], ["penetration_cm", depth],
          ["dT_air_material_1800", float(s1["dTmax"][-1])], ["Cbar_1800", float(s1["Cbar"][-1])],
          ["rate_1800_per_h", float(s1["rate"][-1] * 3600)]]
    write_csv(out / "q1_scales.csv", ["quantity", "value"], q1)
    log(f"问题一：Fo(1800s)={q1[0][1]:.3f}，表层失水深度~{depth:.2f} cm")

    # 时间尺度与 Biot 数：热扩散时间 R0²/alpha、水分扩散时间 R0²/D，Bi_h = hR0/k，Bi_m = h_m R0/D
    T0K, TpK = dm.T0_C + dm.KELVIN, env.T_const_K
    scales = []
    for name, p in (("q1", dm.PropsQ1()), ("q23", dm.PropsQ23()), ("q4", dm.PropsQ4())):
        c0 = np.array([dm.C0])
        k0, b0 = float(p.k(c0)[0]), float(p.b(c0)[0])
        for tag, C, TK in (("C0_T0", dm.C0, T0K), ("C0_Tp", dm.C0, TpK), ("C015_Tp", 0.15, TpK)):
            D = float(p.D(np.array([C]), np.array([TK]))[0])
            kC = float(p.k(np.array([C]))[0])
            scales.append([name, tag, D, dm.R0 ** 2 / D / 3600, dm.HM * dm.R0 / D, dm.H * dm.R0 / kC])
        scales.append([name, "thermal", k0 / b0, dm.R0 ** 2 / (k0 / b0) / 3600, None, dm.H * dm.R0 / k0])
    write_csv(out / "scales.csv", ["question", "state", "D_or_alpha_m2s", "time_scale_h", "Bi_m", "Bi_h"], scales)
    t_oven = float(t1[np.where(np.abs(T1 - (TpK - dm.KELVIN)) > 0.5)[0][-1] + 1])   # 此后烘房与恒温段均值相差 <=0.5 K

    # ---- 5.3 问题二三、问题四基准轨迹（N=800）：两阶段、干燥曲线 ----
    m23 = dm.Model(dm.N_FINAL["q23"], dm.PropsQ23(), env)
    tr23, ts23, _, _ = dm.solve_to_dry(m23, dm.T_CAP_H["q23"])
    s23 = series(m23, tr23, np.append(np.arange(0.0, ts23, 60.0), ts23))
    write_csv(out / "q23_series.csv", list(s23), list(zip(*s23.values())))
    m4 = dm.Model(dm.N_FINAL["q4"], dm.PropsQ4(), env, radius=radius)
    tr4, ts4, _, _ = dm.solve_to_dry(m4, dm.T_CAP_H["q4"])
    s4 = series(m4, tr4, np.append(np.arange(0.0, ts4, 60.0), ts4))
    s4["R_cm"] = np.asarray(radius.R(s4["t"])) * 100
    write_csv(out / "q4_series.csv", list(s4), list(zip(*s4.values())))
    log(f"基准：t*_3={ts23 / 3600:.4f} h，t*_4={ts4 / 3600:.4f} h")

    stages = []
    for name, s, tsx in (("q23", s23, ts23), ("q4", s4, ts4)):
        tpe = preheat_end(s["t"], s["dTmax"])
        k = int(np.searchsorted(s["t"], tpe))
        lost_pe = (dm.C0 - s["Cbar"][k]) / (dm.C0 - s["Cbar"][-1])
        rate_pe = (dm.C0 - s["Cbar"][k]) / (tpe / 3600)
        rate_dry = (s["Cbar"][k] - s["Cbar"][-1]) / ((tsx - tpe) / 3600)
        imax = int(np.argmax(s["rate"]))
        stages.append([name, tpe / 3600, lost_pe, rate_pe, rate_dry, float(s["Tc"][k]), float(s["Tair"][k]),
                       s["t"][imax] / 3600, float(s["rate"][0] * 3600), float(s["Cbar"][-1]), float(s["Cs"][-1])])
        stages[-1] += [float(s["Cbar"][k]), float(np.max(s["Ts"] - s["Tc"])), t_oven / 3600]
    write_csv(out / "stages.csv", ["question", "t_pe_h", "frac_lost_by_tpe", "rate_preheat_per_h",
                                   "rate_isothermal_per_h", "Tc_at_tpe", "Tair_at_tpe", "t_rate_max_h",
                                   "rate0_per_h", "Cbar_end", "Cs_end", "Cbar_at_tpe", "max_Ts_minus_Tc",
                                   "t_oven_plateau_h"], stages)
    log("两阶段：" + "；".join(f"{r[0]} t_pe={r[1]:.2f} h，失水 {100 * r[2]:.1f}%" for r in stages))

    # 烘干标准的影响：最大、表面、平均含水率首次降到各阈值的时刻（60 s 序列线性插值）
    def crossing(t, y, c):
        i = int(np.argmax(y <= c))
        return (t[i - 1] + (y[i - 1] - c) / (y[i - 1] - y[i]) * (t[i] - t[i - 1])) / 3600

    thr = [[c] + [crossing(s["t"], s[key], c) for s in (s23, s4) for key in ("Cmax", "Cs", "Cbar")]
           for c in (0.30, 0.25, 0.20, 0.18, 0.16, 0.15)]
    write_csv(out / "threshold.csv", ["threshold", "t_q3_max_h", "t_q3_surface_h", "t_q3_mean_h",
                                      "t_q4_max_h", "t_q4_surface_h", "t_q4_mean_h"], thr)

    # 干燥曲线的薄层模型拟合：水分比 MR = (Cbar - C_e)/(C0 - C_e)，C_e 取恒温段环境含水浓度
    fits = []
    for name, s in (("q3", s23), ("q4", s4)):
        Ce = env.C_const
        fits += [[name] + r for r in thin_layer_fits(s["t"] / 3600.0, (s["Cbar"] - Ce) / (dm.C0 - Ce))]
    write_csv(out / "drying_fits.csv", ["question", "model", "R2", "RMSE", "p1", "p2", "p3", "p4"], fits)
    log("薄层拟合：" + "；".join(f"{r[0]} {r[1]} R2={r[2]:.5f}" for r in fits))

    # 场分布（供图）：问题二三温度前 6 h、含水率全程；问题四物理坐标含水率
    rr = np.arange(0, 2.0001, 0.05)
    i23 = [int(round(r / 100.0 / m23.grid.d)) for r in rr]
    n23 = m23.N + 1
    fT, fC = [], []
    for tt in np.arange(0.0, 6 * 3600 + 1, 120.0):
        y = tr23.eval([tt])[0]
        fT.append([tt] + [y[n23 + i] - dm.KELVIN for i in i23])
    for tt in np.append(np.arange(0.0, ts23, 600.0), ts23):
        y = tr23.eval([tt])[0]
        fC.append([tt] + [y[i] for i in i23])
    write_csv(out / "q23_field_T.csv", ["t_s"] + [f"{r:.2f}" for r in rr], fT)
    write_csv(out / "q23_field_C.csv", ["t_s"] + [f"{r:.2f}" for r in rr], fC)
    f4 = []
    for tt in np.append(np.arange(0.0, ts4, 600.0), ts4):
        c = tr4.eval([tt])[0][:m4.N + 1]
        f4.append([tt, float(radius.R(tt)) * 100] + dm.q4_row(c, m4, float(radius.R(tt)), list(rr)))
    write_csv(out / "q4_field_C.csv", ["t_s", "R_cm"] + [f"{r:.2f}" for r in rr], f4)

    # ---- 5.4 蒸发吸热：能量一致的恒速—降速干燥 ----
    env_wb = env_variant(t1, T1, C1, wetbulb=True)          # 只取其湿球温度 T_wb(t)
    bracket, crit = [], []
    for name, props, rad, tsx in (("q3", dm.PropsQ23(), None, ts23), ("q4", dm.PropsQ4(), radius, ts4)):
        mev = EvapLimitedModel(dm.N_FINAL["q23" if name == "q3" else "q4"], props, env, env_wb, radius=rad)
        trev, tev, _, _ = dm.solve_to_dry(mev, 300.0)
        sev = series(mev, trev, np.append(np.arange(0.0, tev, 60.0), tev))
        capped = np.array([mev.flux(t, c)[0] < mev.hm * (c - float(env.C_env(t))) - 1e-15
                           for t, c in zip(sev["t"], sev["Cs"])])
        k = int(np.where(capped)[0][-1]) + 1                       # 恒速段（通量受限）结束后的首个采样
        sev["capped"] = capped.astype(int)
        write_csv(out / f"{name}_evap_series.csv", list(sev), list(zip(*sev.values())))
        bracket.append([name, tsx / 3600, tev / 3600])
        crit.append([name, sev["t"][k] / 3600, float(sev["Cbar"][k]), float(sev["Cs"][k]),
                     float(sev["Ts"][k - 1]), float(sev["rate"][k - 1] * 3600), float(sev["Tc"][k])])
    write_csv(out / "evap_tstar.csv", ["question", "t_star_base_h", "t_star_evap_h"], bracket)
    write_csv(out / "evap_critical.csv", ["question", "t_crit_h", "Cbar_crit", "Cs_crit", "Ts_before_crit_C",
                                          "rate_const_per_h", "Tc_at_crit_C"], crit)
    rho_s0 = float(dm.PropsQ23().rho(dm.C0)) / (1.0 + dm.C0)
    ec = []
    for th in (0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, ts23 / 3600):
        k = int(np.argmin(np.abs(s23["t"] - th * 3600)))
        Ta, Cs, Ce = float(s23["Tair"][k]), float(s23["Cs"][k]), float(s23["Cenv"][k])
        Tw = wet_bulb(Ta, Ce)
        jw = rho_s0 * dm.HM * (Cs - Ce)
        q_req, q_avail = latent_heat(Tw) * jw, dm.H * (Ta - Tw)
        ec.append([th, Ta, Tw, Cs, jw, q_req, q_avail, q_req / q_avail if q_avail > 0 else None])
    write_csv(out / "energy_check.csv", ["t_h", "T_air_C", "T_wb_C", "C_s", "j_w_kg_m2s", "q_evap_W_m2",
                                         "q_conv_wb_W_m2", "ratio"], ec)
    log("蒸发吸热：" + "；".join(f"{r[0]} 题设 {r[1]:.2f} h -> 能量一致 {r[2]:.2f} h" for r in bracket)
        + "；恒速段结束 " + "，".join(f"{c[0]} {c[1]:.2f} h（Cbar={c[2]:.3f}）" for c in crit))

    # ---- 5.5 工艺参数（问题二三，N=400） ----
    base400 = tstar(dm.Model(N_SCAN, dm.PropsQ23(), env))
    def pair(kw_env, **kw):
        """同一情景下题设模型与能量一致模型的烘干时长 / h。"""
        e, ew = env_variant(t1, T1, C1, **kw_env), env_variant(t1, T1, C1, wetbulb=True, **kw_env)
        return (tstar(dm.Model(N_SCAN, dm.PropsQ23(), e, **kw)),
                tstar(EvapLimitedModel(N_SCAN, dm.PropsQ23(), e, ew, **kw)), e)

    # 温度下限取 45 °C：恒温段含湿量 0.05 kg/kg 在 40 °C 时已超过饱和含湿量（约 0.049），情景不成立
    rowsT = []
    for Ts in T_SCAN:
        tb, te, e = pair({"T_set": Ts})
        rowsT.append([Ts, tb, degree_hours(e, tb * 3600.0), te])
    write_csv(out / "process_Tset.csv", ["T_set_C", "t_star_h", "degree_hours_Kh", "t_star_evap_h"], rowsT)
    log("温度：" + "，".join(f"{r[0]:.0f}°C->{r[1]:.2f}/{r[3]:.2f} h" for r in rowsT))
    rowsC = [[Cs, *pair({"C_set": Cs})[:2]] for Cs in C_SCAN]
    write_csv(out / "process_Cenv.csv", ["C_set", "t_star_h", "t_star_evap_h"], rowsC)
    rowsR = [[R, *pair({}, R0=R / 100.0)[:2]] for R in (1.0, 1.5, 2.0, 2.5)]
    write_csv(out / "process_R0.csv", ["R0_cm", "t_star_h", "t_star_evap_h"], rowsR)
    # 风速：传质系数单独变化，以及按 Chilton–Colburn 类比 h、h_m 同比例变化
    rowsH = [[f, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env, hm=dm.HM * f)),
              *pair({}, h=dm.H * f, hm=dm.HM * f)[:2]] for f in (0.5, 1.0, 1.5, 2.0)]
    write_csv(out / "process_hm.csv", ["factor", "t_star_hm_only_h", "t_star_h_and_hm_h", "t_star_evap_h"], rowsH)
    grid_map = []                                            # 工艺查询表：初始半径 × 恒温段温度
    for R in (1.0, 1.5, 2.0, 2.5):
        for Ts in T_SCAN:
            grid_map.append([R, Ts, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env_variant(t1, T1, C1, T_set=Ts), R0=R / 100.0))])
    write_csv(out / "process_map.csv", ["R0_cm", "T_set_C", "t_star_h"], grid_map)
    invT = np.array([1.0 / (r[0] + dm.KELVIN) for r in rowsT])
    lnR = np.log([r[0] for r in rowsR])
    fits = [["base_N400_h", base400]]
    for tag, jT, jR in (("", 1, 1), ("evap_", 3, 2)):
        fits += [[f"{tag}arrhenius_slope_K", float(np.polyfit(invT, np.log([r[jT] for r in rowsT]), 1)[0])],
                 [f"{tag}radius_exponent", float(np.polyfit(lnR, np.log([r[jR] for r in rowsR]), 1)[0])]]
    write_csv(out / "process_fits.csv", ["quantity", "value"], fits)
    log("工艺：" + "，".join(f"{q}={v:.4g}" for q, v in fits))

    # ---- 5.6 问题四：收缩律与三情形对照（N=400） ----
    full4 = dm.integrate(dm.Model(N_SCAN, dm.PropsQ4(), env, radius=radius),
                         dm.Model(N_SCAN, dm.PropsQ4(), env, radius=radius).initial_state(), 0.0, 72 * 3600.0)
    m4s = dm.Model(N_SCAN, dm.PropsQ4(), env, radius=radius)
    tl = np.arange(0.0, 72 * 3600.0 + 1, 600.0)
    cb = np.array([m4s.cbar(full4.eval([t])[0][:N_SCAN + 1]) for t in tl])
    Rl = np.asarray(radius.R(tl))
    write_csv(out / "q4_shrinklaw.csv", ["t_s", "Cbar", "R_cm"], list(zip(tl, cb, Rl * 100)))

    def g(cbar):
        return float(np.interp(cbar, cb[::-1], Rl[::-1]))

    t_law = solve_shrink(ShrinkModel(N_SCAN, dm.PropsQ4(), env, g))
    q4rows = [["law_base", 50.0, t_law]]
    for Ts in (45.0, 60.0, 70.0):
        q4rows.append(["law", Ts, solve_shrink(ShrinkModel(N_SCAN, dm.PropsQ4(), env_variant(t1, T1, C1, T_set=Ts), g))])
    write_csv(out / "process_q4.csv", ["kind", "T_set_C", "t_star_h"], q4rows)
    three = [["appendix3_fixedR", tstar(dm.Model(N_SCAN, dm.PropsQ23(), env))],
             ["appendix4_fixedR", tstar(dm.Model(N_SCAN, dm.PropsQ4(), env))],
             ["appendix4_shrinkR", tstar(dm.Model(N_SCAN, dm.PropsQ4(), env, radius=radius))]]
    write_csv(out / "threecase.csv", ["case", "t_star_h"], three)
    log(f"问题四收缩律：基准重现 {t_law:.4f} h；三情形 " + "，".join(f"{r[1]:.4f}" for r in three))

    # ---- 5.7 数值检验 ----
    ana = [[N, *analytic_check(N)] for N in (200, 400, 800, 1600)]
    write_csv(out / "analytic.csv", ["N", "maxerr_T_K", "maxerr_C"], ana)
    log("解析对照：" + "；".join(f"N={r[0]} Delta T={r[1]:.2e} Delta C={r[2]:.2e}" for r in ana))
    ver = [["q23_tstar_N200_h", tstar(dm.Model(200, dm.PropsQ23(), env))],
           ["q23_tstar_N400_h", base400],
           ["q23_tstar_N800_h", ts23 / 3600],
           ["q23_tstar_N400_rtol1e-10_h", tstar(dm.Model(N_SCAN, dm.PropsQ23(), env), rtol=1e-10)]]
    for name, m, tr, tsx in (("q23", m23, tr23, ts23), ("q4", m4, tr4, ts4)):
        integ = integrate_rate(m, tr, tsx, np.concatenate([t1, radius.t_nodes]))
        dcb = dm.C0 - m.cbar(tr.eval([tsx])[0][:m.N + 1])
        ver.append([f"{name}_mass_balance_rel", abs(integ - dcb) / dcb])
    if not a.skip_heavy:
        for name, props, rad, cap in (("q23", dm.PropsQ23(), None, 90.0), ("q4", dm.PropsQ4(), radius, 200.0)):
            fine = dm.Model(1600, props, env, radius=rad)
            coarse = m23 if name == "q23" else m4
            trc = tr23 if name == "q23" else tr4
            trf, tsf, _, _ = dm.solve_to_dry(fine, cap)
            worst = {"C": (0.0, None, None), "T": (0.0, None, None)}    # (最大差, 时刻/s, 位置/cm)
            t_end = min(tsf, trc.t_cross)
            if rad is None:          # 问题二三：前 600 s 逐秒、此后每 60 s，含温度与含水率
                tgrid = np.concatenate([np.arange(1.0, 601.0), np.arange(660.0, t_end + 1, 60.0)])
            else:                    # 问题四：每 60 s（与 result4 一致）
                tgrid = np.arange(60.0, t_end + 1, 60.0)
            for tt in tgrid:
                yc, yf = trc.eval([tt])[0], trf.eval([tt])[0]
                if rad is None:
                    nc, nf = coarse.N + 1, fine.N + 1
                    ic = np.array([int(round(r / 100.0 / coarse.grid.d)) for r in dm.COLS21])
                    jf = np.array([int(round(r / 100.0 / fine.grid.d)) for r in dm.COLS21])
                    for key, e in (("C", np.abs(yc[ic] - yf[jf])), ("T", np.abs(yc[ic + nc] - yf[jf + nf]))):
                        if e.max() > worst[key][0]:
                            worst[key] = (float(e.max()), tt, dm.COLS21[int(e.argmax())])
                else:
                    R_t = float(rad.R(tt))
                    rc = dm.q4_row(yc[:coarse.N + 1], coarse, R_t, dm.COLS20) + [float(yc[coarse.N])]
                    rf = dm.q4_row(yf[:fine.N + 1], fine, R_t, dm.COLS20) + [float(yf[fine.N])]
                    e = [abs(x - z) if x is not None else -1.0 for x, z in zip(rc, rf)]
                    j = int(np.argmax(e))
                    if e[j] > worst["C"][0]:
                        worst["C"] = (e[j], tt, (dm.COLS20 + [R_t * 100])[j])
            for key in ("C", "T") if rad is None else ("C",):
                ver += [[f"{name}_field_maxdiff_{key}_800_1600", worst[key][0]],
                        [f"{name}_field_maxdiff_{key}_at_t_s", worst[key][1]],
                        [f"{name}_field_maxdiff_{key}_at_r_cm", worst[key][2]]]
            ver.append([f"{name}_tstar_N1600_h", tsf / 3600])
    write_csv(out / "verification.csv", ["quantity", "value"], ver, fmt="{:.10g}")
    log("检验：" + "；".join(f"{r[0]}={r[1]:.4g}" for r in ver))


if __name__ == "__main__":
    main()
