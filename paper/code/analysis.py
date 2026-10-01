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
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

import drying_model as dm

P_ATM = 101325.0            # 烘房按标准大气压计 / Pa
N_SCAN = 400                # 情景扫描网格（与 N=800 的烘干时长相差约 1e-4 h）
T_CAP_SCAN = 400.0          # 情景扫描积分上限 / h


# ============================== 1 湿空气性质（ASHRAE 手册第 1 章） ==============================
def p_ws(t_C):
    """饱和水蒸气压 / Pa（Hyland–Wexler 关联式，0～200 ℃）。"""
    T = np.asarray(t_C, dtype=np.float64) + 273.15
    return np.exp(-5.8002206e3 / T + 1.3914993 - 4.8640239e-2 * T + 4.1764768e-5 * T ** 2
                  - 1.4452093e-8 * T ** 3 + 6.5459673 * np.log(T))


def rel_humidity(t_C, W, p=P_ATM):
    """由干球温度与含湿量 W（kg 水/kg 干空气）求相对湿度。"""
    return p * W / (0.621945 + W) / p_ws(t_C)


def wet_bulb(t_C, W, p=P_ATM):
    """湿球温度 / ℃：解 ASHRAE 湿球方程 W = [(2501−2.326t*)W_s* − 1.006(t−t*)]/(2501+1.86t−4.186t*)。"""
    def f(tw):
        ps = p_ws(tw)
        Ws = 0.621945 * ps / (p - ps)
        return ((2501.0 - 2.326 * tw) * Ws - 1.006 * (t_C - tw)) / (2501.0 + 1.86 * t_C - 4.186 * tw) - W
    return float(brentq(f, -20.0, float(t_C), xtol=1e-10))


def latent_heat(t_C):
    """水的汽化潜热近似 (2501 − 2.326t) kJ/kg，与湿球方程一致 → J/kg。"""
    return (2501.0 - 2.326 * t_C) * 1.0e3


# ============================== 2 环境情景 ==============================
def env_variant(t, T_C, C, *, T_set=None, C_set=None, wetbulb=False):
    """在附件 1 基础上构造情景：
    T_set：升温段按 T_air−28 ℃ 等比例缩放，4 h 后取 T_set；C_set：含水浓度等比例缩放，4 h 后取 C_set；
    wetbulb：以湿球温度代替空气温度（物料温度下限情景）。不加参数时与基准环境完全相同。"""
    win = (t >= dm.WINDOW[0]) & (t <= dm.WINDOW[1])
    Tm, Cm = float(np.mean(T_C[win])), float(np.mean(C[win]))
    T_nodes, C_nodes, T_c, C_c = T_C.copy(), C.copy(), Tm, Cm
    if T_set is not None:
        T_nodes = T_C[0] + (T_C - T_C[0]) * (T_set - T_C[0]) / (Tm - T_C[0])
        T_c = float(T_set)
    if C_set is not None:
        C_nodes = C * (C_set / Cm)
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


# ============================== 4 问题四：半径—平均含水率收缩律 ==============================
class ShrinkModel(dm.Model):
    """参考坐标模型，半径由平均含水率决定：R = g(C̄)，C̄ = C0 − I，I 为累计失水（增广状态）。"""
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


# ============================== 5 主流程 ==============================
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
        f"恒温段 T_wb={wet_bulb(Tm, Wm):.2f} ℃")

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
    D0 = 7e-9 * np.exp(-0.89 / dm.C0)
    q1 = [["Fo_1800", alpha * 1800 / dm.R0 ** 2], ["Bi_h", dm.H * dm.R0 / 0.36],
          ["Bi_m0", dm.HM * dm.R0 / D0], ["penetration_cm", depth],
          ["dT_air_material_1800", float(s1["dTmax"][-1])], ["thermal_time_R2_over_alpha_s", dm.R0 ** 2 / alpha]]
    write_csv(out / "q1_scales.csv", ["quantity", "value"], q1)
    log(f"问题一：Fo(1800s)={q1[0][1]:.3f}，表层失水深度≈{depth:.2f} cm")

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
    write_csv(out / "stages.csv", ["question", "t_pe_h", "frac_lost_by_tpe", "rate_preheat_per_h",
                                   "rate_isothermal_per_h", "Tc_at_tpe", "Tair_at_tpe", "t_rate_max_h",
                                   "rate0_per_h", "Cbar_end", "Cs_end"], stages)
    log("两阶段：" + "；".join(f"{r[0]} t_pe={r[1]:.2f} h，失水 {100 * r[2]:.1f}%" for r in stages))

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

    # ---- 5.4 湿球约束与能量一致性 ----
    env_wb = env_variant(t1, T1, C1, wetbulb=True)
    tb23 = tstar(dm.Model(dm.N_FINAL["q23"], dm.PropsQ23(), env_wb), cap_h=200.0)
    tb4 = tstar(dm.Model(dm.N_FINAL["q4"], dm.PropsQ4(), env_wb, radius=radius), cap_h=300.0)
    write_csv(out / "wetbulb_bracket.csv", ["question", "t_star_base_h", "t_star_wetbulb_h", "ratio"],
              [["q3", ts23 / 3600, tb23, tb23 / (ts23 / 3600)], ["q4", ts4 / 3600, tb4, tb4 / (ts4 / 3600)]])
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
    log(f"湿球约束：问题三 [{ts23 / 3600:.2f}, {tb23:.2f}] h，问题四 [{ts4 / 3600:.2f}, {tb4:.2f}] h")

    # ---- 5.5 工艺参数（问题二三，N=400） ----
    base400 = tstar(dm.Model(N_SCAN, dm.PropsQ23(), env))
    rowsT = [[Ts, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env_variant(t1, T1, C1, T_set=Ts)))]
             for Ts in (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0)]
    write_csv(out / "process_Tset.csv", ["T_set_C", "t_star_h"], rowsT)
    log("温度：" + "，".join(f"{r[0]:.0f}℃→{r[1]:.2f} h" for r in rowsT))
    rowsC = [[Cs, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env_variant(t1, T1, C1, C_set=Cs)))]
             for Cs in (0.02, 0.035, 0.05, 0.065, 0.08)]
    write_csv(out / "process_Cenv.csv", ["C_set", "t_star_h"], rowsC)
    rowsR = [[R, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env, R0=R / 100.0))] for R in (1.0, 1.5, 2.0, 2.5)]
    write_csv(out / "process_R0.csv", ["R0_cm", "t_star_h"], rowsR)
    rowsH = [[f, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env, hm=dm.HM * f))] for f in (0.5, 1.0, 2.0)]
    write_csv(out / "process_hm.csv", ["hm_factor", "t_star_h"], rowsH)
    grid_map = []
    for R in (1.0, 1.5, 2.5):
        for Ts in (40.0, 50.0, 60.0, 70.0):
            grid_map.append([R, Ts, tstar(dm.Model(N_SCAN, dm.PropsQ23(), env_variant(t1, T1, C1, T_set=Ts), R0=R / 100.0))])
    write_csv(out / "process_map.csv", ["R0_cm", "T_set_C", "t_star_h"], grid_map)
    invT = np.array([1.0 / (r[0] + dm.KELVIN) for r in rowsT])
    slope = float(np.polyfit(invT, np.log([r[1] for r in rowsT]), 1)[0])
    pR = float(np.polyfit(np.log([r[0] for r in rowsR]), np.log([r[1] for r in rowsR]), 1)[0])
    write_csv(out / "process_fits.csv", ["quantity", "value"],
              [["base_N400_h", base400], ["arrhenius_slope_K", slope], ["radius_exponent", pR]])
    log(f"工艺：ln t*–1/T 斜率 {slope:.0f} K，t*∝R0^{pR:.2f}")

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
    for Ts in (40.0, 60.0, 70.0):
        q4rows.append(["law", Ts, solve_shrink(ShrinkModel(N_SCAN, dm.PropsQ4(), env_variant(t1, T1, C1, T_set=Ts), g))])
    write_csv(out / "process_q4.csv", ["kind", "T_set_C", "t_star_h"], q4rows)
    three = [["appendix3_fixedR", tstar(dm.Model(N_SCAN, dm.PropsQ23(), env))],
             ["appendix4_fixedR", tstar(dm.Model(N_SCAN, dm.PropsQ4(), env))],
             ["appendix4_shrinkR", tstar(dm.Model(N_SCAN, dm.PropsQ4(), env, radius=radius))]]
    write_csv(out / "threecase.csv", ["case", "t_star_h"], three)
    log(f"问题四收缩律：基准重现 {t_law:.4f} h；三情形 " + "，".join(f"{r[1]:.4f}" for r in three))

    # ---- 5.7 数值检验 ----
    ver = [["q23_tstar_N200_h", tstar(dm.Model(200, dm.PropsQ23(), env))],
           ["q23_tstar_N400_h", base400],
           ["q23_tstar_N800_h", ts23 / 3600],
           ["q23_tstar_N400_rtol1e-10_h", tstar(dm.Model(N_SCAN, dm.PropsQ23(), env), rtol=1e-10)]]
    for name, m, tr, tsx in (("q23", m23, tr23, ts23), ("q4", m4, tr4, ts4)):
        n = m.N + 1

        def rate(t, m=m, tr=tr, n=n):
            return m.drying_rate(t, tr.eval([t])[0][:n])
        integ = quad(rate, 0.0, dm.T_SWITCH, epsabs=0, epsrel=1e-12, limit=2000)[0] \
            + quad(rate, dm.T_SWITCH, tsx, epsabs=0, epsrel=1e-12, limit=2000)[0]
        dcb = dm.C0 - m.cbar(tr.eval([tsx])[0][:n])
        ver.append([f"{name}_mass_balance_rel", abs(integ - dcb) / dcb])
    if not a.skip_heavy:
        for name, props, rad, cap in (("q23", dm.PropsQ23(), None, 90.0), ("q4", dm.PropsQ4(), radius, 200.0)):
            fine = dm.Model(1600, props, env, radius=rad)
            coarse = m23 if name == "q23" else m4
            trc = tr23 if name == "q23" else tr4
            trf, tsf, _, _ = dm.solve_to_dry(fine, cap)
            dmax = 0.0
            for tt in np.arange(60.0, min(tsf, trc.t_cross) + 1, 60.0):
                yc, yf = trc.eval([tt])[0], trf.eval([tt])[0]
                if rad is None:
                    ic = [int(round(r / 100.0 / coarse.grid.d)) for r in dm.COLS21]
                    jf = [int(round(r / 100.0 / fine.grid.d)) for r in dm.COLS21]
                    dmax = max(dmax, float(np.max(np.abs(yc[ic] - yf[jf]))))
                else:
                    R_t = float(rad.R(tt))
                    rc = dm.q4_row(yc[:coarse.N + 1], coarse, R_t, dm.COLS20) + [float(yc[coarse.N])]
                    rf = dm.q4_row(yf[:fine.N + 1], fine, R_t, dm.COLS20) + [float(yf[fine.N])]
                    dmax = max(dmax, max(abs(x - z) for x, z in zip(rc, rf) if x is not None))
            ver += [[f"{name}_field_maxdiff_800_1600", dmax], [f"{name}_tstar_N1600_h", tsf / 3600]]
    write_csv(out / "verification.csv", ["quantity", "value"], ver)
    log("检验：" + "；".join(f"{r[0]}={r[1]:.4g}" for r in ver))


if __name__ == "__main__":
    main()
