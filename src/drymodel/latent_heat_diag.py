"""latent_heat_diag.py —— 蒸发吸热影响评估（第六轮返修；独立诊断，不改既有结果导出）。

在与基线完全相同的 N=400 诊断网格、初值、物性、环境、积分界面与 BDF 容差下，
对表面热边界作成对开关对照：

    -k ∂T/∂r|_{R} = h (T_s − T_air) + η L_v(T_s) ρ_s(t) h_m (C_s − C_env)

η=0 复现现有热边界（回归同网格基线），η=1 表示表面蒸发吸热情景。
只新增诊断脚本与数据/图，不切换 config.bc.heat_latent，不触发生产导出。

质量/热量换算：C 为干基含水率；干物质基准密度 ρ_s0=ρ(C0)/(1+C0)，收缩域 ρ_s=ρ_s0 (R0/R)²。
表面蒸发质量通量 j_w=ρ_s h_m (C_s−C_env) [kg/(m²·s)]，蒸发热负荷 q_evap=L_v j_w [W/m²]。
纯水汽化焓 L_v(T) 为诊断选用的近似关系式（输入 K，近 50 °C 约 2.38e6 J/kg；非 IAPWS 拟合，形式见论文附录）。

运行：``python -m drymodel.latent_heat_diag``（写 exports/latent_heat_*.csv/json）。
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import config as cfgmod
from . import solver_bdf as SBDF
from . import runners
from .operators import FVMOperator

ROOT = cfgmod.PROJECT_ROOT
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

N_DIAG = 400
INTERFACE = "integral"
NPTS = 8


# --------------------------------------------------------------------------
# 纯水汽化焓 L_v(T)：诊断选用的近似关系式（J/kg，输入 K；非 IAPWS 拟合，见论文附录）
# --------------------------------------------------------------------------
def Lv(T_K):
    Tc = float(T_K) - 273.15
    return 1.0e3 * (2500.8 - 2.36 * Tc + 0.0016 * Tc ** 2 - 6.0e-5 * Tc ** 3)


class LatentHeatOp(FVMOperator):
    """在基础算子上叠加表面蒸发吸热项（η 开关）。η=0 与基础算子完全一致。"""

    def __init__(self, base, *, eta, rho_s0):
        super().__init__(base.grid, base.props, base.env, h=base.h, hm=base.hm, R0=base.R0,
                         interface=base.interface, integral_npts=base.integral_npts,
                         radius_fn=base.radius_fn, decoupled=base.decoupled,
                         augmented=base.augmented)
        self.eta = float(eta)
        self.rho_s0 = float(rho_s0)

    def rho_s(self, t):
        if self.is_reference:
            R = float(self.radius_fn.R(t))
            return self.rho_s0 * (self.R0 / R) ** 2
        return self.rho_s0

    def q_evap_surface(self, t, C, T):
        """表面蒸发热负荷 q_evap=L_v ρ_s h_m (C_s−C_env) [W/m²]（诊断用，非离散项）。"""
        Cenv = float(self.env.C_env(t))
        return Lv(T[-1]) * self.rho_s(t) * self.hm * (float(C[-1]) - Cenv)

    def rhs(self, t, y):
        dy = super().rhs(t, y)
        if self.eta == 0.0:
            return dy
        C, T = self.split(y)
        Cenv = float(self.env.C_env(t))
        sC, _ = self._surf_coeff(t)
        b = self.props.b(C)
        latent = self.eta * Lv(T[-1]) * self.rho_s(t) * sC * (float(C[-1]) - Cenv)
        dy = np.array(dy, dtype=float, copy=True)
        n = self.N + 1
        dy[n + self.N] -= latent / (b[-1] * self.V[self.N])   # 减到表面温度节点方程
        return dy


def _build_base(cfg, question):
    if question == "q4":
        op, env, radius = runners.build_ref_operator(cfg, "q4", N_DIAG,
                                                      interface=INTERFACE, integral_npts=NPTS)
    else:
        op, env = runners.build_fixed_operator(cfg, question, N_DIAG,
                                               interface=INTERFACE, integral_npts=NPTS)
    return op


def _run_case(cfg, question, eta, *, t_cap_h):
    base = _build_base(cfg, question)
    rho0 = cfg.props(question).rho(cfg.C0) / (1.0 + cfg.C0)
    op = LatentHeatOp(base, eta=eta, rho_s0=rho0)
    n = N_DIAG + 1
    y0 = runners.initial_state(cfg, N_DIAG)
    thr = cfg.threshold
    res = SBDF.integrate_bdf(op, y0, 0.0, t_cap_h * 3600.0, cfg.bdf,
                             breakpoints=(14400.0,), threshold=thr)
    if not res.ok:
        raise RuntimeError(f"{question} η={eta} BDF 失败：{res.message}")
    reached = res.t_cross is not None
    t_star_s = res.t_cross if reached else t_cap_h * 3600.0

    # 时程采样（共同物理时刻）
    ts = np.linspace(0.0, t_star_s, 241)
    states = np.array([res.eval([float(t)])[0] for t in ts])
    C_all = states[:, :n]; T_all = states[:, n:2 * n]
    T_center = T_all[:, 0] - 273.15
    T_surface = T_all[:, -1] - 273.15
    Cmax = C_all.max(axis=1)
    Cbar = np.array([_cbar(op, C_all[i]) for i in range(len(ts))])
    q_evap = np.array([op.q_evap_surface(ts[i], C_all[i], T_all[i]) for i in range(len(ts))])

    # 达标时最大含水率（若超时限）
    Cmax_end = float(Cmax[-1])
    out = dict(question=question, eta=eta, N=N_DIAG, rho_s0=float(rho0),
               reached=reached, t_star_s=float(t_star_s), t_star_h=float(t_star_s / 3600.0),
               T_surface_min_C=float(np.min(T_surface)),
               Cmax_end=Cmax_end,
               t=ts, T_center=T_center, T_surface=T_surface, Cmax=Cmax, Cbar=Cbar,
               q_evap=q_evap)
    return out, op, res


def _cbar(op, C):
    if op.is_reference:
        return float(2.0 * np.sum(op.V * C))          # ΣV_i=1/2 → C̄=2ΣV_iC_i（参考域）
    return float(2.0 / op.R0 ** 2 * np.sum(op.V * C))


def run(cfg=None):
    cfg = cfg or cfgmod.load_config()
    C0 = cfg.C0
    rows = []
    series_out = {}
    load_check = {}

    for question, t_cap in (("q23", 160.0), ("q4", 160.0)):
        base_out, base_op, base_res = _run_case(cfg, question, 0.0, t_cap_h=t_cap)
        lat_out, lat_op, lat_res = _run_case(cfg, question, 1.0, t_cap_h=t_cap)
        dt_h = lat_out["t_star_h"] - base_out["t_star_h"] if (base_out["reached"] and lat_out["reached"]) else None
        # 表面温度最大压低（η=1 相对 η=0，共同时刻，取到较短的 t*）
        tmax = min(base_out["t"][-1], lat_out["t"][-1])
        tt = np.linspace(0.0, tmax, 200)
        def surf_C(res, t):
            st = res.eval([float(t)])[0]
            return st[2 * N_DIAG + 1] - 273.15
        Ts_b = np.array([surf_C(base_res, t) for t in tt])
        Ts_l = np.array([surf_C(lat_res, t) for t in tt])
        max_depress = float(np.max(Ts_b - Ts_l))

        for tag, o in (("eta0", base_out), ("eta1", lat_out)):
            rows.append(dict(
                question=question, scenario=("η=0 无蒸发吸热" if tag == "eta0" else "η=1 表面蒸发吸热"),
                N=o["N"], Lv_form="诊断用近似汽化焓关系式（见论文附录）",
                rho_s0=round(o["rho_s0"], 4),
                reached=o["reached"],
                t_star_h=(round(o["t_star_h"], 4) if o["reached"] else ""),
                note=("" if o["reached"] else f"超过已计算时长 {t_cap:.0f} h（末端 Cmax={o['Cmax_end']:.4f}）"),
                T_surface_min_C=round(o["T_surface_min_C"], 4),
                q_evap_surface_W_m2_at_start=round(float(o["q_evap"][0]), 3),
            ))
        series_out[question] = dict(base=base_out, latent=lat_out,
                                    dt_star_h=dt_h, max_surface_depression_C=max_depress,
                                    depress_t=tt, depress=(Ts_b - Ts_l))

        print(f"[{question}] eta0 t*={base_out['t_star_h']:.4f} h (reached={base_out['reached']}), "
              f"eta1 t*={lat_out['t_star_h']:.4f} h (reached={lat_out['reached']}), "
              f"dt*={('%.4f h' % dt_h) if dt_h is not None else '-'}, "
              f"max surface depression={max_depress:.3f} degC, rho_s0={base_out['rho_s0']:.4f}")

    # --- part 2：已有 η=0 轨迹上的 3 h 蒸发热负荷（问题二三）---
    b23 = series_out["q23"]["base"]
    t3 = 3.0 * 3600.0
    st = _run_case(cfg, "q23", 0.0, t_cap_h=4.0)  # 短时段精确取 3 h 状态
    o3, op3, res3 = st
    s3 = res3.eval([t3])[0]
    n = N_DIAG + 1
    C3 = s3[:n]; T3 = s3[n:2 * n]
    Cs3, Ts3 = float(C3[-1]), float(T3[-1])
    Cenv3 = float(op3.env.C_env(t3))
    jw3 = op3.rho_s(t3) * op3.hm * (Cs3 - Cenv3)
    q3 = Lv(T3[-1]) * jw3
    h_htc = cfg.h
    load_check = dict(question="q23", t_h=3.0, C_s=round(Cs3, 4), C_env=round(Cenv3, 5),
                      T_s_C=round(Ts3 - 273.15, 4), rho_s0=round(op3.rho_s0, 4),
                      Lv_at_Ts=round(Lv(T3[-1]), 1), j_w_kg_m2_s=jw3, q_evap_W_m2=round(q3, 2),
                      q_evap_over_h_C=round(q3 / h_htc, 2), h_W_m2K=h_htc,
                      note="q_evap/h 为承担该热负荷的对流温差尺度，非实际温降或时长误差")
    print(f"[load-check] q23 3h: C_s={Cs3:.4f}, C_env={Cenv3:.5f}, "
          f"q_evap={q3:.2f} W/m^2, q_evap/h={q3/h_htc:.2f} degC, rho_s0={op3.rho_s0:.4f}, "
          f"L_v={Lv(T3[-1]):.0f} J/kg")

    # --- 写汇总 CSV ---
    cols = ["question", "scenario", "N", "Lv_form", "rho_s0", "reached", "t_star_h",
            "note", "T_surface_min_C", "q_evap_surface_W_m2_at_start"]
    with (EXPORTS / "latent_heat_diag.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # --- 写时程 CSV（问题二三、问题四各两情景表面/中心温度）---
    for q in ("q23", "q4"):
        b = series_out[q]["base"]; l = series_out[q]["latent"]
        with (EXPORTS / f"latent_heat_history_{q}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # 每种情景使用其所属的时间列（η=0 与 η=1 的事件时刻不同，不得共用一列）
            w.writerow(["t_eta0_s", "t_eta1_s", "T_center_eta0_C", "T_surface_eta0_C",
                        "T_center_eta1_C", "T_surface_eta1_C", "Cmax_eta0", "Cmax_eta1"])
            m = min(len(b["t"]), len(l["t"]))
            for i in range(m):
                w.writerow([f"{b['t'][i]:.3f}", f"{l['t'][i]:.3f}",
                            f"{b['T_center'][i]:.5f}", f"{b['T_surface'][i]:.5f}",
                            f"{l['T_center'][i]:.5f}", f"{l['T_surface'][i]:.5f}",
                            f"{b['Cmax'][i]:.6f}", f"{l['Cmax'][i]:.6f}"])

    # --- 写 JSON 汇总 ---
    summary = dict(
        aux=dict(N=N_DIAG, interface=INTERFACE, npts=NPTS, C0=C0,
                 Lv_50C=round(Lv(323.15), 1), Lv_form="diagnostic approximate latent-heat correlation (see paper appendix)"),
        rows=rows, load_check=load_check,
        dt_star=dict(q23=series_out["q23"]["dt_star_h"], q4=series_out["q4"]["dt_star_h"]),
        max_surface_depression_C=dict(q23=series_out["q23"]["max_surface_depression_C"],
                                      q4=series_out["q4"]["max_surface_depression_C"]),
    )
    (EXPORTS / "latent_heat_diag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出：exports/latent_heat_diag.csv, latent_heat_history_q23.csv, "
          "latent_heat_history_q4.csv, latent_heat_diag_summary.json")
    return summary


if __name__ == "__main__":
    run()
