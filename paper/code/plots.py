"""plots.py —— matplotlib 候选阶段图 + exports/*.csv（供 MATLAB）。

依赖最终生产配置的图（result 图等）待 D12 后重生成。中文标签在本机（Windows）
用 Microsoft YaHei / SimHei；无中文字体时回退英文不影响数据。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfgmod
from . import data_io

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGS = cfgmod.PROJECT_ROOT / "figs"
EXPORTS = cfgmod.PROJECT_ROOT / "exports"
FIGS.mkdir(exist_ok=True)
EXPORTS.mkdir(exist_ok=True)


def _save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_env(cfg):
    """图 2：环境时序（附件 1 插值 + 断点 + 常值外推）。"""
    t, T, C = data_io.load_attachment1(cfg.air_file())
    env = data_io.make_env_functions(cfg, "base")
    tt = np.linspace(0, 20000, 4001)
    Tair = np.array([env.T_air_K(x) - 273.15 for x in tt])
    Cenv = np.array([env.C_env(x) for x in tt])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tt, Tair, "-", lw=1.2, label="驱动 T_air")
    ax[0].plot(t, T, ".", ms=3, alpha=0.5, label="附件1 原始")
    ax[0].axvline(14400, color="r", ls="--", lw=0.8, label="断点 14400 s")
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("T_air (°C)"); ax[0].legend(fontsize=8)
    ax[1].plot(tt, Cenv, "-", lw=1.2)
    ax[1].plot(t, C, ".", ms=3, alpha=0.5)
    ax[1].axvline(14400, color="r", ls="--", lw=0.8)
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("C_env (kg/kg)")
    fig.suptitle("图2 环境驱动：0–14400 s 线性插值 + 常值外推")
    _save(fig, "fig2_env")
    np.savetxt(EXPORTS / "env_curve.csv",
               np.column_stack([tt, Tair, Cenv]), delimiter=",",
               header="t_s,T_air_C,C_env", comments="")


def plot_radius(cfg):
    """图 3：半径 R(t) 收缩曲线。"""
    t, R = data_io.load_attachment2(cfg.radius_file())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t / 3600.0, R, ".-", ms=4, lw=1)
    ax.axhline(1.2, color="g", ls=":", lw=0.8, label="1.2 cm")
    ax.set_xlabel("t (h)"); ax.set_ylabel("R (cm)")
    ax.set_title("图3 药材半径 R(t)（附件 2）"); ax.legend(fontsize=8)
    _save(fig, "fig3_radius")
    np.savetxt(EXPORTS / "radius_curve.csv",
               np.column_stack([t, R]), delimiter=",",
               header="t_s,R_cm", comments="")


def plot_convergence(study_table, *, cfg=None):
    """图 8：Q2/Q3 t* vs N（harmonic vs integral 界面收敛对照）+ 相邻差表。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for interface, byN in study_table.items():
        Ns = sorted(byN.keys())
        ts = [byN[N] for N in Ns]
        ax.plot(Ns, ts, "o-", label=interface)
        for N in Ns:
            rows.append((interface, N, byN[N]))
    ax.set_xlabel("N"); ax.set_ylabel("t* (h)")
    npts = cfg.raw["numerics"]["quadrature"]["interface_points"] if cfg else 8
    ax.set_title(f"图8 问题Q2/Q3 达标时长 t* 网格收敛\n候选设置：BDF rtol 1e-8, 积分界面 {npts} 点 Gauss（vs 调和平均）")
    ax.legend()
    _save(fig, "fig8_convergence")
    # 数值差表（相邻 N 的 t* 差）
    with open(EXPORTS / "fig8_diff.csv", "w", encoding="utf-8") as f:
        f.write("problem,interface,N,t_star_h,adjacent_diff_h\n")
        for interface, byN in study_table.items():
            Ns = sorted(byN.keys())
            for i, N in enumerate(Ns):
                dd = "" if i == 0 else f"{abs(byN[N]-byN[Ns[i-1]]):.6f}"
                f.write(f"Q2Q3,{interface},{N},{byN[N]:.6f},{dd}\n")
    with open(EXPORTS / "convergence.csv", "w", encoding="utf-8") as f:
        f.write("interface,N,t_star_h\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f}\n")


def plot_s10(s10):
    """图 10：Q4 附录 4 固定半径对照（物性 vs 几何效应）+ 三情形数值差表。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {"case1_app3_R0": "①附录3/R0", "case2_app4_R0": "②附录4/R0",
              "case3_app4_Rt": "③附录4/R(t)"}
    for key, lab in labels.items():
        Ns = sorted(s10[key].keys())
        ax.plot(Ns, [s10[key][N] for N in Ns], "s-", label=lab)
    ax.set_xlabel("N"); ax.set_ylabel("t* (h)")
    ax.set_title("图10 问题Q4 附录4 固定半径对照：分离物性与几何效应\n候选设置：积分界面 8 点, BDF rtol 1e-8")
    ax.legend()
    _save(fig, "fig10_s10")
    with open(EXPORTS / "fig10_diff.csv", "w", encoding="utf-8") as f:
        f.write("problem,case,N,t_star_h\n")
        for key, lab in labels.items():
            for N in sorted(s10[key]):
                f.write(f"Q4,{lab},{N},{s10[key][N]:.6f}\n")


def plot_q1_profiles(q1):
    """图 4：Q1 剖面（不同时刻 C、T vs r）。"""
    cols = q1["cols_cm"]
    t = q1["result1_t"]
    idxs = [np.argmin(np.abs(t - tt)) for tt in (100, 600, 1200, 1800)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i in idxs:
        ax[0].plot(cols, q1["result1_C"][i], "-", label=f"t={int(t[i])}s")
        ax[1].plot(cols, q1["result1_T"][i], "-", label=f"t={int(t[i])}s")
    ax[0].set_xlabel("r (cm)"); ax[0].set_ylabel("C (kg/kg)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("r (cm)"); ax[1].set_ylabel("T (°C)"); ax[1].legend(fontsize=8)
    fig.suptitle("图4 Q1 径向剖面（附录 2）")
    _save(fig, "fig4_q1_profiles")


def plot_q4_profiles(q4):
    """图 7：Q4 剖面（固定厘米位置随时间；域外留空）。"""
    t = np.asarray(q4["result4_t"])
    rows = q4["result4_rows"]
    cols = q4["cols20_cm"]
    surf = [r[-1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, rc in [(0, 0.0), (5, 0.5), (10, 1.0)]:
        vals = [rows[i][j] if rows[i][j] is not None else np.nan for i in range(len(t))]
        ax.plot(t / 3600.0, vals, "-", label=f"r={rc}cm")
    ax.plot(t / 3600.0, surf, "-", label="药材表面")
    ax.axhline(0.15, color="k", ls=":", lw=0.8, label="阈值 0.15")
    ax.set_xlabel("t (h)"); ax.set_ylabel("C (kg/kg)")
    ax.set_title("图7 Q4 固定位置含水率时程"); ax.legend(fontsize=8)
    _save(fig, "fig7_q4_history")


def all_candidate_figs(cfg, *, study_table=None, s10=None, q1=None, q4=None):
    plot_env(cfg)
    plot_radius(cfg)
    if study_table is not None:
        plot_convergence(study_table, cfg=cfg)
    if s10 is not None:
        plot_s10(s10)
    if q1 is not None:
        plot_q1_profiles(q1)
    if q4 is not None:
        plot_q4_profiles(q4)
