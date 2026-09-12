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
    fig.suptitle("图2 环境驱动：0–14400 s 线性插值；4 h 后为常值外推假设")
    _save(fig, "fig2_env")
    np.savetxt(EXPORTS / "env_curve.csv",
               np.column_stack([tt, Tair, Cenv]), delimiter=",",
               header="t_s,T_air_C,C_env", comments="")


def plot_radius(cfg):
    """图 3：半径 R(t) 收缩曲线。"""
    t, R = data_io.load_attachment2(cfg.radius_file())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t / 3600.0, R, ".-", ms=4, lw=1)
    ax.axhline(1.2, color="g", ls=":", lw=0.8, label="1.2 cm 参考线（非下限）")
    ax.set_xlabel("t (h)"); ax.set_ylabel("R (cm)")
    ax.set_title("图3 药材半径 R(t)（附件 2）"); ax.legend(fontsize=8)
    _save(fig, "fig3_radius")
    np.savetxt(EXPORTS / "radius_curve.csv",
               np.column_stack([t, R]), delimiter=",",
               header="t_s,R_cm", comments="")


def plot_convergence(cfg, study_table):
    """图 8：Q23 时长网格趋势；精确值/相邻差与证据边界同时展示。"""
    fig, (ax, ax_table) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    rows = []
    for interface, byN in study_table.items():
        Ns = sorted(byN.keys())
        ts = [byN[N] for N in Ns]
        ax.plot(Ns, ts, "o-", label=interface)
        previous = None
        for N in Ns:
            adjacent = None if previous is None else abs(byN[N] - byN[previous])
            rows.append((interface, N, byN[N], adjacent))
            previous = N
    ax.set_xlabel("N"); ax.set_ylabel("t* (h)")
    ax.set_title("Q23：t* 网格趋势（仅时长证据）"); ax.legend()
    ax.grid(alpha=0.25)

    interfaces = list(study_table)
    Ns = sorted(set().union(*(set(values) for values in study_table.values())))
    cell_text = []
    for row_index, N in enumerate(Ns):
        row = [str(N)]
        row.extend(f"{study_table[name][N]:.9f}" for name in interfaces)
        if row_index == 0:
            row.extend("—" for _ in interfaces)
        else:
            prior = Ns[row_index - 1]
            row.extend(
                f"{abs(study_table[name][N] - study_table[name][prior]):.6f}"
                for name in interfaces
            )
        cell_text.append(row)
    labels = [
        "N", *[f"{name} t*/h" for name in interfaces],
        *[f"{name} 相邻差/h" for name in interfaces],
    ]
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text, colLabels=labels, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.45)
    ax_table.set_title("精确数值（1.2710 h 为 harmonic N200→400 相邻差）", fontsize=9)
    run = cfg.resolve_run("q23", N=max(Ns))
    fig.suptitle(
        f"图8 Q23 候选研究（未授权）：{run.scheme}, rtol={run.rtol:g}, "
        f"integral={run.integral_npts}点；不替代全场/Q4验收",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig8_convergence")
    with open(EXPORTS / "convergence.csv", "w", encoding="utf-8") as f:
        f.write("interface,N,t_star_h,adjacent_delta_h\n")
        for interface, N, value, adjacent in rows:
            adjacent_text = "" if adjacent is None else f"{adjacent:.9f}"
            f.write(f"{interface},{N},{value:.9f},{adjacent_text}\n")


def plot_s10(cfg, s10):
    """图 10：附录 4 固定半径对照（物性 vs 几何效应）。"""
    fig, (ax, ax_table) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.0, 1.05]},
    )
    labels = {"case1_app3_R0": "①附录3/R0", "case2_app4_R0": "②附录4/R0",
              "case3_app4_Rt": "③附录4/R(t)"}
    for key, lab in labels.items():
        Ns = sorted(s10[key].keys())
        ax.plot(Ns, [s10[key][N] for N in Ns], "s-", label=lab)
    ax.set_xlabel("N"); ax.set_ylabel("t* (h)")
    ax.set_title("S10 条件对照（两种作用方向相反）")
    ax.legend()
    ax.grid(alpha=0.25)

    N_show = max(set.intersection(*(set(values) for values in s10.values())))
    values = [s10[key][N_show] for key in labels]
    delta_property = values[1] - values[0]
    delta_geometry = values[2] - values[1]
    ax_table.axis("off")
    cell_text = [
        [labels[key], f"{s10[key][N_show]:.9f}"] for key in labels
    ] + [
        ["②−①（条件物性差）", f"{delta_property:+.9f}"],
        ["③−②（条件几何差）", f"{delta_geometry:+.9f}"],
    ]
    table = ax_table.table(
        cellText=cell_text, colLabels=[f"N={N_show} 情形", "t* 或差值 / h"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 1.45)
    ax_table.set_title("数值依赖比较路径，不是唯一贡献分解", fontsize=9)
    run = cfg.resolve_run("q23", N=N_show)
    fig.suptitle(
        f"图10 Q23/Q4 S10（未授权）：{run.scheme}, integral {run.integral_npts}点",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig10_s10")
    with open(EXPORTS / "s10.csv", "w", encoding="utf-8") as f:
        f.write("case,N,t_star_h\n")
        for key in labels:
            for N in sorted(s10[key]):
                f.write(f"{key},{N},{s10[key][N]:.9f}\n")


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
    run = q1["run_config"]
    fig.suptitle(
        f"图4 Q1 候选径向剖面（未授权）：N={q1['N']}, {run['scheme']}, "
        f"{run['interface']} {run['integral_npts']}点；折线为21个输出位置",
    )
    _save(fig, "fig4_q1_profiles")


def plot_q4_profiles(q4):
    """图 7：Q4 剖面（固定厘米位置随时间；域外留空）。"""
    t = np.asarray(q4["result4_t"])
    plot_t = np.r_[0.0, t]
    rows = q4["result4_rows"]
    cols = q4["cols20_cm"]
    surf = [r[-1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, rc in [(0, 0.0), (5, 0.5), (10, 1.0)]:
        values = [rows[i][j] if rows[i][j] is not None else np.nan
                  for i in range(len(t))]
        ax.plot(plot_t / 3600.0, np.r_[q4["trajectory"].eval([0.0])[0, 0], values],
                "-", label=f"r={rc}cm")
    surface0 = q4["trajectory"].eval([0.0])[0, q4["N"]]
    ax.plot(plot_t / 3600.0, np.r_[surface0, surf], "-", label="药材表面")
    ax.axhline(0.15, color="k", ls=":", lw=0.8, label="阈值 0.15")
    ax.set_xlabel("t (h)"); ax.set_ylabel("C (kg/kg)")
    run = q4["run_config"]
    ax.set_title(
        f"图7 Q4 候选时程：N={q4['N']}, {run['scheme']}, "
        f"{run['interface']} {run['integral_npts']}点（Excel仍从60 s起）"
    ); ax.legend(fontsize=8)
    _save(fig, "fig7_q4_history")


def all_candidate_figs(cfg, *, study_table=None, s10=None, q1=None, q4=None):
    plot_env(cfg)
    plot_radius(cfg)
    if study_table is not None:
        plot_convergence(cfg, study_table)
    if s10 is not None:
        plot_s10(cfg, s10)
    if q1 is not None:
        plot_q1_profiles(q1)
    if q4 is not None:
        plot_q4_profiles(q4)
