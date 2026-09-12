"""paper_figs.py —— 论文正式插图（读取官方 outputs/*.xlsx，输出矢量 PDF 到 paper/figures/）。

重绘入口：``python -m drymodel.paper_figs``。
本模块**只读**已授权生产的官方结果与原始附件，不重跑任何生产计算；
所有定量图使用统一调色板，温度场与水分场各用独立色标与单位。

统一调色板（五色协调，不机械按序拼接为渐变）：
    深蓝 #5271AE  浅蓝 #70ACDE  橙 #FFA660  金 #F5CC7D  红 #D85B59
约定：径向由内到外 中心→表面 用「深蓝→浅蓝→金→橙→红」，同一物理位置在各子图中
颜色与标记一致；水分场用蓝系色带（湿=深蓝），温度场用暖系色带（热=红），互不共用色标。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

from . import config as cfgmod
from . import data_io

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42        # 字体子集嵌入，Overleaf 直接显示
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"

ROOT = cfgmod.PROJECT_ROOT
OUT = ROOT / "outputs"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- 调色板 ----
BLUE_D, BLUE_L, ORANGE, GOLD, RED = "#5271AE", "#70ACDE", "#FFA660", "#F5CC7D", "#D85B59"

# 固定径向位置的颜色与标记（跨子图一致：内→外 = 蓝→红）
POS_STYLE = {
    0.0: dict(color=BLUE_D, marker="o"),
    0.5: dict(color=BLUE_L, marker="s"),
    1.0: dict(color=GOLD,   marker="^"),
    1.5: dict(color=ORANGE, marker="D"),
    2.0: dict(color=RED,    marker="v"),
}
# 时刻→颜色（用于剖面族，早→晚 = 蓝→红）
TIME_COLORS = [BLUE_D, BLUE_L, ORANGE, RED]

# 连续场色带（湿=深蓝，热=红；单调、可黑白区分）
CMAP_MOIST = LinearSegmentedColormap.from_list("moist", ["#F2F7FC", BLUE_L, BLUE_D])
CMAP_TEMP = LinearSegmentedColormap.from_list("temp", ["#FBF0D8", GOLD, ORANGE, RED])
GRAY_OUT = "#E6E6E6"   # 域外留白（淡灰）

THRESH = 0.15


# --------------------------------------------------------------------------
# 读取工具
# --------------------------------------------------------------------------
def _read_sheet(path: Path, sheet_index: int, *, tmax_s=None, tstep=1):
    """读取一个工作表 → (t_s[np], positions[list], M[np, nan for empty])。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet_index]
    it = ws.iter_rows(values_only=True)
    header = next(it)
    positions = list(header[1:])
    ts, M = [], []
    for i, row in enumerate(it):
        t = row[0]
        if t is None:
            continue
        if tmax_s is not None and float(t) > tmax_s:
            break
        if tstep > 1 and (i % tstep) != 0:
            continue
        ts.append(float(t))
        M.append([float(v) if v is not None else np.nan for v in row[1:]])
    wb.close()
    return np.asarray(ts), positions, np.asarray(M, dtype=float)


def _save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", dpi=300)
    fig.savefig(FIGDIR / f"{name}.png", dpi=160)   # 预览用
    plt.close(fig)
    print("  saved", name)


# --------------------------------------------------------------------------
# 图 1：环境与半径输入
# --------------------------------------------------------------------------
def fig_inputs(cfg):
    t1, T1, C1 = data_io.load_attachment1(cfg.air_file())        # s, °C, kg/kg
    t2, R2 = data_io.load_attachment2(cfg.radius_file())         # s, cm
    env = data_io.make_env_functions(cfg, "base")
    tt = np.linspace(0, 20000, 3001)
    Tair = np.array([env.T_air_K(x) - 273.15 for x in tt])
    Cenv = np.array([env.C_env(x) for x in tt])
    tsw = env.t_switch

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))
    # (a) 空气温度
    ax[0].axvspan(tsw / 3600, tt[-1] / 3600, color=GOLD, alpha=0.18, label="外推段")
    ax[0].plot(tt / 3600, Tair, "-", color=BLUE_D, lw=1.4, label="分段线性插值")
    ax[0].plot(t1 / 3600, T1, "o", ms=3, mfc="none", mec=RED, mew=0.8, label="附件1 原始节点")
    ax[0].axvline(tsw / 3600, color=RED, ls="--", lw=0.9)
    ax[0].set_xlabel("时间 t / h"); ax[0].set_ylabel("空气温度 $T_{air}$ / °C")
    ax[0].set_title("(a) 烘房空气温度"); ax[0].legend(fontsize=7, loc="lower right")
    # (b) 环境含水浓度
    ax[1].axvspan(tsw / 3600, tt[-1] / 3600, color=GOLD, alpha=0.18)
    ax[1].plot(tt / 3600, Cenv, "-", color=BLUE_D, lw=1.4)
    ax[1].plot(t1 / 3600, C1, "o", ms=3, mfc="none", mec=RED, mew=0.8)
    ax[1].axvline(tsw / 3600, color=RED, ls="--", lw=0.9, label="断点 4 h")
    ax[1].set_xlabel("时间 t / h"); ax[1].set_ylabel("环境含水浓度 $C_{env}$ / (kg/kg)")
    ax[1].set_title("(b) 环境有效含水浓度"); ax[1].legend(fontsize=7)
    # (c) 半径
    th = t2 / 3600
    ax[2].axvspan(72, th.max() if th.max() > 72 else 72, color=GOLD, alpha=0.18)
    ax[2].plot(th, R2, "-", color=BLUE_D, lw=1.2)
    ax[2].plot(th, R2, "o", ms=3, mfc="none", mec=RED, mew=0.7, label="附件2 原始节点")
    ax[2].axhline(1.2, color=BLUE_L, ls=":", lw=1.0, label="1.20 cm 平台")
    ax[2].set_xlabel("时间 t / h"); ax[2].set_ylabel("半径 $R(t)$ / cm")
    ax[2].set_title("(c) 药材半径（附件2）"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_inputs")


# --------------------------------------------------------------------------
# 图 2：问题一径向剖面
# --------------------------------------------------------------------------
def fig_q1_profiles():
    tsT, pos, T = _read_sheet(OUT / "result1.xlsx", 0)
    tsC, _, C = _read_sheet(OUT / "result1.xlsx", 1)
    pos = np.array([float(p) for p in pos])
    want = [100, 600, 1200, 1800]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, tw in enumerate(want):
        i = int(np.argmin(np.abs(tsT - tw)))
        col = TIME_COLORS[k]
        ax[0].plot(pos, T[i], "-o", color=col, ms=3.5, lw=1.3, label=f"t={tw} s")
        ax[1].plot(pos, C[i], "-o", color=col, ms=3.5, lw=1.3, label=f"t={tw} s")
    ax[0].set_xlabel("径向位置 r / cm"); ax[0].set_ylabel("温度 T / °C")
    ax[0].set_title("(a) 温度径向剖面"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("径向位置 r / cm"); ax[1].set_ylabel("干基含水率 C / (kg/kg)")
    ax[1].set_title("(b) 含水率径向剖面"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_q1_profiles")


# --------------------------------------------------------------------------
# 图 3：问题二/三 场演化 + 最大含水率曲线
# --------------------------------------------------------------------------
def fig_q23_fields():
    # 温度：result2 温度表 0–2 h
    tT, posT, T = _read_sheet(OUT / "result2.xlsx", 0, tmax_s=7200, tstep=6)
    posT = np.array([float(p) for p in posT])
    # 水分：result3 全程
    tC, posC, C = _read_sheet(OUT / "result3.xlsx", 0, tstep=2)
    posC = np.array([float(p) for p in posC])
    hC = tC / 3600.0
    Cmax = np.nanmax(C, axis=1)

    fig = plt.figure(figsize=(11.5, 7.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.28)
    # (a) 温度热力图
    axa = fig.add_subplot(gs[0, 0])
    Rm, Tm = np.meshgrid(posT, tT / 3600.0)
    pa = axa.pcolormesh(Rm, Tm, T, cmap=CMAP_TEMP, shading="auto", rasterized=True)
    cba = fig.colorbar(pa, ax=axa); cba.set_label("温度 / °C")
    axa.set_xlabel("径向位置 r / cm"); axa.set_ylabel("时间 t / h")
    axa.set_title("(a) 温度场演化（0–2 h，附录3）")
    # (b) 水分热力图 + 0.15 等值线
    axb = fig.add_subplot(gs[0, 1])
    Rc, Tc = np.meshgrid(posC, hC)
    pb = axb.pcolormesh(Rc, Tc, C, cmap=CMAP_MOIST, shading="auto", vmin=0.0, vmax=2.55, rasterized=True)
    cbb = fig.colorbar(pb, ax=axb); cbb.set_label("干基含水率 / (kg/kg)")
    cs = axb.contour(Rc, Tc, C, levels=[THRESH], colors=[RED], linewidths=1.4)
    axb.clabel(cs, fmt={THRESH: "0.15 达标线"}, fontsize=7)
    axb.set_xlabel("径向位置 r / cm"); axb.set_ylabel("时间 t / h")
    axb.set_title("(b) 含水率场演化（全程，附录3）")
    # (c) 最大（中心）含水率曲线
    axc = fig.add_subplot(gs[1, :])
    axc.plot(hC, Cmax, "-", color=BLUE_D, lw=1.6, label="全域最大含水率 $C_{\\max}(t)$")
    axc.axhline(THRESH, color=RED, ls="--", lw=1.1, label="阈值 0.15")
    tstar = 57.4740
    axc.axvline(tstar, color=GOLD, ls=":", lw=1.2)
    axc.annotate(f"$t^*$ = {tstar:.4f} h", xy=(tstar, THRESH),
                 xytext=(tstar - 18, 0.6), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=GOLD))
    axc.set_xlabel("时间 t / h"); axc.set_ylabel("干基含水率 / (kg/kg)")
    axc.set_title("(c) 全域最大含水率随时间下降并穿越阈值")
    axc.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.28, 0.98))
    # 放大插图（阈值附近）
    axins = axc.inset_axes([0.60, 0.42, 0.36, 0.5])
    axins.plot(hC, Cmax, "-", color=BLUE_D, lw=1.4)
    axins.axhline(THRESH, color=RED, ls="--", lw=1.0)
    axins.axvline(tstar, color=GOLD, ls=":", lw=1.0)
    axins.set_xlim(54, 59); axins.set_ylim(0.145, 0.165)
    axins.tick_params(labelsize=7); axins.set_title("阈值附近放大", fontsize=7)
    _save(fig, "fig_q23_fields")


# --------------------------------------------------------------------------
# 图 4：问题四 动域场 + 固定位置时程
# --------------------------------------------------------------------------
def fig_q4_fields(cfg):
    tC, pos, C = _read_sheet(OUT / "result4.xlsx", 0, tstep=2)
    # 最后一列为“药材表面”，前 20 列为固定 cm 位置 0..1.9
    pos_cm = np.array([float(p) for p in pos[:-1]])
    Cgrid = C[:, :-1]               # 固定位置场（域外为 nan）
    surf = C[:, -1]                 # 表面值
    hC = tC / 3600.0
    t2, R2 = data_io.load_attachment2(cfg.radius_file())   # s, cm
    rad = data_io.make_radius_function(cfg)
    Rt_cm = np.array([rad.R(t) for t in tC]) * 100.0

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4),
                           gridspec_kw=dict(width_ratios=[1.05, 1.0], wspace=0.28))
    # (a) 动域热力图
    cmap = CMAP_MOIST.copy(); cmap.set_bad(GRAY_OUT)
    Cm = np.ma.masked_invalid(Cgrid)
    Rm, Tm = np.meshgrid(pos_cm, hC)
    pa = ax[0].pcolormesh(Rm, Tm, Cm, cmap=cmap, shading="auto", vmin=0.0, vmax=2.55, rasterized=True)
    cb = fig.colorbar(pa, ax=ax[0]); cb.set_label("干基含水率 / (kg/kg)")
    ax[0].plot(Rt_cm, hC, "-", color=RED, lw=1.6, label="移动边界 $R(t)$")
    ax[0].set_xlabel("固定物理位置 r / cm"); ax[0].set_ylabel("时间 t / h")
    ax[0].set_title("(a) 含水率场与移动边界（域外淡灰）")
    ax[0].legend(fontsize=8, loc="upper right")
    # (b) 固定位置 + 表面时程
    for rc in (0.0, 0.5, 1.0):
        j = int(np.argmin(np.abs(pos_cm - rc)))
        st = POS_STYLE[rc]
        ax[1].plot(hC, Cgrid[:, j], "-", color=st["color"], lw=1.4,
                   label=f"r={rc:.1f} cm")
    ax[1].plot(hC, surf, "-", color=RED, lw=1.4, label="药材表面")
    ax[1].axhline(THRESH, color="k", ls=":", lw=1.0, label="阈值 0.15")
    ax[1].axvline(51.0920, color=GOLD, ls=":", lw=1.2)
    ax[1].annotate("$t^*$ = 51.0920 h", xy=(51.0920, THRESH), xytext=(30, 0.9),
                   fontsize=9, arrowprops=dict(arrowstyle="->", color=GOLD))
    ax[1].set_xlabel("时间 t / h"); ax[1].set_ylabel("干基含水率 / (kg/kg)")
    ax[1].set_title("(b) 固定位置与表面含水率时程")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_q4_fields")


# --------------------------------------------------------------------------
# 图 5：三情形对照（物性 vs 几何）
# --------------------------------------------------------------------------
def fig_threecase():
    # 取自验证研究 exports/fig10_diff.csv（N=400）
    cases = [("① 附录3 物性 / 固定 $R_0$", 57.4741, BLUE_D),
             ("② 附录4 物性 / 固定 $R_0$", 129.8487, GOLD),
             ("③ 附录4 物性 / 收缩 $R(t)$", 51.0921, RED)]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ys = np.arange(len(cases))[::-1]
    for y, (lab, val, col) in zip(ys, cases):
        ax.plot([0, val], [y, y], "-", color=col, lw=1.0, alpha=0.5)
        ax.plot(val, y, "o", color=col, ms=11)
        ax.text(val + 2, y, f"{val:.2f} h", va="center", fontsize=9, color=col)
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in cases], fontsize=9)
    ax.set_xlabel("达标时长 $t^*$ / h"); ax.set_xlim(0, 145)
    ax.set_title("物性变化与尺寸收缩对达标时长的相反作用")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_threecase")


# --------------------------------------------------------------------------
# 图 6：网格/界面收敛
# --------------------------------------------------------------------------
def fig_convergence():
    import csv
    rows = list(csv.DictReader(open(ROOT / "exports" / "convergence.csv", encoding="utf-8")))
    data = {}
    for r in rows:
        data.setdefault(r["interface"], []).append((int(r["N"]), float(r["t_star_h"])))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    styles = {"harmonic": dict(color=RED, marker="s", label="调和平均界面"),
              "integral": dict(color=BLUE_D, marker="o", label="积分界面（8 点 Gauss）")}
    for key, pts in data.items():
        pts.sort()
        Ns = [p[0] for p in pts]; ts = [p[1] for p in pts]
        st = styles.get(key, dict(color=GOLD, marker="^", label=key))
        ax.plot(Ns, ts, "-", **st, ms=7, lw=1.4)
    ax.set_xscale("log", base=2); ax.set_xticks([200, 400, 800])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("网格区间数 N"); ax.set_ylabel("达标时长 $t^*$ / h")
    ax.set_title("达标时长随网格加密的收敛（问题二/三）")
    ax.legend(fontsize=9); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_convergence")


# --------------------------------------------------------------------------
# 图 7：参数扰动（灵敏度）
# --------------------------------------------------------------------------
def fig_sensitivity():
    # 取自 reports/sensitivity.md（基线 57.4741 h）
    items = [
        ("传质系数 $h_m\\times0.5$", +7.455, RED),
        ("传质系数 $h_m\\times2$", -2.802, RED),
        ("外推温度 $-0.39$ °C", +0.718, ORANGE),
        ("外推温度 $+0.39$ °C", -0.706, ORANGE),
        ("外推取末值 hold-last", -0.305, GOLD),
        ("换热系数 $h\\times0.5$", +0.080, BLUE_L),
        ("换热系数 $h\\times2$", -0.039, BLUE_L),
        ("环境含水 $\\pm0.0011$", +0.022, BLUE_D),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ys = np.arange(len(items))[::-1]
    for y, (lab, dv, col) in zip(ys, items):
        ax.barh(y, dv, color=col, alpha=0.85, height=0.62)
        ax.text(dv + (0.15 if dv >= 0 else -0.15), y, f"{dv:+.2f}",
                va="center", ha="left" if dv >= 0 else "right", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([it[0] for it in items], fontsize=8.5)
    ax.set_xlabel("达标时长变化 $\\Delta t^*$ / h（基线 57.4741 h）")
    ax.set_title("参数扰动对达标时长的影响（问题二/三，人为扰动范围内）")
    ax.set_xlim(-4, 9); ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


def main():
    cfg = cfgmod.load_config()
    print("生成论文插图 →", FIGDIR)
    fig_inputs(cfg)
    fig_q1_profiles()
    fig_q23_fields()
    fig_q4_fields(cfg)
    fig_threecase()
    fig_convergence()
    fig_sensitivity()
    print("完成。")


if __name__ == "__main__":
    main()
