"""paper_figs.py —— 论文正式插图（读取官方 outputs 与验证记录，输出矢量 PDF 到 paper/figures/）。

重绘入口：``python -m drymodel.paper_figs``。
本模块**只读**已授权生产的官方结果、原始附件与验证记录，不重跑任何生产计算，
也不在代码中硬编码结果常数：达标时长、三情形、收敛与灵敏度数值均从下列来源读取
    outputs/production_receipt_supplementary.json   达标 t*、采样时刻、未舍入 Cmax
    exports/convergence.csv                          网格/界面收敛 t*
    exports/fig10_diff.csv                           三情形对照 t*
    reports/V1_V2.md                                 解析基准全域最大误差
    reports/sensitivity.md                           参数扰动时长变化
温度场与水分场各用独立色标与单位；连续场对照保持一致归一化。

统一调色板（Ocean Jelly + Warm Coral）：
    冷色 深海蓝 #005BBD / 海洋蓝 #008FF5 / 果冻青 #29C7F6 / 冰蓝 #86E6FF
    暖色 琥珀 #FFC247 / 金橙 #FFA83A / 珊瑚橙 #FF7A45 / 珊瑚红 #FF5B57 / 正红 #F0404F
    中性 炭黑 #252A30 / 薄雾 #EAF3F8
白底、炭黑文字；浅色只用于填充与背景带，不承担白底上的细线、箭头或数字标签。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import openpyxl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec, ticker

from . import config as cfgmod
from . import data_io

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42        # 字体子集嵌入，Overleaf 直接显示
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.edgecolor"] = "#555555"
plt.rcParams["text.color"] = "#222222"
plt.rcParams["axes.labelcolor"] = "#222222"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"

ROOT = cfgmod.PROJECT_ROOT
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "exports"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- 调色板（Ocean Jelly + Warm Coral）----
DEEP, OCEAN, JELLY, ICE = "#005BBD", "#008FF5", "#29C7F6", "#86E6FF"
AMBER, GORANGE, CORAL, CRED, RED = "#FFC247", "#FFA83A", "#FF7A45", "#FF5B57", "#F0404F"
CHAR, MIST = "#252A30", "#EAF3F8"
DARK = CHAR               # 数字/文字标签（深灰）
GRAY_OUT = "#E4E4E4"      # 域外留白（中性淡灰）
# 兼容旧名（映射到新配色）
BLUE_D, BLUE_L, ORANGE, GOLD = DEEP, OCEAN, CORAL, AMBER

# 固定径向位置的颜色与标记（跨子图一致：内→外 = 冷→暖）
POS_STYLE = {
    0.0: dict(color=DEEP,    marker="o", label="中心 $r=0$"),
    0.5: dict(color=OCEAN,   marker="s", label="$r=0.5$ cm"),
    1.0: dict(color=JELLY,   marker="^", label="$r=1.0$ cm"),
    1.5: dict(color=GORANGE, marker="D", label="$r=1.5$ cm"),
    2.0: dict(color=RED,     marker="v", label="表面 $r=2.0$ cm"),
}
TIME_COLORS = [DEEP, OCEAN, CORAL, RED]   # 早→晚

# 连续场色带（湿=深蓝，热=红；单调、可黑白区分）
CMAP_MOIST = LinearSegmentedColormap.from_list("moist", ["#F4FAFE", ICE, OCEAN, DEEP])
CMAP_TEMP = LinearSegmentedColormap.from_list("temp", ["#FFF4D9", AMBER, CORAL, RED])
# 蓝色阶（小影响条形，深→浅）
CMAP_BLUES = LinearSegmentedColormap.from_list("blues", [DEEP, OCEAN, JELLY, ICE])

THRESH = 0.15


# ==========================================================================
# 数据来源（只读，不硬编码结果常数）
# ==========================================================================
def load_receipt():
    d = json.loads((OUT / "production_receipt_supplementary.json").read_text(encoding="utf-8"))
    r = d["results"]
    return {
        "tstar23_h": r["q23"]["t_star_s"] / 3600.0, "tstar23_s": r["q23"]["t_star_s"],
        "tsamp23_s": r["q23"]["t_sample_s"], "cmax23": r["q23"]["cmax_at_tsample_unrounded"],
        "tstar4_h": r["q4"]["t_star_s"] / 3600.0, "tstar4_s": r["q4"]["t_star_s"],
        "tsamp4_s": r["q4"]["t_sample_s"], "cmax4": r["q4"]["cmax_at_tsample_unrounded"],
    }


def load_convergence():
    d = {}
    for r in csv.DictReader((EXPORTS / "convergence.csv").open(encoding="utf-8")):
        d.setdefault(r["interface"], []).append((int(r["N"]), float(r["t_star_h"])))
    for k in d:
        d[k].sort()
    return d


def load_threecase(N=400):
    out = {}
    for r in csv.DictReader((EXPORTS / "fig10_diff.csv").open(encoding="utf-8")):
        if int(r["N"]) == N:
            out[r["case"]] = float(r["t_star_h"])
    return out


def load_v1v2():
    txt = (REPORTS / "V1_V2.md").read_text(encoding="utf-8")

    def grab(sec):
        m = re.search(r"##\s*" + re.escape(sec) + r".*?(?=\n##|\Z)", txt, re.S)
        pts = []
        for line in m.group(0).splitlines():
            if line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip("| \n").split("|")]
                if len(cells) == 3 and re.fullmatch(r"\d+", cells[0]):
                    pts.append((int(cells[0]), float(cells[2])))   # N, 全域最大误差
        return sorted(pts)
    return {"temp": grab("V-1"), "moist": grab("V-2")}


def load_sensitivity():
    txt = (REPORTS / "sensitivity.md").read_text(encoding="utf-8")
    m = re.search(r"基线\s*t\*\s*=\s*([\d.]+)", txt)
    base = float(m.group(1)) if m else 57.4741
    rows = []
    for line in txt.splitlines():
        if line.strip().startswith("| S"):
            cells = [c.strip() for c in line.strip("| \n").split("|")]
            rows.append((cells[0], float(cells[2]), cells[3]))   # 名称, Δt*, 可分辨
    return base, rows


# ==========================================================================
# 读取工具
# ==========================================================================
def _read_sheet(path: Path, sheet_index: int, *, tmax_s=None, tstep=1):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet_index]
    it = ws.iter_rows(values_only=True)
    positions = list(next(it)[1:])
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
    fig.savefig(FIGDIR / f"{name}.png", dpi=160)
    plt.close(fig)
    print("  saved", name)


# ==========================================================================
# 图：环境与半径输入
# ==========================================================================
def fig_inputs(cfg):
    t1, T1, C1 = data_io.load_attachment1(cfg.air_file())
    t2, R2 = data_io.load_attachment2(cfg.radius_file())
    env = data_io.make_env_functions(cfg, "base")
    tsw_h = env.t_switch / 3600.0
    tt = np.linspace(0, 6.5 * 3600, 3001)
    Tair = np.array([env.T_air_K(x) - 273.15 for x in tt])
    Cenv = np.array([env.C_env(x) for x in tt])

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for a, raw_t, raw_y, yl, ttl in (
            (ax[0], t1 / 3600, T1, "空气温度 $T_{\\mathrm{air}}$ / °C", "(a) 烘房空气温度"),
            (ax[1], t1 / 3600, C1, "环境含水浓度 $C_{\\mathrm{env}}$ / (kg·kg$^{-1}$)", "(b) 环境有效含水浓度")):
        y_int = Tair if a is ax[0] else Cenv
        a.axvspan(tsw_h, tt[-1] / 3600, color=GOLD, alpha=0.16, lw=0)
        a.plot(tt / 3600, y_int, "-", color=BLUE_D, lw=1.5, zorder=3)
        a.plot(raw_t, raw_y, "o", ms=2.2, mfc="none", mec="#8a8a8a", mew=0.6, zorder=2)
        a.axvline(tsw_h, color=RED, ls="--", lw=1.0, zorder=4)
        a.set_xlabel("时间 $t$ / h"); a.set_ylabel(yl); a.set_title(ttl)
    ax[0].text(tsw_h + 0.05, ax[0].get_ylim()[0] + 0.12 * (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]),
               "常值外推段\n（4 h 后）", fontsize=7.5, color="#8a6d1f")
    # 图例（用代理）
    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color=BLUE_D, lw=1.5, label="分段线性插值"),
             Line2D([0], [0], marker="o", mfc="none", mec="#8a8a8a", ls="none", ms=4, label="附件1 原始节点"),
             Line2D([0], [0], color=RED, ls="--", lw=1.0, label="断点 $t=4$ h")]
    ax[0].legend(handles=proxy, fontsize=7, loc="lower right", framealpha=0.9)
    # (c) 半径
    th = t2 / 3600
    ax[2].plot(th, R2, "-", color=BLUE_D, lw=1.4, zorder=3, label="附件2 半径 $R(t)$")
    ax[2].plot(th, R2, "o", ms=2.2, mfc="none", mec="#8a8a8a", mew=0.6, zorder=2)
    ax[2].axhline(1.2, color="#9aa7c4", ls=":", lw=1.0, label="约 1.20 cm 平台")
    ax[2].set_xlabel("时间 $t$ / h"); ax[2].set_ylabel("半径 $R(t)$ / cm")
    ax[2].set_title("(c) 药材半径（附件2，≤ 72 h）"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_inputs")


# ==========================================================================
# 图：问题一径向剖面
# ==========================================================================
def fig_q1_profiles():
    tsT, pos, T = _read_sheet(OUT / "result1.xlsx", 0)
    _, _, C = _read_sheet(OUT / "result1.xlsx", 1)
    pos = np.array([float(p) for p in pos])
    want = [100, 600, 1200, 1800]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    mk = ["o", "s", "^", "D"]
    for k, tw in enumerate(want):
        i = int(np.argmin(np.abs(tsT - tw)))
        col = TIME_COLORS[k]
        ax[0].plot(pos, T[i], "-", color=col, marker=mk[k], ms=3.6, lw=1.3, label=f"$t={tw}$ s")
        ax[1].plot(pos, C[i], "-", color=col, marker=mk[k], ms=3.6, lw=1.3, label=f"$t={tw}$ s")
    ax[0].set_xlabel("径向位置 $r$ / cm"); ax[0].set_ylabel("温度 $T$ / °C")
    ax[0].set_title("(a) 温度径向剖面"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("径向位置 $r$ / cm"); ax[1].set_ylabel("干基含水率 $C$ / (kg·kg$^{-1}$)")
    ax[1].set_title("(b) 含水率径向剖面"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_q1_profiles")


# ==========================================================================
# 图：问题二/三 场演化 + 最大含水率曲线
# ==========================================================================
def fig_q23_fields():
    rec = load_receipt()
    tstar_h = rec["tstar23_h"]
    tT, posT, T = _read_sheet(OUT / "result2.xlsx", 0, tmax_s=7200, tstep=6)
    posT = np.array([float(p) for p in posT])
    tC, posC, C = _read_sheet(OUT / "result3.xlsx", 0)
    posC = np.array([float(p) for p in posC])
    hC = tC / 3600.0
    Cmax = np.nanmax(C, axis=1)                 # 输出位置上的最大值（四位小数）

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.92], hspace=0.44, wspace=0.26)
    axa = fig.add_subplot(gs[0, 0])
    Rm, Tm = np.meshgrid(posT, tT / 3600.0)
    pa = axa.pcolormesh(Rm, Tm, T, cmap=CMAP_TEMP, shading="auto", rasterized=True)
    fig.colorbar(pa, ax=axa).set_label("温度 / °C")
    axa.set_xlabel("径向位置 $r$ / cm"); axa.set_ylabel("时间 $t$ / h")
    axa.set_title("(a) 温度场演化（$0$–$2$ h，附录3）")
    axb = fig.add_subplot(gs[0, 1])
    Rc, Tc = np.meshgrid(posC, hC)
    pb = axb.pcolormesh(Rc, Tc, C, cmap=CMAP_MOIST, shading="auto", vmin=0.0, vmax=2.55, rasterized=True)
    fig.colorbar(pb, ax=axb).set_label("干基含水率 / (kg·kg$^{-1}$)")
    cs = axb.contour(Rc, Tc, C, levels=[THRESH], colors=[RED], linewidths=1.4)
    axb.clabel(cs, fmt={THRESH: "0.15"}, fontsize=7)
    axb.set_xlabel("径向位置 $r$ / cm"); axb.set_ylabel("时间 $t$ / h")
    axb.set_title("(b) 含水率场演化（全程，附录3）")
    # (c) 全程最大含水率曲线（独立坐标，不用内嵌窗遮挡数据）
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(hC, Cmax, "-", color=DEEP, lw=1.6, label="输出最大含水率")
    axc.axhline(THRESH, color=RED, ls="--", lw=1.1, label="阈值 0.15")
    axc.axvline(tstar_h, color=CHAR, ls=":", lw=1.0)
    axc.annotate(f"$t^*={tstar_h:.4f}$ h", xy=(tstar_h, THRESH), xytext=(tstar_h - 32, 0.7),
                 fontsize=8.5, color=DARK, arrowprops=dict(arrowstyle="->", color=CHAR))
    axc.set_xlabel("时间 $t$ / h"); axc.set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    axc.set_xlim(0, hC.max()); axc.set_ylim(0, 2.65)
    axc.set_title("(c) 全程最大含水率")
    axc.legend(fontsize=8, loc="upper right")
    # (d) 阈值附近放大：离散 60 s 采样点（不以四位小数平台伪造严格穿越）
    axd = fig.add_subplot(gs[1, 1])
    sel = (hC >= tstar_h - 1.6) & (hC <= hC.max())
    axd.plot(hC[sel], Cmax[sel], "o", ms=3.4, color=OCEAN, label="$60$ s 采样（四位小数）")
    axd.axhline(THRESH, color=RED, ls="--", lw=1.1, label="阈值 0.15")
    axd.axvline(tstar_h, color=CHAR, ls=":", lw=1.2, label="$t^*$（未舍入求根）")
    axd.set_xlim(tstar_h - 1.4, hC.max()); axd.set_ylim(0.1490, 0.1560)
    axd.set_xlabel("时间 $t$ / h"); axd.set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    axd.set_title("(d) 阈值附近（采样分辨率）")
    axd.legend(fontsize=7.5, loc="upper right")
    _save(fig, "fig_q23_fields")


# ==========================================================================
# 图：问题四 动域场（边界严格裁切）+ 固定位置时程
# ==========================================================================
def fig_q4_fields(cfg):
    rec = load_receipt()
    tstar4_h = rec["tstar4_h"]
    tC, pos, C = _read_sheet(OUT / "result4.xlsx", 0)          # 全部 60 s 行（含首末）
    pos_cm = np.array([float(p) for p in pos[:-1]])            # 固定位置 0..1.9
    Cgrid = C[:, :-1]                                          # 域外为 nan
    surf = C[:, -1]                                            # 表面值
    hC = tC / 3600.0
    rad = data_io.make_radius_function(cfg)
    Rt_cm = np.array([float(rad.R(t)) * 100.0 for t in tC])
    R0cm = cfg.R0 * 100.0

    # —— 每时刻仅用域内固定位置 + 真实边界点 (R,C_s)，插值到细网格并严格裁切 ——
    rfine = np.linspace(0.0, R0cm, 240)
    Z = np.full((len(tC), rfine.size), np.nan)
    for i in range(len(tC)):
        Rt = Rt_cm[i]
        m = pos_cm <= Rt - 1e-9
        xp = list(pos_cm[m]); fp = list(Cgrid[i][m])
        if xp and abs(xp[-1] - Rt) <= 1e-9:      # 边界与固定位置重合 → 去重
            fp[-1] = surf[i]
        else:
            xp.append(Rt); fp.append(surf[i])
        xp = np.asarray(xp); fp = np.asarray(fp)
        sel = rfine <= Rt + 1e-9
        Z[i, sel] = np.interp(rfine[sel], xp, fp)
    Zm = np.ma.masked_invalid(Z)
    cmap = CMAP_MOIST.copy(); cmap.set_bad(GRAY_OUT)

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.5),
                           gridspec_kw=dict(width_ratios=[1.06, 1.0], wspace=0.30))
    Rm, Tm = np.meshgrid(rfine, hC)
    pa = ax[0].pcolormesh(Rm, Tm, Zm, cmap=cmap, shading="nearest", vmin=0.0, vmax=2.55,
                          rasterized=True, zorder=1)
    fig.colorbar(pa, ax=ax[0]).set_label("干基含水率 / (kg·kg$^{-1}$)")
    # 域外统一淡灰覆盖（消除任何半格越界），红色移动边界最后叠加
    ax[0].fill_betweenx(hC, Rt_cm, R0cm, color=GRAY_OUT, lw=0, zorder=2)
    ax[0].plot(Rt_cm, hC, "-", color=RED, lw=1.7, zorder=3, label="移动边界 $R(t)$")
    ax[0].set_xlim(0, R0cm); ax[0].set_ylim(hC.min(), hC.max())
    ax[0].set_xlabel("固定物理位置 $r$ / cm"); ax[0].set_ylabel("时间 $t$ / h")
    ax[0].set_title("(a) 含水率场与移动边界（域外淡灰、严格裁切）")
    ax[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    # (b) 固定位置 + 表面时程（用全部行，含首末；事件时刻标注）
    for rc in (0.0, 0.5, 1.0):
        j = int(np.argmin(np.abs(pos_cm - rc)))
        st = POS_STYLE[rc]
        ax[1].plot(hC, Cgrid[:, j], "-", color=st["color"], lw=1.5, label=st["label"])
    ax[1].plot(hC, surf, "-", color=RED, lw=1.5, label="药材表面 $r=R(t)$")
    ax[1].axhline(THRESH, color="#555555", ls=":", lw=1.0, label="阈值 0.15")
    ax[1].axvline(tstar4_h, color="#7a7a7a", ls="--", lw=1.0)
    ax[1].annotate(f"$t^*={tstar4_h:.4f}$ h", xy=(tstar4_h, THRESH), xytext=(28, 0.95),
                   fontsize=9, color=DARK, arrowprops=dict(arrowstyle="->", color="#7a7a7a"))
    ax[1].set_xlim(0, hC.max()); ax[1].set_ylim(0, 2.65)
    ax[1].set_xlabel("时间 $t$ / h"); ax[1].set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    ax[1].set_title("(b) 固定位置与表面含水率时程")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_q4_fields")


# ==========================================================================
# 图：解析基准收敛（V-1/V-2，全域最大误差 vs N）
# ==========================================================================
def fig_analytic_convergence():
    d = load_v1v2()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.95))   # 按最终插入宽度设画布，避免缩放后字过小
    for a, key, ttl, yl, col in (
            (ax[0], "temp", "(a) 温度（附录2 常物性，$t=100$ s）", "沿半径最大误差 / °C", CORAL),
            (ax[1], "moist", "(b) 含水率（$D\\equiv D(C_0)$，$t=100$ s）",
             "沿半径最大误差 / (kg·kg$^{-1}$)", OCEAN)):
        Ns = np.array([p[0] for p in d[key]], float)
        err = np.array([p[1] for p in d[key]], float)
        a.loglog(Ns, err, "o-", color=col, ms=5.5, lw=1.7, label="数值 vs 解析级数解")
        ref = err[0] * (Ns[0] / Ns) ** 2
        a.loglog(Ns, ref, "--", color="#9AA0A6", lw=1.1, label="二阶参考斜率")
        a.set_xlabel("网格区间数 $N$", fontsize=10.5); a.set_ylabel(yl, fontsize=10)
        a.set_title(ttl, fontsize=10.5)
        a.set_xticks(Ns); a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        a.tick_params(labelsize=9.5)
        a.grid(which="both", ls=":", alpha=0.4); a.legend(fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig_analytic_convergence")


# ==========================================================================
# 图：网格/界面收敛（含收敛误差子图，使积分界面可辨识）
# ==========================================================================
def fig_convergence():
    d = load_convergence()
    styles = {"harmonic": dict(color=CORAL, marker="s", label="调和平均界面"),
              "integral": dict(color=OCEAN, marker="o", label="积分界面（8 点 Gauss）")}
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.95))
    for key, pts in d.items():
        Ns = [p[0] for p in pts]; ts = [p[1] for p in pts]
        st = styles.get(key, dict(color=AMBER, marker="^", label=key))
        ax[0].plot(Ns, ts, "-", **st, ms=6.5, lw=1.6)
        ref = ts[-1]
        # (b) 只画比最细网格更粗的两点（最细网格相对自身为 0，不伪造正误差点）
        ax[1].loglog(Ns[:-1], [abs(t - ref) for t in ts[:-1]], "-", **st, ms=6.5, lw=1.6)
    ax[0].set_xscale("log", base=2)
    for a in ax:
        a.set_xticks([200, 400, 800]); a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        a.grid(which="both", ls=":", alpha=0.4); a.tick_params(labelsize=9.5)
    ax[0].set_xlabel("网格区间数 $N$", fontsize=10.5); ax[0].set_ylabel("达标时长 $t^*$ / h", fontsize=10.5)
    ax[0].set_title("(a) 达标时长随网格加密", fontsize=10.5); ax[0].legend(fontsize=9.5)
    ax[1].set_xlabel("网格区间数 $N$", fontsize=10.5); ax[1].set_ylabel("$|t^*_N-t^*_{800}|$ / h", fontsize=10.5)
    ax[1].set_title("(b) 相对最细网格的收敛（对数）", fontsize=10.5); ax[1].legend(fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig_convergence")


# ==========================================================================
# 图：三情形对照（物性 vs 几何；同一网格 N=400）
# ==========================================================================
def fig_threecase():
    tc = load_threecase(400)
    order = [("① 附录3 物性 · 固定 $R_0$", "附录3/R0", DEEP),
             ("② 附录4 物性 · 固定 $R_0$", "附录4/R0", GORANGE),
             ("③ 附录4 物性 · 收缩 $R(t)$", "附录4/R(t)", RED)]
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    ys = [1, 0, -1]           # 紧凑等距三行
    for y, (lab, key, col) in zip(ys, order):
        val = next(v for k, v in tc.items() if key in k)
        ax.plot([0, val], [y, y], "-", color=col, lw=1.2, alpha=0.45)
        ax.plot(val, y, "o", color=col, ms=11)
        ax.text(val + 2, y, f"{val:.2f} h", va="center", fontsize=9.5, color=DARK)
    ax.set_yticks(ys); ax.set_yticklabels([o[0] for o in order], fontsize=9)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("达标时长 $t^*$ / h（同一网格 $N=400$）"); ax.set_xlim(0, 150)
    ax.set_title("物性变化（①→②）延长干燥、尺寸收缩（②→③）缩短干燥")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_threecase")


# ==========================================================================
# 图：参数扰动（成对正负、恢复全部情景、小影响放大）
# ==========================================================================
def _sens_label(name):
    if "hm" in name:
        return ("传质系数 $h_m$ " + ("减半" if "0.5" in name else "加倍"), RED, "hm")
    if "T_air" in name:
        s = "$+0.39$" if "+0.39" in name else "$-0.39$"
        return ("外推空气温度 " + s + " °C", ORANGE, "Tair")
    if "C_env" in name:
        s = "$+0.0011$" if "+0.0011" in name else "$-0.0011$"
        return ("环境含水浓度 " + s, BLUE_L, "Cenv")
    if "hold_last" in name:
        return ("外推取末值 hold-last", GOLD, "hold")
    if "smooth" in name:
        return ("数据平滑 smooth121", "#9a9a9a", "smooth")
    return ("换热系数 $h$ " + ("减半" if "0.5" in name else "加倍"), BLUE_D, "h")


def fig_sensitivity():
    base, rows = load_sensitivity()
    items = [(_sens_label(n)[0], dv, _sens_label(n)[1], disc) for n, dv, disc in rows]
    items.sort(key=lambda z: -abs(z[1]))
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.0),
                           gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.32))
    # (a) 全部情景
    ys = np.arange(len(items))[::-1]
    for y, (lab, dv, col, disc) in zip(ys, items):
        ax[0].barh(y, dv, color=col, alpha=0.9, height=0.64)
        ax[0].text(dv + (0.12 if dv >= 0 else -0.12), y, f"{dv:+.3f}",
                   va="center", ha="left" if dv >= 0 else "right", fontsize=8, color=DARK)
    ax[0].axvline(0, color="#333333", lw=0.9)
    ax[0].set_yticks(ys); ax[0].set_yticklabels([it[0] for it in items], fontsize=8.5)
    ax[0].set_xlabel(f"达标时长变化 $\\Delta t^*$ / h（基线 {base:.4f} h）")
    ax[0].set_title("(a) 全部扰动情景"); ax[0].set_xlim(-4, 9)
    ax[0].grid(axis="x", ls=":", alpha=0.4)
    # (b) 小影响放大（|Δ|<0.4 h）：条形统一蓝色阶（上深下浅），零变化用零值标记
    small = [it for it in items if abs(it[1]) < 0.4]
    ys2 = np.arange(len(small))[::-1]
    blues = [CMAP_BLUES(v) for v in np.linspace(0.05, 0.85, len(small))]  # 深→浅
    band = ax[1].axvspan(-0.02, 0.02, color="#ECECEC", lw=0, label="所设分辨判据 $\\pm0.02$ h")
    for k, (y, (lab, dv, _col, disc)) in enumerate(zip(ys2, small)):
        note = "" if disc.strip() == "是" else "（未分辨）" if "未" in disc else "（临界）"
        if abs(dv) < 5e-5:                       # 零变化情景：零值标记，不画非零长度
            ax[1].plot(0, y, "|", color=DEEP, ms=12, mew=2)
            ax[1].text(0.006, y, f"{dv:+.4f}{note}", va="center", ha="left", fontsize=7.5, color=DARK)
        else:
            ax[1].barh(y, dv, color=blues[k], height=0.6)
            ax[1].text(dv + (0.004 if dv >= 0 else -0.004), y, f"{dv:+.4f}{note}",
                       va="center", ha="left" if dv >= 0 else "right", fontsize=7.5, color=DARK)
    ax[1].axvline(0, color=CHAR, lw=0.9)
    ax[1].set_yticks(ys2); ax[1].set_yticklabels([it[0] for it in small], fontsize=8)
    ax[1].set_xlabel("达标时长变化 $\\Delta t^*$ / h（放大）")
    ax[1].set_title("(b) 小影响情景放大"); ax[1].set_xlim(-0.4, 0.2)
    ax[1].legend(handles=[band], fontsize=7, loc="upper right", framealpha=0.9)
    ax[1].grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


# ==========================================================================
# 图：四问关系（matplotlib：文字边界测量 + 分层布置，替代原 TikZ）
# ==========================================================================
def fig_relation():
    """四问与模型关系图：三列浅色任务分区（模型节点 + 结果节点）+ 跨区关系箭头。

    几何完全由文字外框测量确定：框宽=最宽文字+左右内边距，框高=上内边距+标题高+
    标题正文间距+正文高+下内边距；标题与正文按测得高度用 va='top' 定位（不再按行数猜）。
    生成时断言：文字框被节点框包含、节点框互不相交、箭头不穿越非目标节点框。
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    W, H = 10.0, 8.0
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("auto")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def measure(s, fs, weight="normal", ls=1.5):
        t = ax.text(0, 0, s, ha="center", va="center", fontsize=fs,
                    fontweight=weight, linespacing=ls, alpha=0)
        fig.canvas.draw()
        bb = t.get_window_extent(rend); t.remove()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return abs(x1 - x0), abs(y1 - y0)

    FS_T, FS_B, FS_L, LS = 10.5, 10.0, 9.0, 1.45
    PADX, PADT, PADB = 0.20, 0.16, 0.16          # 节点框内边距
    PPADX, PPADT, PPADB = 0.24, 0.18, 0.24       # 分区面板内边距
    G_TM, G_MR = 0.22, 0.52                      # 标题-模型、模型-结果 间距
    GAPC = 0.80                                  # 列间净距

    # ---- 内容（图中节点用短语；换行控制列宽）----
    cols = [
        dict(key="q1", title="问题一 · 预热过程",
             model=["径向导热与水分扩散", "分别求解"],
             result=["预热温度与", "含水率分布"], warm=False),
        dict(key="q2", title="问题二 · 完整干燥过程",
             model=["变物性传热传质", "耦合模型"],
             result=["全过程温度场", "与含水率场"], warm=False),
        dict(key="q4", title="问题四 · 尺寸收缩",
             model=["更新物性与", "收缩参考坐标"],
             result=["收缩条件下含水率", "与达标时长"], warm=True),
    ]
    for c in cols:
        c["tw"], c["th"] = measure(c["title"], FS_T, "bold")
        c["mtxt"] = "\n".join(c["model"]); c["mw"], c["mh"] = measure(c["mtxt"], FS_B, ls=LS)
        c["rtxt"] = "\n".join(c["result"]); c["rw"], c["rh"] = measure(c["rtxt"], FS_B, ls=LS)
    # 统一列宽/框高（取各列最大，成规则网格）
    box_w = max(max(c["mw"], c["rw"]) for c in cols) + 2 * PADX
    inner_w = max(box_w, max(c["tw"] for c in cols))
    panel_w = inner_w + 2 * PPADX
    mbox_h = max(c["mh"] for c in cols) + PADT + PADB
    rbox_h = max(c["rh"] for c in cols) + PADT + PADB
    th_max = max(c["th"] for c in cols)
    panel_h = PPADT + th_max + G_TM + mbox_h + G_MR + rbox_h + PPADB

    total_w = 3 * panel_w + 2 * GAPC
    x0 = (W - total_w) / 2.0
    for i, c in enumerate(cols):
        c["cx"] = x0 + i * (panel_w + GAPC) + panel_w / 2.0
    panel_top = 6.55
    panel_bot = panel_top - panel_h

    boxes = []                                    # 节点框（供几何检查）：(name,x0,y0,x1,y1)
    panels = []                                   # 分区面板：(name,x0,y0,x1,y1)

    def draw_box(cx, top, w, h, text, fs, fc, ec, tcol, weight="normal", ls=1.0, name=""):
        ax.add_patch(FancyBboxPatch((cx - w / 2, top - h), w, h,
                                    boxstyle="round,pad=0.015", fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(cx, top - PADT, text, ha="center", va="top", fontsize=fs,
                fontweight=weight, color=tcol, linespacing=ls, zorder=4)
        boxes.append((name, cx - w / 2, top - h, cx + w / 2, top))

    # ---- 概览标题（共用框架，文字形式，不再用会溢出的框）----
    ax.text(W / 2, 7.62, "统一径向传热传质模型", ha="center", va="center",
            fontsize=FS_T + 1.5, fontweight="bold", color=DEEP, zorder=5)
    ax.text(W / 2, 7.18, "同一套控制方程与定解条件，按各问替换物性、几何与判据",
            ha="center", va="center", fontsize=FS_L, color="#5A6169", zorder=5)

    # ---- 三列分区 ----
    node_xy = {}
    for c in cols:
        cx = c["cx"]
        pfc = (AMBER + "2E") if c["warm"] else (ICE + "33")
        pec = GORANGE if c["warm"] else OCEAN
        ax.add_patch(FancyBboxPatch((cx - panel_w / 2, panel_bot), panel_w, panel_h,
                                    boxstyle="round,pad=0.02", fc=pfc, ec=pec, lw=1.1,
                                    zorder=1, alpha=0.95))
        panels.append((c["key"], cx - panel_w / 2, panel_bot, cx + panel_w / 2, panel_top))
        # 任务标题
        title_top = panel_top - PPADT
        ax.text(cx, title_top, c["title"], ha="center", va="top",
                fontsize=FS_T, fontweight="bold", color=CHAR, zorder=4)
        # 模型节点
        mtop = title_top - th_max - G_TM
        draw_box(cx, mtop, box_w, mbox_h, c["mtxt"], FS_B, "#FFFFFF", pec, CHAR,
                 ls=LS, name=c["key"] + "_m")
        # 结果节点
        rtop = mtop - mbox_h - G_MR
        draw_box(cx, rtop, box_w, rbox_h, c["rtxt"], FS_B, "#FFFFFF", pec, CHAR,
                 ls=LS, name=c["key"] + "_r")
        node_xy[c["key"]] = dict(cx=cx, m_top=mtop, m_bot=mtop - mbox_h,
                                 r_top=rtop, r_bot=rtop - rbox_h)

    # ---- 问题三（问题二结果节点下方）----
    cx2 = node_xy["q2"]["cx"]
    q3_title, q3_body = "问题三 · 达标判定", "达标时长"
    q3tw, q3th = measure(q3_title, FS_B, "bold")
    q3bw, q3bh = measure(q3_body, FS_B)
    q3w = max(q3tw, q3bw) + 2 * PADX
    q3h = PADT + q3th + 0.14 + q3bh + PADB
    q3_top = panel_bot - 0.62
    ax.add_patch(FancyBboxPatch((cx2 - q3w / 2, q3_top - q3h), q3w, q3h,
                                boxstyle="round,pad=0.015", fc=ICE + "55", ec=OCEAN, lw=1.3, zorder=3))
    ax.text(cx2, q3_top - PADT, q3_title, ha="center", va="top",
            fontsize=FS_B, fontweight="bold", color=DEEP, zorder=4)
    ax.text(cx2, q3_top - PADT - q3th - 0.14, q3_body, ha="center", va="top",
            fontsize=FS_B, fontweight="bold", color=CHAR, zorder=4)
    boxes.append(("q3", cx2 - q3w / 2, q3_top - q3h, cx2 + q3w / 2, q3_top))

    # ---- 箭头 ----
    ar = dict(arrowstyle="-|>", color=DEEP, lw=1.4, mutation_scale=12,
              shrinkA=1, shrinkB=1, zorder=2)
    ar_w = dict(ar, color=GORANGE)
    arrows = []                                   # (p0,p1,{connected boxes})
    # 列内：模型 → 结果（结果提取）
    for k in ("q1", "q2", "q4"):
        p0 = (node_xy[k]["cx"], node_xy[k]["m_bot"]); p1 = (node_xy[k]["cx"], node_xy[k]["r_top"])
        ax.add_patch(FancyArrowPatch(p0, p1, **ar))
        arrows.append((p0, p1, {k + "_m", k + "_r"}))
    # 问题二模型 → 问题四模型（物性与几何修正）
    m_mid = (node_xy["q2"]["m_top"] + node_xy["q2"]["m_bot"]) / 2.0
    p0 = (node_xy["q2"]["cx"] + box_w / 2, m_mid); p1 = (node_xy["q4"]["cx"] - box_w / 2, m_mid)
    ax.add_patch(FancyArrowPatch(p0, p1, **ar_w))
    arrows.append((p0, p1, {"q2_m", "q4_m"}))
    lbl_x = (p0[0] + p1[0]) / 2.0
    ax.text(lbl_x, node_xy["q2"]["m_top"] + 0.30, "物性与\n几何修正", ha="center", va="bottom",
            fontsize=FS_L, color="#8A4B12", linespacing=1.2, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.85))
    # 问题二结果 → 问题三（最大含水率判据）
    p0 = (cx2, node_xy["q2"]["r_bot"]); p1 = (cx2, q3_top)
    ax.add_patch(FancyArrowPatch(p0, p1, **ar))
    arrows.append((p0, p1, {"q2_r", "q3"}))
    ax.text(cx2 + 0.16, (p0[1] + p1[1]) / 2.0, "最大含水率\n判据", ha="left", va="center",
            fontsize=FS_L, color=CHAR, linespacing=1.2, zorder=5)

    # ---- 几何自检（制图必要步骤）----
    def overlap(a, b, tol=1e-6):
        return not (a[3] <= b[1] + tol or b[3] <= a[1] + tol or
                    a[4] <= b[2] + tol or b[4] <= a[2] + tol)

    def seg_hits(p0, p1, rect, tol=0.02):
        for s in np.linspace(0.06, 0.94, 30):
            x = p0[0] + s * (p1[0] - p0[0]); y = p0[1] + s * (p1[1] - p0[1])
            if rect[1] + tol < x < rect[3] - tol and rect[2] + tol < y < rect[4] - tol:
                return True
        return False

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not overlap(boxes[i], boxes[j]), f"节点框相交: {boxes[i][0]} vs {boxes[j][0]}"
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            assert not overlap(panels[i], panels[j]), f"分区面板相交: {panels[i][0]} vs {panels[j][0]}"
    for p0, p1, conn in arrows:
        for bx in boxes:
            if bx[0] in conn:
                continue
            assert not seg_hits(p0, p1, bx), f"箭头穿越非目标节点: {bx[0]}"
    # 内容不超出画布
    all_x0 = min(b[1] for b in boxes + panels); all_x1 = max(b[3] for b in boxes + panels)
    assert 0 <= all_x0 and all_x1 <= W, f"横向越界: [{all_x0:.2f},{all_x1:.2f}]"
    _save(fig, "fig_relation")


# ==========================================================================
# 图：辅助热问题的时间积分对照（向后 Euler vs 自适应 BDF；读诊断 CSV）
# ==========================================================================
def _load_be_bdf_diag():
    diag = list(csv.DictReader((EXPORTS / "be_bdf_diag.csv").open(encoding="utf-8")))
    hist = list(csv.DictReader((EXPORTS / "be_bdf_history.csv").open(encoding="utf-8")))
    return diag, hist


def fig_be_bdf():
    diag, hist = _load_be_bdf_diag()
    t = np.array([float(r["t"]) for r in hist])
    hcols = [c for c in hist[0].keys() if c != "t"]
    series = {c: np.array([float(r[c]) for r in hist]) for c in hcols}
    # 端点差异（相对紧容差 BDF 参考）
    be = [(float(r["control"].split("=")[1].rstrip("s")), float(r["diff_vs_tightBDF_K"]))
          for r in diag if r["method"] == "向后Euler"]
    be.sort(reverse=True)                       # Δt 从大到小
    dts = np.array([b[0] for b in be]); errs = np.array([b[1] for b in be])
    bdf_err = next(float(r["diff_vs_tightBDF_K"]) for r in diag if r["method"] == "自适应BDF")
    bdf_ctrl = next(r["control"] for r in diag if r["method"] == "自适应BDF")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # (a) 表面温度时程（各方法在此标度下基本重合，用一条参考曲线呈现升温）
    Tref_C = series["T_ref_tightBDF"] - 273.15
    ax[0].plot(t, Tref_C, "-", color=DEEP, lw=1.8, label="表面温度（各方法重合）")
    spread = max(float(np.max(np.abs(series[c] - series["T_ref_tightBDF"]))) for c in hcols
                 if c != "T_ref_tightBDF")
    ax[0].plot(t, series["T_be_dt1.0"] - 273.15, "--", color=CORAL, lw=1.0, alpha=0.9,
               label="向后 Euler $\\Delta t=1$ s")
    ax[0].set_xlabel("时间 $t$ / s", fontsize=10.5)
    ax[0].set_ylabel("表面温度 / °C", fontsize=10.5)
    ax[0].set_title("(a) 表面温度时程", fontsize=10.5)
    ax[0].grid(ls=":", alpha=0.4); ax[0].tick_params(labelsize=9.5)
    ax[0].legend(fontsize=8.5, loc="lower right")
    ax[0].text(0.04, 0.94, f"各方法曲线基本重合\n（全程最大偏差 {spread:.1e} K，位于初始升温段）",
               transform=ax[0].transAxes, fontsize=8.0, color=CHAR, va="top", linespacing=1.3)
    # (b) 向后 Euler 终点差异随步长（双对数）+ 一阶参考线；BDF 差异单独标示
    ax[1].loglog(dts, errs, "o-", color=CORAL, ms=6, lw=1.7, label="向后 Euler（100 s 终点）")
    ref = errs[0] * (dts / dts[0])              # 一阶参考：误差 ∝ Δt
    ax[1].loglog(dts, ref, "--", color="#9AA0A6", lw=1.1, label="一阶参考斜率")
    ax[1].axhline(bdf_err, color=OCEAN, lw=1.6, ls="-.",
                  label=f"自适应 BDF（{bdf_ctrl}）")
    ax[1].set_xlabel("时间步长 $\\Delta t$ / s", fontsize=10.5)
    ax[1].set_ylabel("100 s 表面温度差 $|\\Delta T|$ / K", fontsize=10)
    ax[1].set_title("(b) 相对紧容差 BDF 参考的差异", fontsize=10.5)
    ax[1].set_xticks(dts); ax[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax[1].grid(which="both", ls=":", alpha=0.4); ax[1].tick_params(labelsize=9.5)
    ax[1].legend(fontsize=8.2, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_be_bdf")


def main():
    cfg = cfgmod.load_config()
    print("生成论文插图 →", FIGDIR)
    fig_relation()
    fig_inputs(cfg)
    fig_q1_profiles()
    fig_q23_fields()
    fig_q4_fields(cfg)
    fig_analytic_convergence()
    fig_convergence()
    fig_be_bdf()
    fig_threecase()
    fig_sensitivity()
    print("完成。")


if __name__ == "__main__":
    main()
