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

统一调色板（五色协调，不机械按序拼接为渐变）：
    深蓝 #5271AE  浅蓝 #70ACDE  橙 #FFA660  金 #F5CC7D  红 #D85B59
白底、深灰文字；金黄/浅橙只用于填充与背景带，不承担白底上的细线、箭头或数字标签。
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

# ---- 调色板 ----
BLUE_D, BLUE_L, ORANGE, GOLD, RED = "#5271AE", "#70ACDE", "#FFA660", "#F5CC7D", "#D85B59"
DARK = "#222222"          # 数字/文字标签
GRAY_OUT = "#E4E4E4"      # 域外留白（淡灰）

# 固定径向位置的颜色与标记（跨子图一致：内→外 = 蓝→红）
POS_STYLE = {
    0.0: dict(color=BLUE_D, marker="o", label="中心 $r=0$"),
    0.5: dict(color=BLUE_L, marker="s", label="$r=0.5$ cm"),
    1.0: dict(color=ORANGE, marker="^", label="$r=1.0$ cm"),
    1.5: dict(color=GOLD,   marker="D", label="$r=1.5$ cm"),
    2.0: dict(color=RED,    marker="v", label="表面 $r=2.0$ cm"),
}
TIME_COLORS = [BLUE_D, BLUE_L, ORANGE, RED]   # 早→晚

# 连续场色带（湿=深蓝，热=红；单调、可黑白区分）
CMAP_MOIST = LinearSegmentedColormap.from_list("moist", ["#F2F7FC", BLUE_L, BLUE_D])
CMAP_TEMP = LinearSegmentedColormap.from_list("temp", ["#FBF0D8", GOLD, ORANGE, RED])

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
    # (c) 最大含水率曲线（输出为四位小数；严格达标时刻由未舍入求根确定）
    axc = fig.add_subplot(gs[1, :])
    axc.plot(hC, Cmax, "-", color=BLUE_D, lw=1.6, label="输出最大含水率（$60$ s / 四位小数）")
    axc.axhline(THRESH, color=RED, ls="--", lw=1.1, label="阈值 0.15")
    axc.axvline(tstar_h, color="#7a7a7a", ls=":", lw=1.2)
    axc.annotate(f"$t^*={tstar_h:.4f}$ h（未舍入求根）", xy=(tstar_h, THRESH),
                 xytext=(tstar_h - 24, 0.62), fontsize=9, color=DARK,
                 arrowprops=dict(arrowstyle="->", color="#7a7a7a"))
    axc.set_xlabel("时间 $t$ / h"); axc.set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    axc.set_xlim(0, hC.max()); axc.set_ylim(0, 2.65)
    axc.set_title("(c) 输出最大含水率随时间下降；阈值附近仅示采样分辨率")
    axc.legend(fontsize=8, loc="upper right")
    # 放大窗：离散采样点（不以四位小数平台伪造严格穿越）
    axins = axc.inset_axes([0.09, 0.30, 0.34, 0.56])
    sel = (hC >= tstar_h - 2.0) & (hC <= hC.max())
    axins.plot(hC[sel], Cmax[sel], "o", ms=3.0, color=BLUE_D, label="60 s 采样")
    axins.axhline(THRESH, color=RED, ls="--", lw=1.0)
    axins.axvline(tstar_h, color="#7a7a7a", ls=":", lw=1.1)
    axins.set_xlim(tstar_h - 1.6, hC.max()); axins.set_ylim(0.1490, 0.1560)
    axins.tick_params(labelsize=6.5)
    axins.set_title("阈值附近（采样 60 s、四位小数）", fontsize=7)
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
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a, key, ttl, yl, col in (
            (ax[0], "temp", "(a) 温度（附录2 常物性辅助问题）", "全域最大误差 / °C", RED),
            (ax[1], "moist", "(b) 含水率（$D\\equiv D(C_0)$ 辅助问题）",
             "全域最大误差 / (kg·kg$^{-1}$)", BLUE_D)):
        Ns = np.array([p[0] for p in d[key]], float)
        err = np.array([p[1] for p in d[key]], float)
        a.loglog(Ns, err, "o-", color=col, ms=6, lw=1.5, label="数值 vs 解析级数解")
        ref = err[0] * (Ns[0] / Ns) ** 2
        a.loglog(Ns, ref, "--", color="#888888", lw=1.1, label="二阶参考斜率")
        a.set_xlabel("网格区间数 $N$"); a.set_ylabel(yl); a.set_title(ttl)
        a.set_xticks(Ns); a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        a.grid(which="both", ls=":", alpha=0.4); a.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_analytic_convergence")


# ==========================================================================
# 图：网格/界面收敛（含收敛误差子图，使积分界面可辨识）
# ==========================================================================
def fig_convergence():
    d = load_convergence()
    styles = {"harmonic": dict(color=RED, marker="s", label="调和平均界面"),
              "integral": dict(color=BLUE_D, marker="o", label="积分界面（8 点 Gauss）")}
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for key, pts in d.items():
        Ns = [p[0] for p in pts]; ts = [p[1] for p in pts]
        st = styles.get(key, dict(color=GOLD, marker="^", label=key))
        ax[0].plot(Ns, ts, "-", **st, ms=7, lw=1.5)
        ref = ts[-1]
        ax[1].loglog(Ns[:-1], [abs(t - ref) for t in ts[:-1]], "-", **st, ms=7, lw=1.5)
    ax[0].set_xscale("log", base=2)
    for a in ax:
        a.set_xticks([200, 400, 800]); a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        a.grid(which="both", ls=":", alpha=0.4)
    ax[0].set_xlabel("网格区间数 $N$"); ax[0].set_ylabel("达标时长 $t^*$ / h")
    ax[0].set_title("(a) 达标时长随网格加密"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("网格区间数 $N$"); ax[1].set_ylabel("$|t^*_N-t^*_{800}|$ / h")
    ax[1].set_title("(b) 相对最细网格的收敛（对数）"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_convergence")


# ==========================================================================
# 图：三情形对照（物性 vs 几何；同一网格 N=400）
# ==========================================================================
def fig_threecase():
    tc = load_threecase(400)
    order = [("① 附录3 物性 · 固定 $R_0$", "附录3/R0", BLUE_D),
             ("② 附录4 物性 · 固定 $R_0$", "附录4/R0", ORANGE),
             ("③ 附录4 物性 · 收缩 $R(t)$", "附录4/R(t)", RED)]
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    ys = np.arange(len(order))[::-1]
    for y, (lab, key, col) in zip(ys, order):
        val = next(v for k, v in tc.items() if key in k)
        ax.plot([0, val], [y, y], "-", color=col, lw=1.2, alpha=0.45)
        ax.plot(val, y, "o", color=col, ms=11)
        ax.text(val + 2, y, f"{val:.2f} h", va="center", fontsize=9.5, color=DARK)
    ax.set_yticks(ys); ax.set_yticklabels([o[0] for o in order], fontsize=9)
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
    # (b) 小影响放大（|Δ|<0.4 h），显示 ±0.02 h 所设分辨判据带
    small = [it for it in items if abs(it[1]) < 0.4]
    ys2 = np.arange(len(small))[::-1]
    ax[1].axvspan(-0.02, 0.02, color=GOLD, alpha=0.22, lw=0, label="所设分辨判据 $\\pm0.02$ h")
    for y, (lab, dv, col, disc) in zip(ys2, small):
        ax[1].barh(y, dv, color=col, alpha=0.9, height=0.6)
        note = "" if disc.strip() == "是" else "（未分辨）" if "未" in disc else "（临界）"
        ax[1].text(dv + (0.004 if dv >= 0 else -0.004), y, f"{dv:+.4f}{note}",
                   va="center", ha="left" if dv >= 0 else "right", fontsize=7.5, color=DARK)
    ax[1].axvline(0, color="#333333", lw=0.9)
    ax[1].set_yticks(ys2); ax[1].set_yticklabels([it[0] for it in small], fontsize=8)
    ax[1].set_xlabel("达标时长变化 $\\Delta t^*$ / h（放大）")
    ax[1].set_title("(b) 小影响情景放大"); ax[1].set_xlim(-0.4, 0.2)
    ax[1].legend(fontsize=7.5, loc="lower right"); ax[1].grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


def main():
    cfg = cfgmod.load_config()
    print("生成论文插图 →", FIGDIR)
    fig_inputs(cfg)
    fig_q1_profiles()
    fig_q23_fields()
    fig_q4_fields(cfg)
    fig_analytic_convergence()
    fig_convergence()
    fig_threecase()
    fig_sensitivity()
    print("完成。")


if __name__ == "__main__":
    main()
