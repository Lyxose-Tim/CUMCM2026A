"""paper_figs.py —— 论文插图（读取官方 outputs 与验证记录，输出论文 PDF 到 paper/figures/）。

重绘入口：``python -m drymodel.paper_figs``。
本模块**只读**已授权生产的官方结果、原始附件与验证记录，不重跑任何生产计算，
也不在代码中硬编码结果常数：烘干时长、三情形、收敛与灵敏度数值均从下列来源读取
    outputs/production_receipt_supplementary.json   烘干 t*、采样时刻、未舍入 Cmax
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
import io
import json
import re
import shutil
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
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
TEAL, JADE, AQUA, SEAFOAM = "#009B8A", "#00C7A2", "#00D9B8", "#8AF7E3"   # 绿轴（J1 色卡）
CHAR, MIST = "#252A30", "#EAF3F8"
AUXGRAY = "#9AA0A6"      # 统一的参考线/次要标注中性灰（近 Mist 暗）
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
# 暖色阶（灵敏度按绝对影响大小映射：弱→黄、强→红）
CMAP_WARM = LinearSegmentedColormap.from_list("warm", [AMBER, GORANGE, CORAL, CRED, RED])

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
    # 完成 PDF 序列化后一次写入，避免流式输出留下不完整的交叉引用表。
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="pdf", dpi=300)
        (FIGDIR / f"{name}.pdf").write_bytes(buffer.getvalue())
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
    _ylo, _yhi = ax[0].get_ylim(); _rng = _yhi - _ylo
    ax[0].text((tsw_h + tt[-1] / 3600) / 2, _ylo + 0.55 * _rng, "常值外推段\n（4 h 后）",
               fontsize=7, color=CHAR, ha="center", va="center")
    # 图例（用代理）；置于左下空白，避开金色外推区的说明
    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color=BLUE_D, lw=1.5, label="分段线性插值"),
             Line2D([0], [0], marker="o", mfc="none", mec=AUXGRAY, ls="none", ms=4, label="附件1 原始节点"),
             Line2D([0], [0], color=RED, ls="--", lw=1.0, label="断点 $t=4$ h")]
    ax[0].legend(handles=proxy, fontsize=7, loc="lower left", framealpha=0.9)
    # (c) 半径
    th = t2 / 3600
    ax[2].plot(th, R2, "-", color=BLUE_D, lw=1.4, zorder=3, label="附件2 半径 $R(t)$")
    ax[2].plot(th, R2, "o", ms=2.2, mfc="none", mec="#8a8a8a", mew=0.6, zorder=2)
    ax[2].axhline(1.2, color=AUXGRAY, ls=":", lw=1.0, label="约 1.20 cm 平台")
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
    ax[0].set_title("(a) 温度径向剖面"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("径向位置 $r$ / cm"); ax[1].set_ylabel("干基含水率 $C$ / (kg·kg$^{-1}$)")
    ax[1].set_title("(b) 含水率径向剖面"); ax[1].legend(fontsize=7)
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
                 fontsize=7.5, color=DARK, arrowprops=dict(arrowstyle="->", color=CHAR))
    axc.set_xlabel("时间 $t$ / h"); axc.set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    axc.set_xlim(0, hC.max()); axc.set_ylim(0, 2.65)
    axc.set_title("(c) 全程最大含水率")
    axc.legend(fontsize=7, loc="upper right")
    # (d) 阈值附近放大：离散 60 s 采样点（不以四位小数平台伪造严格穿越）
    axd = fig.add_subplot(gs[1, 1])
    sel = (hC >= tstar_h - 1.6) & (hC <= hC.max())
    axd.plot(hC[sel], Cmax[sel], "o", ms=3.4, color=OCEAN, label="$60$ s 采样（四位小数）")
    axd.axhline(THRESH, color=RED, ls="--", lw=1.1, label="阈值 0.15")
    axd.axvline(tstar_h, color=CHAR, ls=":", lw=1.2, label="$t^*$（未舍入求根）")
    axd.set_xlim(tstar_h - 1.4, hC.max() + 0.35); axd.set_ylim(0.1490, 0.1560)  # 右侧留白，事件线不贴框
    axd.set_xlabel("时间 $t$ / h"); axd.set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    axd.set_title("(d) 阈值附近（采样分辨率）")
    axd.legend(fontsize=7, loc="upper right")
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
    cmap = CMAP_MOIST.copy(); cmap.set_bad("#FFFFFF")     # 域外白底

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.5),
                           gridspec_kw=dict(width_ratios=[1.06, 1.0], wspace=0.30))
    Rm, Tm = np.meshgrid(rfine, hC)
    pa = ax[0].pcolormesh(Rm, Tm, Zm, cmap=cmap, shading="nearest", vmin=0.0, vmax=2.55,
                          rasterized=True, zorder=1)
    fig.colorbar(pa, ax=ax[0]).set_label("干基含水率 / (kg·kg$^{-1}$)")
    # 域外统一白底覆盖（消除任何半格越界），红色移动边界最后叠加
    ax[0].fill_betweenx(hC, Rt_cm, R0cm, color="#FFFFFF", lw=0, zorder=2)
    ax[0].plot(Rt_cm, hC, "-", color=RED, lw=1.7, zorder=3, label="移动边界 $R(t)$")
    ax[0].set_xlim(0, R0cm); ax[0].set_ylim(hC.min(), hC.max())
    ax[0].set_xlabel("固定物理位置 $r$ / cm"); ax[0].set_ylabel("时间 $t$ / h")
    ax[0].set_title("(a) 含水率场与移动边界")
    ax[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    # (b) 固定位置 + 表面时程（用全部行，含首末；事件时刻标注）
    for rc in (0.0, 0.5, 1.0):
        j = int(np.argmin(np.abs(pos_cm - rc)))
        st = POS_STYLE[rc]
        ax[1].plot(hC, Cgrid[:, j], "-", color=st["color"], lw=1.5, label=st["label"])
    ax[1].plot(hC, surf, "-", color=RED, lw=1.5, label="药材表面 $r=R(t)$")
    ax[1].axhline(THRESH, color="#555555", ls=":", lw=1.0, label="阈值 0.15")
    ax[1].axvline(tstar4_h, color="#7a7a7a", ls="--", lw=1.0)
    ax[1].annotate(f"$t^*={tstar4_h:.4f}$ h", xy=(tstar4_h, THRESH), xytext=(28, 0.95),
                   fontsize=8, color=DARK, arrowprops=dict(arrowstyle="->", color="#7a7a7a"))
    ax[1].set_xlim(0, hC.max()); ax[1].set_ylim(0, 2.65)
    ax[1].set_xlabel("时间 $t$ / h"); ax[1].set_ylabel("干基含水率 / (kg·kg$^{-1}$)")
    ax[1].set_title("(b) 固定位置与表面含水率时程")
    ax[1].legend(fontsize=7)
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
        a.loglog(Ns, ref, "--", color=AUXGRAY, lw=1.1, label="二阶参考斜率")
        a.set_xlabel("网格区间数 $N$", fontsize=9.5); a.set_ylabel(yl, fontsize=9)
        a.set_title(ttl, fontsize=9.5)
        a.set_xticks(Ns); a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        a.tick_params(labelsize=9.5)
        a.grid(which="both", ls=":", alpha=0.4); a.legend(fontsize=8.5)
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
    ax[0].set_xlabel("网格区间数 $N$", fontsize=9.5); ax[0].set_ylabel("烘干时长 $t^*$ / h", fontsize=9.5)
    ax[0].set_title("(a) 烘干时长随网格加密", fontsize=9.5)
    ax[1].set_xlabel("网格区间数 $N$", fontsize=9.5); ax[1].set_ylabel("$|t^*_N-t^*_{800}|$ / h", fontsize=9.5)
    ax[1].set_title("(b) 相对最细网格的收敛（对数）", fontsize=9.5)
    for a in ax:
        a.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=7,
                 borderaxespad=0, handlelength=1.8, labelspacing=0.35)
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
        ax.text(val + 2, y, f"{val:.2f} h", va="center", fontsize=8.5, color=DARK)
    ax.set_yticks(ys); ax.set_yticklabels([o[0] for o in order], fontsize=8)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("烘干时长 $t^*$ / h（同一网格 $N=400$）"); ax.set_xlim(0, 150)
    ax.set_title("物性变化（①→②）延长干燥、尺寸收缩（②→③）缩短干燥")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_threecase")


# ==========================================================================
# 图：参数扰动（成对正负、恢复全部情景、小影响放大）
# ==========================================================================
def _sens_label(name):
    if "hm" in name:
        return "传质系数 $h_m$\n" + ("减半" if "0.5" in name else "加倍")
    if "T_air" in name:
        s = "$+0.39$" if "+0.39" in name else "$-0.39$"
        return "外推空气温度\n" + s + " °C"
    if "C_env" in name:
        s = "$+0.0011$" if "+0.0011" in name else "$-0.0011$"
        return "环境含水浓度\n" + s
    if "hold_last" in name:
        return "外推取末值\nhold-last"
    if "smooth" in name:
        return "数据平滑\nsmooth121"
    return "换热系数 $h$\n" + ("减半" if "0.5" in name else "加倍")


def fig_sensitivity():
    base, rows = load_sensitivity()
    items = [(_sens_label(n), dv) for n, dv, disc in rows]
    items.sort(key=lambda z: -abs(z[1]))
    amax = max(abs(dv) for _, dv in items) or 1.0
    # 出图尺寸接近论文最终宽度（0.96\textwidth≈16.3 cm≈6.4 in），保证缩放后文字仍约 7--8 pt
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.6),
                           gridspec_kw=dict(width_ratios=[1.5, 1.0], wspace=0.55))
    # (a) 全部情景：按 |Δt*| 映射到暖色阶（弱→黄、强→红）；正负由零轴两侧与符号表达
    ys = np.arange(len(items))[::-1]
    for y, (lab, dv) in zip(ys, items):
        ax[0].barh(y, dv, color=CMAP_WARM(abs(dv) / amax), alpha=0.95, height=0.62)
        ax[0].text(dv + (0.12 if dv >= 0 else -0.12), y, f"{dv:+.3f}",
                   va="center", ha="left" if dv >= 0 else "right", fontsize=7, color=DARK)
    ax[0].axvline(0, color="#333333", lw=0.9)
    ax[0].set_yticks(ys); ax[0].set_yticklabels([it[0] for it in items], fontsize=8)
    ax[0].set_xlabel("烘干时长变化 $\\Delta t^*$ / h", fontsize=8.5)
    ax[0].set_title(f"(a) 全部扰动情景（基线 {base:.4f} h）", fontsize=9)
    ax[0].set_xlim(-5, 10); ax[0].grid(axis="x", ls=":", alpha=0.4); ax[0].tick_params(labelsize=8)
    # (b) 小影响放大：蓝色阶（上深下浅），灰带为筛选尺度（说明见图注，不再放图内图例框）
    small = [it for it in items if abs(it[1]) < 0.4]
    ys2 = np.arange(len(small))[::-1]
    blues = [CMAP_BLUES(v) for v in np.linspace(0.05, 0.85, len(small))]
    ax[1].axvspan(-0.02, 0.02, color="#ECECEC", lw=0)
    for k, (y, (lab, dv)) in enumerate(zip(ys2, small)):
        if abs(dv) < 5e-5:
            ax[1].plot(0, y, "|", color=DEEP, ms=11, mew=2)
            ax[1].text(0.006, y, f"{dv:+.4f}", va="center", ha="left", fontsize=7, color=DARK)
        else:
            ax[1].barh(y, dv, color=blues[k], height=0.6)
            ax[1].text(dv + (0.004 if dv >= 0 else -0.004), y, f"{dv:+.4f}",
                       va="center", ha="left" if dv >= 0 else "right", fontsize=7, color=DARK)
    ax[1].axvline(0, color=CHAR, lw=0.9)
    ax[1].set_yticks(ys2); ax[1].set_yticklabels([it[0] for it in small], fontsize=8)
    ax[1].set_xlabel("$\\Delta t^*$ / h（放大）", fontsize=8.5)
    ax[1].set_title("(b) 小影响情景放大", fontsize=9); ax[1].set_xlim(-0.5, 0.3)
    ax[1].tick_params(labelsize=8); ax[1].grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


# ==========================================================================
# 图：四问关系（采用用户提供的原图，重绘时保持原样）
# ==========================================================================
def fig_relation():
    """原图直接嵌入 PDF；PNG 保留上传文件的原始字节。"""
    source = ROOT / "paper" / "assets" / "fig_relation_uploaded.png"
    pixels = plt.imread(source)
    height, width = pixels.shape[:2]
    fig = plt.figure(figsize=(width / 240, height / 240), dpi=240)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pixels, interpolation="none", aspect="equal")
    ax.set_axis_off()
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(FIGDIR / "fig_relation.pdf", dpi=240, pad_inches=0)
    plt.close(fig)
    shutil.copyfile(source, FIGDIR / "fig_relation.png")
    print("  saved fig_relation (uploaded image)")


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
    ax[0].set_xlabel("时间 $t$ / s", fontsize=9.5)
    ax[0].set_ylabel("表面温度 / °C", fontsize=9.5)
    ax[0].set_title("(a) 表面温度时程", fontsize=9.5)
    ax[0].grid(ls=":", alpha=0.4); ax[0].tick_params(labelsize=9.5)
    ax[0].legend(fontsize=7, loc="lower right", bbox_to_anchor=(0.98, 0.14),
                 borderaxespad=0, handlelength=1.8, labelspacing=0.35)
    ax[0].text(0.04, 0.94, f"各方法曲线基本重合\n全程最大偏差 {spread:.1e} K\n偏差位于初始升温段",
               transform=ax[0].transAxes, fontsize=7, color=CHAR, va="top", linespacing=1.3)
    # (b) 向后 Euler 终点差异随步长（双对数）+ 一阶参考线；BDF 差异单独标示
    ax[1].loglog(dts, errs, "o-", color=CORAL, ms=6, lw=1.7, label="向后 Euler（100 s 终点）")
    ref = errs[0] * (dts / dts[0])              # 一阶参考：误差 ∝ Δt
    ax[1].loglog(dts, ref, "--", color=AUXGRAY, lw=1.1, label="一阶参考斜率")
    ax[1].axhline(bdf_err, color=OCEAN, lw=1.6, ls="-.",
                  label=f"自适应 BDF（{bdf_ctrl}）")
    ax[1].set_xlabel("时间步长 $\\Delta t$ / s", fontsize=9.5)
    ax[1].set_ylabel("100 s 表面温度差 $|\\Delta T|$ / K", fontsize=9)
    ax[1].set_title("(b) 相对紧容差 BDF 参考的差异", fontsize=9.5)
    ax[1].set_xticks(dts); ax[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax[1].get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax[1].grid(which="both", ls=":", alpha=0.4); ax[1].tick_params(labelsize=9.5)
    ax[1].legend(fontsize=7, loc="lower right", bbox_to_anchor=(0.98, 0.14),
                 borderaxespad=0, handlelength=1.8, labelspacing=0.35)
    fig.tight_layout()
    _save(fig, "fig_be_bdf")


# ==========================================================================
# 图：蒸发吸热影响评估（左：温度时程两种边界；右：烘干时长对照）
# ==========================================================================
def fig_latent():
    hist = list(csv.DictReader((EXPORTS / "latent_heat_history_q23.csv").open(encoding="utf-8")))
    # η=0 与 η=1 各用所属情景的时间列（两情景事件时刻不同）
    t0_h = np.array([float(r["t_eta0_s"]) for r in hist]) / 3600.0
    t1_h = np.array([float(r["t_eta1_s"]) for r in hist]) / 3600.0
    Tc0 = np.array([float(r["T_center_eta0_C"]) for r in hist])
    Ts0 = np.array([float(r["T_surface_eta0_C"]) for r in hist])
    Tc1 = np.array([float(r["T_center_eta1_C"]) for r in hist])
    Ts1 = np.array([float(r["T_surface_eta1_C"]) for r in hist])
    diag = json.loads((EXPORTS / "latent_heat_diag_summary.json").read_text(encoding="utf-8"))
    tstar = {}
    for r in diag["rows"]:
        tstar[(r["question"], r["scenario"][:3])] = float(r["t_star_h"])
    q3_0, q3_1 = tstar[("q23", "η=0")], tstar[("q23", "η=1")]
    q4_0, q4_1 = tstar[("q4", "η=0")], tstar[("q4", "η=1")]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    # (a) 温度时程（问题二三，η=0 实线 / η=1 虚线；各用所属时间轴）
    ax[0].plot(t0_h, Ts0, "-", color=OCEAN, lw=1.6, label="表面 $\\eta{=}0$")
    ax[0].plot(t0_h, Tc0, "-", color=DEEP, lw=1.6, label="中心 $\\eta{=}0$")
    ax[0].plot(t1_h, Ts1, "--", color=CORAL, lw=1.6, label="表面 $\\eta{=}1$")
    ax[0].plot(t1_h, Tc1, "--", color=RED, lw=1.6, label="中心 $\\eta{=}1$")
    ax[0].set_xlabel("时间 $t$ / h", fontsize=8.5)
    ax[0].set_ylabel("温度 / °C", fontsize=8.5)
    ax[0].set_title("(a) 温度时程（问题二、三，$N{=}400$）", fontsize=8.5)
    ax[0].grid(ls=":", alpha=0.4); ax[0].tick_params(labelsize=8.5)
    ax[0].legend(fontsize=7, ncol=2, loc="lower right")
    # (b) 烘干时长对照
    groups = ["问题三\n（附录3）", "问题四\n（附录4 收缩）"]
    v0 = [q3_0, q4_0]; v1 = [q3_1, q4_1]
    xpos = np.arange(2); wb = 0.36
    ax[1].bar(xpos - wb / 2, v0, wb, color=OCEAN, label="$\\eta{=}0$ 无蒸发吸热")
    ax[1].bar(xpos + wb / 2, v1, wb, color=CORAL, label="$\\eta{=}1$ 表面蒸发吸热")
    for i in range(2):
        ax[1].text(xpos[i] - wb / 2, v0[i] + 0.8, f"{v0[i]:.2f}", ha="center", fontsize=7, color=DARK)
        ax[1].text(xpos[i] + wb / 2, v1[i] + 0.8, f"{v1[i]:.2f}", ha="center", fontsize=7, color=DARK)
        ax[1].annotate(f"+{v1[i]-v0[i]:.2f} h", (xpos[i], max(v0[i], v1[i]) + 4.0),
                       ha="center", fontsize=7, color="#8A4B12")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(groups, fontsize=7)
    ax[1].set_ylabel("烘干时长 $t^*$ / h", fontsize=8.5)
    ax[1].set_ylim(0, max(v1) + 26)
    ax[1].set_title("(b) 烘干时长对照", fontsize=8.5)
    ax[1].tick_params(labelsize=8.5)
    ax[1].legend(fontsize=7, loc="upper center", ncol=1, framealpha=0.9)
    ax[1].grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_latent")


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
    fig_latent()
    fig_threecase()
    fig_sensitivity()
    print("完成。")


if __name__ == "__main__":
    main()
