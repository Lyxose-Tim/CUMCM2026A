# -*- coding: utf-8 -*-
"""
figures.py —— 论文插图：读取 analysis.py 导出的 CSV 与附件数据，输出 paper/figures/*.pdf

用法：python figures.py [--data 附件目录] [--csv CSV 目录] [--out 图目录] [--preview PNG 预览目录]
配色：冷色 深海蓝 #005BBD / 海洋蓝 #008FF5 / 果冻青 #29C7F6 / 冰蓝 #86E6FF；
      暖色 琥珀 #FFC247 / 金橙 #FFA83A / 珊瑚橙 #FF7A45 / 正红 #F0404F；绿轴 #009B8A / #8AF7E3；中性 #252A30 / #EAF3F8。
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import analysis as an
import drying_model as dm

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.3, "legend.framealpha": 0.92, "legend.edgecolor": "#D0D0D0", "legend.handlelength": 1.8,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8, "text.color": "#222222", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333", "lines.linewidth": 1.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

DEEP, OCEAN, JELLY, ICE = "#005BBD", "#008FF5", "#29C7F6", "#86E6FF"
AMBER, GORANGE, CORAL, RED = "#FFC247", "#FFA83A", "#FF7A45", "#F0404F"
TEAL, JADE, SEAFOAM = "#009B8A", "#00C7A2", "#8AF7E3"
CHAR, MIST, GRAY = "#252A30", "#EAF3F8", "#8C939A"
W = 6.6                                       # 画布宽度 / in（≈ 正文宽度，插入后文字约 8 pt）
CMAP_MOIST = LinearSegmentedColormap.from_list("moist", ["#F4FAFE", ICE, OCEAN, DEEP])
CMAP_TEMP = LinearSegmentedColormap.from_list("temp", ["#FFF4D9", AMBER, CORAL, RED])
SHADE_PRE = dict(color=AMBER, alpha=0.16, lw=0)      # 预热平衡阶段底色
SHADE_CR = dict(color=ICE, alpha=0.28, lw=0)         # 恒速干燥段底色


# ============================== 读取与保存 ==============================
def read(path):
    """读 CSV 为 {列名: ndarray}；数值列转 float（空串为 nan），其余保留字符串。"""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    out = {}
    for j, h in enumerate(head):
        col = [r[j] if j < len(r) else "" for r in body]
        try:
            out[h] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[h] = np.array(col)
    return out


def kv(path):
    d = read(path)
    return {k: float(v) for k, v in zip(d["quantity"], d["value"])}


class Saver:
    def __init__(self, out, preview):
        self.out, self.preview = Path(out), Path(preview) if preview else None
        self.out.mkdir(parents=True, exist_ok=True)
        if self.preview:
            self.preview.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig, name):
        with io.BytesIO() as buf:                 # 序列化完成后一次写入
            fig.savefig(buf, format="pdf")
            (self.out / f"{name}.pdf").write_bytes(buf.getvalue())
        if self.preview:
            fig.savefig(self.preview / f"{name}.png", dpi=200)
        plt.close(fig)
        print("  saved", name, flush=True)


def grid(ax):
    ax.grid(ls=":", lw=0.6, color="#B8BEC4", alpha=0.7)


def panel(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, pad=4)


# ============================== 图 1 研究框架 ==============================
def box(ax, x, y, w, h, *, fc, ec, lw=1.0, r=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p, q, color=GRAY, lw=1.1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def fig_framework(save):
    fig = plt.figure(figsize=(W, 4.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 63); ax.axis("off")
    # 输入
    ins = [("附件 1", "烘房温度与含水浓度（0–4 h）"), ("附录 2–4", "密度、比热、导热与扩散系数"),
           ("附件 2", "药材半径 R(t)（0–72 h）")]
    for k, (a, b) in enumerate(ins):
        x = 2 + 33 * k
        box(ax, x, 54, 30, 7.5, fc=MIST, ec=GRAY, lw=0.8)
        ax.text(x + 15, 59.0, a, ha="center", va="center", fontsize=8.3, weight="bold", color=CHAR)
        ax.text(x + 15, 56.0, b, ha="center", va="center", fontsize=7.4, color=CHAR)
        arrow(ax, (x + 15, 54), (x + 15, 50.6))
    # 统一模型
    box(ax, 2, 40.5, 96, 10, fc="#E6F1FB", ec=DEEP, lw=1.2)
    ax.text(50, 47.4, "一维径向传热传质模型", ha="center", va="center", fontsize=10, weight="bold", color=DEEP)
    ax.text(50, 43.4, "含水率扩散方程 + 导热方程（散度形式）· 表面对流换热与对流传质 · 中心对称"
            " · 有限体积法 + 自适应 BDF", ha="center", va="center", fontsize=7.4, color=CHAR)
    # 四问
    qs = [("问题一 · 预热平衡阶段", "附录 2 常物性\n0–1800 s 温度与含水率", ICE, "#F2FBFF", OCEAN),
          ("问题二 · 烘干全过程", "附录 3 变物性\n两阶段识别与干燥规律", SEAFOAM, "#F0FFFB", TEAL),
          ("问题三 · 烘干时长", "全域最大含水率降到 0.15\n阈值穿越时刻 t*", "#BDF2E6", "#F3FFFC", TEAL),
          ("问题四 · 尺寸收缩", "附录 4 物性 + 附件 2 半径\n参考坐标 x = r/R(t)", AMBER, "#FFF8E8", GORANGE)]
    xs = [2, 26.7, 51.4, 76.1]
    for (title, body, hc, bc, ec), x in zip(qs, xs):
        box(ax, x, 21, 21.9, 15.5, fc=bc, ec=ec, lw=1.1)
        ax.add_patch(Rectangle((x + 0.25, 31.0), 21.4, 5.2, fc=hc, ec="none", alpha=0.75, zorder=2.5))
        ax.text(x + 10.95, 33.6, title, ha="center", va="center", fontsize=8.2, weight="bold", color=CHAR, zorder=3)
        ax.text(x + 10.95, 25.8, body, ha="center", va="center", fontsize=7.3, color=CHAR, linespacing=1.5, zorder=3)
        arrow(ax, (x + 10.95, 40.5), (x + 10.95, 36.6), color=DEEP)
    arrow(ax, (23.9, 28.0), (26.7, 28.0), color=OCEAN)
    arrow(ax, (48.6, 28.0), (51.4, 28.0), color=TEAL)
    # 分析与检验
    outs = [("干燥规律", "两阶段、干燥速率曲线\n双指数干燥方程", TEAL),
            ("工艺参数", "温度、湿度、尺寸、风速\n对烘干时长的影响", GORANGE),
            ("蒸发吸热", "能量一致的恒速—降速\n干燥模型", CORAL),
            ("模型检验", "解析解、网格收敛\n水分收支", DEEP)]
    ax.plot([12.95, 87.05], [17.6, 17.6], color=GRAY, lw=1.0, zorder=1)
    for x in xs:
        ax.plot([x + 10.95, x + 10.95], [21, 17.6], color=GRAY, lw=1.0, zorder=1)
    for (title, body, ec), x in zip(outs, xs):
        arrow(ax, (x + 10.95, 17.6), (x + 10.95, 14.3))
        box(ax, x, 2, 21.9, 12.3, fc="white", ec=ec, lw=1.1)
        ax.text(x + 10.95, 11.0, title, ha="center", va="center", fontsize=8.2, weight="bold", color=ec)
        ax.text(x + 10.95, 6.0, body, ha="center", va="center", fontsize=7.2, color=CHAR, linespacing=1.5)
    save(fig, "fig_framework")


# ============================== 图 2 几何与参考坐标 ==============================
def fig_geometry(save, radius):
    fig = plt.figure(figsize=(W, 2.7))
    a = fig.add_axes([0.0, 0.0, 0.40, 0.86])
    b = fig.add_axes([0.42, 0.0, 0.58, 0.86])
    fig.text(0.01, 0.95, "(a) 横截面与定解条件", fontsize=8.5)
    fig.text(0.43, 0.95, "(b) 问题四：收缩的物理域映射到固定参考域", fontsize=8.5)
    # (a) 横截面：内湿外干的同心环示意含水率分布
    a.set_xlim(-2.4, 2.4); a.set_ylim(-2.15, 1.95); a.set_aspect("equal", adjustable="datalim"); a.axis("off")
    yc = 0.2
    for k, rr in enumerate(np.linspace(1.0, 0.2, 5)):
        a.add_patch(Circle((0, yc), rr, fc=CMAP_MOIST(0.22 + 0.16 * k), ec="white", lw=0.8, zorder=1))
    a.add_patch(Circle((0, yc), 1.0, fc="none", ec=DEEP, lw=1.4, zorder=2))
    a.plot(0, yc, "o", ms=3, color=CHAR, zorder=3)
    a.annotate("", xy=(np.cos(-0.45), yc + np.sin(-0.45)), xytext=(0, yc),
               arrowprops=dict(arrowstyle="-|>", color=CHAR, lw=1.0), zorder=3)
    a.text(0.5, yc - 0.12, "$r$", fontsize=8.5, color=CHAR, zorder=3)
    for ang in (2.35, 3.3):                         # 热量流入
        p = np.array([np.cos(ang), np.sin(ang)])
        a.annotate("", xy=(1.03 * p[0], yc + 1.03 * p[1]), xytext=(1.6 * p[0], yc + 1.6 * p[1]),
                   arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    for ang in (0.8, -0.1):                         # 水分流出
        p = np.array([np.cos(ang), np.sin(ang)])
        a.annotate("", xy=(1.6 * p[0], yc + 1.6 * p[1]), xytext=(1.03 * p[0], yc + 1.03 * p[1]),
                   arrowprops=dict(arrowstyle="-|>", color=OCEAN, lw=1.3))
    a.text(-2.35, 1.9, "热量流入\n$h\\,(T_{air}-T_s)$", fontsize=7, color=RED, va="top")
    a.text(2.35, 1.9, "水分流出\n$h_m(C_s-C_{env})$", fontsize=7, color=OCEAN, va="top", ha="right")
    a.text(0, -1.2, "中心对称：$\\partial T/\\partial r=\\partial C/\\partial r=0$", ha="center", fontsize=7)
    a.text(0, -1.75, "长 $L=25$ cm $\\gg R_0=2$ cm：一维径向", ha="center", fontsize=7)
    # (b) 物理域 [0, R(t)] → 参考域 [0, 1]
    b.axis("off"); b.set_xlim(-1.5, 9.3); b.set_ylim(-1.0, 3.25)
    for k, (t, lab) in enumerate(((0.0, "$t=0$"), (6 * 3600.0, "$t=6$ h"), (51.09 * 3600.0, "$t=51.1$ h"))):
        y = 2.45 - 1.0 * k
        R = float(radius.R(t)) * 100
        xs = np.linspace(0, 2 * R, 61)
        for i in range(60):
            b.add_patch(Rectangle((xs[i], y), xs[i + 1] - xs[i], 0.4,
                                  fc=CMAP_MOIST(0.8 - 0.25 * k - 0.25 * i / 60), ec="none", zorder=1))
        b.add_patch(Rectangle((0, y), 2 * R, 0.4, fc="none", ec=DEEP, lw=0.9, zorder=2))
        b.plot([2 * R, 2 * R], [y - 0.06, y + 0.46], color=RED, lw=1.4, zorder=3)
        b.text(-0.2, y + 0.2, lab, ha="right", va="center", fontsize=7.2)
        b.text(2 * R - 0.05, y + 0.47, f"$R={R:.2f}$ cm", ha="right", va="bottom", fontsize=6.8, color=RED)
        b.annotate("", xy=(5.95, 1.65), xytext=(2 * R + 0.08, y + 0.2),
                   arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8, ls="--"))
    b.plot([0, 4], [-0.05, -0.05], color=CHAR, lw=0.7)
    for v in (0, 0.5, 1.0, 1.5, 2.0):
        b.plot([2 * v, 2 * v], [-0.15, -0.05], color=CHAR, lw=0.7)
        b.text(2 * v, -0.42, f"{v:g}", ha="center", fontsize=6.8)
    b.text(2, -0.88, "物理坐标 $r$ / cm", ha="center", fontsize=7.2)
    b.add_patch(Rectangle((6.0, 1.45), 3.0, 0.4, fc=SEAFOAM, ec=TEAL, lw=1.0))
    for v in (0, 0.5, 1.0):
        b.plot([6 + 3 * v, 6 + 3 * v], [1.33, 1.43], color=CHAR, lw=0.7)
        b.text(6 + 3 * v, 1.05, f"{v:g}", ha="center", fontsize=6.8)
    b.text(7.5, 0.62, "参考坐标 $x=r/R(t)$", ha="center", fontsize=7.2)
    b.text(7.5, 2.1, "固定计算域", ha="center", fontsize=7.2, color=TEAL)
    save(fig, "fig_geometry")


# ============================== 图 3 烘房湿空气状态 ==============================
def fig_environment(save, C):
    e = read(C / "env_psychro.csv")
    num = np.array([s != "plateau" for s in np.asarray(e["t_s"]).astype(str)])
    t = np.asarray(e["t_s"][num], dtype=float) / 3600.0
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw=dict(wspace=0.42))
    Tp, Wp, RHp, Twp = (float(e[k][~num][0]) for k in ("T_air_C", "W", "RH", "T_wb_C"))
    for ax in (a, b):
        ax.axvspan(4, 8, color=MIST, lw=0)
        ax.axvline(4, color=GRAY, lw=0.8, ls="--")
        ax.set_xlim(0, 8); ax.set_xlabel("时间 $t$ / h"); grid(ax)
    a.plot(t, e["T_air_C"][num], color=CHAR, label="烘房温度 $T_{air}$")
    a.plot([4, 8], [Tp, Tp], color=CHAR, ls="--", lw=1.2)
    a.plot(t, e["T_wb_C"][num], color=TEAL, label="湿球温度 $T_{wb}$")
    a.plot([4, 8], [Twp, Twp], color=TEAL, ls="--", lw=1.2)
    a.text(6, 33, "4 h 后取\n末小时均值", ha="center", fontsize=7, color=CHAR)
    a.set_ylabel("温度 / °C"); a.set_ylim(22, 54); a.legend(loc="lower right")
    panel(a, "(a) 烘房温度与湿球温度")
    b.plot(t, e["W"][num], color=OCEAN, label="含水浓度（含湿量）")
    b.plot([4, 8], [Wp, Wp], color=OCEAN, ls="--", lw=1.2)
    b.set_ylabel("含水浓度 / (kg·kg$^{-1}$)", color=OCEAN); b.tick_params(axis="y", colors=OCEAN)
    b2 = b.twinx()
    b2.plot(t, 100 * e["RH"][num], color=CORAL, label="相对湿度")
    b2.plot([4, 8], [100 * RHp, 100 * RHp], color=CORAL, ls="--", lw=1.2)
    b2.set_ylabel("相对湿度 / %", color=CORAL); b2.tick_params(axis="y", colors=CORAL); b2.set_ylim(40, 90)
    b.set_ylim(0.0, 0.06)
    b.legend(handles=[Line2D([], [], color=OCEAN, label="含水浓度"), Line2D([], [], color=CORAL, label="相对湿度")],
             loc="lower right")
    panel(b, "(b) 含水浓度与相对湿度")
    save(fig, "fig_environment")


# ============================== 图 4 问题一剖面 ==============================
def fig_q1(save, C):
    p = read(C / "q1_profiles.csv")
    times = [100, 300, 600, 900, 1200, 1500, 1800]
    cols = [DEEP, OCEAN, JELLY, TEAL, GORANGE, CORAL, RED]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.3))
    for tt, c in zip(times, cols):
        m = p["t_s"] == tt
        a.plot(p["r_cm"][m], p["T_C"][m], color=c, lw=1.4, label=f"{tt} s")
        b.plot(p["r_cm"][m], p["C"][m], color=c, lw=1.4)
    for ax in (a, b):
        ax.set_xlabel("到中心的距离 $r$ / cm"); ax.set_xlim(0, 2); grid(ax)
    a.set_ylabel("温度 / °C"); b.set_ylabel("含水率 / (kg·kg$^{-1}$)")
    b.legend(*a.get_legend_handles_labels(), ncol=2, loc="lower left", columnspacing=0.8, fontsize=6.9,
             title="时刻", title_fontsize=7)
    panel(a, "(a) 温度剖面"); panel(b, "(b) 含水率剖面")
    save(fig, "fig_q1")


# ============================== 图 5 问题二两阶段 ==============================
def fig_q2_stages(save, C):
    s, st = read(C / "q23_series.csv"), read(C / "stages.csv")
    tpe = float(st["t_pe_h"][0])
    h = s["t"] / 3600.0
    tstar = h[-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(wspace=0.3))
    a.axvspan(0, tpe, **SHADE_PRE)
    m = h <= 6.0
    a.plot(h[m], s["Tair"][m], color=CHAR, lw=1.2, ls="--", label="烘房 $T_{air}$")
    a.plot(h[m], s["Ts"][m], color=RED, label="表面")
    a.plot(h[m], s["Tc"][m], color=DEEP, label="中心")
    a.axvline(tpe, color=GRAY, lw=0.8)
    a.text(tpe + 0.12, 29.0, f"$t_{{pe}}={tpe:.2f}$ h", fontsize=7.2, color=CHAR)
    a.text(tpe / 2, 52.3, "预热平衡", ha="center", fontsize=7.2, color="#8A5A00")
    a.text((tpe + 6) / 2, 52.3, "恒温干燥", ha="center", fontsize=7.2, color=CHAR)
    a.set_xlim(0, 6); a.set_ylim(27, 54.5); a.set_xlabel("时间 $t$ / h"); a.set_ylabel("温度 / °C")
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.42)); grid(a); panel(a, "(a) 温度（前 6 h）")
    b.axvspan(0, tpe, **SHADE_PRE)
    b.plot(h, s["Cc"], color=DEEP, label="中心")
    b.plot(h, s["Cbar"], color=TEAL, label="平均 $\\bar C$")
    b.plot(h, s["Cs"], color=RED, label="表面")
    b.axhline(0.15, color=GRAY, lw=0.9, ls=":")
    b.text(1, 0.2, "0.15", fontsize=7, color=GRAY)
    b.axvline(tstar, color=GRAY, lw=0.8)
    b.text(tstar - 1, 1.1, f"$t^*={tstar:.2f}$ h", ha="right", fontsize=7.2)
    b.set_xlim(0, 60); b.set_ylim(0, 2.65); b.set_xlabel("时间 $t$ / h"); b.set_ylabel("含水率 / (kg·kg$^{-1}$)")
    b.legend(loc="upper right"); grid(b); panel(b, "(b) 含水率（全过程）")
    save(fig, "fig_q2_stages")


# ============================== 图 6 问题二三场分布 ==============================
def fig_q23_fields(save, C):
    fT, fC = read(C / "q23_field_T.csv"), read(C / "q23_field_C.csv")
    rr = np.array([float(k) for k in fT if k != "t_s"])
    keep = fT["t_s"] <= 3 * 3600
    fT = {k: v[keep] for k, v in fT.items()}
    T = np.column_stack([fT[k] for k in fT if k != "t_s"])
    Cf = np.column_stack([fC[k] for k in fC if k != "t_s"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw=dict(wspace=0.38))
    pa = a.pcolormesh(rr, fT["t_s"] / 3600, T, cmap=CMAP_TEMP, shading="gouraud", rasterized=True)
    cs = a.contour(rr, fT["t_s"] / 3600, T, levels=[30, 35, 40, 45, 49.5], colors="white", linewidths=0.6)
    a.clabel(cs, fmt="%g", fontsize=6.2)
    fig.colorbar(pa, ax=a, pad=0.03).set_label("温度 / °C")
    a.set_xlabel("到中心的距离 $r$ / cm"); a.set_ylabel("时间 $t$ / h"); panel(a, "(a) 温度场（前 3 h）")
    hb = fC["t_s"] / 3600
    pb = b.pcolormesh(rr, hb, Cf, cmap=CMAP_MOIST, shading="gouraud", norm=PowerNorm(0.45, 0, 2.55),
                      rasterized=True)
    cs = b.contour(rr, hb, Cf, levels=[0.25, 0.5, 1.0, 2.0], colors="white", linewidths=0.6)
    b.clabel(cs, fmt="%g", fontsize=6.2)
    c15 = b.contour(rr, hb, Cf, levels=[0.15], colors=[RED], linewidths=1.3)
    b.clabel(c15, fmt={0.15: "0.15"}, fontsize=6.5)
    fig.colorbar(pb, ax=b, pad=0.03).set_label("含水率 / (kg·kg$^{-1}$)")
    b.set_xlabel("到中心的距离 $r$ / cm"); b.set_ylabel("时间 $t$ / h"); panel(b, "(b) 含水率场（全过程）")
    save(fig, "fig_q23_fields")


# ============================== 图 7 干燥曲线与速率曲线 ==============================
def two_term(t, a, k0, b, k1):
    return a * np.exp(-k0 * t) + b * np.exp(-k1 * t)


def fig_drying(save, C):
    fits = read(C / "drying_fits.csv")
    st = read(C / "stages.csv")
    Ce = float(read(C / "q23_series.csv")["Cenv"][-1])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw=dict(wspace=0.3))
    for q, col, lab, k in (("q23", DEEP, "问题二、三", 0), ("q4", CORAL, "问题四", 1)):
        s = read(C / f"{q}_series.csv")
        h = s["t"] / 3600
        MR = (s["Cbar"] - Ce) / (dm.C0 - Ce)
        qq = "q3" if q == "q23" else "q4"
        m = (fits["question"] == qq) & (fits["model"] == "Two-term")
        p = [float(fits[f"p{i}"][m][0]) for i in (1, 2, 3, 4)]
        a.semilogy(h, MR, color=col, label=f"{lab} 数值解")
        a.semilogy(h[::30], two_term(h[::30], *p), "o", ms=2.6, mfc="white", mec=col, mew=0.8,
                   label=f"双指数：$\\tau_1$={1 / p[1]:.1f} h，$\\tau_2$={1 / p[3]:.0f} h")
        b.semilogy(s["Cbar"], s["rate"] * 3600, color=col, label=lab)
        tpe = float(st["t_pe_h"][k])
        i = int(np.argmin(np.abs(h - tpe)))
        b.plot(s["Cbar"][i], s["rate"][i] * 3600, "o", ms=4.5, color=col, mec="white", zorder=4)
    a.set_xlim(0, 60); a.set_ylim(0.02, 1.05); a.set_xlabel("时间 $t$ / h")
    a.set_ylabel("水分比 MR"); a.legend(loc="upper right", fontsize=6.8); grid(a)
    panel(a, "(a) 干燥曲线（对数纵轴）")
    hd, lb = b.get_legend_handles_labels()
    hd.append(Line2D([], [], marker="o", ls="none", ms=4.5, color=GRAY, mec="white"))
    lb.append("预热平衡结束")
    b.set_xlim(2.6, 0); b.set_xlabel("平均含水率 $\\bar C$ / (kg·kg$^{-1}$)（干燥方向 →）")
    b.set_ylabel("干燥速率 $-\\mathrm{d}\\bar C/\\mathrm{d}t$ / h$^{-1}$"); b.legend(hd, lb, loc="lower left"); grid(b)
    panel(b, "(b) 干燥速率曲线")
    save(fig, "fig_drying")


# ============================== 图 8 问题四 ==============================
def fig_q4(save, C, radius):
    f, s, tc = read(C / "q4_field_C.csv"), read(C / "q4_series.csv"), read(C / "threecase.csv")
    rr = np.array([float(k) for k in f if k not in ("t_s", "R_cm")])
    Z = np.column_stack([f[k] for k in f if k not in ("t_s", "R_cm")])
    h = f["t_s"] / 3600
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.9), gridspec_kw=dict(wspace=0.36, hspace=0.5))
    a, b, c, d = ax.ravel()
    # (a) 物理坐标下的含水率场：在每一时刻把表面值放在 R(t) 处，细网格插值后裁去域外
    rf = np.linspace(0, 2.0, 201)
    Zf = np.full((len(h), rf.size), np.nan)
    for i in range(len(h)):
        R = f["R_cm"][i]
        ok = ~np.isnan(Z[i])
        xp, fp = list(rr[ok]), list(Z[i][ok])
        if xp[-1] < R - 1e-9:
            s_i = int(np.argmin(np.abs(s["t"] - f["t_s"][i])))
            xp.append(R); fp.append(float(s["Cs"][s_i]))
        sel = rf <= R + 1e-9
        Zf[i, sel] = np.interp(rf[sel], xp, fp)
    cmap = CMAP_MOIST.copy(); cmap.set_bad("white")
    pa = a.pcolormesh(rf, h, np.ma.masked_invalid(Zf), cmap=cmap, shading="nearest",
                      norm=PowerNorm(0.45, 0, 2.55), rasterized=True)
    a.plot(f["R_cm"], h, color=RED, lw=1.5, label="表面 $R(t)$")
    a.contour(rf, h, np.ma.masked_invalid(Zf), levels=[0.15], colors=[CORAL], linewidths=1.0, linestyles="--")
    fig.colorbar(pa, ax=a, pad=0.03).set_label("含水率 / (kg·kg$^{-1}$)")
    a.set_xlim(0, 2); a.set_xlabel("到中心的距离 $r$ / cm"); a.set_ylabel("时间 $t$ / h")
    a.legend(loc="upper right"); panel(a, "(a) 含水率场与移动边界")
    # (b) 固定位置与表面时程
    for key, col, lab in (("0.00", DEEP, "中心"), ("0.50", OCEAN, "$r=0.5$ cm"), ("1.00", JELLY, "$r=1.0$ cm")):
        b.plot(h, f[key], color=col, label=lab)
    b.plot(s["t"] / 3600, s["Cs"], color=RED, label="表面")
    b.axhline(0.15, color=GRAY, lw=0.9, ls=":")
    t4 = s["t"][-1] / 3600
    b.axvline(t4, color=GRAY, lw=0.8)
    b.text(t4 - 1, 0.45, f"$t^*={t4:.2f}$ h", ha="right", fontsize=7.2)
    b.set_xlim(0, 54); b.set_ylim(0, 2.65); b.set_xlabel("时间 $t$ / h"); b.set_ylabel("含水率 / (kg·kg$^{-1}$)")
    b.legend(loc="upper right", fontsize=6.9); grid(b); panel(b, "(b) 固定位置与表面的含水率")
    # (c) 半径与平均含水率
    tn = radius.t_nodes / 3600
    c.plot(tn, radius.R_nodes_m * 100, "o", ms=2.2, mfc="white", mec=DEEP, mew=0.7, label="附件 2 半径")
    c.set_xlim(0, 60); c.set_ylim(1.0, 2.1); c.set_xlabel("时间 $t$ / h")
    c.set_ylabel("半径 $R$ / cm", color=DEEP); c.tick_params(axis="y", colors=DEEP); grid(c)
    c2 = c.twinx()
    c2.plot(s["t"] / 3600, s["Cbar"], color=TEAL, label="平均含水率")
    c2.set_ylabel("平均含水率 / (kg·kg$^{-1}$)", color=TEAL); c2.tick_params(axis="y", colors=TEAL)
    c2.set_ylim(0, 2.7)
    c.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=DEEP, ms=3.5, label="附件 2 半径"),
                      Line2D([], [], color=TEAL, label="平均含水率")], loc="upper right")
    panel(c, "(c) 收缩与失水同步进行")
    # (d) 三情形对照
    names = ["附录 3 物性\n固定半径", "附录 4 物性\n固定半径", "附录 4 物性\n随附件 2 收缩"]
    vals = [float(v) for v in tc["t_star_h"]]
    cols = [DEEP, GORANGE, RED]
    y = np.arange(3)[::-1] * 1.25
    d.barh(y, vals, color=cols, height=0.5, alpha=0.9)
    for yi, v, nm in zip(y, vals, names):
        d.text(v + 2, yi, f"{v:.2f} h", va="center", fontsize=7.4)
        d.text(0.5, yi + 0.3, nm.replace("\n", "，"), va="bottom", fontsize=7.2, color=CHAR)
    d.set_yticks([]); d.set_ylim(-0.45, 3.2); d.set_xlim(0, 150)
    d.set_xlabel("烘干时长 / h（$N=400$）"); d.grid(axis="x", ls=":", lw=0.6, color="#B8BEC4")
    panel(d, "(d) 物性与收缩的分别影响")
    save(fig, "fig_q4")


# ============================== 图 9 工艺参数 ==============================
def fig_process(save, C):
    T, Cs, R, H = (read(C / f"process_{k}.csv") for k in ("Tset", "Cenv", "R0", "hm"))
    fits = kv(C / "process_fits.csv")
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.75), gridspec_kw=dict(wspace=0.32, hspace=0.5))
    a, b, c, d = ax.ravel()
    sty_b = dict(color=DEEP, marker="o", ms=3.8, label="题设模型")
    sty_e = dict(color=CORAL, marker="s", ms=3.6, ls="--", label="能量一致模型")
    a.plot(T["T_set_C"], T["t_star_h"], **sty_b); a.plot(T["T_set_C"], T["t_star_evap_h"], **sty_e)
    a.axvspan(60, 72, color=MIST, lw=0)
    a.text(66, 95, "超出药典\n低温干燥\n上限 60 °C", ha="center", fontsize=6.8, color=CHAR)
    a.set_xlim(43, 72); a.set_ylim(0, 120)
    a.set_xlabel("恒温段温度 $T_{set}$ / °C"); a.set_ylabel("烘干时长 / h"); a.legend(loc="lower left"); grid(a)
    panel(a, "(a) 烘房温度")
    b.plot(Cs["C_set"], Cs["t_star_h"], **sty_b); b.plot(Cs["C_set"], Cs["t_star_evap_h"], **sty_e)
    b.set_xlabel("恒温段含水浓度 / (kg·kg$^{-1}$)"); b.set_ylabel("烘干时长 / h"); b.set_ylim(50, 100)
    b.legend(loc="upper left"); grid(b); panel(b, "(b) 烘房湿度")
    c.loglog(R["R0_cm"], R["t_star_h"], **sty_b); c.loglog(R["R0_cm"], R["t_star_evap_h"], **sty_e)
    rr = np.array([1.0, 2.5])
    c.loglog(rr, R["t_star_h"][0] * (rr / rr[0]) ** 2, color=GRAY, lw=0.9, ls=":", label="斜率 2 参考")
    c.text(1.05, 60, f"拟合指数：{fits['radius_exponent']:.2f}（题设）\n"
           f"　　　　　{fits['evap_radius_exponent']:.2f}（能量一致）", fontsize=6.8)
    c.set_xticks([1.0, 1.5, 2.0, 2.5]); c.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    c.xaxis.set_minor_formatter(ticker.NullFormatter())
    c.set_yticks([10, 20, 50, 100]); c.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    c.set_xlabel("初始半径 $R_0$ / cm"); c.set_ylabel("烘干时长 / h"); c.legend(loc="lower right"); grid(c)
    panel(c, "(c) 药材尺寸（双对数）")
    d.plot(H["factor"], H["t_star_h_and_hm_h"], **sty_b)
    d.plot(H["factor"], H["t_star_hm_only_h"], color=GRAY, marker="^", ms=3.4, lw=1.0, ls=":",
           label="题设模型（仅 $h_m$ 变化）")
    d.plot(H["factor"], H["t_star_evap_h"], **sty_e)
    d.set_xlabel("风速倍率（$h$、$h_m$ 同比例）"); d.set_ylabel("烘干时长 / h"); d.set_ylim(40, 110)
    d.legend(loc="upper right", fontsize=6.8); grid(d); panel(d, "(d) 风速")
    save(fig, "fig_process")


# ============================== 图 10 蒸发吸热 ==============================
def fig_evap(save, C):
    s3, e3 = read(C / "q23_series.csv"), read(C / "q3_evap_series.csv")
    s4, e4 = read(C / "q4_series.csv"), read(C / "q4_evap_series.csv")
    cr = read(C / "evap_critical.csv")
    fig, ax = plt.subplots(2, 2, figsize=(W, 4.75), gridspec_kw=dict(wspace=0.32, hspace=0.5))
    a, b, c, d = ax.ravel()
    he = e3["t"] / 3600
    tcr = float(cr["t_crit_h"][0])
    Tw = an.wet_bulb(float(s3["Tair"][-1]), float(s3["Cenv"][-1]))
    a.axvspan(0, tcr, **SHADE_CR)
    m3, me = s3["t"] / 3600 <= 30, he <= 30
    a.plot(s3["t"][m3] / 3600, s3["Tair"][m3], color=CHAR, lw=1.1, ls="--", label="烘房 $T_{air}$")
    a.axhline(Tw, color=TEAL, lw=0.9, ls=":")
    a.text(8, Tw - 2.4, f"$T_{{wb}}$={Tw:.1f} °C", ha="center", fontsize=7, color=TEAL)
    a.plot(s3["t"][m3] / 3600, s3["Ts"][m3], color=DEEP, label="表面（题设模型）")
    a.plot(he[me], e3["Ts"][me], color=CORAL, label="表面（能量一致）")
    a.plot(he[me], e3["Tc"][me], color=RED, ls="--", lw=1.1, label="中心（能量一致）")
    a.text(tcr / 2, 30.5, "恒速干燥段", ha="center", fontsize=7.2, color=DEEP)
    a.set_xlim(0, 30); a.set_ylim(27, 54); a.set_xlabel("时间 $t$ / h"); a.set_ylabel("温度 / °C")
    a.legend(loc="lower right", fontsize=6.7); grid(a); panel(a, "(a) 问题三物料温度")
    for s, e, ls, lab in ((s3, e3, "-", "问题三"), (s4, e4, "--", "问题四")):
        b.semilogy(s["Cbar"], s["rate"] * 3600, color=DEEP, ls=ls, lw=1.3, label=f"{lab} 题设模型")
        b.semilogy(e["Cbar"], e["rate"] * 3600, color=CORAL, ls=ls, lw=1.3, label=f"{lab} 能量一致")
    for k in range(2):
        b.plot(cr["Cbar_crit"][k], cr["rate_const_per_h"][k], "o", ms=4.2, color=CORAL, mec="white", zorder=4)
    hd, lb = b.get_legend_handles_labels()
    hd.append(Line2D([], [], marker="o", ls="none", ms=4.2, color=CORAL, mec="white"))
    lb.append("临界含水率")
    b.set_xlim(2.6, 0); b.set_xlabel("平均含水率 $\\bar C$ / (kg·kg$^{-1}$)（干燥方向 →）")
    b.set_ylabel("干燥速率 / h$^{-1}$"); b.legend(hd, lb, loc="lower left", fontsize=6.5); grid(b)
    panel(b, "(b) 干燥速率曲线")
    # (c) 题设传质律所需蒸发热与湿球状态下对流供热之比
    k = s3["t"] <= 12 * 3600
    Tair, Cenv, Cs = s3["Tair"][k], s3["Cenv"][k], s3["Cs"][k]
    Twb = np.array([an.wet_bulb(x, y) for x, y in zip(Tair, Cenv)])
    rho_s0 = float(dm.PropsQ23().rho(dm.C0)) / (1 + dm.C0)
    ratio = an.latent_heat(Twb) * rho_s0 * dm.HM * (Cs - Cenv) / (dm.H * (Tair - Twb))
    hh = s3["t"][k] / 3600
    c.semilogy(hh, ratio, color=RED)
    c.axhline(1, color=GRAY, lw=0.9, ls="--")
    c.fill_between(hh, 1, ratio, where=ratio > 1, color=CORAL, alpha=0.15, lw=0)
    i1 = int(np.argmax(ratio < 1))
    c.text(hh[i1] + 0.3, 1.25, f"{hh[i1]:.1f} h 后供热充足", fontsize=7, color=CHAR)
    c.text(11.7, 9, "比值 > 1：按题设传质律的蒸发量\n所需汽化热超过对流供热", ha="right", fontsize=6.8, color=CHAR)
    c.set_xlim(0, 12); c.set_xlabel("时间 $t$ / h"); c.set_ylabel("所需汽化热 / 可供对流热")
    grid(c); panel(c, "(c) 题设模型的表面能量核算")
    # (d) 最大含水率与烘干时长
    for s, e, ls, lab in ((s3, e3, "-", "问题三"), (s4, e4, "--", "问题四")):
        d.plot(s["t"] / 3600, s["Cmax"], color=DEEP, ls=ls, lw=1.3, label=f"{lab} 题设 {s['t'][-1] / 3600:.2f} h")
        d.plot(e["t"] / 3600, e["Cmax"], color=CORAL, ls=ls, lw=1.3, label=f"{lab} 能量一致 {e['t'][-1] / 3600:.2f} h")
    d.axhline(0.15, color=GRAY, lw=0.9, ls=":")
    d.set_xlim(0, 75); d.set_ylim(0, 2.7); d.set_xlabel("时间 $t$ / h"); d.set_ylabel("最大含水率 / (kg·kg$^{-1}$)")
    d.legend(loc="upper right", fontsize=6.5); grid(d); panel(d, "(d) 最大含水率与烘干时长")
    save(fig, "fig_evap")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here.parent.parent / "附件"))
    ap.add_argument("--csv", default=str(here.parent.parent / "exports" / "redo"))
    ap.add_argument("--out", default=str(here.parent / "figures"))
    ap.add_argument("--preview", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    C, save, radius = Path(a.csv), Saver(a.out, a.preview), dm.load_radius(a.data)
    jobs = {"framework": lambda: fig_framework(save), "geometry": lambda: fig_geometry(save, radius),
            "environment": lambda: fig_environment(save, C), "q1": lambda: fig_q1(save, C),
            "q2": lambda: fig_q2_stages(save, C), "fields": lambda: fig_q23_fields(save, C),
            "drying": lambda: fig_drying(save, C), "q4": lambda: fig_q4(save, C, radius),
            "process": lambda: fig_process(save, C), "evap": lambda: fig_evap(save, C)}
    for k, job in jobs.items():
        if a.only is None or k in a.only:
            job()


if __name__ == "__main__":
    main()
