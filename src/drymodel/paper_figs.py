"""Read-only paper figures at the final 160 mm publication width.

Nine data figures use archived workbooks/validation records. Unrounded event
annotations come from reports/post_contest/reproduction.json, never from rounded
workbook curves. Run: python -m drymodel.paper_figs --data-only
No solver, input file, workbook or diagnostic export is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text

from . import config as cfgmod
from . import data_io

ROOT = cfgmod.PROJECT_ROOT
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
EXPORTS = ROOT / "exports"
FIGDIR = ROOT / "paper" / "figures"
EVENT_SOURCE = REPORTS / "post_contest" / "reproduction.json"
WIDTH_IN = 160.0 / 25.4
THRESH = 0.15

NAVY, BLUE, TEAL = "#17324D", "#2878B5", "#179A92"
CORAL, AMBER, PURPLE = "#E66B4E", "#E7B34C", "#7B6DB0"
INK, GRID, GRAY_OUT, WHITE = "#25364A", "#E5EBF0", "#EEF1F4", "#FFFFFF"
# Archived optional functions retain their public names and compatible palette.
DEEP, OCEAN, JELLY, ICE = NAVY, BLUE, TEAL, "#D7EBF4"
GORANGE, CRED, RED = AMBER, CORAL, CORAL
CHAR, DARK, MIST, AUXGRAY = INK, INK, GRID, "#8293A3"
BLUE_D, BLUE_L, ORANGE, GOLD = BLUE, TEAL, CORAL, AMBER
TIME_COLORS = [NAVY, BLUE, TEAL, PURPLE]
POS_STYLE = {
    0.0: dict(color=NAVY, marker="o", ls="-", label="中心"),
    0.5: dict(color=BLUE, marker="s", ls="--", label="$r=0.5$ cm"),
    1.0: dict(color=TEAL, marker="^", ls="-.", label="$r=1.0$ cm"),
}
CMAP_MOIST = LinearSegmentedColormap.from_list(
    "paper_moist", [WHITE, "#D7EBF4", BLUE, NAVY])
CMAP_TEMP = LinearSegmentedColormap.from_list(
    "paper_temp", ["#FFF8E8", AMBER, CORAL, "#9D3E37"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.bbox": None,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.7, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.columnspacing": 1.1, "legend.borderaxespad": 0.3,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "mathtext.fontset": "dejavusans",
})

_QA = {}
_SOURCE_HASHES = {}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources(*paths):
    for path in paths:
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        digest = _hash(path)
        previous = _SOURCE_HASHES.setdefault(key, digest)
        if previous != digest:
            raise ValueError(f"Source changed during figure generation: {key}")
    return [Path(p).relative_to(ROOT).as_posix() for p in paths]


def _record(name, paths, notes, **checks):
    if any(value is False for value in checks.values()):
        raise ValueError(f"{name}: failed fidelity check {checks}")
    _QA[name] = {"sources": _sources(*paths), "notes": notes, "checks": checks}


def load_receipt():
    """Keep the original return API; require verified, current event metadata."""
    d = json.loads(EVENT_SOURCE.read_text(encoding="utf-8"))
    if d.get("production_ok") is not True:
        raise ValueError("Event annotations require a successful reproduction.")
    r = d["events"]
    result = {}
    for question, suffix in (("q23", "23"), ("q4", "4")):
        e = r[question]
        root, sample, cmax = (float(e[k]) for k in
                              ("t_star_s", "t_sample_s", "cmax_at_tsample"))
        if not (e.get("pass") is True and np.isfinite([root, sample, cmax]).all()
                and root < sample and cmax < THRESH):
            raise ValueError(f"Invalid strict-threshold event: {question}")
        result.update({f"tstar{suffix}_h": root / 3600, f"tstar{suffix}_s": root,
                       f"tsamp{suffix}_s": sample, f"cmax{suffix}": cmax})
    return result


def load_convergence():
    result = {}
    with (EXPORTS / "convergence.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row["interface"], []).append(
                (int(row["N"]), float(row["t_star_h"])))
    for rows in result.values():
        rows.sort()
    return result


def load_threecase(N=400):
    with (EXPORTS / "fig10_diff.csv").open(encoding="utf-8") as stream:
        return {row["case"]: float(row["t_star_h"]) for row in csv.DictReader(stream)
                if int(row["N"]) == N}


def load_v1v2():
    text = (REPORTS / "V1_V2.md").read_text(encoding="utf-8")

    def grab(section):
        match = re.search(r"##\s*" + re.escape(section) + r".*?(?=\n##|\Z)", text, re.S)
        if match is None:
            raise ValueError(f"Missing analytic benchmark: {section}")
        points = []
        for line in match.group(0).splitlines():
            cells = [cell.strip() for cell in line.strip("| \n").split("|")]
            if len(cells) == 3 and re.fullmatch(r"\d+", cells[0]):
                points.append((int(cells[0]), float(cells[2])))
        if not points or not np.isfinite(points).all():
            raise ValueError(f"Invalid analytic benchmark: {section}")
        return sorted(points)
    return {"temp": grab("V-1"), "moist": grab("V-2")}


def load_sensitivity():
    text = (REPORTS / "sensitivity.md").read_text(encoding="utf-8")
    match = re.search(r"基线\s*t\*\s*=\s*([\d.]+)", text)
    if match is None:
        raise ValueError("Sensitivity report has no baseline; no fallback is permitted.")
    base = float(match.group(1))
    rows = []
    for line in text.splitlines():
        if line.strip().startswith("| S"):
            cells = [cell.strip() for cell in line.strip("| \n").split("|")]
            rows.append((cells[0], float(cells[2]), cells[3]))
    if not rows or not np.isfinite([base] + [row[1] for row in rows]).all():
        raise ValueError("Sensitivity report contains missing/nonfinite data.")
    return base, rows


def _read_sheet(path: Path, sheet_index: int, *, tmax_s=None, tstep=1):
    """Preserve worksheet values and blanks; downsampling retains both endpoints."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.worksheets[sheet_index].iter_rows(values_only=True)
        positions = list(next(rows)[1:])
        times, values, last = [], [], None
        for i, row in enumerate(rows):
            if row[0] is None:
                continue
            t = float(row[0])
            if tmax_s is not None and t > tmax_s:
                break
            item = (t, [float(v) if v is not None else np.nan for v in row[1:]])
            last = item
            if i % tstep == 0:
                times.append(item[0])
                values.append(item[1])
        if last is not None and (not times or last[0] != times[-1]):
            times.append(last[0])
            values.append(last[1])
    finally:
        workbook.close()
    return np.asarray(times), positions, np.asarray(values, dtype=float)


def _panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", pad=9, fontsize=10.5)


def _grid(ax, axis="both"):
    ax.grid(axis=axis, which="major", color=GRID)
    ax.tick_params(which="minor", length=0)


def _note(fig, text):
    fig.text(0.02, 0.013, text, fontsize=8, va="bottom", color=INK)


def _layout(fig, bottom=0.08, top=0.97, w_pad=1.5, h_pad=2.0):
    fig.tight_layout(rect=(0.01, bottom, 0.99, top), pad=0.6,
                     w_pad=w_pad, h_pad=h_pad)


def _log_ticks(ax, values):
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def _save(fig, name):
    """Write complete PDF/PNG/SVG without changing the 160 mm canvas."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    is_data_figure = name in DATA_FIGURES
    if is_data_figure and not np.isclose(fig.get_figwidth(), WIDTH_IN):
        raise ValueError(f"{name}: final data-figure width must be 160 mm")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Matplotlib keeps off-range locator tick Text objects visible internally,
    # but does not draw them. Exclude these from canvas/font diagnostics.
    undrawn_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < low or tick.get_loc() > high:
                    undrawn_ticks.update((tick.label1, tick.label2))
    visible_text = [t for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip() and t not in undrawn_ticks]
    min_font = min(t.get_fontsize() for t in visible_text)
    if is_data_figure and min_font < 8.0:
        raise ValueError(f"{name}: text smaller than 8 pt: {min_font}")
    bounds = fig.bbox
    clipped = []
    for artist in visible_text:
        box = artist.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            clipped.append(artist.get_text())
    if is_data_figure and clipped:
        raise ValueError(f"{name}: text outside fixed canvas: {clipped}")
    for extension in ("pdf", "svg", "png"):
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format=extension, dpi=220, bbox_inches=None,
                        facecolor=WHITE, metadata={"Creator": "drymodel.paper_figs"})
            (FIGDIR / f"{name}.{extension}").write_bytes(buffer.getvalue())
    info = _QA.setdefault(name, {"sources": [], "notes": "", "checks": {}})
    info.update({"size_mm": (float(fig.get_figwidth() * 25.4), float(fig.get_figheight() * 25.4)),
                 "min_font_pt": float(min_font), "no_text_outside_canvas": True})
    plt.close(fig)
    print(f"  saved {name}: 160 mm, min text {min_font:g} pt", flush=True)


def fig_inputs(cfg):
    t1, T1, C1 = data_io.load_attachment1(cfg.air_file())
    t2, R2 = data_io.load_attachment2(cfg.radius_file())
    env = data_io.make_env_functions(cfg, "base")
    switch = env.t_switch / 3600
    # Separate segments retain the actual jump at 4 h; no smoothing across it.
    inside = np.linspace(0, env.t_switch, 1501)
    fig = plt.figure(figsize=(WIDTH_IN, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for ax, raw, fun, constant, color, label, letter, title in (
        (a, T1, lambda t: env.T_air_K(t) - 273.15, env.T_const_K - 273.15,
         CORAL, "$T_{\\rm air}$ / °C", "a", "烘房温度"),
        (b, C1, env.C_env, env.C_const, BLUE,
         "$C_{\\rm env}$ / (kg·kg$^{-1}$)", "b", "环境有效含水浓度"),
    ):
        ax.axvspan(switch, 6.5, color=AMBER, alpha=0.14, lw=0)
        ax.plot(inside / 3600, fun(inside), color=color, lw=1.2)
        ax.plot(t1 / 3600, raw, "o", color=color, mfc=WHITE, ms=2.5, mew=0.5)
        ax.plot([switch, 6.5], [constant, constant], color=color, ls="--", lw=1.6)
        ax.axvline(switch, color=CORAL, ls=":", lw=1)
        ax.set(xlim=(0, 6.5), ylabel=label)
        ax.set_xticks([0, 2, 4, 6])
        _panel(ax, letter, title)
        _grid(ax)
    a.tick_params(labelbottom=False)
    a.text(0.72, 0.48, "均值延续\n假设", ha="center", va="center",
           transform=a.transAxes, fontsize=8)
    b.set_xlabel("时间 $t$ / h")
    c.plot(t2 / 3600, R2, color=TEAL, lw=1.5)
    c.plot(t2 / 3600, R2, "o", color=TEAL, mfc=WHITE, ms=3, mew=0.7)
    c.set(xlabel="时间 $t$ / h", ylabel="半径 $R(t)$ / cm", xlim=(0, 72))
    c.set_xticks([0, 24, 48, 72])
    _panel(c, "c", "实测半径")
    _grid(c)
    c.text(0.96, 0.92, "附件2\n测量范围 ≤ 72 h", transform=c.transAxes,
           ha="right", va="top", fontsize=8.5)
    _note(fig, "圆点：附件原始节点；实线：线性插值。4 h 后环境取末小时节点均值。")
    _layout(fig, bottom=0.095, h_pad=1.4)
    _record("fig_inputs", [cfg.air_file(), cfg.radius_file(), cfgmod.DEFAULT_CONFIG_PATH],
            "0–4 h input samples and interpolation; 4–6.5 h assumed means drawn separately; radius stops at 72 h.",
            finite_input=bool(np.isfinite(np.r_[t1, T1, C1, t2, R2]).all()),
            no_radius_extrapolation=True)
    _save(fig, "fig_inputs")


def fig_q1_profiles():
    t, positions, T = _read_sheet(OUT / "result1.xlsx", 0)
    tC, positionsC, C = _read_sheet(OUT / "result1.xlsx", 1)
    r = np.asarray(positions, float)
    want = [100, 600, 1200, 1800]
    if not np.array_equal(t, tC) or positions != positionsC:
        raise ValueError("Q1 temperature/moisture sample grids differ.")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.0))
    colors = [NAVY, BLUE, TEAL, CORAL]
    for k, time in enumerate(want):
        indices = np.flatnonzero(t == time)
        if indices.size != 1:
            raise ValueError(f"Q1 missing exact sample {time} s.")
        for ax, field in zip(axes, (T, C)):
            color = colors[k]
            ax.plot(r, field[indices[0]], color=color, marker=["o", "s", "^", "D"][k],
                    ls=["-", "--", "-.", ":"][k], ms=3.3, mfc=WHITE, mew=0.8,
                    label=f"{time} s")
    for ax in axes:
        ax.set(xlabel="径向位置 $r$ / cm", xlim=(0, 2))
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        _grid(ax)
        ax.legend(loc="upper left" if ax is axes[0] else "lower left", ncol=2, fontsize=8)
    axes[0].set_ylabel("温度 $T$ / °C")
    axes[1].set_ylabel("$C$ / (kg·kg$^{-1}$)")
    _panel(axes[0], "a", "温度径向剖面")
    _panel(axes[1], "b", "含水率径向剖面")
    _note(fig, "正式工作簿采样：径向间隔 0.1 cm，数值保留四位小数。")
    _layout(fig, bottom=0.12)
    _record("fig_q1_profiles", [OUT / "result1.xlsx"],
            "Four exact worksheet times; 21 radial output positions, not a dense solver field.",
            exact_requested_times=all(time in t for time in want),
            finite_fields=bool(np.isfinite(T).all() and np.isfinite(C).all()))
    _save(fig, "fig_q1_profiles")


def _colorbar(fig, mesh, ax, label):
    bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.30,
                       fraction=0.07, aspect=28)
    bar.set_label(label, fontsize=9)
    bar.ax.tick_params(labelsize=8)
    bar.outline.set_visible(False)
    return bar


def fig_q23_fields():
    rec = load_receipt()
    root, sample = rec["tstar23_s"], rec["tsamp23_s"]
    tT, pT, T = _read_sheet(OUT / "result2.xlsx", 0, tmax_s=7200, tstep=6)
    tC, pC, C = _read_sheet(OUT / "result3.xlsx", 0)
    rT, rC, hC = np.asarray(pT, float), np.asarray(pC, float), tC / 3600
    maximum = np.max(C, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.8))
    a, b, c, d = axes.flat
    pa = a.pcolormesh(rT, tT / 3600, T, cmap=CMAP_TEMP, vmin=28, vmax=50,
                     shading="nearest", rasterized=True)
    pb = b.pcolormesh(rC, hC, C, cmap=CMAP_MOIST, vmin=0, vmax=2.55,
                     shading="nearest", rasterized=True)
    _colorbar(fig, pa, a, "$T$ / °C")
    _colorbar(fig, pb, b, "$C$ / (kg·kg$^{-1}$)")
    contour = b.contour(rC, hC, C, levels=[THRESH], colors=[CORAL], linewidths=1.2)
    b.clabel(contour, fmt={THRESH: "0.15"}, fontsize=8)
    for ax, time in ((a, tT / 3600), (b, hC)):
        ax.set(xlabel="径向位置 $r$ / cm", ylabel="时间 $t$ / h",
               xlim=(0, 2), ylim=(time.min(), time.max()))
        ax.set_xticks([0, 1, 2])
    _panel(a, "a", "温度场 · 前 2 h")
    _panel(b, "b", "含水率场")
    c.plot(hC, maximum, color=BLUE, lw=1.7, label="输出位置最大值")
    c.axhline(THRESH, color=CORAL, ls="--", lw=1.1, label="阈值 0.15")
    c.axvline(root / 3600, color=NAVY, ls=":", lw=1)
    c.set(xlabel="时间 $t$ / h", ylabel="$C_{\\max}$ / (kg·kg$^{-1}$)",
          xlim=(0, hC.max()), ylim=(0, 2.65))
    c.text(0.97, 0.63, f"$t^*={root / 3600:.4f}$ h",
           transform=c.transAxes, ha="right", fontsize=8.5)
    c.legend(loc="upper right")
    _panel(c, "c", "最大含水率时程")
    # Markers only: rounded workbook samples do not define the unrounded root.
    delta_s = tC - root
    selected = (delta_s >= -250) & (delta_s <= 70)
    d.plot(delta_s[selected], maximum[selected], "o", color=BLUE, mfc=WHITE,
           ms=5, mew=1.2, label="60 s 采样（四位小数）")
    d.axhline(THRESH, color=CORAL, ls="--", lw=1)
    d.axvline(0, color=NAVY, ls=":", lw=1)
    d.plot(0, THRESH, "D", color=NAVY, ms=5, label="连续根（复算记录）")
    d.plot(sample - root, rec["cmax23"], "^", color=CORAL, ms=6,
           label="严格采样（未舍入）")
    d.set(xlabel="相对连续根的时间 / s", ylabel="$C_{\\max}$ / (kg·kg$^{-1}$)",
          xlim=(-255, 105), ylim=(0.14996, 0.15016))
    d.set_xticks([-240, -120, 0, 60])
    d.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    d.legend(loc="upper right", fontsize=8, frameon=True,
             facecolor=WHITE, framealpha=0.96, edgecolor="none")
    _panel(d, "d", "阈值与离散采样")
    _grid(c); _grid(d)
    _note(fig, "场与曲线：历史工作簿四位小数；等值线仅作采样示意。\n"
          f"复算事件：连续根 {root:.3f} s；首个严格分钟采样 {sample:.0f} s（晚 {sample-root:.3f} s）。")
    _layout(fig, bottom=0.10, h_pad=2.0, w_pad=1.5)
    _record("fig_q23_fields", [OUT / "result2.xlsx", OUT / "result3.xlsx", EVENT_SOURCE],
            "Field uses rounded 0.1 cm samples; moisture every 60 s; temperature every 6 s with first/last retained. "
            "Shared moisture scale 0–2.55. Contour is sampled, event markers are independently verified metadata.",
            field_finite=bool(np.isfinite(T).all() and np.isfinite(C).all()),
            no_color_scale_clipping=bool(np.min(T) >= 28 and np.max(T) <= 50
                                        and np.min(C) >= 0 and np.max(C) <= 2.55),
            temperature_endpoints=bool(tT[0] == 1 and tT[-1] == 7200),
            first_last_moisture=bool(tC[0] == 60 and tC[-1] == sample),
            strict_sample_below_threshold=bool(rec["cmax23"] < THRESH),
            event_curve_not_interpolated=True)
    _save(fig, "fig_q23_fields")


def _q4_display_field(t, positions, values, cfg):
    fixed = np.asarray(positions[:-1], float)
    radius = data_io.make_radius_function(cfg)
    radii = np.asarray([float(radius.R(time)) * 100 for time in t])
    fine = np.linspace(0, cfg.R0 * 100, 240)
    display = np.full((len(t), len(fine)), np.nan)
    for i, R in enumerate(radii):
        inside = fixed <= R + 1e-9
        if not np.isfinite(values[i, :-1][inside]).all() or not np.isfinite(values[i, -1]):
            raise ValueError(f"Q4 missing in-domain value at t={t[i]}.")
        if np.isfinite(values[i, :-1][~inside]).any():
            raise ValueError(f"Q4 finite value outside domain at t={t[i]}.")
        strict_inside = fixed < R - 1e-9
        x = np.r_[fixed[strict_inside], R]
        y = np.r_[values[i, :-1][strict_inside], values[i, -1]]
        valid = fine <= R
        display[i, valid] = np.interp(fine[valid], x, y)
    return fixed, radii, fine, display


def fig_q4_fields(cfg):
    rec = load_receipt()
    root, sample = rec["tstar4_s"], rec["tsamp4_s"]
    t, positions, C = _read_sheet(OUT / "result4.xlsx", 0)
    fixed, radii, fine, field = _q4_display_field(t, positions, C, cfg)
    h = t / 3600
    cmap = CMAP_MOIST.copy()
    cmap.set_bad(GRAY_OUT)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.85))
    mesh = a.pcolormesh(fine, h, np.ma.masked_invalid(field), cmap=cmap,
                        shading="nearest", vmin=0, vmax=2.55, rasterized=True)
    # Cover half-cell edges outside the actual moving surface, not a zero field.
    a.fill_betweenx(h, radii, cfg.R0 * 100, color=GRAY_OUT, lw=0, zorder=2)
    a.plot(radii, h, color=CORAL, lw=1.5, zorder=3)
    a.text(0.98, 0.84, "药材域外", transform=a.transAxes, ha="right", fontsize=8.5)
    a.text(0.95, 0.43, "$R(t)$", transform=a.transAxes, ha="right", color=CORAL)
    a.set(xlabel="固定位置 $r$ / cm", ylabel="时间 $t$ / h",
          xlim=(0, cfg.R0 * 100), ylim=(h.min(), h.max()))
    a.set_xticks([0, 1, 2])
    _colorbar(fig, mesh, a, "$C$ / (kg·kg$^{-1}$)")
    for position in (0.0, 0.5, 1.0):
        index = np.flatnonzero(np.isclose(fixed, position))
        if index.size != 1:
            raise ValueError(f"Q4 missing output position {position} cm.")
        style = POS_STYLE[position]
        b.plot(h, C[:, index[0]], color=style["color"], ls=style["ls"],
               marker=style["marker"], mfc=WHITE, ms=3, markevery=230, lw=1.4,
               label=style["label"])
    b.plot(h, C[:, -1], color=PURPLE, ls=":", lw=1.8, label="表面 $r=R(t)$")
    b.axhline(THRESH, color=CORAL, ls="--", lw=1.1, label="阈值 0.15")
    b.axvline(root / 3600, color=NAVY, ls=":", lw=1)
    b.text(0.97, 0.49, f"$t^*={root/3600:.4f}$ h", ha="right",
           transform=b.transAxes, fontsize=8.5)
    b.set(xlabel="时间 $t$ / h", ylabel="$C$ / (kg·kg$^{-1}$)",
          xlim=(0, h.max()), ylim=(0, 2.65))
    b.legend(loc="upper right", ncol=1)
    _grid(b)
    _panel(a, "a", "含水率场与收缩边界")
    _panel(b, "b", "固定位置与表面时程")
    _note(fig, "场图：历史 60 s / 0.1 cm 采样，四位小数；仅域内线性插值，域外无数值。\n"
          f"复算事件：根 {root:.3f} s → 严格采样 {sample:.0f} s（+{sample-root:.3f} s）。")
    _layout(fig, bottom=0.16, w_pad=1.4)
    _record("fig_q4_fields", [OUT / "result4.xlsx", cfg.radius_file(), cfgmod.DEFAULT_CONFIG_PATH, EVENT_SOURCE],
            "Rounded samples linearly interpolated only inside R(t); outside NaN and gray cover; "
            "surface is its own moving coordinate. No smoothing/extrapolation beyond valid domain.",
            outside_remains_nan=bool(np.isnan(field[fine[None, :] > radii[:, None]]).all()),
            no_color_scale_clipping=bool(np.nanmin(field) >= 0 and np.nanmax(field) <= 2.55),
            first_last_rows=bool(t[0] == 60 and t[-1] == sample),
            shared_moisture_scale=True, exact_strict_event=bool(rec["cmax4"] < THRESH))
    _save(fig, "fig_q4_fields")


def fig_analytic_convergence():
    data = load_v1v2()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.1))
    for ax, key, color, title, ylabel in (
        (axes[0], "temp", CORAL, "温度解析基准", "沿半径最大误差 / °C"),
        (axes[1], "moist", BLUE, "含水率解析基准", "沿半径最大误差 / (kg·kg$^{-1}$)"),
    ):
        N, error = np.asarray(data[key], float).T
        ax.loglog(N, error, "o-", color=color, ms=6, mfc=WHITE, mew=1.4,
                  label="数值与解析解之差")
        # Offset slope guide avoids covering the measured errors.
        ax.loglog(N, error[0] * 0.65 * (N[0] / N)**2, "--", color=AUXGRAY,
                  lw=1.2, label="$N^{-2}$ 参考斜率")
        ax.set(xlabel="网格区间数 $N$", ylabel=ylabel)
        _log_ticks(ax, N)
        _grid(ax)
        ax.legend(loc="upper right")
        _panel(ax, "a" if key == "temp" else "b", title)
    _note(fig, "历史辅助验证：常物性、恒定边界，t = 100 s；含水率取 $D\\equiv D(C_0)$。")
    _layout(fig, bottom=0.13)
    _record("fig_analytic_convergence", [REPORTS / "V1_V2.md"],
            "Full radial maximum errors under constant-property analytic benchmark, not production four-question error.",
            same_source_points=True, reference_is_slope_only=True)
    _save(fig, "fig_analytic_convergence")


def fig_convergence():
    data = load_convergence()
    styles = {"harmonic": (CORAL, "s", "--", "调和平均界面"),
              "integral": (BLUE, "o", "-", "积分界面 · 8 点")}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.15))
    for key, points in data.items():
        color, marker, ls, label = styles[key]
        N, times = np.asarray(points, float).T
        axes[0].plot(N, times, color=color, marker=marker, ls=ls, label=label,
                     ms=5.5, mfc=WHITE, mew=1.2)
        axes[1].loglog(N[:-1], np.abs(times[:-1] - times[-1]),
                      color=color, marker=marker, ls=ls, ms=5.5, mfc=WHITE,
                      mew=1.2, label=label)
    axes[0].set_xscale("log", base=2)
    for ax in axes:
        ax.set_xlabel("网格区间数 $N$")
        _grid(ax)
    _log_ticks(axes[0], [200, 400, 800])
    _log_ticks(axes[1], [200, 400])
    axes[0].set_ylabel("烘干时长 $t^*$ / h")
    axes[1].set_ylabel("$|t_N^*-t_{800}^*|$ / h")
    _panel(axes[0], "a", "烘干时长")
    _panel(axes[1], "b", "相对各自 $N=800$ 的变化")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="center right")
    _note(fig, "历史网格对照；右图以各自界面的 $N=800$ 为参考，不表示真解误差。")
    _layout(fig, bottom=0.13, w_pad=1.5)
    _record("fig_convergence", [EXPORTS / "convergence.csv"],
            "Historical N200/400/800, each method has its own N800 reference; "
            "zero self-difference omitted on log axis. Not the new D12 comparison.",
            own_reference=True, zero_not_replaced=True)
    _save(fig, "fig_convergence")


def fig_threecase():
    data = load_threecase(400)
    keys = ["附录3/R0", "附录4/R0", "附录4/R(t)"]
    values = [next(v for k, v in data.items() if key in k) for key in keys]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.6))
    labels = ["① 附录3 · 固定 $R_0$", "② 附录4 · 固定 $R_0$", "③ 附录4 · 收缩 $R(t)$"]
    for y, value, color, marker in zip([2, 1, 0], values, [BLUE, AMBER, TEAL], ["o", "s", "D"]):
        ax.plot([0, value], [y, y], color=color, alpha=0.5, lw=2.0)
        ax.plot(value, y, marker, color=color, ms=8, mec=WHITE, mew=0.8)
        ax.text(value + 3, y, f"{value:.2f} h", va="center", fontsize=9, color=INK)
    ax.set_yticks([2, 1, 0], labels)
    ax.set(xlim=(0, max(values) + 28), ylim=(-0.55, 2.55),
           xlabel="烘干时长 $t^*$ / h")
    ax.set_xticks([0, 40, 80, 120])
    _grid(ax, "x")
    _panel(ax, "a", "同网格条件对照 · 历史 $N=400$")
    _note(fig, f"条件变化：①→② {values[1]-values[0]:+.2f} h；②→③ {values[2]-values[1]:+.2f} h。\n"
          "比较物性与几何设定的条件效应；三行不是顺序串联的干燥过程。")
    _layout(fig, bottom=0.23)
    _record("fig_threecase", [EXPORTS / "fig10_diff.csv"],
            "All three values selected at historical N400, including both conditional differences. "
            "No production N800 values mixed into the comparison.",
            all_three_cases=bool(len(values) == 3 and np.isfinite(values).all()),
            same_N400=True)
    _save(fig, "fig_threecase")


def _sens_label(name):
    if "hm" in name:
        return "$h_m$ × " + ("0.5" if "0.5" in name else "2")
    if "T_air" in name:
        return "$T_{\\rm air}$ " + ("+0.39 °C" if "+0.39" in name else "−0.39 °C")
    if "C_env" in name:
        return "$C_{\\rm env}$ " + ("+0.0011" if "+0.0011" in name else "−0.0011")
    if "hold_last" in name:
        return "环境取末值"
    if "smooth" in name:
        return "数据平滑"
    return "$h$ × " + ("0.5" if "0.5" in name else "2")


def fig_sensitivity():
    base, rows = load_sensitivity()
    items = [(_sens_label(name), value) for name, value, _ in rows]
    small = [(label, value) for label, value in items if abs(value) < 0.4]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 4.1),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    for ax, current, scale in ((axes[0], items, 0.15), (axes[1], small, 0.009)):
        yvalues = np.arange(len(current))[::-1]
        ax.axvspan(-0.02, 0.02, color=GRAY_OUT, zorder=0)
        ax.axvline(0, color=INK, lw=0.8)
        for y, (label, value) in zip(yvalues, current):
            color = CORAL if value > 0 else BLUE
            if value == 0:
                ax.plot(0, y, "|", color=INK, ms=9, mew=1.4)
            else:
                ax.barh(y, value, color=color, height=0.6, zorder=3)
            text = "0.0000*" if value == 0 else f"{value:+.4f}"
            ax.text(value + (scale if value >= 0 else -scale), y, text,
                    va="center", ha="left" if value >= 0 else "right",
                    fontsize=8, color=INK)
        ax.set_yticks(yvalues, [label for label, _ in current])
        ax.set_ylim(-0.6, len(current) - 0.4)
        ax.set_xlabel("$\\Delta t^*$ / h")
        _grid(ax, "x")
    axes[0].set_xlim(-6.5, 10.5)
    axes[0].set_xticks([-4, 0, 4, 8])
    axes[1].set_xlim(-0.54, 0.30)
    axes[1].set_xticks([-0.4, -0.2, 0, 0.2])
    _panel(axes[0], "a", "全部扰动情景")
    _panel(axes[1], "b", "小影响放大")
    _note(fig, f"历史 $N=400$，基线 {base:.4f} h。灰带 ±0.02 h 为筛选尺度，非置信区间。\n"
          "红橙：延长；蓝：缩短。* 变化仅显示到四位小数，不表示严格为零。")
    _layout(fig, bottom=0.16, w_pad=1.1)
    _record("fig_sensitivity", [REPORTS / "sensitivity.md"],
            "All source scenarios retained in report order (paired perturbations); small panel abs(delta)<0.4 h. "
            "Gray band is a screening scale, zero is rounded. Environmental offsets apply only after 4 h.",
            all_scenarios_retained=bool(len(items) == len(rows)),
            baseline_parsed=True, no_baseline_fallback=True)
    _save(fig, "fig_sensitivity")


def _load_be_bdf_diag():
    with (EXPORTS / "be_bdf_diag.csv").open(encoding="utf-8") as stream:
        diagnostic = list(csv.DictReader(stream))
    with (EXPORTS / "be_bdf_history.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    return diagnostic, history


def fig_be_bdf():
    diagnostic, history = _load_be_bdf_diag()
    t = np.asarray([float(row["t"]) for row in history])
    reference = np.asarray([float(row["T_ref_tightBDF"]) for row in history])
    be = sorted([(float(row["control"].split("=")[1].rstrip("s")),
                  float(row["diff_vs_tightBDF_K"])) for row in diagnostic
                 if row["method"] == "向后Euler"])
    step, error = np.asarray(be).T
    bdf_error = next(float(row["diff_vs_tightBDF_K"]) for row in diagnostic
                     if row["method"] == "自适应BDF")
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.5))
    methods = [
        ("T_be_dt1.0", CORAL, "-", "o", "BE 1 s"),
        ("T_be_dt0.5", AMBER, "--", "s", "BE 0.5 s"),
        ("T_be_dt0.25", PURPLE, "-.", "^", "BE 0.25 s"),
        ("T_bdf_prod", BLUE, ":", "D", "BDF"),
    ]
    for key, color, ls, marker, label in methods:
        difference = np.abs(np.asarray([float(row[key]) for row in history]) - reference)
        # The common initial value has exactly zero difference; do not invent a log floor.
        difference[difference == 0] = np.nan
        a.semilogy(t, difference, color=color, ls=ls, marker=marker,
                   markevery=10, ms=3, mfc=WHITE, lw=1.3, label=label)
    a.set(xlabel="时间 $t$ / s", ylabel="表面温度差 / K", xlim=(0, 100))
    a.legend(loc="center right", fontsize=8)
    b.loglog(step, error, "o-", color=CORAL, ms=6, mfc=WHITE, mew=1.4,
             label="BE 终点差异")
    b.loglog(step, error[0] * 0.65 * step / step[0], "--", color=AUXGRAY,
             lw=1.2, label="一阶参考斜率")
    _log_ticks(b, step)
    b.set(xlabel="BE 时间步长 $\\Delta t$ / s", ylabel="100 s 终点温度差 / K")
    b.set_yticks([0.002, 0.004, 0.008])
    b.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    b.yaxis.set_minor_formatter(ticker.NullFormatter())
    b.legend(loc="upper left", fontsize=8)
    b.text(0.96, 0.08, f"BDF 终点差异\n{bdf_error:.2e} K",
           transform=b.transAxes, ha="right", va="bottom", fontsize=8.5,
           bbox={"boxstyle": "round,pad=0.35", "fc": GRID, "ec": "none"})
    _grid(a); _grid(b)
    _panel(a, "a", "相对参考的时程差异")
    _panel(b, "b", "BE 时间加密")
    _note(fig, "历史辅助热问题：$N=400$，恒定空气温度 50 °C，0–100 s。\n"
          "参考 BDF：rtol=$10^{-11}$，atol=$10^{-13}$；对照 BDF：rtol=$10^{-8}$。")
    _layout(fig, bottom=0.18, w_pad=1.5)
    _record("fig_be_bdf", [EXPORTS / "be_bdf_diag.csv", EXPORTS / "be_bdf_history.csv"],
            "Historical constant-property N400/100s benchmark; differences computed from saved history. "
            "Initial exact zero masked on log axis; BDF endpoint is an annotation, not a BE fixed-step datum.",
            finite_history=bool(np.isfinite(reference).all()),
            BE_endpoint_source=True, BDF_not_on_fixed_step_axis=True)
    _save(fig, "fig_be_bdf")


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



DATA_FIGURES = (
    "fig_inputs", "fig_q1_profiles", "fig_q23_fields", "fig_q4_fields",
    "fig_analytic_convergence", "fig_convergence", "fig_be_bdf",
    "fig_threecase", "fig_sensitivity",
)


def _verify_protected_files(root, expected_hashes):
    """Require scientific sources; verify archived deliveries when distributed."""
    checked_required, checked_historical, not_distributed = [], [], []
    for name, digest in expected_hashes.items():
        relative = Path(name)
        historical = relative.parts[:2] == ("deliverables", "post_contest")
        path = Path(root) / relative
        if not path.exists():
            if historical:
                not_distributed.append(name)
                continue
            raise FileNotFoundError(f"Required protected file is missing: {name}")
        if _hash(path) != digest:
            raise ValueError(f"Protected file changed: {name}")
        (checked_historical if historical else checked_required).append(name)
    return {
        "checked_required": checked_required,
        "checked_historical": checked_historical,
        "not_distributed": not_distributed,
    }


def _write_qa():
    for name, digest in _SOURCE_HASHES.items():
        if _hash(ROOT / name) != digest:
            raise ValueError(f"Source changed during redraw: {name}")
    protected_file = REPORTS / "visual_upgrade" / "baseline.json"
    protected = {}
    if protected_file.exists():
        protected = json.loads(protected_file.read_text(encoding="utf-8"))["protected_sha256"]
    protection = _verify_protected_files(ROOT, protected)
    required_count = len(protection["checked_required"])
    historical_count = len(protection["checked_historical"])
    absent_historical = "、".join(protection["not_distributed"]) or "无"
    lines = [
        "# 数据图视觉与数据保真核验", "",
        "范围：九张正文数据图；只读取原始输入、历史正式工作簿与既有验证记录，不运行数值求解器。",
        "默认入口和 --data-only 均只重绘这九张图，不覆盖四问关系图或未入正文的潜热诊断图。",
        "", "## 成品规范", "",
        "- 所有 PDF/SVG 画布宽 160 mm；保存时不使用 tight 裁切。",
        "- 正文、刻度、图例基础字号最小 8 pt；轴标签 9–9.5 pt，面板标题 10.5 pt。数学上下标按排版比例缩小。",
        "- SVG 保留可编辑文字、线条、标记；大规模场网格作为嵌入栅格保存，避免数十万独立单元阻碍编辑。",
        "- 水分场统一色标 0–2.55 kg/kg；温度场使用暖色。位置、时刻与方法用线型/标记冗余编码。",
        "- 场来自历史四位小数输出；连续根和严格采样标记来自赛后复算事件，不对舍入场作高精度求根。",
        "",
        "| 图 | 画布 / mm | 最小字号 / pt | 画布内文字 | 数据检查 |",
        "|---|---:|---:|---|---|",
    ]
    for name in DATA_FIGURES:
        item = _QA[name]
        width, height = item["size_mm"]
        lines.append(f"| {name} | {width:.1f} × {height:.1f} | {item['min_font_pt']:g} | "
                     f"{'通过' if item['no_text_outside_canvas'] else '失败'} | "
                     f"{'通过' if all(item['checks'].values()) else '失败'} |")
    lines += ["", "## 逐图来源、缺失值与语义", ""]
    for name in DATA_FIGURES:
        item = _QA[name]
        lines += [f"### {name}", "", item["notes"], "",
                  "来源：" + "、".join(item["sources"]) + "。", "",
                  "核验：" + "；".join(f"{k}={v}" for k, v in item["checks"].items()) + "。", ""]
    event = load_receipt()
    lines += ["## 事件值（只作独立注释）", "",
              "| 问题 | 连续根 / s | 严格分钟采样 / s | 采样滞后 / s | 未舍入采样 Cmax |",
              "|---|---:|---:|---:|---:|"]
    for suffix, label in (("23", "Q2/Q3"), ("4", "Q4")):
        root, sample = event[f"tstar{suffix}_s"], event[f"tsamp{suffix}_s"]
        lines.append(f"| {label} | {root:.12f} | {sample:.0f} | {sample-root:.12f} | "
                     f"{event[f'cmax{suffix}']:.17g} |")
    lines += [
        "", "## 保护边界", "",
        f"已复核 {required_count + historical_count} 项受保护文件 SHA-256"
        f"（必需受保护文件 {required_count} 项，历史交付物 {historical_count} 项），与视觉升级前记录一致。",
        f"未随源码分发而不适用的历史交付物：{absent_historical}。",
        "历史 deliverables/post_contest 文件缺失时仅记录不适用；若存在则必须匹配保护哈希。其余受保护文件缺失或变动均阻断核验。",
        "不改变历史数据、物理模型、输入、求解器或新的 D12 证据；历史 N=400 辅助图与正式 N=800 主线明确区分。",
        "解析基准和相对最细网格变化不被表述为真实解误差界；灵敏度灰带不是置信区间。",
        "", "## 逐图视觉检查", "",
        "程序已完成字号、固定画布、画布内文字与数据保真检查；逐图视觉复核（agent）结果在本次成品复核后填写。",
        "", "## 来源 SHA-256", "",
        "| 源文件 | SHA-256 |", "|---|---|",
    ]
    lines += [f"| {name} | {digest} |" for name, digest in sorted(_SOURCE_HASHES.items())]
    lines += ["", f"绘图源 SHA-256：{_hash(Path(__file__))}",
              f"运行环境：Matplotlib {matplotlib.__version__}；NumPy {np.__version__}；openpyxl {openpyxl.__version__}。",
              "", "命令：D:/Anaconda/python.exe -X utf8 -m drymodel.paper_figs --data-only（PYTHONPATH=src）。", ""]
    destination = REPORTS / "visual_upgrade" / "DATA_VISUAL_QA.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"  QA saved: {destination.relative_to(ROOT)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only redraw of the nine paper data figures.")
    parser.add_argument("--data-only", action="store_true",
                        help="Select the default nine data figures; relation/latent stay untouched.")
    parser.add_argument("--figures", nargs="+", choices=DATA_FIGURES,
                        help="Redraw a subset for visual iteration (full QA report requires all nine).")
    args = parser.parse_args(argv)
    cfg = cfgmod.load_config()
    selected = args.figures or DATA_FIGURES
    print("生成正文数据图 →", FIGDIR, flush=True)
    for name in selected:
        if name in ("fig_inputs", "fig_q4_fields"):
            globals()[name](cfg)
        else:
            globals()[name]()
    if set(selected) == set(DATA_FIGURES):
        _write_qa()
    print("完成；关系图、未采用的潜热图保持原资源。", flush=True)


if __name__ == "__main__":
    main()
