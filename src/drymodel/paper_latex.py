"""paper_latex.py —— 由官方产出 CSV 生成 LaTeX 表格片段（论文表 1–6，同源可复现）。

生成到 paper/tables/*.tex，由 paper/sections/*.tex \\input。不重跑生产，仅读 outputs/*.csv。
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import config as cfgmod

OUTPUTS = cfgmod.PROJECT_ROOT / "outputs"
PAPER_TABLES = cfgmod.PROJECT_ROOT / "paper" / "tables"


def _read(name):
    with open(OUTPUTS / name, encoding="utf-8") as f:
        return list(csv.reader(f))


def _tabular(headers, rows, colspec):
    L = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
         " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def build_latex_tables():
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    rcols = [r"$r{=}0$", r"$r{=}0.5$", r"$r{=}1.0$", r"$r{=}1.5$", r"$r{=}2.0$"]

    specs = [
        ("table1_temp.csv", "table1.tex", r"$t$/s", rcols, "cccccc"),
        ("table2_moist.csv", "table2.tex", r"$t$/s", rcols, "cccccc"),
        ("table3_temp.csv", "table3.tex", r"$t$/h", rcols, "cccccc"),
        ("table4_moist.csv", "table4.tex", r"$t$/h", rcols, "cccccc"),
        ("table5_moist.csv", "table5.tex", r"$t$/h", rcols, "cccccc"),
    ]
    for src, dst, thead, cols, spec in specs:
        rows = _read(src)[1:]
        (PAPER_TABLES / dst).write_text(_tabular([thead] + cols, rows, spec), encoding="utf-8")

    # 表 6：列 0/0.5/1.0/药材表面
    rows6 = _read("table6_moist.csv")[1:]
    (PAPER_TABLES / "table6.tex").write_text(
        _tabular([r"$t$/h", r"$r{=}0$", r"$r{=}0.5$", r"$r{=}1.0$", r"药材表面"], rows6, "ccccc"),
        encoding="utf-8")
    # 表 6 配套半径小表
    rows6r = _read("table6_radius.csv")[1:]
    rows6r = [[r[0], r[1], ("是" if r[2] == "1" else "否")] for r in rows6r]
    (PAPER_TABLES / "table6_radius.tex").write_text(
        _tabular([r"$t$/h", r"$R$/cm", r"外推"], rows6r, "ccc"), encoding="utf-8")
    return sorted(p.name for p in PAPER_TABLES.glob("*.tex"))


if __name__ == "__main__":
    names = build_latex_tables()
    print("LaTeX 表格片段 → paper/tables/：", names)
