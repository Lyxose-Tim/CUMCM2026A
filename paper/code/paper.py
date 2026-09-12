"""paper.py —— 由官方产出（outputs/）整理论文表 1–6 与相关文本。

严格区分**阈值穿越时间**（t*，首次向下穿越 0.15 = 严格达标集合下确界）与
**严格合格采样时间 t_sample**（首个实测未舍入 max C<0.15 的 60 s 网格点）；
说明四位小数显示 0.1500 与未舍入判据的关系。数据源为已授权生产的官方文件。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config as cfgmod

OUTPUTS = cfgmod.PROJECT_ROOT / "outputs"
REPORTS = cfgmod.PROJECT_ROOT / "reports"


def _read_csv(name):
    with open(OUTPUTS / name, encoding="utf-8") as f:
        return list(csv.reader(f))


def _md_table(rows, headers):
    L = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(L)


def build_paper_tables():
    # 优先读原生产 JSON 回执；否则读后续复核补充回执
    p = OUTPUTS / "production_receipt.json"
    if not p.exists():
        p = OUTPUTS / "production_receipt_supplementary.json"
    receipt = json.loads(p.read_text(encoding="utf-8"))
    t23 = receipt["results"]["q23"]
    t4 = receipt["results"]["q4"]
    tstar3_h = t23["t_star_s"] / 3600.0
    tstar4_h = t4["t_star_s"] / 3600.0
    tsamp3_h = t23["t_sample_s"] / 3600.0
    tsamp4_h = t4["t_sample_s"] / 3600.0

    t1 = _read_csv("table1_temp.csv")
    t2 = _read_csv("table2_moist.csv")
    t3 = _read_csv("table3_temp.csv")
    t4c = _read_csv("table4_moist.csv")
    t5 = _read_csv("table5_moist.csv")
    t6 = _read_csv("table6_moist.csv")
    t6r = _read_csv("table6_radius.csv")

    rcols = ["0", "0.5", "1.0", "1.5", "2.0"]
    L = [
        "# 论文表 1–6 与相关文本（据已授权生产官方文件）\n",
        f"> 数据源：`outputs/result1–4.xlsx`（config_id `{receipt['config_id']}`，代码 `{receipt.get('reviewed_at_commit', receipt.get('code_commit',''))[:10]}`）。",
        "> 全部为未舍入解在指定时刻/位置重构后取四位小数；判据与验证一律用未舍入值。\n",
        "## 达标时间与采样时刻的区分（重要）\n",
        f"- **阈值穿越时间**（t*，= 首次向下穿越 max C=0.15 = 严格达标集合下确界）：",
        f"  - Q3：t*₃ = **{tstar3_h:.4f} h**（{t23['t_star_s']:.3f} s）",
        f"  - Q4：t*₄ = **{tstar4_h:.4f} h**（{t4['t_star_s']:.3f} s）",
        f"- **严格合格采样时间 t_sample**（首个实测未舍入 max C<0.15 的 60 s 网格点，用于 result3/4 末行）：",
        f"  - Q3：t_sample,3 = {t23['t_sample_s']} s（{tsamp3_h:.4f} h）；Q4：t_sample,4 = {t4['t_sample_s']} s（{tsamp4_h:.4f} h）",
        "- 二者关系：t* 是连续解在重构场上求根得到的**阈值穿越时刻**（0.15 恰好被触及）；",
        "  t_sample 是其后首个**实测严格 <0.15** 的 60 s 采样点，故 t_sample ≥ t*，二者相差 < 60 s。\n",
        "### 显示 0.1500 与未舍入判据\n",
        "- 表 5/6 末行取 t* 时刻：中心含水率**未舍入值恰为 0.15**（阈值穿越定义），四位小数显示为 `0.1500`。",
        "- 「各处 C<0.15」为**严格不等式**：t* 处 max C=0.15（边界），达标（<0.15）在 t* 之后瞬间发生；",
        "  显示的 `0.1500` 不表示已严格达标，判据一律以未舍入值为准（续算 600 s 复核 max C≤0.15+1e-9，无回穿）。\n",
        "## 表 1　Q1 温度场（°C，附录 2，0–1800 s）\n",
        _md_table([[r[0]] + r[1:] for r in t1[1:]], ["t (s)"] + [f"r={c} cm" for c in rcols]),
        "\n## 表 2　Q1 水分浓度（kg/kg，干基，附录 2）\n",
        _md_table([[r[0]] + r[1:] for r in t2[1:]], ["t (s)"] + [f"r={c} cm" for c in rcols]),
        "\n## 表 3　Q2 温度场（°C，附录 3，0.5–3 h）\n",
        _md_table([[r[0]] + r[1:] for r in t3[1:]], ["t (h)"] + [f"r={c} cm" for c in rcols]),
        "\n## 表 4　Q2 水分浓度（kg/kg，附录 3）\n",
        _md_table([[r[0]] + r[1:] for r in t4c[1:]], ["t (h)"] + [f"r={c} cm" for c in rcols]),
        f"\n## 表 5　Q3 水分浓度（kg/kg，每 6 h + 末行烘干结束时间 t*₃={tstar3_h:.4f} h）\n",
        _md_table([[r[0]] + r[1:] for r in t5[1:]], ["t (h)"] + [f"r={c} cm" for c in rcols]),
        "\n> 末行 t*₃ 处中心（r=0）未舍入含水率 = 0.15（阈值穿越），显示 0.1500。\n",
        f"## 表 6　Q4 水分浓度（kg/kg，列 0/0.5/1.0 cm + 药材表面；每 6 h + 末行 t*₄={tstar4_h:.4f} h）\n",
        _md_table([[r[0]] + r[1:] for r in t6[1:]], ["t (h)", "r=0 cm", "r=0.5 cm", "r=1.0 cm", "药材表面"]),
        "\n> 1.5、2.0 cm 列在 t≥6 h 全部位于收缩后药材表面之外（域外），故表 6 不列；见 result4 域外留空规则。\n",
        "### 表 6 配套　各行半径 R(t)（cm）\n",
        _md_table([[r[0], r[1], ("是" if r[2] == "1" else "否")] for r in t6r[1:]],
                  ["t (h)", "R (cm)", "外推"]),
        "\n> 达标时 R(t*₄)=1.20 cm，未超附件 2 的 72 h 数据范围（无外推）。\n",
        "## 说明\n",
        "- result1（1 s×0.1 cm，0–1800 s）、result2（1 s，至 t_end,1s）、result3（60 s，至 t_sample,3）、",
        "  result4（60 s，至 t_sample,4，含药材表面列、域外留空）与上表同源（同一未舍入解）。",
        "- 半径小表不并入 result4 工作表；Q4 温度场求解并参与 D(C,T) 耦合，但不写入正式 result4。",
    ]
    (REPORTS / "paper_tables.md").write_text("\n".join(L), encoding="utf-8")
    return {"tstar3_h": tstar3_h, "tstar4_h": tstar4_h,
            "tsample3_h": tsamp3_h, "tsample4_h": tsamp4_h}


if __name__ == "__main__":
    r = build_paper_tables()
    print("论文表 1–6 → reports/paper_tables.md")
    print(f"  t*3={r['tstar3_h']:.4f}h t_sample3={r['tsample3_h']:.4f}h ; "
          f"t*4={r['tstar4_h']:.4f}h t_sample4={r['tsample4_h']:.4f}h")
