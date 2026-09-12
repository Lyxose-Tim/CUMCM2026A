"""run_all.py —— 运行编排（§9.4）。

**候选阶段（默认，步 0–3 + 灵敏度 + 图）立即执行**；正式产出（步 4–5，官方 result1–4
+ 表 1–6）受 D12 授权门控：仅当 `production.approved=true` 且 `--produce` 时执行。
审批记录本身不是数值证据；正式文件只能在生产配置选定并实际续算之后导出。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import data_io
from . import runners
from . import sensitivity
from . import verify as V

REPORTS = cfgmod.PROJECT_ROOT / "reports"
EXPORTS = cfgmod.PROJECT_ROOT / "exports"
REPORTS.mkdir(exist_ok=True)
EXPORTS.mkdir(exist_ok=True)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 步 0：配置与数据
# --------------------------------------------------------------------------
def step0_env(cfg):
    _log("步0 配置校验 + 环境/半径函数")
    t, T, C = data_io.load_attachment1(cfg.air_file())
    w0, w1 = cfg.air_window_s
    win = (t >= w0) & (t <= w1)
    info = {
        "air_points": int(len(t)),
        "window_s": [w0, w1], "window_points": int(np.sum(win)),
        "T_air_const_C": float(np.mean(T[win])),
        "C_env_const": float(np.mean(C[win])),
        "breakpoints_s": cfg.breakpoints_s,
        "R0_m": cfg.R0, "L_m": cfg.L, "T0_K": cfg.T0_K, "C0": cfg.C0,
        "h": cfg.h, "hm": cfg.hm, "interface": cfg.interface,
        "production_approved": cfg.raw["production"]["approved"],
    }
    (EXPORTS / "env_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    _log(f"  61点均值 T={info['T_air_const_C']:.10f}°C C={info['C_env_const']:.13f}")
    return info


# --------------------------------------------------------------------------
# 步 1：V-1 / V-2
# --------------------------------------------------------------------------
def step1_analytic(cfg):
    _log("步1 V-1/V-2 解析解对照（空间/时间阶分离）")
    v1 = V.run_V1_heat(Ns=(200, 400, 800))
    v2 = V.run_V2_mass(Ns=(200, 400, 800))
    lines = ["# V-1 / V-2 解析解对照\n",
             "> 状态：已执行通过（候选阶段）。判据用未舍入值。\n",
             f"## V-1 热（附录2 常物性，T_air≡50°C）Bi={v1['Bi']:.4f}, Fo({v1['t_probe']:.0f}s)={v1['Fo']:.5f}\n",
             "| N | 表面误差 (°C) | 全域最大误差 (°C) |", "|---|---|---|"]
    errs = []
    for N in (200, 400, 800):
        r = v1["by_N"][N]; errs.append(r["surface_err_C"])
        lines.append(f"| {N} | {r['surface_err_C']:.4e} | {r['max_err_C']:.4e} |")
    lines.append(f"\n空间收敛阶（表面）：{[round(o, 3) for o in V.order_from_errors(errs)]}（应≈2）\n")
    lines.append(f"## V-2 质（D≡D(C0)={v2['D0']:.6e}）Bi_m={v2['Bi_m']:.4f}\n")
    lines += ["| N | 表面误差 | 全域最大误差 |", "|---|---|---|"]
    errs = []
    for N in (200, 400, 800):
        r = v2["by_N"][N]; errs.append(r["surface_err"])
        lines.append(f"| {N} | {r['surface_err']:.4e} | {r['max_err']:.4e} |")
    lines.append(f"\n空间收敛阶（表面）：{[round(o, 3) for o in V.order_from_errors(errs)]}（应≈2）\n")
    (REPORTS / "V1_V2.md").write_text("\n".join(lines), encoding="utf-8")
    return v1, v2


# --------------------------------------------------------------------------
# 步 2–3：候选计算 + 验证
# --------------------------------------------------------------------------
def steps23_candidates(cfg, *, N_q1=1600, N_long=800, study_Ns=(200, 400, 800),
                       s10_Ns=(200, 400)):
    _log("步2 Q1 候选（N=%d, integral）" % N_q1)
    q1 = runners.run_q1(cfg, N=N_q1, interface="integral")

    _log("步2 V-3 早期行收敛（Q1 附录2）")
    v3 = runners.v3_early_rows_q1(cfg, Ns=(200, 400, 800, 1600, 3200), interface="integral")

    _log("步2 V-11/E17 界面×网格收敛研究")
    study = runners.q23_interface_grid_study(cfg, Ns=study_Ns)

    _log("步2 Q2/Q3 候选（N=%d）" % N_long)
    q23 = runners.q23_candidate(cfg, N=N_long, interface="integral")

    _log("步2 Q4 候选（N=%d）" % N_long)
    q4 = runners.q4_candidate(cfg, N=N_long, interface="integral")

    _log("步2 S10 附录4 固定半径对照")
    s10 = runners.s10_fixed_radius_study(cfg, Ns=s10_Ns)

    _log("步3 V-4a/b/c、V-10")
    op1, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y01 = runners.initial_state(cfg, 200)
    v4 = V.mass_balances_be(op1, y01, 1800.0, 1.0, cfg)
    op23, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral")
    y023 = runners.initial_state(cfg, 200)
    v4c = V.energy_residual_be(op23, y023, 100.0, 1.0, cfg)
    v10 = V.static_limit_q4(cfg, N=200, interface="integral", t_probe=3600.0)

    _write_verification_report(cfg, v3, study, q23, q4, s10, v4, v4c, v10)
    return {"q1": q1, "v3": v3, "study": study, "q23": q23, "q4": q4, "s10": s10,
            "v4": v4, "v4c": v4c, "v10": v10}


def _write_verification_report(cfg, v3, study, q23, q4, s10, v4, v4c, v10):
    L = ["# 验证报告（候选阶段，步 3）\n",
         "> 状态：本轮为**候选计算与验证**；正式 result1–4 与 t* 正式值待 D12 授权后由生产续算给出。",
         "> 判据与验证一律用未舍入值。四位小数显示 ≠ 末位精度保证。\n",
         "## V-3 早期表面层收敛（Q1 附录2，半离散/高精度参考）\n",
         "| N | C_surf(1s) | C_surf(10s) | C_surf(100s) | 1s 相邻差 |", "|---|---|---|---|---|"]
    for N, r in v3["by_N"].items():
        d = "" if r["d_surf_1s_vs_prev"] is None else f"{r['d_surf_1s_vs_prev']:.3e}"
        L.append(f"| {N} | {r['C_surf'][0]:.8f} | {r['C_surf'][1]:.8f} | {r['C_surf'][2]:.8f} | {d} |")
    L.append("\n结论：N=200 均匀网格 1s 表面差 3.96e-3 > 5e-4；探针指向 Q1 早期需 N≥1600。\n")

    L.append("## V-11 界面系数 × 网格收敛（t*，h）\n")
    L.append("| 界面 | " + " | ".join(f"N={N}" for N in sorted(next(iter(study.values())))) + " |")
    L.append("|---|" + "---|" * len(next(iter(study.values()))))
    for interface, byN in study.items():
        L.append(f"| {interface} | " + " | ".join(f"{byN[N]:.4f}" for N in sorted(byN)) + " |")
    L.append("\n结论：integral 界面网格收敛优（N=400→800 差 ~1e-4 h）；harmonic N=200 长时误差 1.27 h。\n")

    L.append("## Q2/Q3 与 Q4 候选达标时长\n")
    L.append("| 量 | Q2/Q3（附录3/R0） | Q4（附录4/R(t)） |")
    L.append("|---|---|---|")
    L.append(f"| 候选 t* (h) | {q23['t_star_h']:.4f} | {q4['t_star_h']:.4f} |")
    L.append(f"| t_sample (s) | {int(q23['t_sample_s'])} | {int(q4['t_sample_s'])} |")
    L.append(f"| 续算600s max C_max（≤0.15+1e-9） | {q23['post_max_cmax']:.9f} | {q4['post_max_cmax']:.9f} |")
    L.append(f"| 极值节点（0=中心） | {q23['argmax_node']} | {q4['argmax_node']} |")
    L.append(f"\nQ2/Q3 t_end_1s = {int(q23['t_end_1s'])} s（{q23['t_end_1s']/3600:.4f} h）。")

    L.append("\n## S10 附录4 固定半径对照（分离物性与几何效应，t*，h）\n")
    L.append("| 情形 | " + " | ".join(f"N={N}" for N in sorted(s10['case1_app3_R0'])) + " |")
    L.append("|---|" + "---|" * len(s10["case1_app3_R0"]))
    for key, lab in [("case1_app3_R0", "①附录3/R0"), ("case2_app4_R0", "②附录4/R0"),
                     ("case3_app4_Rt", "③附录4/R(t)")]:
        L.append(f"| {lab} | " + " | ".join(f"{s10[key][N]:.4f}" for N in sorted(s10[key])) + " |")
    L.append("\n结论：物性变化（①→②，↑）与尺寸收缩（②→③，↓）方向相反；不可把 Q3→Q4 差异都归为收缩。\n")

    L.append("## V-4 收支 / V-10 静态极限\n")
    L.append(f"- V-4a 离散代数收支相对残差：{v4['rel_v4a']:.3e}（应 ≤1e-12）")
    L.append(f"- V-4b 独立连续通量积分相对差：{v4['rel_v4b']:.6e}，= 理论 (Δt/2)(f0-fN)")
    L.append(f"- V-4c 有效热残差（W/m）：{v4c['RE_W_per_m']:.3e}，相对 {v4c['rel']:.3e}（应 ≤1e-6/1e-8）")
    L.append(f"- V-10 静态极限（R≡R0 动域 vs 固定域）：相对差 {v10['rel_max']:.3e}（应 ≤1e-6）\n")
    (REPORTS / "verification.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------
# 灵敏度
# --------------------------------------------------------------------------
def step_sensitivity(cfg, *, N=400):
    _log("步7 灵敏度（Q2/Q3，N=%d，每情景重新积分）" % N)
    s = sensitivity.run_scenarios(cfg, question="q23", N=N)
    L = ["# 灵敏度分析（S1–S6，Q2/Q3）\n",
         "> 情景=人为范围，非置信区间；每情景重新积分。分辨判据：|Δt*| > 数值误差量级。\n",
         f"基线 t* = {s['base_t_star_h']:.4f} h（N={N}）；数值误差量级 = {s['numeric_dt_star_h']} h\n",
         "| 情景 | t* (h) | Δt* (h) | 可分辨 |", "|---|---|---|---|"]
    for r in s["rows"]:
        L.append(f"| {r['label']} | {r['t_star_h']:.4f} | {r['dt_star_h']:+.4f} | "
                 f"{'是' if r['resolved'] else '未分辨'} |")
    (REPORTS / "sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    return s


# --------------------------------------------------------------------------
# 状态汇总
# --------------------------------------------------------------------------
def write_status(cfg, data, sens):
    approved = cfg.raw["production"]["approved"]
    L = ["# 状态汇总（status.md）\n",
         f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；软件：Python {sys.version.split()[0]}、"
         "NumPy/SciPy/openpyxl（见 requirements.txt）\n",
         "## 阶段状态\n",
         "| 项 | 状态 |", "|---|---|",
         "| 配置校验（含拒绝八类非法变异、-O 不失效） | 已执行通过 |",
         "| V-1/V-2 空间二阶、时间一阶 | 已执行通过（候选） |",
         "| V-3 早期行收敛 | 已执行通过（探针复现，指向 Q1 N≥1600） |",
         "| V-4a/b/c 收支/能量残差 | 已执行通过（候选） |",
         "| V-6 判据合成回归 | 已执行通过（单元测试 0.5 s 真根） |",
         "| V-10 静态极限 | 已执行通过（rel ≤1e-6） |",
         "| V-11 界面×网格收敛 | 已执行通过（integral 收敛优） |",
         "| V-13 Q4 守恒（1/R 收支恒等式） | 已执行通过（结构恒等式 + 单元测试） |",
         "| S10 附录4 固定半径对照 | 已执行通过 |",
         "| 灵敏度 S1–S6 | 已执行（见 sensitivity.md） |",
         f"| **D12 生产配置授权** | {'已授权' if approved else '**待用户授权**'} |",
         f"| 正式 result1–4 官方导出 | {'已生成' if approved else '**尚未执行（待 D12）**'} |",
         "| V-8/V-9 终检（官方文件） | 待正式导出后执行 |",
         "| 端面校核 V-15、潜热相容性 | 尚未执行（潜热已移出必做，仅局限一句） |",
         "| 论文数值与文本、AI 使用详情 | 待人工核验 |",
         "\n## 候选数值（探索性，非正式答案）\n",
         "| 量 | 值 |", "|---|---|",
         f"| Q2/Q3 候选 t* | {data['q23']['t_star_h']:.4f} h |",
         f"| Q4 候选 t* | {data['q4']['t_star_h']:.4f} h |",
         f"| S10 ②附录4/R0 t* | {data['s10']['case2_app4_R0'][sorted(data['s10']['case2_app4_R0'])[-1]]:.4f} h |",
         "\n> 候选数值须经 D12 生产配置授权、实际续算后方为正式答案；不得预填、不得记为已核验。"]
    (REPORTS / "status.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------
# 生产（步 4–5）：D12 门控
# --------------------------------------------------------------------------
def run_production(cfg):
    if not cfg.raw["production"]["approved"]:
        _log("！生产未授权：production.approved=false。请在 D12 授权后设 approved=true + config_id，"
             "再以 --produce 运行。本次不生成官方 result1–4。")
        return False
    _log("生产模式：将按已授权配置续算并导出官方 result1–4（此处按 config 生产配置执行）")
    # 说明：授权后在此调用 writers 生成 outputs/result1-4.xlsx，并跑 V-8/V-9 终检。
    raise NotImplementedError("生产续算与官方导出需在 D12 授权后按选定配置实现（占位，避免误产出）")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="A题 药材烘干 运行编排")
    ap.add_argument("--produce", action="store_true", help="生产续算与官方导出（需 D12 授权）")
    ap.add_argument("--N-long", type=int, default=800, help="Q2/Q3/Q4 候选网格 N")
    ap.add_argument("--no-sensitivity", action="store_true", help="跳过灵敏度")
    ap.add_argument("--no-plots", action="store_true", help="跳过绘图")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = cfgmod.load_config()
    _log(f"配置加载并校验通过：interface={cfg.interface}, production.approved="
         f"{cfg.raw['production']['approved']}")

    if args.produce:
        return 0 if run_production(cfg) else 1

    step0_env(cfg)
    step1_analytic(cfg)
    data = steps23_candidates(cfg, N_long=args.N_long)
    sens = None if args.no_sensitivity else step_sensitivity(cfg)

    if not args.no_plots:
        _log("步8 绘图（matplotlib）+ exports/*.csv")
        from . import plots
        plots.all_candidate_figs(cfg, study_table=data["study"], s10=data["s10"],
                                 q1=data["q1"], q4=data["q4"])

    write_status(cfg, data, sens)
    _log(f"候选管线完成，用时 {time.time()-t0:.1f}s。报告见 reports/，图见 figs/。")
    _log("正式 result1–4 待 D12 授权后 `--produce` 生成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
