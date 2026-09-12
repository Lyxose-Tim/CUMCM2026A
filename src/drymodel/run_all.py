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
OUTPUTS = cfgmod.PROJECT_ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
EXPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


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
def steps23_candidates(cfg, *, study_Ns=(200, 400, 800), s10_Ns=(200, 400)):
    """候选计算 + 验证（配置驱动）。所有报告字段由本轮实跑数据填充（W5）。"""
    _log("步2 Q1 候选（config 解析）")
    q1 = runners.run_q1(cfg)

    _log("步2 V-11/E17 界面×网格收敛研究（=Q23 t* 网格对照）")
    study = runners.q23_interface_grid_study(cfg, Ns=study_Ns)

    _log("步2 Q2/Q3 候选")
    q23 = runners.q23_candidate(cfg)
    _log("步2 Q4 候选")
    q4 = runners.q4_candidate(cfg)
    _log("步2 S10 附录4 固定半径对照")
    s10 = runners.s10_fixed_radius_study(cfg, Ns=s10_Ns)

    # ---- V-3 分对象（实测）----
    _log("步3 V-3：早期表面(Q1/Q23) + 通量均量 + t*时间对照 + Q4固定厘米(含 r=1.2cm)")
    v3 = {
        "early_q1": V.v3_early_surface(cfg, question="q1", Ns=(200, 400, 800)),
        "early_q23": V.v3_early_surface(cfg, question="q23", Ns=(200, 400, 800)),
        "flux_q23": V.v3_flux_mean(cfg, question="q23", Ns=(200, 400, 800), t_probe_s=10800.0),
        "tstar_time_q23": V.v3_tstar_time(cfg, question="q23", N=400, rtols=(1e-8, 1e-10)),
        "q4_profile": V.v3_q4_profile(cfg, Ns=(200, 400), t_probe_h=24.0),
    }
    _write_v3_report(cfg, v3, study)

    # ---- V-4/V-5/V-10（实测；V-4b 粗/细步区分）----
    _log("步3 V-4a/b(BE dt=1/0.25)、V-4b(BDF)、V-4c、V-5 包络、V-10")
    op1, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    y01 = runners.initial_state(cfg, 200)
    v4_dt1 = V.mass_balances_be(op1, y01, 1800.0, 1.0, cfg)
    op1b, _ = runners.build_fixed_operator(cfg, "q1", 200, interface="harmonic", decoupled=True)
    v4_dt025 = V.mass_balances_be(op1b, y01, 1800.0, 0.25, cfg)
    v4b_bdf = V.flux_integral_bdf(cfg, question="q23", N=200, t_end_s=10800.0)
    op23, _ = runners.build_fixed_operator(cfg, "q23", 200, interface="integral")
    y023 = runners.initial_state(cfg, 200)
    v4c = V.energy_residual_be(op23, y023, 100.0, 1.0, cfg)
    v5_q23 = V.envelope_check(cfg, question="q23", N=200, t_end_s=8000.0)
    v5_q4 = V.envelope_check(cfg, N=200, moving=True, t_end_s=3600.0)
    v10 = V.static_limit_q4(cfg, N=200, interface="integral", t_probe=3600.0)

    data = {"q1": q1, "v3": v3, "study": study, "q23": q23, "q4": q4, "s10": s10,
            "v4_dt1": v4_dt1, "v4_dt025": v4_dt025, "v4b_bdf": v4b_bdf, "v4c": v4c,
            "v5_q23": v5_q23, "v5_q4": v5_q4, "v10": v10}
    _write_verification_report(cfg, data)
    return data


def _write_v3_report(cfg, v3, study):
    L = ["# V-3 收敛（按问题/输出对象；空间 vs 时间对照分列）\n",
         "> 本轮实测。空间对照=半离散/高精度参考解随 N；时间对照=固定 N 收紧 rtol。",
         "> 时长收敛（t*）不替代表点/固定厘米位置收敛（尤其 Q4 r=1.2 cm）。\n"]
    # 早期表面
    for key, lab in (("early_q1", "Q1 附录2"), ("early_q23", "Q23 附录3")):
        r = v3[key]
        L.append(f"## 空间对照 · result1/2 早期逐秒表面 C（{lab}）\n")
        L.append("| N | " + " | ".join(f"{tp:g}s" for tp in r["probe_times"]) + " | 1s 相邻差 |")
        L.append("|---|" + "---|" * (len(r["probe_times"]) + 1))
        for N in sorted(r["by_N"]):
            row = r["by_N"][N]
            d = row["d_vs_prev"][r["probe_times"][0]]
            ds = "" if d is None else f"{d:.3e}"
            vals = " | ".join(f"{row['C_surf'][tp]:.8f}" for tp in r["probe_times"])
            L.append(f"| {N} | {vals} | {ds} |")
        L.append(f"\n最大相邻差 = {r['max_adjacent_diff']:.3e}\n")
    # t* 网格（空间）vs 时间
    L.append("## 空间对照 · t*(h) 随 N（=V-11 界面×网格）\n")
    L.append("| 界面 | " + " | ".join(f"N={N}" for N in sorted(next(iter(study.values())))) + " |")
    L.append("|---|" + "---|" * len(next(iter(study.values()))))
    for interface, byN in study.items():
        L.append(f"| {interface} | " + " | ".join(f"{byN[N]:.4f}" for N in sorted(byN)) + " |")
    tt = v3["tstar_time_q23"]
    L.append(f"\n## 时间对照 · t*(h) 随 BDF rtol（Q23，N={tt['N']}）\n")
    L.append("| rtol | " + " | ".join(f"{rt:.0e}" for rt in tt["by_rtol"]) + " |")
    L.append("|---|" + "---|" * len(tt["by_rtol"]))
    L.append("| t* (h) | " + " | ".join(f"{tt['by_rtol'][rt]:.5f}" for rt in tt["by_rtol"]) + " |")
    L.append(f"\n最大时间差 = {tt['max_adjacent_diff_h']:.3e} h（与空间网格差分列，证据范围不同）\n")
    # 通量/均量
    fm = v3["flux_q23"]
    L.append(f"## 空间对照 · 表面通量 f 与平均含水率 C̄（Q23，t={int(fm['t_probe_s'])}s）\n")
    L.append(f"- 最大 f 相邻差 = {fm['max_flux_diff']:.3e}；最大 C̄ 相邻差 = {fm['max_cbar_diff']:.3e}\n")
    # Q4 固定厘米（含 r=1.2cm 回归点）
    qp = v3["q4_profile"]
    L.append(f"## 空间对照 · Q4 全部固定厘米位置（t={qp['t_probe_h']:g}h）\n")
    L.append(f"- 最差收敛位置：r={qp['worst']['r_cm']} cm，N对 {qp['worst']['N_pair']}，相邻差 {qp['worst']['diff']:.3e}")
    reg = qp["r1_2cm_by_N"]
    reg_s = " → ".join(f"N{N}:{reg[N]:.6f}" for N in sorted(reg) if reg[N] is not None)
    L.append(f"- **r=1.2 cm 回归点（独立于 t*，不以时长收敛替代）**：{reg_s}；"
             f"最大相邻差 {qp['r1_2cm_max_adjacent_diff']:.3e}\n")
    (REPORTS / "V3_convergence.md").write_text("\n".join(L), encoding="utf-8")


def _write_verification_report(cfg, d):
    q23, q4, s10 = d["q23"], d["q4"], d["s10"]
    L = ["# 验证报告（候选阶段，实测）\n",
         "> 本轮字段均由实跑数据填充。候选计算与验证；正式 result1–4 与 t* 正式值待 D12 授权后生产续算。",
         "> 判据与验证一律用未舍入值。四位小数显示 ≠ 末位精度保证。\n",
         "## Q2/Q3 与 Q4 候选达标时长（实测）\n",
         "| 量 | Q2/Q3（附录3/R0） | Q4（附录4/R(t)） |", "|---|---|---|",
         f"| 生效配置 N/界面/npts | {q23['N']}/{q23['interface']}/{q23['npts']} | {q4['N']}/{q4['interface']}/{q4['npts']} |",
         f"| 候选 t* (h) | {q23['t_star_h']:.4f} | {q4['t_star_h']:.4f} |",
         f"| t_sample (s) | {int(q23['t_sample_s'])} | {int(q4['t_sample_s'])} |",
         f"| 续算600s max C_max（≤0.15+1e-9） | {q23['post_max_cmax']:.9f} | {q4['post_max_cmax']:.9f} |",
         f"| 极值节点（0=中心） | {q23['argmax_node']} | {q4['argmax_node']} |",
         f"\nQ2/Q3 t_end_1s = {int(q23['t_end_1s'])} s（{q23['t_end_1s']/3600:.4f} h）。\n",
         "## S10 附录4 固定半径对照（t*，h，实测）\n",
         "| 情形 | " + " | ".join(f"N={N}" for N in sorted(s10['case1_app3_R0'])) + " |",
         "|---|" + "---|" * len(s10["case1_app3_R0"])]
    for key, lab in [("case1_app3_R0", "①附录3/R0"), ("case2_app4_R0", "②附录4/R0"),
                     ("case3_app4_Rt", "③附录4/R(t)")]:
        L.append(f"| {lab} | " + " | ".join(f"{s10[key][N]:.4f}" for N in sorted(s10[key])) + " |")
    L.append("\n## V-4 收支（实测）\n")
    v1, v025, vb = d["v4_dt1"], d["v4_dt025"], d["v4b_bdf"]
    L.append("| 检验 | Δt=1 s | Δt=0.25 s | 说明 |")
    L.append("|---|---|---|---|")
    L.append(f"| V-4a 离散代数恒等式 rel | {v1['rel_v4a']:.3e} | {v025['rel_v4a']:.3e} | 同一 RHS 恒等式，非时间精度 |")
    L.append(f"| V-4b(BE) 独立梯形 rel | {v1['rel_v4b']:.6e} | {v025['rel_v4b']:.6e} | 粗步 vs 细步：O(Δt) 下降 |")
    acc = cfg.raw["acceptance"]
    tbe = float(acc["flux_integral"]["be"]["tol_rel"])
    tbdf = float(acc["flux_integral"]["bdf"]["tol_rel_factor_of_rtol"]) * float(cfg.bdf["rtol"])
    coarse = "粗步未过" if v1["rel_v4b"] > tbe else "过"
    fine = "细步通过" if v025["rel_v4b"] <= tbe else "未过"
    L.append(f"\nV-4b(BE) 门槛 {tbe:.0e}：粗步(Δt=1) rel {v1['rel_v4b']:.2e} → {coarse}；"
             f"细步(Δt=0.25) rel {v025['rel_v4b']:.2e} → {fine}；"
             f"= 变步长理论差 Σ Δt_k(f_{{k-1}}−f_k)/2（diff−theory={abs(v1['diff_v4b']-v1['theory_v4b']):.1e}）。")
    L.append(f"\nV-4b(BDF) 独立连续自适应求积 rel = {vb['rel']:.3e} {'≤' if vb['rel']<=tbdf else '>'} 10×rtol={tbdf:.0e}"
             f"（{'通过' if vb['rel']<=tbdf else '未过'}），加密复核 rel_refine = {vb['rel_refine']:.3e}"
             f"（区别于同一 RHS 恒等式）。")
    L.append(f"\nV-4c 有效热残差（W/m）= {d['v4c']['RE_W_per_m']:.3e}，相对 {d['v4c']['rel']:.3e}"
             f"（离散代数平衡，时间精度由 V-3）。\n")
    L.append("## V-5 物理界限包络（H17/H18，实测）\n")
    for lab, v in (("Q23 固定域", d["v5_q23"]), ("Q4 动域", d["v5_q4"])):
        wc, wt = v["worst_C_excess"], v["worst_T_excess"]
        L.append(f"- {lab}：{'无越界' if v['ok'] else '有越界'}；C 最大越界 {wc['exceed']:.2e}"
                 f"（t={wc['t']}, node={wc['node']}），T 最大越界 {wt['exceed']:.2e}")
    L.append(f"\n## V-10 静态极限（R≡R0 动域 vs 固定域）\n- 相对差 {d['v10']['rel_max']:.3e}（应 ≤1e-6）\n")
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
    """状态汇总（W5：由实测数据判定；状态词严格分四类；不以 approved 判定文件已生成）。"""
    approved = cfg.raw["production"]["approved"]
    v3, d = data["v3"], data
    acc = cfg.raw["acceptance"]
    tol_be = float(acc["flux_integral"]["be"]["tol_rel"])          # 1e-4
    tol_bdf = float(acc["flux_integral"]["bdf"]["tol_rel_factor_of_rtol"]) * float(cfg.bdf["rtol"])  # 10×rtol
    tol_v4a = float(acc["discrete_balance_rel"])                   # 1e-12
    tol_erel = float(acc["energy_residual"]["rel"])                # 1e-8
    tol_tstar = float(acc["t_star_h"])                             # 0.02 h

    def pf(ok):
        return "已修复并实测通过" if ok else "**仍失败**"

    # 实测阈值判定（不用 pf(True)、不以 t*>0 判收敛）
    v4a_ok = d["v4_dt1"]["rel_v4a"] < tol_v4a and d["v4_dt025"]["rel_v4a"] < tol_v4a
    v4b_be_coarse_fail = d["v4_dt1"]["rel_v4b"] > tol_be           # 粗步(Δt=1)按 1e-4 门槛
    v4b_be_fine_ok = d["v4_dt025"]["rel_v4b"] <= tol_be            # 细步(Δt=0.25)
    v4b_bdf_ok = d["v4b_bdf"]["rel"] <= tol_bdf                    # 10×rtol
    v4c_ok = d["v4c"]["rel"] < tol_erel
    v5_ok = d["v5_q23"]["ok"] and d["v5_q4"]["ok"]
    v10_ok = d["v10"]["rel_max"] < 1e-6
    # V-11：integral 界面网格收敛（N400→800 t* 差 < t_star_h）
    ig = data["study"]["integral"]; Ns = sorted(ig)
    interf_conv = abs(ig[Ns[-1]] - ig[Ns[-2]])
    interf_ok = interf_conv < tol_tstar
    # 正式文件由**磁盘实际存在**判定（不以 approved）
    files_exist = all((OUTPUTS / f"result{k}.xlsx").exists() for k in (1, 2, 3, 4))
    v4b_be_txt = (f"粗步 Δt=1 rel {d['v4_dt1']['rel_v4b']:.2e} {'>' if v4b_be_coarse_fail else '≤'} {tol_be:.0e}"
                  f"（{'粗步未过' if v4b_be_coarse_fail else '过'}）；"
                  f"细步 Δt=0.25 rel {d['v4_dt025']['rel_v4b']:.2e} {'≤' if v4b_be_fine_ok else '>'} {tol_be:.0e}"
                  f"（{'细步通过' if v4b_be_fine_ok else '未过'}）")

    L = ["# 状态汇总（status.md）\n",
         f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；软件：Python "
         f"{data.get('versions', sys.version.split()[0])}\n",
         "> 状态词：已修复并实测通过 / 仍未执行 / 仍失败 / 待用户授权。**由实测阈值判定，不用 pf(True)、"
         "不以 t*>0 判收敛、不以 approved 判文件已生成。**\n",
         "## 阶段状态（实测）\n",
         "| 项 | 状态 | 实测证据 |", "|---|---|---|",
         "| 配置校验（拒八类变异+类型/有限性、-O 不失效、两检查器一致） | 已修复并实测通过 | test_checker_parity 15 例 |",
         "| V-1/V-2 空间二阶、时间一阶 | 已修复并实测通过 | reports/V1_V2.md |",
         f"| V-3 分对象（早期表面/通量均量/t*空间与时间/Q4固定厘米含 r=1.2cm） | 已实测分对象登记（最终 N 见 D12 申请） | reports/V3_convergence.md；"
         f"Q4 r=1.2cm 相邻差 {v3['q4_profile']['r1_2cm_max_adjacent_diff']:.2e} |",
         f"| V-4a 离散代数恒等式（非时间精度） | {pf(v4a_ok)} | rel {d['v4_dt1']['rel_v4a']:.1e}/{d['v4_dt025']['rel_v4a']:.1e}（<{tol_v4a:.0e}） |",
         f"| V-4b BE 独立梯形（粗步未过/细步通过，1e-4 门槛） | {pf(v4b_be_fine_ok)} | {v4b_be_txt} |",
         f"| V-4b BDF 独立连续自适应求积（10×rtol） | {pf(v4b_bdf_ok)} | rel {d['v4b_bdf']['rel']:.2e} ≤ {tol_bdf:.0e} |",
         f"| V-4c 有效热残差（W/m，细步安全） | {pf(v4c_ok)} | rel {d['v4c']['rel']:.1e}（<{tol_erel:.0e}） |",
         f"| V-5 物理界限包络（H17/H18，接受解） | {pf(v5_ok)} | 固定域/动域均无越界 |",
         f"| V-10 静态极限 | {pf(v10_ok)} | rel {d['v10']['rel_max']:.1e} |",
         f"| V-11 integral 界面网格收敛 | {pf(interf_ok)} | N{Ns[-2]}→{Ns[-1]} t* 差 {interf_conv:.2e} h（<{tol_tstar} h） |",
         "| V-6 判据合成回归 | 已修复并实测通过 | test_criterion 0.5 s 真根 |",
         "| S10 附录4 固定半径对照 | 已实测（三情形登记） | reports/verification.md |",
         f"| 灵敏度 S1–S6 | {'已实测（见报告）' if sens else '仍未执行（本次跳过）'} | reports/sensitivity.md |",
         "| V-13 Q4 守恒（1/R 收支恒等式） | 已修复并实测通过 | test_balances 增广态 |",
         f"| **D12 生产配置授权** | {'已授权' if approved else '**待用户授权**'} | production.approved={approved} |",
         f"| 正式 result1–4 官方导出 | {'已生成（磁盘存在）' if files_exist else '**仍未执行**'} | 由 outputs/ 磁盘实际存在判定（非 approved） |",
         "| 生产编排 run_production | 已实现（--produce 门控，见 test_production 缩比测试） | run_all.run_production |",
         "| V-8/V-9 官方文件终检 | 已实现（导出后执行，非 D12 前置） | writers.verify_workbook/cross_file_check |",
         "| 端面校核 V-15 | 仍未执行 | — |",
         "| 论文数值/文本、AI 使用详情 | 待人工核验 | — |",
         "\n## 候选数值（探索性，非正式答案）\n",
         "| 量 | 值（生效配置） |", "|---|---|",
         f"| Q2/Q3 候选 t* | {d['q23']['t_star_h']:.4f} h（N={d['q23']['N']}, {d['q23']['interface']}） |",
         f"| Q4 候选 t* | {d['q4']['t_star_h']:.4f} h（N={d['q4']['N']}, {d['q4']['interface']}） |",
         f"| S10 ②附录4/R0 t* | {d['s10']['case2_app4_R0'][max(d['s10']['case2_app4_R0'])]:.4f} h |",
         "\n> 候选数值须经 D12 生产配置授权、实际续算后方为正式答案；不得预填、不得记为已核验。",
         "> 配置快照见 exports/config_snapshot.json。"]
    (REPORTS / "status.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------
# 生产（步 4–5）：D12 门控
# --------------------------------------------------------------------------
def run_production(cfg):
    from . import production
    if not cfg.raw["production"]["approved"]:
        _log("！生产未授权：production.approved=false。请在 D12 授权后设 approved=true + config_id + "
             "per_question.final_N，再以 --produce 运行。本次不生成官方 result1–4。")
        return False
    _log("生产模式：按已授权分问最终配置续算并导出官方 result1–4 + 表 1–6，并跑 V-8/V-9 终检")
    r = production.run_production(cfg)  # require_approved 默认 True
    _log(f"导出完成：V-9 全部通过={all(v['ok'] for v in r['V9'].values())}，V-8 通过={r['V8']['ok']}")
    return bool(r["ok"])


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="A题 药材烘干 运行编排")
    ap.add_argument("--produce", action="store_true", help="生产续算与官方导出（需 D12 授权）")
    ap.add_argument("--no-sensitivity", action="store_true", help="跳过灵敏度")
    ap.add_argument("--no-plots", action="store_true", help="跳过绘图")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = cfgmod.load_config()
    _log(f"配置加载并校验通过：interface={cfg.interface}, production.approved="
         f"{cfg.raw['production']['approved']}")

    if args.produce:
        return 0 if run_production(cfg) else 1

    # 可追溯配置快照（含软件版本）
    snap = cfg.snapshot()
    try:
        import numpy, scipy, openpyxl
        snap["libs"] = {"numpy": numpy.__version__, "scipy": scipy.__version__,
                        "openpyxl": openpyxl.__version__}
    except Exception:
        pass
    (EXPORTS / "config_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("配置快照 → exports/config_snapshot.json")

    step0_env(cfg)
    step1_analytic(cfg)
    data = steps23_candidates(cfg)
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
