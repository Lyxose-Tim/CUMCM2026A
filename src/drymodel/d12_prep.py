"""d12_prep.py —— D12 生产配置申请：按各问实际误差选定最终 N，登记全部场点验收。

对照口径（acceptance.table_points）：**生产配置 vs 更精细对照**（同 BDF rtol，时间误差相消 → 空间误差）。
- Q1：N=1600 vs 3200，覆盖 result1 全部 1 s 行×21 列（C、T）+ 表 1/2。
- Q23：N=800 vs 1600，覆盖早期 1 s 行 + 60 s 网格 + 表点 + t* + 表面通量 + C̄。
- Q4：N=800 vs 1600，覆盖**全部固定厘米列 + 表面 + r=1.2 cm** + t*。
另登记：独立时间加密（收紧 rtol）、8/16 点界面求积对照。
**不以 t* 收敛替代场点验收**；最终 N 由各问 max 误差与 acceptance 门槛比较选定。
"""
from __future__ import annotations

import numpy as np

from . import config as cfgmod
from . import criterion as CR
from . import data_io
from . import postprocess as PP
from . import production as PROD
from . import runners
from . import solver_bdf as SBDF
from . import verify as V

REPORTS = cfgmod.PROJECT_ROOT / "reports"
COLS21 = runners.RESULT_COLS_CM
COLS20 = [round(0.1 * j, 4) for j in range(20)]
KELVIN = 273.15


def _sample_cols(full, ts, node_idx, n, chunk=4000):
    """分块求值，仅提取输出列的 C、T（控制内存）。返回 (C[len,ncol], T[len,ncol])。"""
    ts = np.asarray(ts, float)
    Cc, Tc = [], []
    for i in range(0, len(ts), chunk):
        Y = full.eval(ts[i:i + chunk])
        Cc.append(Y[:, node_idx]); Tc.append(Y[:, node_idx + n])
    return np.vstack(Cc), np.vstack(Tc)


def _argmax_loc(diff2d, ts, cols):
    """返回 (最大绝对差, t, r_cm)。"""
    k = int(np.nanargmax(np.abs(diff2d)))
    i, j = np.unravel_index(k, diff2d.shape)
    return float(abs(diff2d[i, j])), float(ts[i]), float(cols[j])


# --------------------------------------------------------------------------
def q1_selection(cfg, Ns=(1600, 3200)):
    acc = cfg.raw["acceptance"]["table_points"]
    tolC, tolT = float(acc["dC"]), float(acc["dT_degC"])
    ts = np.arange(1, 1801)
    grids = {}
    for N in Ns:
        op, _ = runners.build_fixed_operator(cfg, "q1", N, interface="integral", decoupled=True)
        n = N + 1
        res = SBDF.integrate_bdf(op, runners.initial_state(cfg, N), 0.0, 1800.0, cfg.bdf,
                                 breakpoints=(), threshold=None)
        node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
        C, T = _sample_cols(res, ts, node_idx, n)
        grids[N] = (C, T - KELVIN)
    dC = grids[Ns[0]][0] - grids[Ns[1]][0]
    dT = grids[Ns[0]][1] - grids[Ns[1]][1]
    mC = _argmax_loc(dC, ts, COLS21)
    mT = _argmax_loc(dT, ts, COLS21)
    ok = mC[0] <= tolC and mT[0] <= tolT
    final_N = Ns[0] if ok else Ns[1]
    return {"question": "q1", "compare": Ns, "maxdC": mC, "maxdT": mT,
            "tolC": tolC, "tolT": tolT, "pass": ok, "final_N": final_N,
            "note": "覆盖 result1 全部 1 s 行×21 列（C、T）"}


def _q_traj(cfg, question, N, moving):
    fc = {"N": N, "interface": "integral", "npts": 8,
          "t_cap_h": cfg.resolved(question, "candidate")["t_cap_h"]}
    full, op, det = PROD._single_trajectory(cfg, question, fc, moving)
    return full, op, det


def q23_selection(cfg, Ns=(800, 1600)):
    acc = cfg.raw["acceptance"]["table_points"]
    tolC = float(acc["dC"])
    thr = cfg.threshold
    data = {}
    for N in Ns:
        full, op, det = _q_traj(cfg, "q23", N, moving=False)
        n = op.N + 1
        node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
        t_sample = det.t_sample
        # 覆盖：早期 1 s 行(1..600) + 60 s 网格(60..t_sample) 全部正式采样行
        ts = np.union1d(np.arange(1, 601), np.arange(60, int(t_sample) + 1, 60))
        C, _ = _sample_cols(full, ts, node_idx, n)
        surf_f = float(op.flux_cbar(10800.0, full.eval([10800.0])[0][:n]))
        cbar = float(op.grid.cbar(full.eval([10800.0])[0][:n]))
        data[N] = {"ts": ts, "C": C, "t_star_h": det.t_cross / 3600.0,
                   "surf_f": surf_f, "cbar": cbar}
    # 场点最大差（对齐公共时间）
    tcommon = np.intersect1d(data[Ns[0]]["ts"], data[Ns[1]]["ts"])
    i0 = np.searchsorted(data[Ns[0]]["ts"], tcommon)
    i1 = np.searchsorted(data[Ns[1]]["ts"], tcommon)
    dC = data[Ns[0]]["C"][i0] - data[Ns[1]]["C"][i1]
    mC = _argmax_loc(dC, tcommon, COLS21)
    dtstar = abs(data[Ns[0]]["t_star_h"] - data[Ns[1]]["t_star_h"])
    ok = mC[0] <= tolC
    return {"question": "q23", "compare": Ns, "maxdC": mC, "tolC": tolC,
            "dtstar_h": dtstar, "d_surf_f": abs(data[Ns[0]]["surf_f"] - data[Ns[1]]["surf_f"]),
            "d_cbar": abs(data[Ns[0]]["cbar"] - data[Ns[1]]["cbar"]),
            "pass": ok, "final_N": Ns[0] if ok else Ns[1],
            "note": "覆盖早期 1 s 行 + 60 s 网格全部行 + 表面通量 + C̄ + t*"}


def q4_selection(cfg, Ns=(800, 1600)):
    tolC = float(cfg.raw["acceptance"]["table_points"]["dC"])
    data = {}
    for N in Ns:
        full, op, det = _q_traj(cfg, "q4", N, moving=True)
        n = op.N + 1
        radius = data_io.make_radius_function(cfg)
        t_sample = det.t_sample
        ts = np.arange(60, int(t_sample) + 1, 60)
        # 全部固定厘米列(0..1.9) + 表面 + r=1.2cm；域外 NaN
        rows = np.full((len(ts), len(COLS20) + 1), np.nan)
        for i, t in enumerate(ts):
            c = full.eval([float(t)])[0][:n]
            R_t = float(radius.R(float(t)))
            row = PP.sample_q4_row(c, op.grid, R_t, COLS20)
            for j, v in enumerate(row):
                rows[i, j] = np.nan if v is None else v
            rows[i, -1] = c[-1]                       # 表面
        data[N] = {"ts": ts, "rows": rows, "t_star_h": det.t_cross / 3600.0}
    tcommon = np.intersect1d(data[Ns[0]]["ts"], data[Ns[1]]["ts"])
    i0 = np.searchsorted(data[Ns[0]]["ts"], tcommon)
    i1 = np.searchsorted(data[Ns[1]]["ts"], tcommon)
    d = data[Ns[0]]["rows"][i0] - data[Ns[1]]["rows"][i1]   # NaN 处（域外）自动跳过
    colnames = COLS20 + ["surface"]
    with np.errstate(invalid="ignore"):
        k = int(np.nanargmax(np.abs(d)))
    i, j = np.unravel_index(k, d.shape)
    m_all = (float(abs(d[i, j])), float(tcommon[i]), colnames[j])
    # r=1.2 cm 专列（COLS20 index 12）
    j12 = COLS20.index(1.2)
    with np.errstate(invalid="ignore"):
        d12 = np.nanmax(np.abs(d[:, j12]))
    dtstar = abs(data[Ns[0]]["t_star_h"] - data[Ns[1]]["t_star_h"])
    ok = m_all[0] <= tolC
    return {"question": "q4", "compare": Ns, "max_all": m_all, "max_r1_2cm": float(d12),
            "tolC": tolC, "dtstar_h": dtstar, "pass": ok, "final_N": Ns[0] if ok else Ns[1],
            "note": "覆盖全部固定厘米列(0..1.9)+表面+r=1.2cm 的 60 s 全部行；不以 t* 替代"}


def interface_8_16(cfg, question, N, moving):
    """8 点 vs 16 点界面求积 t* 对照（独立数值键）。"""
    out = {}
    for npts in (8, 16):
        det, *_ = runners.q23_detect(cfg, N=N, interface="integral", npts=npts,
                                     question=question, moving=moving,
                                     t_cap_h=cfg.resolved(question, "candidate")["t_cap_h"])
        out[npts] = det.t_cross / 3600.0
    return {"question": question, "N": N, "t8": out[8], "t16": out[16],
            "diff_h": abs(out[8] - out[16])}


def run_d12_prep(cfg):
    """执行三问选型 + 时间/界面对照，写 reports/D12_config_application.md。返回结构化结果。"""
    q1 = q1_selection(cfg)
    q23 = q23_selection(cfg)
    q4 = q4_selection(cfg)
    time_q23 = V.v3_tstar_time(cfg, question="q23", N=q23["final_N"], rtols=(1e-8, 1e-10))
    quad_q23 = interface_8_16(cfg, "q23", q23["final_N"], moving=False)
    quad_q4 = interface_8_16(cfg, "q4", q4["final_N"], moving=True)
    res = {"q1": q1, "q23": q23, "q4": q4, "time_q23": time_q23,
           "quad_q23": quad_q23, "quad_q4": quad_q4}
    _write_application(cfg, res)
    return res


def _write_application(cfg, r):
    rt = float(cfg.bdf["rtol"])
    L = ["# D12 生产配置申请（供审批；approved 仍为 false，未生成正式 result1–4）\n",
         "> 由各问实际误差选定最终 N（生产配置 vs 更精细对照，同 BDF rtol）。"
         "空间/独立时间/8-16 点界面对照分别登记；不以 t* 收敛替代场点验收。\n",
         "## 拟定配置（待审批写入 config production.approved=true + per_question.final_N + config_id）\n",
         "| 问 | 界面 | 求积点 | BDF rtol/atol | max_step | 对照 N | 拟定最终 N | 场点验收 |",
         "|---|---|---|---|---|---|---|---|"]
    for q, lab in (("q1", "Q1(附录2)"), ("q23", "Q2/Q3(附录3)"), ("q4", "Q4(附录4)")):
        b = cfg.bdf
        L.append(f"| {lab} | integral | 8(16对照) | {rt:.0e}/{b['atol_C']:.0e},{b['atol_T_K']:.0e} | "
                 f"{int(b['max_step_data_s'])}/{int(b['max_step_after_s'])}s | "
                 f"{r[q]['compare']} | **{r[q]['final_N']}** | {'通过' if r[q]['pass'] else '**未过→取更细 N**'} |")
    L.append("\n## Q1 场点误差（result1 全部 1 s 行×21 列）\n")
    q1 = r["q1"]
    L.append(f"- max|ΔC| = {q1['maxdC'][0]:.3e}（t={q1['maxdC'][1]:.0f}s, r={q1['maxdC'][2]}cm），门槛 {q1['tolC']:.0e}")
    L.append(f"- max|ΔT| = {q1['maxdT'][0]:.3e} °C（t={q1['maxdT'][1]:.0f}s, r={q1['maxdT'][2]}cm），门槛 {q1['tolT']:.0e}")
    L.append(f"- 结论：{'满足 → 最终 N=1600' if q1['pass'] else '不满足 → 最终 N=3200'}\n")
    L.append("## Q2/Q3 场点误差（早期 1 s 行 + 60 s 全部行 + 通量/均量 + t*）\n")
    q23 = r["q23"]
    L.append(f"- max|ΔC| = {q23['maxdC'][0]:.3e}（t={q23['maxdC'][1]:.0f}s, r={q23['maxdC'][2]}cm），门槛 {q23['tolC']:.0e}")
    L.append(f"- 表面通量差 {q23['d_surf_f']:.3e}；C̄ 差 {q23['d_cbar']:.3e}；Δt*(N对照) {q23['dtstar_h']:.3e} h")
    L.append(f"- 结论：{'满足 → 最终 N=800' if q23['pass'] else '不满足 → 最终 N=1600'}\n")
    L.append("## Q4 场点误差（全部固定厘米列 + 表面 + r=1.2 cm，60 s 全部行）\n")
    q4 = r["q4"]
    L.append(f"- max|Δ| = {q4['max_all'][0]:.3e}（t={q4['max_all'][1]:.0f}s, 列={q4['max_all'][2]}），门槛 {q4['tolC']:.0e}")
    L.append(f"- **r=1.2 cm 专列 max|Δ| = {q4['max_r1_2cm']:.3e}**（独立登记，不以 t* 替代）；Δt*(N对照) {q4['dtstar_h']:.3e} h")
    L.append(f"- 结论：{'满足 → 最终 N=800' if q4['pass'] else '不满足 → 最终 N=1600'}\n")
    L.append("## 独立时间加密 与 8/16 点界面求积对照\n")
    L.append(f"- Q23 时间对照（N={r['time_q23']['N']}，rtol 1e-8→1e-10）：Δt* = {r['time_q23']['max_adjacent_diff_h']:.3e} h")
    L.append(f"- Q23 界面 8 vs 16 点（N={r['quad_q23']['N']}）：Δt* = {r['quad_q23']['diff_h']:.3e} h")
    L.append(f"- Q4 界面 8 vs 16 点（N={r['quad_q4']['N']}）：Δt* = {r['quad_q4']['diff_h']:.3e} h\n")
    L.append("## 审批说明\n")
    L.append("- 审批通过后：置 `production.approved=true`、填 `per_question.{q1,q23,q4}.final_N` 与 `production.config_id`，"
             "再运行 `python -m drymodel.run_all --produce` 生成官方 result1–4 + 表 1–6 并跑 V-8/V-9。")
    L.append("- 正式文件缺席不作否决理由；本申请仅提交配置与验收证据，approved 保持 false。")
    (REPORTS / "D12_config_application.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import time
    _cfg = cfgmod.load_config()
    _t0 = time.time()
    _r = run_d12_prep(_cfg)
    print(f"D12 config application written to reports/D12_config_application.md "
          f"[{time.time()-_t0:.1f}s]")
    for _q in ("q1", "q23", "q4"):
        print(f"  {_q}: final_N={_r[_q]['final_N']} pass={_r[_q]['pass']}")
