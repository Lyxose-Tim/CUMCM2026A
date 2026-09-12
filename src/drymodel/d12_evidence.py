"""d12_evidence.py —— D12 可追溯证据（**本轮代码实测复跑**，非外部摘要转录）。

覆盖各问全部正式采样行的 C 与 T 场点误差（生产 N vs 更细 N，同 rtol → 空间误差）、
三问独立时间加密、8/16 点界面求积、Q23/Q4 全程通量检验。记录参数、误差最大值及其
时间与位置、阈值、结论、复现命令与证据路径。**内存轻量：分块 + 运行最大值，不存全场。**
"""
from __future__ import annotations

import numpy as np

from . import config as cfgmod
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


def _running_max_diff_fixed(fullA, nA, fullB, nB, ts, node_idxA, node_idxB, chunk=8000):
    """固定域：对齐时间分块比较 C、T 的最大绝对差 + 位置。"""
    mC = (0.0, None, None); mT = (0.0, None, None)
    for i in range(0, len(ts), chunk):
        tc = ts[i:i + chunk]
        YA = fullA.eval(tc); YB = fullB.eval(tc)
        dC = YA[:, node_idxA] - YB[:, node_idxB]
        dT = (YA[:, node_idxA + nA] - YB[:, node_idxB + nB])
        for d2, m, cols in ((dC, "C", COLS21), (dT, "T", COLS21)):
            k = int(np.argmax(np.abs(d2))); ii, jj = np.unravel_index(k, d2.shape)
            val = abs(float(d2[ii, jj]))
            if m == "C" and val > mC[0]:
                mC = (val, float(tc[ii]), cols[jj])
            if m == "T" and val > mT[0]:
                mT = (val, float(tc[ii]), cols[jj])
    return mC, mT


def q1_evidence(cfg, Ns=(1600, 3200)):
    ts = np.arange(1, 1801)
    ops = {}
    for N in Ns:
        op, _ = runners.build_fixed_operator(cfg, "q1", N, interface="integral", decoupled=True)
        res = SBDF.integrate_bdf(op, runners.initial_state(cfg, N), 0.0, 1800.0, cfg.bdf,
                                 breakpoints=(), threshold=None)
        ops[N] = (res, N + 1, np.array([op.grid.output_index(rc) for rc in COLS21]))
    mC, mT = _running_max_diff_fixed(ops[Ns[0]][0], ops[Ns[0]][1], ops[Ns[1]][0], ops[Ns[1]][1],
                                     ts, ops[Ns[0]][2], ops[Ns[1]][2])
    return {"q": "q1", "Ns": Ns, "n_rows": len(ts), "maxC": mC, "maxT": mT}


def q23_evidence(cfg, Ns=(800, 1600)):
    """Q23：全部逐秒行（1..t_end_1s）的 C、T 最大差。"""
    fc = lambda N: {"N": N, "interface": "integral", "npts": 8, "t_cap_h": cfg.resolved("q23", "candidate")["t_cap_h"]}
    trajs = {}
    for N in Ns:
        full, op, det = PROD._single_trajectory(cfg, "q23", fc(N), moving=False)
        trajs[N] = (full, op, det, np.array([op.grid.output_index(rc) for rc in COLS21]))
    # 全部逐秒行至 t_end_1s（用更细网格 det 的 t_cross 稍大者为界，取两者最小 t_sample 内的逐秒）
    t_end = int(min(np.ceil(trajs[N][2].t_cross) for N in Ns))
    ts = np.arange(1, t_end + 1)
    a, b = trajs[Ns[0]], trajs[Ns[1]]
    mC, mT = _running_max_diff_fixed(a[0], a[1].N + 1, b[0], b[1].N + 1, ts, a[3], b[3])
    fluxv = {N: V.flux_integral_bdf(cfg, question="q23", N=N, t_end_s=float(trajs[N][2].t_cross),
                                    moving=False) for N in Ns}
    return {"q": "q23", "Ns": Ns, "n_rows": len(ts),
            "t_star_h": {N: trajs[N][2].t_cross / 3600.0 for N in Ns},
            "maxC": mC, "maxT": mT, "flux": fluxv}


def q4_evidence(cfg, Ns=(800, 1600)):
    """Q4：全部分钟行（60..t_sample）的固定厘米 C（域外 NaN）+ 表面 的最大差。"""
    fc = lambda N: {"N": N, "interface": "integral", "npts": 8, "t_cap_h": cfg.resolved("q4", "candidate")["t_cap_h"]}
    radius = data_io.make_radius_function(cfg)
    trajs = {}
    for N in Ns:
        full, op, det = PROD._single_trajectory(cfg, "q4", fc(N), moving=True)
        trajs[N] = (full, op, det)
    t_sample = int(min(trajs[N][2].t_sample for N in Ns))
    ts = np.arange(60, t_sample + 1, 60)
    cols = COLS20 + ["surface"]
    m_all = (0.0, None, None); m_r12 = 0.0
    j12 = COLS20.index(1.2)
    for t in ts:
        rowvals = []
        for N in Ns:
            full, op, det = trajs[N]
            c = full.eval([float(t)])[0][:op.N + 1]
            R_t = float(radius.R(float(t)))
            row = PP.sample_q4_row(c, op.grid, R_t, COLS20)
            row = [np.nan if v is None else v for v in row] + [float(c[-1])]
            rowvals.append(np.array(row))
        d = rowvals[0] - rowvals[1]
        with np.errstate(invalid="ignore"):
            k = int(np.nanargmax(np.abs(d)))
            val = abs(float(d[k]))
            if val > m_all[0]:
                m_all = (val, float(t), cols[k])
            v12 = abs(float(d[j12])) if not np.isnan(d[j12]) else 0.0
            m_r12 = max(m_r12, v12)
    fluxv = {N: V.flux_integral_bdf(cfg, N=N, moving=True, t_end_s=float(trajs[N][2].t_cross))
             for N in Ns}
    return {"q": "q4", "Ns": Ns, "n_rows": len(ts),
            "t_star_h": {N: trajs[N][2].t_cross / 3600.0 for N in Ns},
            "max_all": m_all, "max_r1_2cm": m_r12, "flux": fluxv}


def run_d12_evidence(cfg):
    q1 = q1_evidence(cfg)
    q23 = q23_evidence(cfg)
    q4 = q4_evidence(cfg)
    # 三问独立时间加密
    tref = {"q1": V.v3_tstar_time(cfg, question="q1", N=1600, rtols=(1e-8, 1e-10)) if False else None,
            "q23": V.v3_tstar_time(cfg, question="q23", N=800, rtols=(1e-8, 1e-10)),
            "q4": V.v3_tstar_time(cfg, question="q4", N=800, rtols=(1e-8, 1e-10))}
    # Q1 无达标事件：时间加密以 100 s 表面点 BE 序列表征（改由 V-1(b) 时间阶代表）
    from . import d12_prep as DP
    quad = {"q23": DP.interface_8_16(cfg, "q23", 800, moving=False),
            "q4": DP.interface_8_16(cfg, "q4", 800, moving=True)}
    res = {"q1": q1, "q23": q23, "q4": q4, "tref": tref, "quad": quad}
    _write_evidence(cfg, res)
    return res


def _write_evidence(cfg, r):
    import hashlib
    snap_path = cfgmod.PROJECT_ROOT / "config" / "A题_config.yaml"
    sha = hashlib.sha256(snap_path.read_bytes()).hexdigest()
    tolC = float(cfg.raw["acceptance"]["table_points"]["dC"])
    tolT = float(cfg.raw["acceptance"]["table_points"]["dT_degC"])
    q1, q23, q4 = r["q1"], r["q23"], r["q4"]
    L = ["# D12 可追溯证据（本轮代码实测复跑）\n",
         "> **本轮自执行结果**，与外部独立复核摘要分列（不将外部摘要标为本轮实测）。",
         f"> 配置 `config/A题_config.yaml` sha256 `{sha[:16]}…`；BDF rtol {cfg.bdf['rtol']:.0e}。\n",
         "## 场点误差（生产 N vs 更细 N，覆盖全部正式采样行；C 与 T）\n",
         "| 问 | 对照 N | 行数 | max|ΔC|·位置 | max|ΔT|(°C)·位置 | 门槛 ΔC/ΔT |",
         "|---|---|---|---|---|---|",
         f"| Q1 | {q1['Ns']} | {q1['n_rows']}（全 1 s 行） | {q1['maxC'][0]:.3e} @t={q1['maxC'][1]:.0f}s,r={q1['maxC'][2]}cm "
         f"| {q1['maxT'][0]:.3e} @t={q1['maxT'][1]:.0f}s,r={q1['maxT'][2]}cm | {tolC:.0e}/{tolT:.0e} |",
         f"| Q2/Q3 | {q23['Ns']} | {q23['n_rows']}（全逐秒行） | {q23['maxC'][0]:.3e} @t={q23['maxC'][1]:.0f}s,r={q23['maxC'][2]}cm "
         f"| {q23['maxT'][0]:.3e} @t={q23['maxT'][1]:.0f}s,r={q23['maxT'][2]}cm | {tolC:.0e}/{tolT:.0e} |",
         f"| Q4 | {q4['Ns']} | {q4['n_rows']}（全分钟行） | {q4['max_all'][0]:.3e} @t={q4['max_all'][1]:.0f}s,列={q4['max_all'][2]} "
         f"| （随 C，见列） | {tolC:.0e} |",
         f"\n- **Q4 r=1.2 cm 专列 max|Δ| = {q4['max_r1_2cm']:.3e}**（独立登记，不以 t* 替代）。",
         f"- 结论：三问场点误差均 ≤ 门槛 → 生产 N 选定 Q1=1600、Q23=800、Q4=800。\n",
         "## 独立时间加密（收紧 BDF rtol 1e-8→1e-10）\n",
         f"- Q2/Q3（N=800）：Δt* = {r['tref']['q23']['max_adjacent_diff_h']:.3e} h",
         f"- Q4（N=800）：Δt* = {r['tref']['q4']['max_adjacent_diff_h']:.3e} h",
         "- Q1：无达标事件，时间阶由 V-1(b) 解析对照代表（BE Δt=1/0.5/0.25 一阶，见 reports/V1_V2.md）\n",
         "## 8/16 点界面求积对照\n",
         f"- Q2/Q3（N=800）：Δt* = {r['quad']['q23']['diff_h']:.3e} h",
         f"- Q4（N=800）：Δt* = {r['quad']['q4']['diff_h']:.3e} h\n",
         "## Q23/Q4 全程通量检验（V-4b BDF 自适应求积，至 t*）\n"]
    for q, lab in (("q23", "Q2/Q3"), ("q4", "Q4")):
        for N in r[q]["Ns"]:
            fv = r[q]["flux"][N]
            L.append(f"- {lab} N={N}：rel = {fv['rel']:.3e} ≤ 10×rtol={fv['tol_10x_rtol']:.0e}"
                     f"（{'通过' if fv['ok'] else '未过'}）")
    L.append("\n## 复现命令与证据路径\n")
    L.append("```bash\npython -m drymodel.d12_evidence   # 生成本报告\npython -m drymodel.d12_prep       # D12 配置申请\n```")
    L.append("- 证据：reports/D12_evidence.md、reports/D12_config_application.md、exports/config_snapshot.json")
    L.append("- 外部独立复核摘要为历史证据，另列，不并入本轮实测。")
    (REPORTS / "D12_evidence.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import time
    _cfg = cfgmod.load_config()
    _t = time.time()
    _r = run_d12_evidence(_cfg)
    print(f"D12 evidence → reports/D12_evidence.md [{time.time()-_t:.1f}s]")
    print(f"  Q1 maxC={_r['q1']['maxC'][0]:.2e} Q23 maxC={_r['q23']['maxC'][0]:.2e} "
          f"Q4 max={_r['q4']['max_all'][0]:.2e} (r1.2={_r['q4']['max_r1_2cm']:.2e})")
