"""d12_evidence.py —— D12 可追溯证据（**本轮代码实测复跑**，非外部摘要转录）。

对每问在**全部正式采样场点**上分别登记（生产配置 vs 更细对照，逐项与 acceptance 门槛比较）：
  - 空间：生产 N vs 更细 N（同 rtol、同求积点）；
  - 独立时间加密：生产 N，rtol 1e-8 vs 1e-10（**Q1 亦用 BDF 场点，不用 BE 时间阶代替**）；
  - 8/16 点界面求积：生产 N，8 点 vs 16 点；
  - Q23/Q4 全程通量（V-4b BDF 自适应求积，至 t*）。
**逐项 pass/fail；任一超差或缺项 → 如实报告并失败退出（非零）。** 外部审计与本轮自执行分列。
内存：Q23 逐秒场样存为 21 列数组（~35 MB），比较项分块 running-max。
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
from .grid import RadialGrid, RefGrid
from .operators import FVMOperator

REPORTS = cfgmod.PROJECT_ROOT / "reports"
COLS21 = runners.RESULT_COLS_CM
COLS20 = [round(0.1 * j, 4) for j in range(20)]
KELVIN = 273.15


def _bdf_with_rtol(cfg, rtol):
    b = dict(cfg.bdf); b["rtol"] = rtol
    return b


def _integrate(cfg, question, N, moving, rtol, npts, t_end_s):
    """在 (N, rtol, npts) 下积分到 t_end_s（无事件），返回 (BDFResult, op)。"""
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4") if moving else cfg.props(question)
    if moving:
        op = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface="integral", integral_npts=npts,
                         radius_fn=data_io.make_radius_function(cfg))
    else:
        op = FVMOperator(RadialGrid(N, cfg.R0), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface="integral", integral_npts=npts,
                         decoupled=(question == "q1"))
    n = N + 1
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
    res = SBDF.integrate_bdf(op, y0, 0.0, t_end_s, _bdf_with_rtol(cfg, rtol), breakpoints=(14400.0,))
    if not res.ok:
        raise RuntimeError(f"{question} 积分失败(N={N},rtol={rtol},npts={npts})：{res.message}")
    return res, op


def _sample_fixed(res, op, ts, want_T, chunk=8000):
    node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
    n = op.N + 1
    Cc, Tc = [], []
    for i in range(0, len(ts), chunk):
        Y = res.eval(ts[i:i + chunk])
        Cc.append(Y[:, node_idx].copy())
        if want_T:
            Tc.append(Y[:, node_idx + n].copy())
    C = np.vstack(Cc)
    T = (np.vstack(Tc) - KELVIN) if want_T else None
    return C, T


def _sample_q4(res, op, ts, radius):
    n = op.N + 1
    rows = np.full((len(ts), len(COLS20) + 1), np.nan)
    for i, t in enumerate(ts):
        c = res.eval([float(t)])[0][:n]
        R_t = float(radius.R(float(t)))
        row = PP.sample_q4_row(c, op.grid, R_t, COLS20)
        for j, v in enumerate(row):
            rows[i, j] = np.nan if v is None else v
        rows[i, -1] = c[-1]
    return rows


def _maxloc(diff, ts, cols):
    with np.errstate(invalid="ignore"):
        k = int(np.nanargmax(np.abs(diff)))
    i, j = np.unravel_index(k, diff.shape)
    return abs(float(diff[i, j])), float(ts[i]), (cols[j] if isinstance(cols[j], str) else float(cols[j]))


def evidence_fixed(cfg, question, N_final, N_finer, ts, tolC, tolT):
    """Q1/Q23：固定域场点 C、T 的 空间/时间/8-16 逐项对照 + pass/fail。"""
    base, opb = _integrate(cfg, question, N_final, False, 1e-8, 8, ts[-1])
    Cb, Tb = _sample_fixed(base, opb, ts, want_T=True)
    out = {}
    # 空间：更细 N
    sp, ops = _integrate(cfg, question, N_finer, False, 1e-8, 8, ts[-1])
    Cs, Ts = _sample_fixed(sp, ops, ts, want_T=True)
    out["spatial"] = {"maxC": _maxloc(Cb - Cs, ts, COLS21), "maxT": _maxloc(Tb - Ts, ts, COLS21),
                      "vs": f"N={N_final} vs {N_finer}"}
    del sp, Cs, Ts
    # 时间：rtol 1e-10（BDF 场点，Q1 亦然）
    tm, opt = _integrate(cfg, question, N_final, False, 1e-10, 8, ts[-1])
    Ct, Tt = _sample_fixed(tm, opt, ts, want_T=True)
    out["time"] = {"maxC": _maxloc(Cb - Ct, ts, COLS21), "maxT": _maxloc(Tb - Tt, ts, COLS21),
                   "vs": "rtol 1e-8 vs 1e-10"}
    del tm, Ct, Tt
    # 界面：16 点
    it, opi = _integrate(cfg, question, N_final, False, 1e-8, 16, ts[-1])
    Ci, Ti = _sample_fixed(it, opi, ts, want_T=True)
    out["interface"] = {"maxC": _maxloc(Cb - Ci, ts, COLS21), "maxT": _maxloc(Tb - Ti, ts, COLS21),
                        "vs": "8点 vs 16点"}
    # pass/fail：所有对照 maxC≤tolC 且 maxT≤tolT
    ok = all(o["maxC"][0] <= tolC and o["maxT"][0] <= tolT for o in out.values())
    return {"q": question, "N_final": N_final, "n_rows": len(ts), "items": out,
            "tolC": tolC, "tolT": tolT, "pass": ok}


def evidence_q4(cfg, N_final, N_finer, ts, tolC):
    radius = data_io.make_radius_function(cfg)
    cols = COLS20 + ["surface"]
    base, opb = _integrate(cfg, "q4", N_final, True, 1e-8, 8, ts[-1])
    Rb = _sample_q4(base, opb, ts, radius)
    out = {}
    for key, (N, rtol, npts, lab) in {
            "spatial": (N_finer, 1e-8, 8, f"N={N_final} vs {N_finer}"),
            "time": (N_final, 1e-10, 8, "rtol 1e-8 vs 1e-10"),
            "interface": (N_final, 1e-8, 16, "8点 vs 16点")}.items():
        cmp_, opc = _integrate(cfg, "q4", N, True, rtol, npts, ts[-1])
        Rc = _sample_q4(cmp_, opc, ts, radius)
        d = Rb - Rc
        m = _maxloc(d, ts, cols)
        j12 = COLS20.index(1.2)
        with np.errstate(invalid="ignore"):
            d12 = float(np.nanmax(np.abs(d[:, j12])))
        out[key] = {"max": m, "r1_2cm": d12, "vs": lab}
        del cmp_, Rc, d
    ok = all(o["max"][0] <= tolC for o in out.values())
    return {"q": "q4", "N_final": N_final, "n_rows": len(ts), "items": out,
            "tolC": tolC, "pass": ok}


def run_d12_evidence(cfg):
    acc = cfg.raw["acceptance"]["table_points"]
    tolC, tolT = float(acc["dC"]), float(acc["dT_degC"])
    # 采样时间轴 = 官方正式采样行
    ts_q1 = np.arange(1, 1801)
    # Q23/Q4 至 t_sample（由生产轨迹）
    det23, *_ = runners.q23_detect(cfg, N=800, interface="integral", npts=8, question="q23", t_cap_h=90.0)
    det4, *_ = runners.q23_detect(cfg, N=800, interface="integral", npts=8, question="q4", t_cap_h=200.0)
    t_end1s = int(np.ceil(det23.t_cross))
    ts_q23 = np.arange(1, t_end1s + 1)                 # result2 全逐秒行
    ts_q4 = np.arange(60, int(det4.t_sample) + 1, 60)  # result4 全分钟行

    q1 = evidence_fixed(cfg, "q1", 1600, 3200, ts_q1, tolC, tolT)
    q23 = evidence_fixed(cfg, "q23", 800, 1600, ts_q23, tolC, tolT)
    q4 = evidence_q4(cfg, 800, 1600, ts_q4, tolC)
    flux = {"q23": {N: V.flux_integral_bdf(cfg, question="q23", N=N, t_end_s=float(det23.t_cross))
                    for N in (800, 1600)},
            "q4": {N: V.flux_integral_bdf(cfg, N=N, moving=True, t_end_s=float(det4.t_cross))
                   for N in (800, 1600)}}
    res = {"q1": q1, "q23": q23, "q4": q4, "flux": flux}
    _write_evidence(cfg, res)
    all_ok = q1["pass"] and q23["pass"] and q4["pass"] and \
        all(fv["ok"] for q in flux.values() for fv in q.values())
    res["all_ok"] = all_ok
    return res


def _fmt(m):
    return f"{m[0]:.3e} @t={m[1]:.0f}s,列={m[2]}"


def _write_evidence(cfg, r):
    import hashlib
    sha = hashlib.sha256((cfgmod.PROJECT_ROOT / "config" / "A题_config.yaml").read_bytes()).hexdigest()
    q1, q23, q4 = r["q1"], r["q23"], r["q4"]
    L = ["# D12 可追溯证据（本轮代码实测复跑）\n",
         "> **本轮自执行结果**，与外部独立复核摘要分列（不将外部摘要标为本轮实测）。",
         f"> 配置 sha256 `{sha[:16]}…`；BDF rtol {cfg.bdf['rtol']:.0e}；门槛 ΔC={q1['tolC']:.0e}、ΔT={q1['tolT']:.0e} °C。\n",
         "## 各问验证覆盖范围（如实）\n",
         "| 问 | 验证场量 | 位置 | 时间范围 |", "|---|---|---|---|",
         f"| Q1 | **温度 与 水分浓度**（result1 两工作表） | r=0–2.0 cm（21 列） | 全 1 s 行 1–1800 s |",
         f"| Q2/Q3 | **温度 与 水分浓度**（result2 两工作表） | r=0–2.0 cm（21 列） | 全逐秒行 1–t_end,1s |",
         f"| Q4 | **仅水分浓度**（result4；温度求解参与 D(C,T) 耦合但不写入 result4，未作输出场验证） | 固定 0–1.9 cm+表面（域外留空） | 全 60 s 行 60–t_sample,4 |",
         "\n> 注：跨文件 result3 覆盖至 t_sample,3=206940 s（末行 = 严格合格采样时刻）；表 5/6 末行对应事件行 t*。\n",
         "## 逐项对照（全部正式采样场点；每项 pass/fail）\n"]
    for q, lab, rows in ((q1, "Q1(附录2)", q1["n_rows"]), (q23, "Q2/Q3(附录3)", q23["n_rows"])):
        L.append(f"### {lab}：final_N={q['N_final']}，覆盖 {rows} 行（**C 与 T** 场点）\n")
        L.append("| 对照 | max|ΔC|·位置 | max|ΔT|(°C)·位置 | 判定 |")
        L.append("|---|---|---|---|")
        for key, o in q["items"].items():
            ok = o["maxC"][0] <= q["tolC"] and o["maxT"][0] <= q["tolT"]
            L.append(f"| {o['vs']} | {_fmt(o['maxC'])} | {_fmt(o['maxT'])} | {'✅过' if ok else '❌超差'} |")
        L.append(f"\n{lab} 总判定：{'✅ 通过' if q['pass'] else '❌ 失败'}\n")
    L.append(f"### Q4(附录4)：final_N={q4['N_final']}，覆盖 {q4['n_rows']} 行（**仅水分浓度**：固定厘米+表面，域外 NaN）\n")
    L.append("| 对照 | max|Δ|·位置 | r=1.2cm max|Δ| | 判定 |")
    L.append("|---|---|---|---|")
    for key, o in q4["items"].items():
        ok = o["max"][0] <= q4["tolC"]
        L.append(f"| {o['vs']} | {_fmt(o['max'])} | {o['r1_2cm']:.3e} | {'✅过' if ok else '❌超差'} |")
    L.append(f"\nQ4 总判定：{'✅ 通过' if q4['pass'] else '❌ 失败'}；**r=1.2 cm 独立登记，不以 t* 替代**。\n")
    L.append("## Q23/Q4 全程通量检验（V-4b BDF 自适应求积，至 t*）\n")
    for q, lab in (("q23", "Q2/Q3"), ("q4", "Q4")):
        for N, fv in r["flux"][q].items():
            L.append(f"- {lab} N={N}：rel {fv['rel']:.3e} ≤ 10×rtol={fv['tol_10x_rtol']:.0e}"
                     f"（{'✅过' if fv['ok'] else '❌超差'}）")
    L.append("\n## 复现命令与证据路径\n")
    L.append("```bash\npython -m drymodel.d12_evidence   # 生成本报告（任一超差→非零退出）\n"
             "python -m drymodel.d12_prep       # D12 配置申请\n```")
    L.append("- 证据：reports/D12_evidence.md、reports/D12_config_application.md、"
             "outputs/production_receipt.json、exports/config_snapshot.json")
    L.append("- 外部独立复核摘要为历史证据，另列，不并入本轮实测。")
    (REPORTS / "D12_evidence.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    import time
    _cfg = cfgmod.load_config()
    _t = time.time()
    _r = run_d12_evidence(_cfg)
    print(f"D12 evidence → reports/D12_evidence.md [{time.time()-_t:.1f}s]；all_ok={_r['all_ok']}")
    for _q in ("q1", "q23", "q4"):
        print(f"  {_q}: pass={_r[_q]['pass']}")
    sys.exit(0 if _r["all_ok"] else 2)     # 任一超差 → 非零退出
