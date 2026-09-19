"""D12：按实际生产配置独立加密，登记全部正式采样场点的数值证据。

空间只加密 N，时间只收紧 BDF 容差和最大步长，积分界面只加密求积点。
不改变物理参数、题给采样规则或验收阈值。JSON 与 Markdown 来自同一次计算；
任一超差、非有限值或求解失败均不能形成通过报告。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import criterion as CR
from . import data_io
from . import postprocess as PP
from . import production as PROD
from . import runners
from . import solver_bdf as SBDF
from .grid import RadialGrid, RefGrid
from .operators import FVMOperator

REPORTS = cfgmod.PROJECT_ROOT / "reports"
COLS21 = runners.RESULT_COLS_CM
COLS20 = [round(0.1 * j, 4) for j in range(20)]
KELVIN = 273.15


def _bdf_with_rtol(cfg, rtol):
    b = dict(cfg.bdf)
    b["rtol"] = float(rtol)
    return b


def _time_refinement(bdf):
    """独立时间对照：相对/绝对容差均缩至 1/100，两个最大步长均减半。"""
    out = deepcopy(bdf)
    for key in ("rtol", "atol_C", "atol_T_K", "atol_I"):
        if key in out:
            out[key] = float(out[key]) * 0.01
    for key in ("max_step_data_s", "max_step_after_s"):
        out[key] = float(out[key]) * 0.5
    # SciPy 会把更小的 rtol 提高至 100*eps；不能把未生效值写成证据。
    if out["rtol"] < 100.0 * np.finfo(float).eps:
        raise ValueError("时间加密 rtol 低于 SciPy 可执行下限，不能声称已按该容差验证")
    return out


def _settings(cfg, question, N_final, N_finer):
    fc = PROD._final_config(cfg, question)
    if int(N_finer) <= int(N_final):
        raise ValueError("空间对照必须比生产网格更细")
    base = {"N": int(N_final), "interface": fc["interface"],
            "npts": int(fc["npts"]), "bdf": deepcopy(cfg.bdf)}
    spatial = dict(base, N=int(N_finer))
    temporal = dict(base, bdf=_time_refinement(base["bdf"]))
    ncheck = max(2 * base["npts"], int(cfg.numerics["quadrature"]["interface_points_check"]))
    interface = dict(base, npts=ncheck)
    return base, {"spatial": spatial, "time": temporal, "interface": interface}


def _integrate(cfg, question, N, moving, rtol, npts, t_end_s, *, interface=None, bdf=None):
    """保留原位置参数 API；完整 BDF 参数和界面方式显式传入实际求解器。"""
    if interface is None:
        interface = PROD._final_config(cfg, question)["interface"]
    control = _bdf_with_rtol(cfg, rtol) if bdf is None else deepcopy(bdf)
    env = data_io.make_env_functions(cfg, "base")
    props = cfg.props("q4") if moving else cfg.props(question)
    if moving:
        op = FVMOperator(RefGrid(N), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, integral_npts=npts,
                         radius_fn=data_io.make_radius_function(cfg))
    else:
        op = FVMOperator(RadialGrid(N, cfg.R0), props, env, h=cfg.h, hm=cfg.hm, R0=cfg.R0,
                         interface=interface, integral_npts=npts, decoupled=(question == "q1"))
    n = N + 1
    y0 = np.concatenate([np.full(n, cfg.C0), np.full(n, cfg.T0_K)])
    res = SBDF.integrate_bdf(op, y0, 0.0, float(t_end_s), control,
                             breakpoints=tuple(cfg.breakpoints_s))
    if not res.ok:
        raise RuntimeError(f"{question} 积分失败(N={N},rtol={control['rtol']},"
                           f"interface={interface},npts={npts})：{res.message}")
    return res, op


def _integrate_setting(cfg, question, setting, t_end_s):
    return _integrate(cfg, question, setting["N"], question == "q4",
                      setting["bdf"]["rtol"], setting["npts"], t_end_s,
                      interface=setting["interface"], bdf=setting["bdf"])


def _sample_fixed(res, op, ts, want_T, chunk=8000):
    node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
    n = op.N + 1
    # 一次稠密求值最多约 64 MB，避免细网格时形成几百 MB 的临时全场数组。
    chunk = min(chunk, max(1, 8_000_000 // (2 * n)))
    Cc, Tc = [], []
    for i in range(0, len(ts), chunk):
        Y = res.eval(ts[i:i + chunk])
        Cc.append(Y[:, node_idx].copy())
        if want_T:
            Tc.append(Y[:, node_idx + n].copy())
    C = np.vstack(Cc)
    T = (np.vstack(Tc) - KELVIN) if want_T else None
    if not np.all(np.isfinite(C)) or (want_T and not np.all(np.isfinite(T))):
        raise ValueError("固定域正式场点含非有限值")
    return C, T


def _sample_q4(res, op, ts, radius):
    n = op.N + 1
    rows = np.full((len(ts), len(COLS20) + 1), np.nan)
    for i, t in enumerate(ts):
        c = res.eval([float(t)])[0][:n]
        if not np.all(np.isfinite(c)):
            raise ValueError("Q4 正式场点轨迹含非有限值")
        R_t = float(radius.R(float(t)))
        row = PP.sample_q4_row(c, op.grid, R_t, COLS20)
        for j, v in enumerate(row):
            rows[i, j] = np.nan if v is None else v
        rows[i, -1] = c[-1]
    return rows


def _maxloc(diff, ts, cols):
    if np.any(np.isinf(diff)) or not np.any(np.isfinite(diff)):
        raise ValueError("误差数组无有效有限值，不能通过验收")
    k = int(np.nanargmax(np.abs(diff)))
    i, j = np.unravel_index(k, diff.shape)
    return abs(float(diff[i, j])), float(ts[i]), (cols[j] if isinstance(cols[j], str) else float(cols[j]))


def _label(key, base, check):
    if key == "spatial":
        return f"N={base['N']} vs {check['N']}"
    if key == "interface":
        return f"{base['npts']}点 vs {check['npts']}点（{base['interface']}）"
    b, c = base["bdf"], check["bdf"]
    return (f"rtol {b['rtol']:.3g} vs {c['rtol']:.3g}；绝对容差×0.01；"
            f"max_step {b['max_step_data_s']:g}/{b['max_step_after_s']:g} s vs "
            f"{c['max_step_data_s']:g}/{c['max_step_after_s']:g} s")


def _not_applicable(base):
    return {"applicable": False, "pass": None, "status": "not_applicable",
            "reason": f"生产界面为 {base['interface']}，不使用积分界面求积点；未执行求积加密"}


def _event_time(cfg, res, op):
    """在同一未舍入稠密轨迹上求全节点最大值的阈值根；未覆盖时明确失败。"""
    n = op.N + 1
    start, end = float(res.segments[0].t[0]), float(res.segments[-1].t[-1])

    def cmax(t):
        c = res.eval([float(t)])[0][:n]
        if not np.all(np.isfinite(c)):
            raise ValueError("事件轨迹含非有限浓度")
        return float(np.max(c))

    if cmax(start) <= cfg.threshold or cmax(end) > cfg.threshold:
        return {"covered": False, "t_star_s": None, "interval_s": [start, end],
                "reason": "该积分区间未覆盖从初态到达标阈值的根；不能通过事件精度验收"}
    return {"covered": True, "t_star_s": float(CR.locate_cross_dense(cmax, start, end, cfg.threshold)),
            "interval_s": [start, end]}


def _event_comparison(cfg, baseline, comparison):
    tol = float(cfg.raw["acceptance"]["t_star_h"])
    covered = baseline["covered"] and comparison["covered"]
    delta_h = abs(baseline["t_star_s"] - comparison["t_star_s"]) / 3600.0 if covered else None
    return {"baseline": baseline, "comparison": comparison, "delta_h": delta_h,
            "tol_h": tol, "pass": bool(covered and delta_h <= tol)}


def evidence_fixed(cfg, question, N_final, N_finer, ts, tolC, tolT, *, flux_t_end_s=None):
    """Q1/Q23：沿用函数参数，所有未显式覆盖的控制取自实际生产配置。"""
    base_setting, checks = _settings(cfg, question, N_final, N_finer)
    base, opb = _integrate_setting(cfg, question, base_setting, ts[-1])
    Cb, Tb = _sample_fixed(base, opb, ts, want_T=True)
    base_event = _event_time(cfg, base, opb) if question == "q23" else None
    flux = {}
    if flux_t_end_s is not None:
        flux[N_final] = _flux_evidence(cfg, question, base_setting, flux_t_end_s,
                                       trajectory=(base, opb))
    print(f"D12 {question} baseline sampled ({len(ts)} rows)", flush=True)
    del base, opb
    out = {}
    for key, setting in checks.items():
        if key == "interface" and base_setting["interface"] != "integral":
            out[key] = _not_applicable(base_setting)
            continue
        cmp_, opc = _integrate_setting(cfg, question, setting, ts[-1])
        Cc, Tc = _sample_fixed(cmp_, opc, ts, want_T=True)
        maxC, maxT = _maxloc(Cb - Cc, ts, COLS21), _maxloc(Tb - Tc, ts, COLS21)
        out[key] = {"maxC": maxC, "maxT": maxT, "vs": _label(key, base_setting, setting),
                    "baseline": base_setting, "comparison": setting, "applicable": True,
                    "pass_C": bool(maxC[0] <= tolC), "pass_T": bool(maxT[0] <= tolT),
                    "pass": bool(maxC[0] <= tolC and maxT[0] <= tolT)}
        if base_event is not None:
            out[key]["event"] = _event_comparison(cfg, base_event, _event_time(cfg, cmp_, opc))
            out[key]["pass"] = out[key]["pass"] and out[key]["event"]["pass"]
        if key == "spatial" and flux_t_end_s is not None:
            flux[N_finer] = _flux_evidence(cfg, question, setting, flux_t_end_s,
                                           trajectory=(cmp_, opc))
        print(f"D12 {question} {key}: pass={out[key]['pass']}", flush=True)
        del cmp_, opc, Cc, Tc
    return {"q": question, "N_final": N_final, "n_rows": len(ts), "items": out,
            "baseline": base_setting, "tolC": tolC, "tolT": tolT, "flux": flux,
            "event_baseline": base_event,
            "pass": all(o["pass"] for o in out.values() if o["applicable"])}


def evidence_q4(cfg, N_final, N_finer, ts, tolC, *, flux_t_end_s=None):
    base_setting, checks = _settings(cfg, "q4", N_final, N_finer)
    radius = data_io.make_radius_function(cfg)
    cols = COLS20 + ["surface"]
    base, opb = _integrate_setting(cfg, "q4", base_setting, ts[-1])
    Rb = _sample_q4(base, opb, ts, radius)
    base_event = _event_time(cfg, base, opb)
    flux = {}
    if flux_t_end_s is not None:
        flux[N_final] = _flux_evidence(cfg, "q4", base_setting, flux_t_end_s,
                                       trajectory=(base, opb))
    print(f"D12 q4 baseline sampled ({len(ts)} rows)", flush=True)
    del base, opb
    out = {}
    for key, setting in checks.items():
        if key == "interface" and base_setting["interface"] != "integral":
            out[key] = _not_applicable(base_setting)
            continue
        cmp_, opc = _integrate_setting(cfg, "q4", setting, ts[-1])
        Rc = _sample_q4(cmp_, opc, ts, radius)
        if not np.array_equal(np.isnan(Rb), np.isnan(Rc)):
            raise ValueError("Q4 生产/加密轨迹的域外留空位置不一致")
        d = Rb - Rc
        m = _maxloc(d, ts, cols)
        j12 = COLS20.index(1.2)
        valid12 = np.isfinite(d[:, j12])
        d12 = float(np.max(np.abs(d[valid12, j12]))) if np.any(valid12) else None
        out[key] = {"max": m, "r1_2cm": d12, "vs": _label(key, base_setting, setting),
                    "baseline": base_setting, "comparison": setting, "applicable": True,
                    "pass": bool(m[0] <= tolC)}
        out[key]["event"] = _event_comparison(cfg, base_event, _event_time(cfg, cmp_, opc))
        out[key]["pass"] = out[key]["pass"] and out[key]["event"]["pass"]
        if key == "spatial" and flux_t_end_s is not None:
            flux[N_finer] = _flux_evidence(cfg, "q4", setting, flux_t_end_s,
                                           trajectory=(cmp_, opc))
        print(f"D12 q4 {key}: pass={out[key]['pass']}", flush=True)
        del cmp_, opc, Rc, d
    return {"q": "q4", "N_final": N_final, "n_rows": len(ts), "items": out,
            "baseline": base_setting, "tolC": tolC, "flux": flux,
            "event_baseline": base_event,
            "pass": all(o["pass"] for o in out.values() if o["applicable"])}


def _flux_evidence(cfg, question, setting, t_end_s, *, trajectory=None):
    """从已有轨迹独立对连续通量求积；不启用累计通量辅助态、不额外重积分。"""
    import warnings
    from scipy.integrate import IntegrationWarning, quad

    res, op = trajectory if trajectory is not None else _integrate_setting(cfg, question, setting, t_end_s)
    n = op.N + 1
    dCbar = float(op.grid.cbar(res.eval([t_end_s])[0][:n]) - cfg.C0)
    # 在附件插值节点、半径节点及 BDF 接受节点处分段，避开分段函数的折点。
    knots = [np.asarray([0.0, t_end_s]), np.asarray(op.env.t_nodes),
             np.asarray(cfg.breakpoints_s), *[seg.t for seg in res.segments]]
    if op.is_reference:
        knots.append(np.asarray(op.radius_fn.t_nodes))
    edges = np.unique(np.concatenate(knots))
    edges = edges[(edges >= 0.0) & (edges <= t_end_s)]
    factor = float(cfg.raw["acceptance"]["flux_integral"]["bdf"]["tol_rel_factor_of_rtol"])
    tol = factor * float(setting["bdf"]["rtol"])
    epsrel = min(1e-9, 0.1 * tol)
    epsabs = min(1e-12, epsrel * max(abs(dCbar), 1e-300)) / (len(edges) - 1)

    def integral(rel):
        total, estimated_error = 0.0, 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            for a, b in zip(edges[:-1], edges[1:]):
                val, err = quad(lambda t: op.flux_cbar(t, res.eval([t])[0][:n]),
                                float(a), float(b), epsabs=epsabs, epsrel=rel, limit=100)
                total += val
                estimated_error += err
        return total, estimated_error, sorted({str(w.message) for w in caught})

    Iq, err1, warning1 = integral(epsrel)
    Iq2, err2, warning2 = integral(epsrel * 0.01)
    scale = max(abs(dCbar), 1e-300)
    rel, rel_refine = abs(Iq + dCbar) / scale, abs(Iq2 - Iq) / scale
    finite = bool(np.all(np.isfinite([Iq, Iq2, rel, rel_refine, err1, err2])))
    if not finite:
        raise ValueError("独立通量求积产生非有限结果")
    ok_balance = bool(rel <= tol and abs(Iq2 + dCbar) / scale <= tol)
    ok_refine = bool(rel_refine <= tol and max(err1, err2) / scale <= tol)
    return {"rel": float(rel), "rel_refine": float(rel_refine), "tol_10x_rtol": tol,
            "tolerance_factor_of_rtol": factor, "tol_rel": tol,
            "ok": bool(ok_balance and ok_refine and not warning1 and not warning2),
            "pass_balance": ok_balance, "pass_refinement": ok_refine,
            "I_quad": float(Iq), "I_quad_refined": float(Iq2), "dCbar": dCbar,
            "quadrature_error_estimates": [float(err1), float(err2)],
            "quadrature_epsrel": [epsrel, epsrel * 0.01],
            "quadrature_warnings": sorted(set(warning1 + warning2)),
            "quadrature_intervals": len(edges) - 1, "moving": question == "q4",
            "N": setting["N"], "t_end_s": float(t_end_s), "configuration": setting,
            "note": "独立连续通量自适应求积，收紧求积容差复核；非同一 RHS 代数恒等式"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(cfg):
    effective = json.dumps(cfg.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    code = sorted((cfgmod.PROJECT_ROOT / "src" / "drymodel").glob("*.py"))
    return {"config_path": str(cfg.source_path) if cfg.source_path else None,
            "config_file_sha256": _sha(cfg.source_path) if cfg.source_path else None,
            "effective_config_sha256": hashlib.sha256(effective).hexdigest(),
            "effective_config": deepcopy(cfg.raw),
            "code_sha256": {p.relative_to(cfgmod.PROJECT_ROOT).as_posix(): _sha(p) for p in code},
            "input_sha256": {str(p): _sha(p) for p in (cfg.air_file(), cfg.radius_file())},
            "software": {"python": platform.python_version(), "platform": platform.platform(),
                         **{name: version(name) for name in ("numpy", "scipy", "PyYAML", "openpyxl")}}}


def run_d12_evidence(cfg, output_dir=None):
    """无参数路径保持兼容；--output-dir 可隔离赛后报告，不覆盖历史证据。"""
    outdir = Path(output_dir) if output_dir is not None else REPORTS
    acc = cfg.raw["acceptance"]["table_points"]
    tolC, tolT = float(acc["dC"]), float(acc["dT_degC"])
    final = {q: PROD._final_config(cfg, q) for q in ("q1", "q23", "q4")}
    r = {"schema_version": 2, "started_at_utc": datetime.now(timezone.utc).isoformat(),
         "provenance": _provenance(cfg), "production_config": final,
         "all_ok": False, "complete": False, "flux": {}, "sampling": {}}
    started = time.perf_counter()
    try:
        detections = {}
        times = {"q1": np.arange(1, 1801, dtype=float)}
        for q in ("q23", "q4"):
            fc = final[q]
            det, *_ = runners.q23_detect(cfg, question=q, N=fc["N"],
                                         interface=fc["interface"], npts=fc["npts"],
                                         t_cap_h=fc["t_cap_h"])
            print(f"D12 {q} detection: t*={det.t_cross:.9g}s, t_sample={det.t_sample}", flush=True)
            if (det.t_sample is None or not np.isfinite(det.t_sample)
                    or not np.isfinite(det.t_cross) or det.t_sample <= det.t_cross
                    or not det.post_ok or not np.isfinite(det.post_max_cmax)
                    or det.post_max_cmax > cfg.threshold + 1e-9):
                raise RuntimeError(f"{q} 生产事件/严格合格采样/续算检查失败")
            detections[q] = det
            step = 1 if q == "q23" else 60
            times[q] = np.unique(np.append(np.arange(step, int(det.t_sample) + 1, step,
                                                     dtype=float), float(det.t_cross)))
            r["sampling"][q] = {"t_cross_s": float(det.t_cross), "t_sample_s": float(det.t_sample),
                                  "regular_step_s": step, "event_row_included": True,
                                  "post_ok": bool(det.post_ok), "post_max_cmax": float(det.post_max_cmax)}
        r["sampling"]["q1"] = {"start_s": 1, "end_s": 1800, "regular_step_s": 1,
                               "event_row_included": False}
        for q in ("q1", "q23", "q4"):
            N = final[q]["N"]
            if q == "q4":
                r[q] = evidence_q4(cfg, N, 2 * N, times[q], tolC,
                                     flux_t_end_s=float(detections[q].t_cross))
            else:
                flux_end = float(detections[q].t_cross) if q == "q23" else None
                r[q] = evidence_fixed(cfg, q, N, 2 * N, times[q], tolC, tolT,
                                        flux_t_end_s=flux_end)
            flux = r[q].pop("flux")
            if flux:
                r["flux"][q] = flux
            print(f"D12 {q}: N={N}, pass={r[q]['pass']}", flush=True)
        r["complete"] = True
        r["all_ok"] = bool(all(r[q]["pass"] for q in ("q1", "q23", "q4")) and
                            all(fv["ok"] for rows in r["flux"].values() for fv in rows.values()))
    except Exception as exc:
        r["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        r["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        r["elapsed_seconds"] = time.perf_counter() - started
        _write_evidence(cfg, r, output_dir=outdir)
    return r


def _fmt(m):
    return f"{m[0]:.3e} @t={m[1]:.9g}s,列={m[2]}"


def _write_evidence(cfg, r, output_dir=None):
    outdir = Path(output_dir) if output_dir is not None else REPORTS
    outdir.mkdir(parents=True, exist_ok=True)
    # 严格 JSON 禁止 NaN/Infinity；域外位置不进入误差最大值，仅报告真实有效场点。
    (outdir / "D12_evidence.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    p = r["provenance"]
    L = ["# D12 可追溯证据（本轮代码实测复跑）\n",
         "> 本轮自执行证据；不将历史摘要登记为本轮实测。",
         f"> 实际配置 sha256 `{p['effective_config_sha256']}`；开始 {r['started_at_utc']}。",
         f"> 运行完整：{r['complete']}；总验收：{'通过' if r['all_ok'] else '失败'}。\n",
         "空间对照只将生产网格加倍；时间对照只将 BDF 相对/绝对容差缩至 1/100、最大步长减半；"
         "积分界面求积点数至少加倍。三项分别独立重算，物理配置与验收阈值不变。\n",
         "## 实际生产配置\n",
         "| 问 | final_N | 界面 | 求积点 | BDF rtol | max_step 数据段/外推段 (s) |",
         "|---|---:|---|---:|---:|---|"]
    for q, fc in r["production_config"].items():
        b = cfg.bdf
        L.append(f"| {q} | {fc['N']} | {fc['interface']} | {fc['npts']} | {b['rtol']:.3g} | "
                 f"{b['max_step_data_s']:g}/{b['max_step_after_s']:g} |")
    L.extend(["\n## 验证覆盖范围\n",
              "Q1：C 与 T，r=0–2.0 cm 共 21 列，全 1 s 行 1–1800 s。",
              "Q2/Q3：C 与 T，同 21 列，逐秒覆盖至 result3 末行 t_sample，并显式并入事件 t*。",
              "Q4：仅 C，固定 0–1.9 cm 与药材表面，全 60 s 行至 t_sample，并入事件 t*；域外留空。",
              "Q4 温度参与耦合求解，但不属于 result4 输出场验收。\n"])
    for q, sampling in r["sampling"].items():
        if "t_cross_s" in sampling:
            L.append(f"- {q}：t*={sampling['t_cross_s']:.12g} s；"
                     f"t_sample={sampling['t_sample_s']:.12g} s；包含事件行。")
    L.append("\n## 逐项独立对照\n")
    for question in ("q1", "q23", "q4"):
        if question not in r:
            L.append(f"### {question}：未完成，不能通过\n")
            continue
        q = r[question]
        L.append(f"### {question}：final_N={q['N_final']}，覆盖 {q['n_rows']} 行\n")
        L.append(f"浓度门槛 ΔC≤{q['tolC']:.3g}；" +
                 (f"温度门槛 ΔT≤{q['tolT']:.3g} °C。" if question != "q4" else "仅浓度验收。"))
        L.extend(["| 对照 | 最大浓度误差·位置 | 温度误差·位置 / r=1.2cm 误差 | 判定 |",
                  "|---|---|---|---|"])
        for key, o in q["items"].items():
            if not o["applicable"]:
                L.append(f"| {key} | — | {o['reason']} | 不适用（未执行） |")
                continue
            if question == "q4":
                first = _fmt(o["max"])
                second = f"{o['r1_2cm']:.3e}" if o["r1_2cm"] is not None else "无域内采样"
            else:
                first, second = _fmt(o["maxC"]), _fmt(o["maxT"])
            L.append(f"| {o['vs']} | {first} | {second} | {'通过' if o['pass'] else '失败'} |")
        L.append(f"\n{question} 判定：{'通过' if q['pass'] else '失败'}。\n")
        if q.get("event_baseline") is not None:
            L.extend(["| 事件对照 | 基准 t* (s) | 对照 t* (s) | 误差 (h) | 门槛 (h) | 判定 |",
                      "|---|---:|---:|---:|---:|---|"])
            for key, o in q["items"].items():
                if not o["applicable"]:
                    continue
                ev = o["event"]
                values = [ev["baseline"]["t_star_s"], ev["comparison"]["t_star_s"], ev["delta_h"]]
                a, b, d = [f"{v:.12g}" if v is not None else "未覆盖" for v in values]
                L.append(f"| {key} | {a} | {b} | {d} | {ev['tol_h']:.3g} | "
                         f"{'通过' if ev['pass'] else '失败'} |")
    L.append("## 全程通量检验（独立连续通量求积至 t*）\n")
    for question, rows in r["flux"].items():
        for N, fv in rows.items():
            L.append(f"- {question} N={N}：收支相对误差 {fv['rel']:.3e}；"
                     f"求积加密差 {fv['rel_refine']:.3e}；门槛 {fv['tol_rel']:.3e}；"
                     f"{'通过' if fv['ok'] else '失败'}。")
            if fv["quadrature_warnings"]:
                L.append(f"  求积告警：{'；'.join(fv['quadrature_warnings'])}")
    if "error" in r:
        L.extend(["\n## 未完成原因\n", f"{r['error']['type']}: {r['error']['message']}"])
    L.extend(["\n## 复现与追溯\n", "```text",
              f'python -m drymodel.d12_evidence --output-dir "{outdir}"' +
              (f' --config "{cfg.source_path}"' if cfg.source_path else ""), "```",
              "同目录 D12_evidence.json 保存实际配置全文、每项基准/对照参数、最大误差及其时空位置、"
              "逐项 pass/fail、Python/依赖版本、全部模型源文件与输入附件 SHA-256。",
              "配置文件哈希和实际内存配置哈希分列；若调用方更改内存配置，复现时应使用 JSON 中"
              "provenance.effective_config，不能直接沿用原配置文件。",
              f"本次耗时 {r['elapsed_seconds']:.3f} s。"])
    (outdir / "D12_evidence.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="使用指定 YAML；默认项目正式配置")
    parser.add_argument("--output-dir", type=Path, default=REPORTS, help="证据报告目录，默认 reports")
    args = parser.parse_args(argv)
    try:
        r = run_d12_evidence(cfgmod.load_config(args.config), output_dir=args.output_dir)
    except Exception as exc:
        print(f"D12 evidence 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"D12 evidence → {args.output_dir / 'D12_evidence.md'}；all_ok={r['all_ok']}")
    return 0 if r["all_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
