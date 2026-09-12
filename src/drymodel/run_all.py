"""候选计算、结构化验证与报告编排。

正式 result1–4 仍受 D12 授权和未实现的生产编排门控。本模块不会因为
``production.approved`` 的布尔值而宣称文件已生成或验证已通过。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import openpyxl
import scipy
import yaml

from . import config as cfgmod
from . import data_io
from . import runners
from . import sensitivity
from . import verify as V

REPORTS = cfgmod.PROJECT_ROOT / "reports"
EXPORTS = cfgmod.PROJECT_ROOT / "exports"
REPORTS.mkdir(exist_ok=True)
EXPORTS.mkdir(exist_ok=True)


def _log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _status_cn(status):
    return {
        "pass": "已实测通过",
        "fail": "仍失败",
        "partial": "局部通过",
        "not_run": "仍未执行",
    }.get(status, str(status))


def _next_candidate(cfg, question, current):
    candidates = cfg.raw["numerics"]["per_question"][question]["candidate_N"]
    finer = [value for value in candidates if value > current]
    if not finer:
        raise ValueError(f"{question}: 候选 N={current} 没有更细网格可作独立空间对照")
    return min(finer)


def step0_env(cfg, *, N_q1, N_long):
    _log("步0 配置校验 + 环境/半径函数 + 有效配置快照")
    t, T, C = data_io.load_attachment1(cfg.air_file())
    w0, w1 = cfg.air_window_s
    window = (t >= w0) & (t <= w1)
    runs = {
        "q1": cfg.resolve_run("q1", N=N_q1),
        "q23": cfg.resolve_run("q23", N=N_long, augmented=True),
        "q4": cfg.resolve_run("q4", N=N_long, augmented=True),
    }
    effective = {
        question: {"digest": run.digest, **run.snapshot()}
        for question, run in runs.items()
    }
    source = Path(cfg.source_path or cfgmod.DEFAULT_CONFIG_PATH)
    mirror = cfgmod.PROJECT_ROOT / "建模方案v1.1" / "A题_config.yaml"
    mirror_exists = mirror.is_file()
    configs_identical = mirror_exists and source.read_bytes() == mirror.read_bytes()
    config_record = V.check_record(
        "C-1-config", "global", "pass" if configs_identical else "fail",
        metric={
            "authoritative_config_validated": True,
            "mirror_exists": bool(mirror_exists),
            "yaml_files_byte_identical": bool(configs_identical),
            "effective_config_digests": {
                question: run.digest for question, run in runs.items()
            },
        },
        limit={
            "authoritative_config_validated_required": True,
            "mirror_exists_required": True,
            "yaml_files_byte_identical_required": True,
        },
        config={"source": str(source), "mirror": str(mirror)},
        coverage={
            "questions": list(effective),
            "fields": sorted(next(iter(effective.values())).keys()),
        },
        message=("" if configs_identical else "权威 YAML 与方案目录镜像不一致或镜像缺失"),
    )
    info = {
        "air_points": int(len(t)),
        "window_s": [w0, w1],
        "window_points": int(np.sum(window)),
        "T_air_const_C": float(np.mean(T[window])),
        "C_env_const": float(np.mean(C[window])),
        "breakpoints_s": cfg.breakpoints_s,
        "R0_m": cfg.R0,
        "L_m": cfg.L,
        "T0_K": cfg.T0_K,
        "C0": cfg.C0,
        "h": cfg.h,
        "hm": cfg.hm,
        "effective_candidate_configs": effective,
        "production_approved": cfg.raw["production"]["approved"],
        "record": config_record,
    }
    _write_json(EXPORTS / "env_check.json", info)
    _log(f"  61点均值 T={info['T_air_const_C']:.10f}°C C={info['C_env_const']:.13f}")
    return info


def step1_analytic(cfg):
    _log("步1 V-1/V-2：空间解析解与 BE 时间阶分栏")
    suite = V.analytic_records(cfg)
    by_id = {record["check_id"]: record for record in suite["records"]}
    v1, v2 = suite["v1"], suite["v2"]
    lines = [
        "# V-1 / V-2 解析解对照",
        "",
        "> 空间与时间证据分别计算、分别判定；所有状态由下列实测指标生成。",
        "",
        "| 检查 | 状态 | 收敛阶 |",
        "|---|---|---|",
    ]
    for check_id in ("V-1-spatial", "V-1-temporal", "V-2-spatial", "V-2-temporal"):
        record = by_id[check_id]
        lines.append(
            f"| {check_id} | {_status_cn(record['status'])} | "
            f"{', '.join(f'{value:.4f}' for value in record['metric']['orders'])} |"
        )
    lines += [
        "",
        f"## V-1 热空间对照（Bi={v1['Bi']:.6f}, Fo={v1['Fo']:.8f}）",
        "",
        "| N | 表面误差 / °C | 全域最大误差 / °C |",
        "|---:|---:|---:|",
    ]
    for N, row in v1["by_N"].items():
        lines.append(f"| {N} | {row['surface_err_C']:.8e} | {row['max_err_C']:.8e} |")
    lines += [
        "",
        f"## V-2 质空间对照（D0={v2['D0']:.8e}, Bi_m={v2['Bi_m']:.6f}）",
        "",
        "| N | 表面误差 | 全域最大误差 |",
        "|---:|---:|---:|",
    ]
    for N, row in v2["by_N"].items():
        lines.append(f"| {N} | {row['surface_err']:.8e} | {row['max_err']:.8e} |")
    (REPORTS / "V1_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return suite


def _be_operator(cfg, question, N, dt, *, interface, decoupled, purpose):
    run = cfg.resolve_run(
        question, N=N, purpose=purpose, interface=interface,
        scheme="backward_euler", dt_be_s=dt,
    )
    op, _ = runners.build_fixed_operator(
        cfg, question, N, decoupled=decoupled, run_config=run,
    )
    return op


def steps23_candidates(cfg, *, N_q1=800, N_long=800,
                       study_Ns=(200, 400, 800), s10_Ns=(200, 400)):
    _log(f"步2 Q1 候选（N={N_q1}）")
    q1 = runners.run_q1(cfg, N=N_q1)

    _log("步2 V-3 早期表面探针（仅作局部证据）")
    v3_probe = runners.v3_early_rows_q1(
        cfg,
        Ns=tuple(cfg.raw["numerics"]["per_question"]["q1"]["candidate_N"]),
        interface="integral",
    )

    _log("步2 Q23 界面×网格时长研究（局部证据）")
    study = runners.q23_interface_grid_study(cfg, Ns=study_Ns)

    _log(f"步2 Q2/Q3 候选（N={N_long}）")
    q23 = runners.q23_candidate(cfg, N=N_long)
    _log(f"步2 Q4 候选（N={N_long}）")
    q4 = runners.q4_candidate(cfg, N=N_long)

    _log("步2 S10 附录4固定半径对照")
    s10 = runners.s10_fixed_radius_study(cfg, Ns=s10_Ns)

    _log("步3 BE V-4a/b/c 真实步长与粗细步对照")
    be = {}
    for dt in (1.0, 0.25):
        op = _be_operator(
            cfg, "q1", 200, dt, interface="harmonic", decoupled=True,
            purpose=f"V-4-BE-dt-{dt:g}",
        )
        be[f"mass_dt_{dt:g}"] = V.mass_balances_be(
            op, runners.initial_state(cfg, 200), 1800.0, dt, cfg,
        )
    for dt in (1.0, 0.5):
        op = _be_operator(
            cfg, "q23", 200, dt, interface="integral", decoupled=False,
            purpose=f"V-4c-BE-dt-{dt:g}",
        )
        be[f"energy_dt_{dt:g}"] = V.energy_residual_be(
            op, runners.initial_state(cfg, 200), 100.0, dt, cfg,
        )

    v10 = V.static_limit_q4(cfg, N=200, interface="integral", t_probe=3600.0)
    return {
        "q1": q1,
        "q23": q23,
        "q4": q4,
        "v3_probe": v3_probe,
        "study": study,
        "s10": s10,
        "be": be,
        "v10": v10,
    }


def _comparison_times_q1(candidate, reference):
    end = min(candidate["t_end"], reference["t_end"])
    return np.arange(1.0, int(end) + 1.0)


def _comparison_times_q23(candidate, reference):
    end = min(candidate["t_end_1s"], reference["t_end_1s"])
    seconds = np.arange(1.0, int(end) + 1.0)
    extras = np.r_[np.asarray(candidate["result3_t"], dtype=float),
                   np.asarray(candidate["table34_ts_h"], dtype=float) * 3600.0,
                   np.asarray(candidate["table5"], dtype=float)[:, 0] * 3600.0]
    return np.unique(np.r_[seconds, extras[extras <= min(candidate["trajectory"].t_end,
                                                         reference["trajectory"].t_end)]])


def _comparison_times_q4(candidate, reference):
    end = min(candidate["t_sample_s"], reference["t_sample_s"])
    minutes = np.arange(60.0, int(end) + 1.0, 60.0)
    extras = np.asarray([row[0] * 3600.0 for row in candidate["table6"]], dtype=float)
    return np.unique(np.r_[minutes, extras[extras <= min(candidate["trajectory"].t_end,
                                                         reference["trajectory"].t_end)]])


def _gate(records, *, production_approved=False):
    required = {
        "q1": [
            ("V-1-spatial", "q1-heat"), ("V-1-temporal", "q1-heat"),
            ("V-2-spatial", "q1-mass"), ("V-2-temporal", "q1-mass"),
            ("V-3-q1-space", "q1"), ("V-3-q1-time", "q1"),
            ("V-5", "q1"),
            ("V-11-q1-quadrature", "q1"),
        ],
        "q23": [
            ("V-3-q23-space", "q23"), ("V-3-q23-time", "q23"),
            ("V-4-BDF", "q23"), ("V-5", "q23"), ("V-6", "q23"),
            ("V-11-q23-quadrature", "q23"),
        ],
        "q4": [
            ("V-3-q4-space", "q4"), ("V-3-q4-time", "q4"),
            ("V-4-BDF", "q4"), ("V-5", "q4"), ("V-6", "q4"),
            ("V-10", "q4"), ("V-13", "q4"),
            ("V-11-q4-quadrature", "q4"),
        ],
    }
    index = {(row["check_id"], row["question"]): row for row in records}
    config_row = index.get(("C-1-config", "global"))
    config_ready = config_row is not None and config_row["status"] == "pass"
    per_question = {}
    for question, keys in required.items():
        checks = []
        for key in keys:
            row = index.get(key)
            checks.append({
                "check_id": key[0], "question": key[1],
                "status": "not_run" if row is None else row["status"],
            })
        per_question[question] = {
            "ready": config_ready and all(row["status"] == "pass" for row in checks),
            "checks": checks,
        }
    return {
        "ready_for_D12_authorization": all(row["ready"] for row in per_question.values()),
        "production_approved": bool(production_approved),
        "configuration_ready": config_ready,
        "per_question": per_question,
    }


def _production_not_run_records():
    """V-8/V-9 only become runnable after formal files have been exported."""
    return [
        V.check_record(
            "V-8", "production", "not_run", metric={}, limit={}, config={},
            coverage={"cross_file_result2_result3": True},
            message="正式 result2/3 未生成；V-8 跨文件一致性按计划留到 D12 后",
        ),
        V.check_record(
            "V-9", "production", "not_run", metric={}, limit={}, config={},
            coverage={
                "formal_outputs": ["result1", "result2", "result3", "result4"],
                "workbook_readback": True,
            },
            message="正式工作簿未生成；V-9 检查器已实现但尚无正式文件可验",
        ),
    ]


def run_candidate_validation(cfg, analytic, data, environment):
    """运行审计要求的空间、时间、求积、收支、包络和事件证据。"""
    q1, q23, q4 = data["q1"], data["q23"], data["q4"]
    refined = dict(cfg.raw["numerics"]["bdf_refined"])
    q1_fine_N = _next_candidate(cfg, "q1", q1["N"])
    long_fine_N = _next_candidate(cfg, "q23", q23["N"])
    q4_fine_N = _next_candidate(cfg, "q4", q4["N"])

    _log(f"步3 V-3 Q1 空间 N={q1['N']}→{q1_fine_N} 与独立时间加密")
    q1_space = runners.run_q1(cfg, N=q1_fine_N, save=False, purpose="space-reference")
    q1_time = runners.run_q1(
        cfg, N=q1["N"], save=False, purpose="time-reference",
        bdf_overrides=refined,
    )
    _log(f"步3 V-3 Q23 空间 N={q23['N']}→{long_fine_N} 与独立时间加密")
    q23_space = runners.q23_candidate(
        cfg, N=long_fine_N, save=False, purpose="space-reference",
    )
    q23_time = runners.q23_candidate(
        cfg, N=q23["N"], save=False, purpose="time-reference",
        bdf_overrides=refined,
    )
    _log(f"步3 V-3 Q4 空间 N={q4['N']}→{q4_fine_N} 与独立时间加密")
    q4_space = runners.q4_candidate(
        cfg, N=q4_fine_N, save=False, purpose="space-reference",
    )
    q4_time = runners.q4_candidate(
        cfg, N=q4["N"], save=False, purpose="time-reference",
        bdf_overrides=refined,
    )

    times_q1_space = _comparison_times_q1(q1, q1_space)
    times_q1_time = _comparison_times_q1(q1, q1_time)
    times_q23_space = _comparison_times_q23(q23, q23_space)
    times_q23_time = _comparison_times_q23(q23, q23_time)
    times_q4_space = _comparison_times_q4(q4, q4_space)
    times_q4_time = _comparison_times_q4(q4, q4_time)

    records = [environment["record"], *analytic["records"]]
    records += [
        V.compare_fixed_trajectories(
            cfg, q1, q1_space, question="q1", check_id="V-3-q1-space",
            comparison="space", times_s=times_q1_space,
        ),
        V.compare_fixed_trajectories(
            cfg, q1, q1_time, question="q1", check_id="V-3-q1-time",
            comparison="time", times_s=times_q1_time,
        ),
        V.compare_fixed_trajectories(
            cfg, q23, q23_space, question="q23", check_id="V-3-q23-space",
            comparison="space", times_s=times_q23_space,
        ),
        V.compare_fixed_trajectories(
            cfg, q23, q23_time, question="q23", check_id="V-3-q23-time",
            comparison="time", times_s=times_q23_time,
        ),
        V.compare_moving_trajectories(
            cfg, q4, q4_space, check_id="V-3-q4-space",
            comparison="space", times_s=times_q4_space,
        ),
        V.compare_moving_trajectories(
            cfg, q4, q4_time, check_id="V-3-q4-time",
            comparison="time", times_s=times_q4_time,
        ),
    ]

    _log("步3 V-4 BDF 增广收支 + 独立 Simpson 通量求积")
    q23_balance = V.bdf_mass_balance_record(
        cfg, q23, question="q23", time_reference=q23_time,
    )
    q4_balance = V.bdf_mass_balance_record(
        cfg, q4, question="q4", time_reference=q4_time,
    )
    records += [q23_balance, q4_balance]

    records += [
        V.envelope_record(cfg, q1, question="q1", sample_times_s=times_q1_space),
        V.envelope_record(cfg, q23, question="q23", sample_times_s=times_q23_space),
        V.envelope_record(cfg, q4, question="q4", sample_times_s=times_q4_space),
        V.event_accuracy_record(cfg, q23, q23_space, q23_time, question="q23"),
        V.event_accuracy_record(cfg, q4, q4_space, q4_time, question="q4"),
        data["v10"]["record"],
    ]
    records.append(V.moving_geometry_record(cfg, q4))

    _log("步3 V-11 各问 8/16 点界面求积对照")
    npts_check = int(cfg.raw["numerics"]["quadrature"]["interface_points_check"])
    q1_quad = runners.run_q1(
        cfg, N=q1["N"], integral_npts=npts_check, save=False,
        purpose="quadrature-reference",
    )
    q23_quad = runners.q23_candidate(
        cfg, N=q23["N"], integral_npts=npts_check, save=False,
        purpose="quadrature-reference",
    )
    q4_quad = runners.q4_candidate(
        cfg, N=q4["N"], integral_npts=npts_check, save=False,
        purpose="quadrature-reference",
    )
    records += [
        V.compare_fixed_trajectories(
            cfg, q1, q1_quad, question="q1", check_id="V-11-q1-quadrature",
            comparison="interface_quadrature", times_s=_comparison_times_q1(q1, q1_quad),
        ),
        V.compare_fixed_trajectories(
            cfg, q23, q23_quad, question="q23", check_id="V-11-q23-quadrature",
            comparison="interface_quadrature", times_s=_comparison_times_q23(q23, q23_quad),
        ),
        V.compare_moving_trajectories(
            cfg, q4, q4_quad, check_id="V-11-q4-quadrature",
            comparison="interface_quadrature", times_s=_comparison_times_q4(q4, q4_quad),
        ),
    ]

    be = data["be"]
    records += [
        be["mass_dt_1"]["v4a"], be["mass_dt_1"]["v4b"],
        be["mass_dt_0.25"]["v4a"], be["mass_dt_0.25"]["v4b"],
        be["energy_dt_1"]["record"], be["energy_dt_0.5"]["record"],
    ]
    # 同一 check_id 下用配置中的 dt 区分粗细步；报告器不合并或覆盖这些记录。

    interface_partial = V.check_record(
        "V-11-interface-duration", "q23", "partial",
        metric={interface: {str(N): value for N, value in by_n.items()}
                for interface, by_n in data["study"].items()},
        limit={"scope": "duration trend only; not full-field acceptance"},
        config={"interfaces": list(data["study"]),
                "Ns": sorted(next(iter(data["study"].values())).keys())},
        coverage={"variables": ["t_star"], "question": "q23"},
        message="仅证明 Q23 时长趋势；不替代全场或 Q4 检验",
    )
    records.append(interface_partial)
    records += _production_not_run_records()

    bundle = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__,
            "pyyaml": yaml.__version__,
        },
        "records": records,
        "gate": _gate(
            records, production_approved=cfg.raw["production"]["approved"],
        ),
        "candidate_summary": {
            "q1": {"N": q1["N"], "config_digest": q1["config_digest"]},
            "q23": {"N": q23["N"], "t_star_h": q23["t_star_h"],
                    "t_sample_s": q23["t_sample_s"], "delta_C": q23.get("delta"),
                    "dt_star_h": q23.get("dt_star_h"),
                    "config_digest": q23["config_digest"]},
            "q4": {"N": q4["N"], "t_star_h": q4["t_star_h"],
                   "t_sample_s": q4["t_sample_s"], "delta_C": q4.get("delta"),
                   "dt_star_h": q4.get("dt_star_h"),
                   "config_digest": q4["config_digest"]},
        },
    }
    _write_json(REPORTS / "verification.json", bundle)
    _write_verification_report(bundle)
    return bundle


def _write_verification_report(bundle):
    lines = [
        "# 验证报告（候选阶段）",
        "",
        "> 状态由 `reports/verification.json` 的实际指标生成。局部检查、未运行和失败不会被提升为通过。",
        "> 本报告不构成 D12 授权；正式 result1–4 尚未生成。",
        "",
        "## 验证状态总表",
        "",
        "| 检查 | 问题 | 状态 | 覆盖 |",
        "|---|---|---|---|",
    ]
    for record in bundle["records"]:
        coverage = record["coverage"]
        summary = ", ".join(
            f"{key}={value}" for key, value in coverage.items()
            if key in ("comparison", "rows", "time_start_s", "time_end_s", "t_end_s",
                       "accepted_substeps", "moving_domain")
        ) or "见 JSON"
        lines.append(
            f"| {record['check_id']} | {record['question']} | "
            f"{_status_cn(record['status'])} | {summary} |"
        )

    lines += [
        "",
        "## 全输出对象空间/时间/求积对照",
        "",
        "| 检查 | 状态 | max ΔC | max ΔT / °C | Δt* / h | ΔC 最差位置 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in bundle["records"]:
        metric = record["metric"]
        if "dC" not in metric or "dT_C" not in metric:
            continue
        worst = (record.get("worst_location") or {}).get("dC")
        lines.append(
            f"| {record['check_id']} | {_status_cn(record['status'])} | "
            f"{metric['dC']:.8e} | {metric['dT_C']:.8e} | "
            f"{'' if metric.get('dt_star_h') is None else f'{metric['dt_star_h']:.8e}'} | "
            f"{worst} |"
        )

    lines += [
        "",
        "## V-4 BE 粗细步与热残差",
        "",
        "| 项 | dt / s | 指标 | 限值 | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    v4b_rows = sorted(
        (row for row in bundle["records"]
         if row["check_id"] == "V-4b" and row["question"] == "fixed"),
        key=lambda row: float(row["config"].get("dt_be_s", 0.0)), reverse=True,
    )
    for row in v4b_rows:
        dt = float(row["config"]["dt_be_s"])
        value = row["metric"].get("relative_difference")
        limit = row["limit"]["relative_difference_max"]
        value_text = "" if value is None else f"{value:.10e}"
        lines.append(
            f"| V-4b 独立梯形通量 | {dt:g} | {value_text} | "
            f"{limit:.1e} | {_status_cn(row['status'])} |"
        )
    if not v4b_rows:
        lines.append("| V-4b 独立梯形通量 | — | — | — | 未运行 |")
    v4c_rows = sorted(
        (row for row in bundle["records"]
         if row["check_id"] == "V-4c" and row["question"] == "fixed"),
        key=lambda row: float(row["config"].get("dt_be_s", 0.0)), reverse=True,
    )
    for row in v4c_rows:
        dt = float(row["config"]["dt_be_s"])
        residual = row["metric"].get("RE_W_per_m")
        relative = row["metric"].get("relative_residual")
        metric_text = (
            "" if residual is None or relative is None
            else f"{residual:.10e} W/m；rel={relative:.10e}"
        )
        lines.append(
            f"| V-4c 有效热残差 | {dt:g} | {metric_text} | "
            f"abs≤{row['limit']['abs_W_per_m_max']:.1e} 且 "
            f"rel≤{row['limit']['relative_residual_max']:.1e} | "
            f"{_status_cn(row['status'])} |"
        )
    if not v4c_rows:
        lines.append("| V-4c 有效热残差 | — | — | — | 未运行 |")
    coarse = (
        v4b_rows[0]["metric"].get("relative_difference")
        if len(v4b_rows) >= 2 else None
    )
    fine = (
        v4b_rows[-1]["metric"].get("relative_difference")
        if len(v4b_rows) >= 2 else None
    )
    ratio = None if not coarse or not fine else coarse / fine
    dt_ratio = None if ratio is None else (
        float(v4b_rows[0]["config"]["dt_be_s"])
        / float(v4b_rows[-1]["config"]["dt_be_s"])
    )
    observed_order = (
        None if ratio is None else np.log(ratio) / np.log(dt_ratio)
    )
    lines += [
        "",
        ("V-4b 未形成可计算的粗细步比值。" if ratio is None else
         f"V-4b 实测粗细步比值：{ratio:.8f}（dt 缩小 {dt_ratio:g} 倍）；"
         f"对应收敛阶 {observed_order:.6f}。粗步失败状态保留，细步独立判定通过。"),
        "",
        "## 候选有效配置",
        "",
        "| 问题 | N | 配置摘要 |",
        "|---|---:|---|",
    ]
    for question, summary in bundle["candidate_summary"].items():
        lines.append(
            f"| {question} | {summary['N']} | `{summary['config_digest']}` |"
        )

    gate = bundle["gate"]
    lines += [
        "",
        "## 未通过、局部与未运行记录",
        "",
    ]
    unresolved = [row for row in bundle["records"] if row["status"] != "pass"]
    if unresolved:
        for row in unresolved:
            detail = row["message"] or "详见机器可读 metric/limit/coverage"
            lines.append(
                f"- {row['check_id']} / {row['question']}："
                f"{_status_cn(row['status'])}；{detail}"
            )
    else:
        lines.append("- 无。")
    lines += ["", "## D12 前候选门控", ""]
    lines.append(
        f"- 全局配置证据：{'通过' if gate['configuration_ready'] else '未通过'}"
    )
    for question, state in gate["per_question"].items():
        lines.append(
            f"- {question}: {'适用数值检查已通过' if state['ready'] else '仍有失败或未运行项'}"
        )
    lines += [
        "",
        f"总状态：**{'可提交用户考虑 D12 授权，但本次仍未授权' if gate['ready_for_D12_authorization'] and not gate['production_approved'] else ('适用数值检查通过，但配置已标授权；本轮仍未执行生产' if gate['ready_for_D12_authorization'] else '不可提交 D12 授权')}**。",
        "",
        "机器可读的每项 metric/limit/config/coverage/worst_location 见 `reports/verification.json`。",
    ]
    (REPORTS / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def step_sensitivity(cfg, *, N=400, refine=True):
    _log(f"步7 灵敏度（仅 Q23，N={N}，数值误差{'实测加密' if refine else '未运行'}）")
    result = sensitivity.run_scenarios(cfg, question="q23", N=N, refine=refine)
    result["execution_status"] = (
        "measured" if all(row["resolution_status"] == "measured"
                          for row in result["rows"])
        else "partial"
    )
    lines = [
        "# 灵敏度分析（实际启用范围：Q23 S1–S6）",
        "",
        "> 情景范围、验收门槛与实测数值误差分开记录；Q1/Q4 未执行。",
        "> 可分辨要求候选/加密差同向，且加密后的情景差大于基线与情景两次求解的合成误差。",
        "",
        f"基线 t*={result['base_t_star_h']:.8f} h；"
        f"验收门槛={result['acceptance_threshold_h']:.6g} h；"
        f"基线实测时间加密差={result['base_numeric_error_h']!r} h。",
        "",
        "| 情景 | Δt* / h | 加密后 Δt* / h | 合成不确定性 / h | 判断依据状态 | 可分辨 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in result["rows"]:
        resolved = "未判断" if row["resolved"] is None else ("是" if row["resolved"] else "否")
        resolution_status = {
            "measured": "已实测",
            "not_run": "未执行",
        }.get(row["resolution_status"], row["resolution_status"])
        lines.append(
            f"| {row['label']} | {row['dt_star_h']:+.8f} | "
            f"{'' if row['refined_dt_star_h'] is None else f'{row['refined_dt_star_h']:+.8f}'} | "
            f"{'' if row['combined_uncertainty_h'] is None else f'{row['combined_uncertainty_h']:.8e}'} | "
            f"{resolution_status} | {resolved} |"
        )
    (REPORTS / "sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(REPORTS / "sensitivity.json", result)
    return result


def write_sensitivity_not_run(cfg, *, N=400):
    """Overwrite any stale sensitivity artifact with an explicit skipped record."""
    result = {
        "question": "q23",
        "enabled_scope": [],
        "not_run_scope": ["q1", "q23", "q4"],
        "N": int(N),
        "interface": cfg.raw["numerics"]["per_question"]["q23"]["candidate_interface"],
        "acceptance_threshold_h": float(cfg.raw["acceptance"]["t_star_h"]),
        "numeric_error_source": "not_run",
        "execution_status": "not_run",
        "rows": [],
        "message": "本轮通过 --no-sensitivity 显式跳过；未沿用旧报告状态",
    }
    _write_json(REPORTS / "sensitivity.json", result)
    (REPORTS / "sensitivity.md").write_text(
        "# 灵敏度分析\n\n本轮未执行（`--no-sensitivity`）；未沿用旧结果。\n",
        encoding="utf-8",
    )
    return result


def write_status(cfg, data, validation, sensitivity_result):
    gate = validation["gate"]
    records = {
        (row["check_id"], row["question"]): row for row in validation["records"]
    }
    def record_state(check_id, question):
        row = records.get((check_id, question))
        return _status_cn("not_run" if row is None else row["status"])

    config_state = record_state("C-1-config", "global")
    v8_state = record_state("V-8", "production")
    v9_state = record_state("V-9", "production")
    sensitivity_status = sensitivity_result.get("execution_status", "not_run")
    sensitivity_state = {
        "measured": "已实测",
        "partial": "局部执行（数值误差未测）",
        "not_run": "仍未执行",
    }.get(sensitivity_status, sensitivity_status)
    if gate["production_approved"]:
        d12 = "配置标志已授权；但本轮未执行生产，不能据此推断文件已生成"
    elif gate["ready_for_D12_authorization"]:
        d12 = "待用户授权（候选适用数值检查通过）"
    else:
        d12 = "尚不可授权（存在失败/局部/未运行的必检项）"
    lines = [
        "# 状态汇总（候选阶段）",
        "",
        f"生成时间：{validation['generated_at']}；Python {validation['software']['python']}；"
        f"NumPy {validation['software']['numpy']}；SciPy {validation['software']['scipy']}；"
        f"openpyxl {validation['software']['openpyxl']}；PyYAML {validation['software']['pyyaml']}。",
        "",
        "| 项 | 实际状态 | 证据 |",
        "|---|---|---|",
        f"| 当前配置校验、镜像一致性与有效快照 | {config_state} | reports/verification.json#C-1-config |",
        f"| 候选数值适用检查 | {'已实测通过' if gate['ready_for_D12_authorization'] else '仍有未通过项'} | reports/verification.json |",
        f"| 灵敏度 Q23 S1–S6 | {sensitivity_state} | reports/sensitivity.json |",
        f"| D12 生产配置授权 | {d12} | production.approved={str(gate['production_approved']).lower()} |",
        "| 生产续算/导出编排 | 仍未实现 | run_production() 明确抛 NotImplementedError |",
        "| 正式 result1–4 | 仍未生成 | 未调用生产导出，不由 approved 推断 |",
        f"| V-8 result2/3 跨文件一致性 | {v8_state} | 检查器已实现，待正式文件后执行 |",
        f"| V-9 工作簿结构与格式终检 | {v9_state} | 检查器已实现，待正式文件后执行 |",
        "| 论文数值与文本 | 待人工核验 | 不在本次代码返修内 |",
        "",
        "## 候选数值（未授权，非正式答案）",
        "",
        "| 量 | 值 | 配置摘要 |",
        "|---|---:|---|",
        f"| Q1 N | {data['q1']['N']} | {data['q1']['config_digest']} |",
        f"| Q23 t* | {data['q23']['t_star_h']:.8f} h | {data['q23']['config_digest']} |",
        f"| Q4 t* | {data['q4']['t_star_h']:.8f} h | {data['q4']['config_digest']} |",
        "",
        f"> `production.approved={str(gate['production_approved']).lower()}`；"
        "本轮未执行生产，本报告不声称正式文件存在。",
    ]
    (REPORTS / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_production(cfg):
    if not cfg.raw["production"]["approved"]:
        _log("生产未授权：production.approved=false；不会生成正式 result1–4。")
        return False
    raise NotImplementedError(
        "生产续算、配置摘要/验证证据绑定与正式 result1–4 导出尚未实现"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="A题药材烘干候选计算与验证")
    parser.add_argument("--produce", action="store_true", help="正式生产（当前仍未实现且需 D12）")
    parser.add_argument("--N-q1", type=int, default=800, help="Q1 候选网格")
    parser.add_argument("--N-long", type=int, default=800, help="Q23/Q4 候选网格")
    parser.add_argument("--no-sensitivity", action="store_true", help="跳过灵敏度并如实记未运行")
    parser.add_argument("--no-plots", action="store_true", help="跳过绘图")
    args = parser.parse_args(argv)

    started = time.time()
    cfg = cfgmod.load_config()
    if args.produce:
        return 0 if run_production(cfg) else 1

    environment = step0_env(cfg, N_q1=args.N_q1, N_long=args.N_long)
    analytic = step1_analytic(cfg)
    data = steps23_candidates(cfg, N_q1=args.N_q1, N_long=args.N_long)
    validation = run_candidate_validation(cfg, analytic, data, environment)
    sens = (
        write_sensitivity_not_run(cfg)
        if args.no_sensitivity else step_sensitivity(cfg)
    )

    if not args.no_plots:
        _log("步8 从本轮候选数据重绘图表")
        from . import plots
        plots.all_candidate_figs(
            cfg, study_table=data["study"], s10=data["s10"],
            q1=data["q1"], q4=data["q4"],
        )
    write_status(cfg, data, validation, sens)
    _log(f"候选管线完成，用时 {time.time() - started:.1f}s。")
    _log("正式 result1–4 仍未生成；production.approved=false 保持。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
