"""候选状态与 D12 门控必须完全由结构化实测记录生成。"""
from types import SimpleNamespace

import numpy as np

from drymodel import config as cfgmod
from drymodel import run_all
from drymodel import verify


def _record(check_id, question, status="pass"):
    return verify.check_record(
        check_id, question, status, metric={}, limit={}, config={}, coverage={},
    )


def _all_required_records():
    keys = [
        ("C-1-config", "global"),
        ("V-1-spatial", "q1-heat"), ("V-1-temporal", "q1-heat"),
        ("V-2-spatial", "q1-mass"), ("V-2-temporal", "q1-mass"),
        ("V-3-q1-space", "q1"), ("V-3-q1-time", "q1"),
        ("V-5", "q1"), ("V-11-q1-quadrature", "q1"),
        ("V-3-q23-space", "q23"), ("V-3-q23-time", "q23"),
        ("V-4-BDF", "q23"), ("V-5", "q23"), ("V-6", "q23"),
        ("V-11-q23-quadrature", "q23"),
        ("V-3-q4-space", "q4"), ("V-3-q4-time", "q4"),
        ("V-4-BDF", "q4"), ("V-5", "q4"), ("V-6", "q4"),
        ("V-10", "q4"), ("V-13", "q4"),
        ("V-11-q4-quadrature", "q4"),
    ]
    return [_record(*key) for key in keys]


def test_gate_preserves_fail_partial_and_not_run_statuses():
    records = _all_required_records()
    assert run_all._gate(records)["ready_for_D12_authorization"] is True

    failed = [dict(row) for row in records]
    target = next(row for row in failed if row["check_id"] == "V-3-q4-space")
    target["status"] = "fail"
    gate = run_all._gate(failed)
    assert gate["ready_for_D12_authorization"] is False
    assert gate["per_question"]["q4"]["ready"] is False

    partial = [dict(row) for row in records]
    next(row for row in partial if row["check_id"] == "V-5" and row["question"] == "q1")[
        "status"
    ] = "partial"
    assert run_all._gate(partial)["ready_for_D12_authorization"] is False

    missing = [row for row in records if not (
        row["check_id"] == "V-11-q23-quadrature" and row["question"] == "q23"
    )]
    missing_gate = run_all._gate(missing)
    status = next(
        row["status"] for row in missing_gate["per_question"]["q23"]["checks"]
        if row["check_id"] == "V-11-q23-quadrature"
    )
    assert status == "not_run"
    assert missing_gate["ready_for_D12_authorization"] is False


def test_q23_comparison_includes_strict_final_minute():
    candidate = {
        "t_end_1s": 107,
        "result3_t": np.array([60.0, 120.0]),
        "table34_ts_h": np.array([0.01]),
        "table5": np.array([[0.02, 0, 0, 0, 0, 0]], dtype=float),
        "trajectory": SimpleNamespace(t_end=130.0),
    }
    reference = {
        "t_end_1s": 108,
        "trajectory": SimpleNamespace(t_end=130.0),
    }
    times = run_all._comparison_times_q23(candidate, reference)
    assert 107.0 in times
    assert 120.0 in times


def test_status_report_uses_records_and_overwrites_skipped_sensitivity(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "REPORTS", tmp_path)
    cfg = cfgmod.load_config()
    records = _all_required_records()
    next(row for row in records if row["check_id"] == "V-3-q4-space")["status"] = "fail"
    records.extend([_record("V-8", "production", "not_run"),
                    _record("V-9", "production", "not_run")])
    validation = {
        "generated_at": "2026-09-12 00:00:00",
        "software": {
            "python": "test", "numpy": "test", "scipy": "test",
            "openpyxl": "test", "pyyaml": "test",
        },
        "records": records,
        "gate": run_all._gate(records, production_approved=False),
    }
    data = {
        "q1": {"N": 800, "config_digest": "q1"},
        "q23": {"t_star_h": 1.0, "config_digest": "q23"},
        "q4": {"t_star_h": 2.0, "config_digest": "q4"},
    }
    skipped = run_all.write_sensitivity_not_run(cfg)
    run_all.write_status(cfg, data, validation, skipped)
    text = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "候选数值适用检查 | 仍有未通过项" in text
    assert "灵敏度 Q23 S1–S6 | 仍未执行" in text
    assert "正式 result1–4 | 仍未生成" in text
    assert skipped["execution_status"] == "not_run"


def test_report_renderers_preserve_missing_checks_as_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "REPORTS", tmp_path)
    cfg = cfgmod.load_config()
    gate = run_all._gate([], production_approved=False)
    bundle = {"records": [], "candidate_summary": {}, "gate": gate}
    run_all._write_verification_report(bundle)

    validation = {
        "generated_at": "2026-09-12 00:00:00",
        "software": {
            "python": "test", "numpy": "test", "scipy": "test",
            "openpyxl": "test", "pyyaml": "test",
        },
        "records": [],
        "gate": gate,
    }
    data = {
        "q1": {"N": 800, "config_digest": "q1"},
        "q23": {"t_star_h": 1.0, "config_digest": "q23"},
        "q4": {"t_star_h": 2.0, "config_digest": "q4"},
    }
    run_all.write_status(
        cfg, data, validation, {"execution_status": "not_run"},
    )

    verification_text = (tmp_path / "verification.md").read_text(
        encoding="utf-8",
    )
    status_text = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "V-4b 独立梯形通量 | — | — | — | 未运行" in verification_text
    assert "V-4c 有效热残差 | — | — | — | 未运行" in verification_text
    assert "当前配置校验、镜像一致性与有效快照 | 仍未执行" in status_text
    assert "V-8 result2/3 跨文件一致性 | 仍未执行" in status_text
