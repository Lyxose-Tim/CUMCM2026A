"""D12 配置接线/失败门测试；替身轨迹只用于单测，不是物理计算证据。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from drymodel import config
from drymodel import d12_evidence as evidence


@pytest.fixture
def cfg():
    base = config.load_config()
    raw = deepcopy(base.raw)
    for q, N in zip(("q1", "q23", "q4"), (20, 40, 60)):
        raw["numerics"]["per_question"][q]["final_N"] = N
    raw["numerics"]["candidate"].update(interface="integral", integral_npts=6)
    raw["numerics"]["candidate"]["t_cap_h"] = {"q23": 73.0, "q4": 89.0}
    raw["numerics"]["quadrature"]["interface_points_check"] = 10
    raw["numerics"]["bdf"].update(rtol=3e-7, atol_C=2e-10, atol_T_K=4e-7,
                                   atol_I=6e-10, max_step_data_s=24.0,
                                   max_step_after_s=180.0)
    return config.Config(raw, source_path=base.source_path)


@pytest.fixture
def solver_spy(monkeypatch):
    calls = []

    def integrate(op, y0, t0, t_end, bdf, **kwargs):
        calls.append({"N": op.N, "interface": op.interface, "npts": op.integral_npts,
                      "bdf": deepcopy(bdf), "t_end": t_end, "kwargs": kwargs,
                      "moving": op.is_reference, "decoupled": op.decoupled})
        # 合成单调场给出已知可求根事件；只检验配置接线，不模拟 PDE。
        rate = np.log(y0[0] / 0.15) / (0.95 * t_end)

        def evaluate(ts):
            values = np.tile(y0, (len(ts), 1))
            values[:, :op.N + 1] *= np.exp(-rate * np.asarray(ts))[:, None]
            return values

        return SimpleNamespace(ok=True, eval=evaluate,
                               segments=[SimpleNamespace(t=np.array([t0, t_end]))])

    monkeypatch.setattr(evidence.SBDF, "integrate_bdf", integrate)
    return calls


def assert_controls(calls, cfg, N):
    assert [c["N"] for c in calls] == [N, 2 * N, N, N]
    assert [c["npts"] for c in calls] == [6, 6, 6, 12]
    assert all(c["interface"] == "integral" for c in calls)
    assert calls[0]["bdf"] == calls[1]["bdf"] == calls[3]["bdf"] == cfg.bdf
    for key in ("rtol", "atol_C", "atol_T_K", "atol_I"):
        assert calls[2]["bdf"][key] == pytest.approx(cfg.bdf[key] / 100)
    for key in ("max_step_data_s", "max_step_after_s"):
        assert calls[2]["bdf"][key] == pytest.approx(cfg.bdf[key] / 2)


@pytest.mark.parametrize("question,N", [("q1", 20), ("q23", 40), ("q4", 60)])
def test_nondefault_controls_reach_solver_independently(cfg, solver_spy, question, N):
    before = deepcopy(cfg.raw)
    ts = np.array([1., 2.])
    if question == "q4":
        result = evidence.evidence_q4(cfg, N, 2 * N, ts, 5e-4)
    else:
        result = evidence.evidence_fixed(cfg, question, N, 2 * N, ts, 5e-4, 1e-2)
    assert_controls(solver_spy, cfg, N)
    assert all(c["moving"] == (question == "q4") for c in solver_spy)
    assert all(c["decoupled"] == (question == "q1") for c in solver_spy)
    assert result["baseline"]["bdf"] == cfg.bdf
    assert result["items"]["interface"]["comparison"]["npts"] == 12
    assert result["items"]["time"]["comparison"]["bdf"] == solver_spy[2]["bdf"]
    assert cfg.raw == before


def test_harmonic_is_used_and_quadrature_is_not_claimed(cfg, solver_spy):
    cfg.raw["numerics"]["candidate"]["interface"] = "harmonic"
    result = evidence.evidence_fixed(cfg, "q1", 20, 40, np.array([1.]), 5e-4, 1e-2)
    assert len(solver_spy) == 3
    assert all(c["interface"] == "harmonic" for c in solver_spy)
    assert result["items"]["interface"]["status"] == "not_applicable"
    assert result["items"]["interface"]["pass"] is None


def flux_stub(cfg, question, setting, t_end_s, *, trajectory=None):
    assert trajectory is not None  # 正式场点轨迹复用，不能额外积分一条不同基准轨迹。
    return {"ok": True, "rel": 0., "rel_refine": 0., "tol_rel": 10 * cfg.bdf["rtol"],
            "configuration": setting, "quadrature_warnings": [], "t_end_s": t_end_s}


def test_run_and_json_report_use_actual_production_config(cfg, solver_spy, monkeypatch, tmp_path):
    detects, flux_calls = [], []

    def detect(cfg_in, **kw):
        assert cfg_in is cfg
        detects.append(kw)
        return (SimpleNamespace(t_cross=119.5, t_sample=120., post_ok=True,
                                post_max_cmax=0.15), None, None, None)

    def flux(*args, **kwargs):
        flux_calls.append((args[1], deepcopy(args[2]), args[3]))
        return flux_stub(*args, **kwargs)

    monkeypatch.setattr(evidence.runners, "q23_detect", detect)
    monkeypatch.setattr(evidence, "_flux_evidence", flux)
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "D12_evidence.md").write_text("历史报告", encoding="utf-8")
    monkeypatch.setattr(evidence, "REPORTS", historical)
    out = tmp_path / "round2_evidence"
    result = evidence.run_d12_evidence(cfg, output_dir=out)

    assert [(d["N"], d["interface"], d["npts"], d["t_cap_h"]) for d in detects] == [
        (40, "integral", 6, 73.), (60, "integral", 6, 89.)]
    for i, N in enumerate((20, 40, 60)):
        assert_controls(solver_spy[4 * i:4 * i + 4], cfg, N)
    assert [(q, s["N"], s["npts"], t) for q, s, t in flux_calls] == [
        ("q23", 40, 6, 119.5), ("q23", 80, 6, 119.5),
        ("q4", 60, 6, 119.5), ("q4", 120, 6, 119.5)]

    saved = json.loads((out / "D12_evidence.json").read_text(encoding="utf-8"))
    assert saved["complete"] and saved["all_ok"]
    assert result["q23"]["n_rows"] == 121  # 全逐秒行和非整数事件行。
    assert saved["q4"]["n_rows"] == 3  # 60、120、事件 119.5。
    assert saved["q23"]["items"]["time"]["comparison"]["bdf"]["max_step_data_s"] == 12.
    assert saved["q23"]["items"]["time"]["event"]["tol_h"] == cfg.raw["acceptance"]["t_star_h"]
    assert saved["q23"]["items"]["time"]["event"]["baseline"]["t_star_s"] == pytest.approx(114.)
    p = saved["provenance"]
    assert p["effective_config"] == cfg.raw
    assert p["config_file_sha256"] == hashlib.sha256(cfg.source_path.read_bytes()).hexdigest()
    effective_bytes = json.dumps(cfg.raw, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert p["effective_config_sha256"] == hashlib.sha256(effective_bytes).hexdigest()
    assert "src/drymodel/d12_evidence.py" in p["code_sha256"]
    assert len(p["input_sha256"]) == 2
    assert "scipy" in p["software"]
    report = (out / "D12_evidence.md").read_text(encoding="utf-8")
    assert "N=40 vs 80" in report and "6点 vs 12点" in report
    assert "119.5" in report and "206940" not in report
    assert (historical / "D12_evidence.md").read_text(encoding="utf-8") == "历史报告"


def test_solver_failure_writes_failed_incomplete_evidence(cfg, monkeypatch, tmp_path):
    def failed(*a, **k):
        raise RuntimeError("测试积分失败")

    monkeypatch.setattr(evidence.runners, "q23_detect", failed)
    with pytest.raises(RuntimeError, match="测试积分失败"):
        evidence.run_d12_evidence(cfg, output_dir=tmp_path)
    saved = json.loads((tmp_path / "D12_evidence.json").read_text(encoding="utf-8"))
    assert not saved["complete"] and not saved["all_ok"]
    assert saved["error"]["message"] == "测试积分失败"
    assert "未完成，不能通过" in (tmp_path / "D12_evidence.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("outcome,expected", [(True, 0), (False, 2), (None, 2)])
def test_cli_forwards_paths_and_returns_failure_code(cfg, monkeypatch, tmp_path, outcome, expected):
    paths = []

    def load(path):
        paths.append(path)
        return cfg

    def run(c, output_dir):
        assert c is cfg and output_dir == tmp_path
        if outcome is None:
            raise RuntimeError("failure")
        return {"all_ok": outcome}

    monkeypatch.setattr(evidence.cfgmod, "load_config", load)
    monkeypatch.setattr(evidence, "run_d12_evidence", run)
    assert evidence.main(["--config", "custom.yaml", "--output-dir", str(tmp_path)]) == expected
    assert str(paths[0]) == "custom.yaml"


def test_refinement_does_not_claim_unachievable_rtol(cfg):
    cfg.bdf["rtol"] = 1e-13
    with pytest.raises(ValueError, match="SciPy"):
        evidence._settings(cfg, "q1", 20, 40)


@pytest.mark.parametrize("question", ["q23", "q4"])
def test_missing_event_cannot_pass_identical_field_comparisons(cfg, monkeypatch, question):
    def no_cross(op, y0, t0, t_end, bdf, **kwargs):
        return SimpleNamespace(ok=True, eval=lambda ts: np.tile(y0, (len(ts), 1)),
                               segments=[SimpleNamespace(t=np.array([t0, t_end]))])

    monkeypatch.setattr(evidence.SBDF, "integrate_bdf", no_cross)
    if question == "q23":
        result = evidence.evidence_fixed(cfg, question, 40, 80, np.array([1., 2.]), 5e-4, 1e-2)
    else:
        result = evidence.evidence_q4(cfg, 60, 120, np.array([1., 2.]), 5e-4)
    assert not result["pass"]
    for item in result["items"].values():
        assert not item["event"]["baseline"]["covered"]
        assert item["event"]["delta_h"] is None
        assert not item["pass"]


def test_event_error_uses_configured_tolerance(cfg, solver_spy, monkeypatch):
    cfg.raw["acceptance"]["t_star_h"] = 1e-4  # 0.36 s；1 s 偏差必须失败。
    events = iter((100., 101., 100., 100.))
    monkeypatch.setattr(evidence, "_event_time", lambda *a: {"covered": True, "t_star_s": next(events)})
    result = evidence.evidence_fixed(cfg, "q23", 40, 80, np.array([120.]), 5e-4, 1e-2)
    spatial = result["items"]["spatial"]
    assert spatial["pass_C"] and spatial["pass_T"]
    assert spatial["event"]["delta_h"] == pytest.approx(1 / 3600)
    assert not spatial["event"]["pass"] and not result["pass"]


def test_flux_quadrature_has_independent_balance_and_refinement_gates(cfg, monkeypatch):
    """解析标量收支单测；不是 A 题空间模型的物理结果。"""
    from scipy import integrate

    rate = 0.1
    trajectory = SimpleNamespace(
        eval=lambda ts: np.repeat((cfg.C0 * np.exp(-rate * np.asarray(ts)))[:, None], 4, axis=1),
        segments=[SimpleNamespace(t=np.array([0., 1.]))])
    op = SimpleNamespace(N=1, grid=SimpleNamespace(cbar=np.mean),
                         env=SimpleNamespace(t_nodes=np.array([0., 1.])), is_reference=False,
                         flux_cbar=lambda t, C: rate * C[-1])
    setting = {"N": 1, "interface": "integral", "npts": 6, "bdf": cfg.bdf}
    result = evidence._flux_evidence(cfg, "q23", setting, 1., trajectory=(trajectory, op))
    assert result["ok"] and result["pass_balance"] and result["pass_refinement"]
    assert result["I_quad"] == pytest.approx(cfg.C0 * (1 - np.exp(-rate)))
    assert result["quadrature_epsrel"][1] < result["quadrature_epsrel"][0]

    original_quad = integrate.quad

    def inaccurate(*args, **kwargs):
        val, _ = original_quad(*args, **kwargs)
        return val, 1e-2

    monkeypatch.setattr(integrate, "quad", inaccurate)
    result = evidence._flux_evidence(cfg, "q23", setting, 1., trajectory=(trajectory, op))
    assert result["pass_balance"] and not result["pass_refinement"] and not result["ok"]

    def warned(*args, **kwargs):
        warnings.warn("test quadrature warning", integrate.IntegrationWarning)
        return original_quad(*args, **kwargs)

    monkeypatch.setattr(integrate, "quad", warned)
    result = evidence._flux_evidence(cfg, "q23", setting, 1., trajectory=(trajectory, op))
    assert result["pass_balance"] and not result["ok"]
    assert result["quadrature_warnings"] == ["test quadrature warning"]
