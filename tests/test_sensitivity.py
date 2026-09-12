"""灵敏度分辨判据必须使用实测加密误差，而非验收门槛。"""
import copy
from types import SimpleNamespace

from drymodel import config as cfgmod
from drymodel import sensitivity


def test_scenario_labels_and_amplitudes_follow_config():
    cfg = cfgmod.load_config()
    raw = copy.deepcopy(cfg.raw)
    raw["air"]["sensitivity"].update(dT_degC=0.9, dC=0.002)
    labels = [label for label, _ in sensitivity.scenario_definitions(cfgmod.Config(raw))]
    assert "S3 T_air+0.9" in labels
    assert "S4 C_env-0.002" in labels


def test_resolution_uses_measured_uncertainty_not_acceptance(monkeypatch):
    cfg = cfgmod.load_config()

    def fake_detect(cfg, *, bdf_overrides=None, scenario="base", h_mult=1.0, **kwargs):
        refined = bdf_overrides is not None
        if h_mult == 2.0:       # Δ=0.015 h (< 0.02 验收门槛)，但实测可分辨
            hours = 10.015
        elif h_mult == 0.5:     # Δ=0.03 h (> 0.02)，但加密不稳定，不能分辨
            hours = 9.99 if refined else 10.03
        else:
            hours = 10.001 if refined else 10.0
        return SimpleNamespace(t_cross=hours * 3600.0), None, None, None

    monkeypatch.setattr(sensitivity.runners, "q23_detect", fake_detect)
    result = sensitivity.run_scenarios(
        cfg,
        scenarios=[("small-but-stable", {"h_mult": 2.0}),
                   ("large-but-unstable", {"h_mult": 0.5})],
        refine=True,
    )
    stable, unstable = result["rows"]
    assert result["acceptance_threshold_h"] == 0.02
    assert stable["resolved"] is True
    assert abs(stable["dt_star_h"]) < result["acceptance_threshold_h"]
    assert unstable["resolved"] is False
    assert abs(unstable["dt_star_h"]) > result["acceptance_threshold_h"]
    assert all(row["resolution_status"] == "measured" for row in result["rows"])


def test_unmeasured_sensitivity_is_explicitly_not_run(monkeypatch):
    cfg = cfgmod.load_config()

    def fake_detect(*args, h_mult=1.0, **kwargs):
        return SimpleNamespace(t_cross=(10.0 + 0.1 * (h_mult - 1.0)) * 3600), None, None, None

    monkeypatch.setattr(sensitivity.runners, "q23_detect", fake_detect)
    result = sensitivity.run_scenarios(
        cfg, scenarios=[("one", {"h_mult": 2.0})], refine=False,
    )
    assert result["numeric_error_source"] == "not_run"
    assert result["rows"][0]["resolved"] is None
    assert result["rows"][0]["resolution_status"] == "not_run"
