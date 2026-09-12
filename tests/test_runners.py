"""候选编排回归：有效配置、同源轨迹和论文表结构。"""
import copy

import pytest

from drymodel import config as cfgmod
from drymodel import runners
from drymodel import verify


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_interface_quadrature_points_control_actual_operator(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["numerics"]["quadrature"]["interface_points"] = 12
    raw["numerics"]["quadrature"]["interface_points_check"] = 16
    changed = cfgmod.Config(raw)
    cfgmod.check_config(changed)
    op, _ = runners.build_fixed_operator(changed, "q23", 20)
    assert op.interface == "integral"
    assert op.integral_npts == 12
    assert op.run_config.snapshot()["integral_npts"] == 12


def test_regular_table_rows_do_not_duplicate_terminal():
    exact = runners._regular_plus_terminal(12.0 * 3600.0)
    assert exact == [6.0 * 3600.0, 12.0 * 3600.0]
    non_exact = runners._regular_plus_terminal(12.5 * 3600.0)
    assert non_exact == [6.0 * 3600.0, 12.0 * 3600.0, 12.5 * 3600.0]


def test_q23_table5_has_time_plus_five_radii_and_separate_cmax(cfg):
    out = runners.q23_candidate(cfg, N=20, save=False)
    assert out["table5"].shape[1] == 6
    assert out["table5_columns"].tolist() == [
        "time_h", "r=0cm", "r=0.5cm", "r=1cm", "r=1.5cm", "r=2cm",
    ]
    assert len(out["table5_cmax"]) == len(out["table5"])
    assert out["trajectory"].t_start == pytest.approx(0.0)
    assert out["trajectory"].t_end >= out["t_sample_s"]
    assert out["trajectory"].eval([out["t_sample_s"]])[0].shape[0] == 43


def test_q4_table6_terminal_row_is_unique(cfg):
    assert len(runners._regular_plus_terminal(6.0 * 3600.0)) == 1


def test_q4_v13_checks_actual_geometry_and_extrapolation_markers(cfg):
    out = runners.q4_candidate(cfg, N=20, save=False)
    record = verify.moving_geometry_record(cfg, out)
    assert record["status"] == "pass", record
    assert record["metric"]["diffusion_scale_abs_error"] <= 1e-12
    assert record["metric"]["surface_mass_coeff_abs_error"] <= 1e-12
    assert record["metric"]["extrapolation_marked"] is True
