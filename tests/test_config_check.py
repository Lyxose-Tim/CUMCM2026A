"""config_check 单元测试：正例通过 + 八类非法变异被拒。"""
import copy
import importlib.util
import subprocess
import sys

import pytest
import yaml

from drymodel import config as cfgmod
from drymodel import config_check as cc


@pytest.fixture(scope="module")
def base_cfg():
    with open(cfgmod.DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_base_config_passes(base_cfg):
    assert cc.check(base_cfg) is True


def test_props_consistency(base_cfg):
    assert cc.check_props_against_formulas() is True


# --- 八类非法变异（§9.3），每类须抛 ConfigError ---
def _mutate(base, path, value):
    d = copy.deepcopy(base)
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return d


def test_reject_1_q1_rho_negative(base_cfg):
    bad = _mutate(base_cfg, "props.q1.rho.value", -820.0)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_2_q23_D_prefactor(base_cfg):
    bad = _mutate(base_cfg, "props.q23.D_formula", "2.4e3*exp(-0.45/C)*exp(-3850/T_K)")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_3_q4_D_zero(base_cfg):
    bad = _mutate(base_cfg, "props.q4.D_formula", "0")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_4_N_default_negative(base_cfg):
    bad = _mutate(base_cfg, "numerics.N_default", -200)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_5_bdf_rtol_negative(base_cfg):
    bad = _mutate(base_cfg, "numerics.bdf.rtol", -1e-8)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_6_result2_endpoint_3h(base_cfg):
    bad = _mutate(base_cfg, "output.result2.t_end", "3h")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_7_energy_form_enthalpy(base_cfg):
    bad = _mutate(base_cfg, "props.energy_form", "enthalpy_advective")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_8_q4_axial_shrinking(base_cfg):
    bad = _mutate(base_cfg, "q4.axial", "shrinking_L")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


# --- 定点返修新增负例（PATCH-04/05）---
def test_reject_energy_ref_scale_with_L(base_cfg):
    bad = _mutate(base_cfg, "acceptance.energy_residual.ref_scale", "2*pi*R*L*h*max(dT,1K)")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_q1_applies_event(base_cfg):
    bad = _mutate(base_cfg, "numerics.per_question.q1.applies_event", True)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_unauthorized_final_N(base_cfg):
    bad = _mutate(base_cfg, "numerics.per_question.q23.final_N", 800)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_quadrature_points_zero(base_cfg):
    bad = _mutate(base_cfg, "numerics.quadrature.interface_points", 0)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_bool_as_integer_mesh(base_cfg):
    """布尔不能冒充整数网格数。"""
    bad = _mutate(base_cfg, "numerics.N_default", True)
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_nan_tolerance(base_cfg):
    bad = _mutate(base_cfg, "numerics.bdf.rtol", float("nan"))
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_reject_result2_test_mode_as_official(base_cfg):
    bad = _mutate(base_cfg, "output.result2.t_end", "72h")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)


def test_load_config_validates():
    c = cfgmod.load_config()
    assert c.R0 == 0.02 and c.L == 0.25
    assert c.T0_K == pytest.approx(301.15)
    assert c.h == 25.0 and c.hm == 8e-7
    # rho_s0 = rho(2.55)/3.55（附录 3）
    assert c.rho_s0("q23") == pytest.approx((650 + 128 * 2.55) / 3.55)


@pytest.mark.parametrize(("path", "value"), [
    ("numerics.bdf.atol_I", -1.0),
    ("numerics.bdf.atol_C", float("inf")),
    ("numerics.bdf.atol_C", True),
    ("numerics.bdf.atol_T_K", "1e-8"),
    ("numerics.bdf.max_step_after_s", -1.0),
    ("numerics.picard.max_iter", 1.5),
    ("numerics.picard.tol_dC", -1.0),
    ("numerics.retry.halvings_max", 1.5),
    ("air.sensitivity.dT_degC", float("nan")),
    ("output.result1.t_s", "1..100"),
    ("numerics.interface", "harmonic"),
])
def test_reject_audit_schema_mutations(base_cfg, path, value):
    with pytest.raises(cc.ConfigError):
        cc.check(_mutate(base_cfg, path, value))


def test_approved_config_requires_concrete_parameters_and_evidence(base_cfg):
    bad = copy.deepcopy(base_cfg)
    bad["production"].update(approved=True, config_id="candidate-x")
    with pytest.raises(cc.ConfigError):
        cc.check(bad)

    complete = copy.deepcopy(bad)
    complete["production"].update(
        config_digest="0123456789abcdef",
        verification_record="reports/verification.json",
    )
    for block in complete["numerics"]["per_question"].values():
        block.update(
            final_N=800,
            final_interface="integral",
            final_scheme="BDF",
            final_quadrature_points=8,
        )
    assert cc.check(complete) is True

    malformed_digest = copy.deepcopy(complete)
    malformed_digest["production"]["config_digest"] = "not-a-digest"
    with pytest.raises(cc.ConfigError):
        cc.check(malformed_digest)

    wrong_record = copy.deepcopy(complete)
    wrong_record["production"]["verification_record"] = "somewhere/pass.txt"
    with pytest.raises(cc.ConfigError):
        cc.check(wrong_record)

    unverified_grid = copy.deepcopy(complete)
    unverified_grid["numerics"]["per_question"]["q1"]["final_N"] = 123
    with pytest.raises(cc.ConfigError):
        cc.check(unverified_grid)

    unverified_quadrature = copy.deepcopy(complete)
    unverified_quadrature["numerics"]["per_question"]["q4"][
        "final_quadrature_points"
    ] = 12
    with pytest.raises(cc.ConfigError):
        cc.check(unverified_quadrature)

    final_run = cfgmod.Config(complete).resolve_run("q4", purpose="production")
    assert final_run.N == 800
    assert final_run.interface == "integral"
    assert final_run.scheme == "BDF"
    assert final_run.integral_npts == 8


def test_plan_checker_is_thin_authoritative_entrypoint():
    path = cfgmod.PROJECT_ROOT / "建模方案v1.1" / "config_check.py"
    spec = importlib.util.spec_from_file_location("plan_config_check", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.check is cc.check
    assert module.ConfigError is cc.ConfigError


def test_two_yaml_files_are_byte_identical():
    plan = cfgmod.PROJECT_ROOT / "建模方案v1.1" / "A题_config.yaml"
    assert cfgmod.DEFAULT_CONFIG_PATH.read_bytes() == plan.read_bytes()


def test_invalid_config_is_rejected_under_python_optimized(base_cfg, tmp_path):
    bad = _mutate(base_cfg, "numerics.bdf.atol_I", -1.0)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    checker = cfgmod.PROJECT_ROOT / "建模方案v1.1" / "config_check.py"
    proc = subprocess.run(
        [sys.executable, "-O", str(checker), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "atol_I" in (proc.stderr + proc.stdout)


def test_effective_run_config_is_complete_and_stable():
    cfg = cfgmod.load_config()
    run = cfg.resolve_run("q23", N=800, purpose="candidate", augmented=True)
    snap = run.snapshot()
    assert snap["N"] == 800
    assert snap["interface"] == "integral"
    assert snap["integral_npts"] == 8
    assert snap["scheme"] == "BDF"
    assert snap["max_step_after_s"] == 600.0
    assert snap["breakpoints_s"] == (14400.0,)
    assert snap["air_data_end_s"] == 14400.0
    assert snap["augmented"] is True
    assert len(run.digest) == 16
    assert run.digest == cfg.resolve_run("q23", N=800, augmented=True).derive(
        purpose="candidate"
    ).digest
