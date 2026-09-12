"""config_check 单元测试：正例通过 + 八类非法变异被拒。"""
import copy

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
