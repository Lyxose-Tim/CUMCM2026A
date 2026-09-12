"""data_io 单元测试：加载、61 点均值、断点切换、单位、smooth121、半径。"""
import copy

import numpy as np
import pytest

from drymodel import config as cfgmod
from drymodel import data_io


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_load_attachment1(cfg):
    t, T, C = data_io.load_attachment1(cfg.air_file())
    assert len(t) == 241
    assert t[0] == 0.0 and t[-1] == 14400.0
    assert T[0] == pytest.approx(28.0)
    assert C[-1] == pytest.approx(0.04986)


def test_load_attachment2(cfg):
    t, R = data_io.load_attachment2(cfg.radius_file())
    assert len(t) == 145
    assert R[0] == pytest.approx(2.0)
    assert R[-1] == pytest.approx(1.198)
    assert t[-1] == 259200.0


def test_env_window_mean_and_breakpoint(cfg):
    env = data_io.make_env_functions(cfg, "base")
    # 断点两侧：14400 s 取原始节点 50.165 °C，>14400 s 取窗口均值 49.9989...
    assert env.T_air_K(14400.0) == pytest.approx(50.165 + 273.15)
    assert env.T_air_K(14401.0) == pytest.approx(49.9989344262 + 273.15, abs=1e-6)
    assert env.C_env(20000.0) == pytest.approx(0.0499875409836, abs=1e-9)
    assert env.breakpoints == (14400.0,)


def test_env_interpolation_midpoint(cfg):
    env = data_io.make_env_functions(cfg, "base")
    # 0 s: 28 °C；60 s: 28.528 °C；30 s 线性中点
    assert env.T_air_K(0.0) == pytest.approx(28.0 + 273.15)
    assert env.T_air_K(30.0) == pytest.approx((28.0 + 28.528) / 2 + 273.15)


def test_env_scenarios_extrap_only(cfg):
    """外推扰动仅作用于外推段，不改 0–14400 s。"""
    base = data_io.make_env_functions(cfg, "base")
    dTp = data_io.make_env_functions(cfg, "dT_air+")
    # 内段不变
    assert dTp.T_air_K(1000.0) == pytest.approx(base.T_air_K(1000.0))
    # 外推段 +0.39 °C
    assert dTp.T_air_K(20000.0) - base.T_air_K(20000.0) == pytest.approx(0.39, abs=1e-9)


def test_env_scenario_amplitudes_are_config_driven(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["air"]["sensitivity"]["dT_degC"] = 0.9
    raw["air"]["sensitivity"]["dC"] = 0.002
    changed = cfgmod.Config(raw)
    base = data_io.make_env_functions(changed, "base")
    plus_t = data_io.make_env_functions(changed, "dT_air+")
    minus_c = data_io.make_env_functions(changed, "dC_env-")
    assert plus_t.T_air_K(20000.0) - base.T_air_K(20000.0) == pytest.approx(0.9)
    assert minus_c.C_env(20000.0) - base.C_env(20000.0) == pytest.approx(-0.002)


def test_smooth121_keeps_endpoints_and_window_mean(cfg):
    base = data_io.make_env_functions(cfg, "base")
    sm = data_io.make_env_functions(cfg, "smooth121")
    # 端点原值（t=0）不变；外推均值不变
    assert sm.T_air_K(0.0) == pytest.approx(base.T_air_K(0.0))
    assert sm.T_const_K == pytest.approx(base.T_const_K)


def test_radius_function(cfg):
    R = data_io.make_radius_function(cfg)
    assert R.R(0.0) == pytest.approx(0.02)
    assert R.R(259200.0) == pytest.approx(0.01198)
    # t>72 h 保持末值并标注外推
    assert R.R(300000.0) == pytest.approx(0.01198)
    assert R.is_extrapolated(300000.0) is True
    assert R.is_extrapolated(100000.0) is False
