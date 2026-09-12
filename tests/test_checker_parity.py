"""统一两份检查器：src/drymodel/config_check 与 建模方案v1.1/config_check 行为一致。

对基例 + 全部负例断言两者接受/拒绝完全一致（W2）。
"""
import copy
import importlib.util

import pytest
import yaml

from drymodel import config as cfgmod
from drymodel import config_check as src_cc

# 载入独立交付版检查器（自包含）
_STANDALONE = cfgmod.PROJECT_ROOT / "建模方案v1.1" / "config_check.py"
_spec = importlib.util.spec_from_file_location("_standalone_config_check", _STANDALONE)
std_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(std_cc)


@pytest.fixture(scope="module")
def base_cfg():
    with open(cfgmod.DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mutate(base, path, value):
    d = copy.deepcopy(base)
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return d


# (路径, 值) —— 覆盖八类变异 + 类型/有限性/范围负例
NEGATIVES = [
    ("props.q1.rho.value", -820.0),
    ("props.q23.D_formula", "2.4e3*exp(-0.45/C)*exp(-3850/T_K)"),
    ("props.q4.D_formula", "0"),
    ("numerics.N_default", -200),
    ("numerics.bdf.rtol", -1e-8),
    ("output.result2.t_end", "3h"),
    ("props.energy_form", "enthalpy_advective"),
    ("q4.axial", "shrinking_L"),
    ("numerics.N_default", True),               # 布尔冒充整数网格
    ("numerics.bdf.rtol", float("nan")),        # NaN 容差
    ("numerics.quadrature.interface_points", 0),
    ("acceptance.energy_residual.ref_scale", "2*pi*R*L*h*max(dT,1K)"),  # 含 L
    ("numerics.per_question.q1.applies_event", True),
]


# 两字段负例：approved=false 但 final_N 已填（未授权配置）→ 两检查器均应拒
def _unauthorized_with_final_N(base):
    d = _mutate(base, "production.approved", False)
    return _mutate(d, "production.config_id", None)


def _accepts(check_fn, cfg):
    try:
        check_fn(cfg)
        return True
    except Exception:
        return False


def test_base_accepted_by_both(base_cfg):
    assert _accepts(src_cc.check, base_cfg) is True
    assert _accepts(std_cc.check, base_cfg) is True


@pytest.mark.parametrize("path,value", NEGATIVES)
def test_negatives_rejected_by_both(base_cfg, path, value):
    bad = _mutate(base_cfg, path, value)
    src_ok = _accepts(src_cc.check, bad)
    std_ok = _accepts(std_cc.check, bad)
    assert src_ok == std_ok == False, (
        f"检查器不一致 {path}={value!r}: src_accepts={src_ok}, std_accepts={std_ok}")


def test_unauthorized_final_N_rejected_by_both(base_cfg):
    bad = _unauthorized_with_final_N(base_cfg)
    assert _accepts(src_cc.check, bad) is False
    assert _accepts(std_cc.check, bad) is False
