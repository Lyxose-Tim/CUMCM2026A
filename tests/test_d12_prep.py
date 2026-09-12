"""D12 选型（轻量：仅 Q1，快速）。Q23/Q4 全量由 `python -m drymodel.d12_prep` 运行。"""
import pytest

from drymodel import config as cfgmod, d12_prep as D


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_q1_selection_field_points(cfg):
    """Q1 按 result1 全部场点误差选 N（不以 t* 替代）；1600 vs 3200。"""
    r = D.q1_selection(cfg, Ns=(1600, 3200))
    assert r["maxdC"][0] <= r["tolC"]        # 场点 C 误差达标
    assert r["maxdT"][0] <= r["tolT"]
    assert r["final_N"] == 1600 and r["pass"]
    # 最差位置应在早期表面（r=2.0cm）
    assert r["maxdC"][2] == 2.0
