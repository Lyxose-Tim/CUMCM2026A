"""postprocess 单元测试：域掩码、表面一致性、round4；V-10 静态极限。"""
import numpy as np
import pytest

from drymodel import config as cfgmod, data_io, verify as V
from drymodel import postprocess as PP
from drymodel.grid import RefGrid


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_inside_boundary():
    assert PP.inside(0.012, 0.012) is True           # 相等在域内
    assert PP.inside(0.012, 0.012 * (1 - 1e-6)) is False
    assert PP.inside(0.010, 0.012) is True


def test_round4_none():
    assert PP.round4(None) is None
    assert PP.round4(0.149963) == 0.15


def test_sample_q4_masking_and_surface():
    g = RefGrid(200)
    c = np.linspace(0.20, 0.14, 201)                 # 中心 0.20 → 表面 0.14
    R_t = 0.012                                      # 1.2 cm
    row = PP.sample_q4_row(c, g, R_t, [0.0, 0.5, 1.0, 1.5, 1.9])
    assert row[0] == pytest.approx(0.20)             # 中心
    assert row[3] is None and row[4] is None         # 1.5/1.9 cm 域外
    # 1.0 cm: x=1.0/1.2=0.8333 → 插值
    assert 0.14 < row[2] < 0.20


def test_sample_q4_surface_equals_cN():
    g = RefGrid(100)
    c = np.linspace(0.3, 0.1, 101)
    R_t = 0.010                                      # 1.0 cm 恰在最外固定列
    row = PP.sample_q4_row(c, g, R_t, [1.0])
    assert row[0] == pytest.approx(c[-1])            # x=1 → c_N（表面）


def test_last_valid_time_matches_scheme(cfg):
    """§6.3：1.5 cm 列最后有效约 3.6515 h（1 s 网格精度内）。"""
    R = data_io.make_radius_function(cfg)
    tg = np.arange(0, 10 * 3600, 1.0)
    last = PP.last_valid_time_for_column(1.5, R, tg)
    assert last / 3600.0 == pytest.approx(3.6515, abs=1e-3)


def test_v10_static_limit(cfg):
    """V-10：动域(R≡R0) vs 固定域，相对差 ≤1e-6。"""
    r = V.static_limit_q4(cfg, N=200, interface="integral", t_probe=3600.0)
    assert r["rel_max"] < 1e-6
