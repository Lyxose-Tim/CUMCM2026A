"""V-3 分对象收敛（W3）：空间/时间对照证据分列；Q4 r=1.2 cm 回归点独立登记。"""
import pytest

from drymodel import config as cfgmod, verify as V


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_v3_early_surface_is_spatial(cfg):
    """result1/2 早期表面：空间对照，随 N 二阶收敛（Q1 附录2）。"""
    r = V.v3_early_surface(cfg, question="q1", Ns=(200, 400, 800))
    assert r["kind"] == "spatial"
    # 1s 表面差随 N 减小
    d = [r["by_N"][N]["d_vs_prev"][1.0] for N in (400, 800)]
    assert d[1] < d[0]


def test_v3_flux_mean_converges(cfg):
    r = V.v3_flux_mean(cfg, question="q23", Ns=(200, 400), t_probe_s=10800.0)
    assert r["kind"] == "spatial"
    assert r["max_cbar_diff"] < 1e-4


def test_v3_tstar_time_is_temporal(cfg):
    """t* 时间对照：收紧 rtol → 独立时间证据（与网格对照分列）。"""
    r = V.v3_tstar_time(cfg, question="q23", N=400, rtols=(1e-8, 1e-10))
    assert r["kind"] == "temporal"
    assert r["max_adjacent_diff_h"] < 1e-4          # 时间误差小


def test_v3_q4_r12_regression_registered(cfg):
    """Q4 r=1.2 cm 回归点独立登记，不以时长收敛替代。"""
    r = V.v3_q4_profile(cfg, Ns=(200, 400), t_probe_h=24.0)
    assert 1.2 in r["cols_cm"]
    assert r["r1_2cm_by_N"][200] is not None and r["r1_2cm_by_N"][400] is not None
    assert r["r1_2cm_max_adjacent_diff"] > 0.0      # 确有网格误差需追踪
