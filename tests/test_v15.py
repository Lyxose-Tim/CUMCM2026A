"""V-15 端面增量校核（§8.2）：平板中心余量与增量判据。"""
import pytest

from drymodel import config as cfgmod, verify as V


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_slab_center_initial_unity():
    """Fo=0 平板中心 θ=Σc_n=1。"""
    assert V.analytic_slab_center(0.0, 20.25, 120) == pytest.approx(1.0, abs=5e-3)


def test_v15_matches_scheme_margin(cfg):
    """72 h 质平板中心余量 ≈0.980（方案 §8.2）。"""
    r = V.v15_end_effect(cfg)
    assert r["Fo_L_72h"] == pytest.approx(0.0819, abs=1e-3)
    assert r["slab_center_margin_72h"] == pytest.approx(0.980, abs=2e-3)


def test_v15_passes_criteria(cfg):
    """端面增量 |ΔT|<0.1 °C、|ΔC|<0.005 → 线性辅助情景通过。"""
    r = V.v15_end_effect(cfg)
    assert r["worstT"]["dT"] < 0.1
    assert r["worstC"]["dC"] < 0.005
    assert r["pass"]
