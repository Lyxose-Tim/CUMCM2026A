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


def test_v15_covers_q2_and_reproduces_external(cfg):
    """覆盖 Q2 表点：热最大在 1.5 h ≈9.1013e-4 °C；质最大在 42 h ≈1.1281e-5（外部复核值）。"""
    r = V.v15_end_effect(cfg, nterms=120)
    assert r["worstT"]["t_h"] == pytest.approx(1.5, abs=1e-6)
    assert r["worstT"]["dT"] == pytest.approx(9.1013e-4, rel=1e-3)
    assert r["worstC"]["t_h"] == pytest.approx(42.0, abs=1e-6)
    assert r["worstC"]["dC"] == pytest.approx(1.1281e-5, rel=1e-3)


def test_v15_truncation_converges(cfg):
    """max|ΔC| 随 nterms 收敛（60 截断底噪 → 120/200 稳定物理值）。"""
    c120 = V.v15_end_effect(cfg, nterms=120)["worstC"]["dC"]
    c200 = V.v15_end_effect(cfg, nterms=200)["worstC"]["dC"]
    assert c120 == pytest.approx(c200, abs=1e-6)     # 已收敛，非截断底噪
