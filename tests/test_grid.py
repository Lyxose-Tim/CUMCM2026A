"""grid 单元测试：体积、守恒度量、输出下标。"""
import numpy as np
import pytest

from drymodel.grid import RadialGrid, RefGrid


def test_radial_volume_sum():
    """ΣV_i = R0²/2。"""
    g = RadialGrid(200, 0.02)
    assert np.sum(g.V) == pytest.approx(0.02 ** 2 / 2.0, rel=1e-14)


def test_radial_cbar_constant_field():
    """常值场 C≡c0 → C̄ = c0。"""
    g = RadialGrid(200, 0.02)
    C = np.full(g.N + 1, 2.55)
    assert g.cbar(C) == pytest.approx(2.55, rel=1e-13)


def test_radial_faces_and_areas():
    g = RadialGrid(20, 0.02)
    assert len(g.r) == 21 and len(g.r_face) == 20
    assert g.r_face[0] == pytest.approx(0.5 * g.dr)
    assert np.allclose(g.A_face, g.r_face)


def test_output_index_on_nodes():
    g = RadialGrid(200, 0.02)  # 200 是 20 的倍数 → 0.1 cm 落在节点
    assert g.output_index(0.0) == 0
    assert g.output_index(2.0) == 200
    assert g.output_index(0.1) == 10
    assert g.output_index(0.5) == 50


def test_output_index_offgrid_raises():
    g = RadialGrid(30, 0.02)  # 30 不是 20 的倍数
    with pytest.raises(ValueError):
        g.output_index(0.1)


def test_ref_volume_sum():
    g = RefGrid(200)
    assert np.sum(g.V) == pytest.approx(0.5, rel=1e-14)


def test_ref_cbar_constant_field():
    g = RefGrid(200)
    c = np.full(g.N + 1, 2.55)
    assert g.cbar(c) == pytest.approx(2.55, rel=1e-13)
