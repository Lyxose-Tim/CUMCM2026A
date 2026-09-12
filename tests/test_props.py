"""物性单元测试：附录 2/3/4 公式复现 + 界面系数性质。"""
import math

import numpy as np
import pytest

from drymodel import props as P


def test_q1_constants_and_D():
    q1 = P.PropsQ1()
    for C in (0.05, 0.5, 2.55):
        assert q1.rho(C) == pytest.approx(820.0)
        assert q1.cp(C) == pytest.approx(2600.0)
        assert q1.k(C) == pytest.approx(0.36)
        assert float(q1.D(C)) == pytest.approx(7e-9 * math.exp(-0.89 / C), rel=1e-13)
    # b = rho*cp
    assert float(q1.b(2.55)) == pytest.approx(820.0 * 2600.0)


def test_q23_formulas():
    q = P.PropsQ23()
    for C in (0.05, 1.0, 2.55):
        assert float(q.rho(C)) == pytest.approx(650 + 128 * C)
        assert float(q.cp(C)) == pytest.approx(1450 + 2736 * C / (C + 1))
        assert float(q.k(C)) == pytest.approx(0.21 + 0.38 * C / (C + 1))
        for T in (301.15, 323.1489344262):
            expected = 2.4e-3 * math.exp(-0.45 / C) * math.exp(-3850 / T)
            assert float(q.D(C, T)) == pytest.approx(expected, rel=1e-12)


def test_q4_formulas():
    q = P.PropsQ4()
    for C in (0.05, 1.0, 2.55):
        assert float(q.rho(C)) == pytest.approx(760 + 90 * C)
        assert float(q.cp(C)) == pytest.approx(1850 + 2150 * C / (C + 1))
        assert float(q.k(C)) == pytest.approx(0.12 + 0.20 * C / (C + 1))
        for T in (301.15, 323.15):
            expected = 4.2e-4 * math.exp(-0.30 / C) * math.exp(-3850 / T)
            assert float(q.D(C, T)) == pytest.approx(expected, rel=1e-12)


def test_D_guard_nonpositive_C():
    """C<=0 → D:=0，不报错、不裁剪。"""
    q1 = P.PropsQ1()
    D = q1.D(np.array([-0.1, 0.0, 0.5]))
    assert D[0] == 0.0 and D[1] == 0.0 and D[2] > 0.0


def test_D_guard_exp_underflow():
    """极小正 C 使指数 <-700 → D 取 0（不下溢告警）。"""
    q1 = P.PropsQ1()
    D = q1.D(np.array([1e-4]))  # -0.89/1e-4 = -8900 < -700
    assert D[0] == 0.0


def test_harmonic_interface():
    a = np.array([1.0, 0.0, 2.0])
    b = np.array([3.0, 5.0, 2.0])
    Df = P.D_face_harmonic(a, b)
    assert Df[0] == pytest.approx(2 * 1 * 3 / 4)
    assert Df[1] == 0.0            # 任一为 0 → 0
    assert Df[2] == pytest.approx(2.0)


def test_integral_interface_degenerates_to_pointwise():
    """近等值极限 Cj→Ci 时积分界面 → D(Ci, Tb)。"""
    q = P.PropsQ23()
    Ci = np.array([1.2345])
    Cj = np.array([1.2345])
    Tb = np.array([320.0])
    Dint = P.D_face_integral(Ci, Cj, Tb, q.D, npts=8)
    assert float(Dint[0]) == pytest.approx(float(q.D(Ci, Tb)[0]), rel=1e-12)


def test_integral_8_vs_16_close():
    q = P.PropsQ23()
    Ci = np.array([0.3])
    Cj = np.array([2.5])
    Tb = np.array([310.0])
    d8 = P.D_face_integral(Ci, Cj, Tb, q.D, npts=8)
    d16 = P.D_face_integral(Ci, Cj, Tb, q.D, npts=16)
    assert float(d8[0]) == pytest.approx(float(d16[0]), rel=1e-6)
