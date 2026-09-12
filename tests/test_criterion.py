"""criterion 单元测试：合成回归（重构场根 vs 端点极值插值）。"""
import numpy as np
import pytest

from drymodel import criterion as CR


def test_synthetic_regression_true_root():
    """两节点 (0.151,0.1505)→(0.148,0.1495)，Δt=1 s：重构场真根 0.5 s。"""
    C_n = np.array([0.151, 0.1505])
    C_np1 = np.array([0.148, 0.1495])
    t_cross = CR.event_root_reconstructed(0.0, C_n, 1.0, C_np1, thr=0.15)
    assert t_cross == pytest.approx(0.5, abs=1e-9)


def test_endpoint_extremum_interp_would_give_wrong_root():
    """对照：先取端点极值再线性插值给 0.6667 s（错误方法，不采用）。"""
    max_n = max(0.151, 0.1505)      # 0.151
    max_np1 = max(0.148, 0.1495)    # 0.1495
    a_wrong = (max_n - 0.15) / (max_n - max_np1)
    assert a_wrong == pytest.approx(2.0 / 3.0, abs=1e-9)
    # 与正确根 0.5 不同 → 证明必须在重构场上求根
    assert abs(a_wrong - 0.5) > 0.1


def test_no_crossing_returns_none():
    C_n = np.array([0.20, 0.19])
    C_np1 = np.array([0.16, 0.17])   # 终点仍 >0.15
    assert CR.event_root_reconstructed(0.0, C_n, 1.0, C_np1) is None


def test_already_below_returns_none():
    C_n = np.array([0.14, 0.13])     # 起点已达标
    C_np1 = np.array([0.12, 0.11])
    assert CR.event_root_reconstructed(0.0, C_n, 1.0, C_np1) is None


def test_locate_cross_dense_linear():
    """连续解 cmax(t)=0.2-0.001 t，过 0.15 于 t=50。"""
    def cmax_fn(t):
        return 0.2 - 0.001 * t
    tc = CR.locate_cross_dense(cmax_fn, 0.0, 100.0, thr=0.15)
    assert tc == pytest.approx(50.0, abs=1e-6)


def test_first_sample_below():
    def cmax_fn(t):
        return 0.2 - 0.001 * t       # <0.15 当 t>50
    ts = CR.first_sample_below(cmax_fn, 50.0, 60.0, thr=0.15, t_max=400)
    assert ts == 60.0  # 首个 60 s 倍数 ≥50 且 cmax(60)=0.14<0.15
