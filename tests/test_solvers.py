"""solver 单元测试：BE 对解析解一阶收敛；BDF 与 BE/半离散一致。"""
import numpy as np
import pytest

from drymodel import config as cfgmod, verify as V
from drymodel.grid import RadialGrid
from drymodel.operators import FVMOperator
from drymodel import solver_be as SBE
from drymodel import solver_bdf as SBDF


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def _const_heat_op(N, R0=0.02):
    props = V.ConstProps(820.0, 2600.0, 0.36, D_v=7e-9 * np.exp(-0.89 / 2.55))
    env = V.ConstEnv(323.15, 0.05)
    grid = RadialGrid(N, R0)
    return FVMOperator(grid, props, env, h=25.0, hm=8e-7, R0=R0,
                       interface="harmonic", decoupled=True)


def test_be_first_order_time(cfg):
    N = 400
    op = _const_heat_op(N)
    y0 = np.concatenate([np.full(N + 1, 2.55), np.full(N + 1, 301.15)])
    ref = V.hi_precision_reference(op, y0, [100.0])[0][N + 1:][-1]
    errs = []
    for dt in (1.0, 0.5, 0.25):
        r = SBE.integrate_be(op, y0, 100.0, dt, C0_ref=2.55, picard=cfg.picard,
                             retry=cfg.retry, record_times={100})
        errs.append(abs(r.T[-1][-1] - ref))
    orders = V.order_from_errors(errs)
    assert all(0.9 < o < 1.1 for o in orders)


def test_be_matches_analytic_spatial(cfg):
    """BE 细步 + N=400 逼近解析解（空间二阶主导）。"""
    N = 400
    op = _const_heat_op(N)
    y0 = np.concatenate([np.full(N + 1, 2.55), np.full(N + 1, 301.15)])
    r = SBE.integrate_be(op, y0, 100.0, 0.25, C0_ref=2.55, picard=cfg.picard,
                         retry=cfg.retry, record_times={100})
    Bi = 25.0 * 0.02 / 0.36
    alpha = 0.36 / (820.0 * 2600.0)
    Fo = alpha * 100.0 / 0.02 ** 2
    eta = op.grid.r / 0.02
    theta = V.analytic_cylinder_robin(eta, Fo, Bi, 80)
    T_ana = 323.15 + (301.15 - 323.15) * theta
    err = np.max(np.abs(r.T[-1] - T_ana))
    assert err < 5e-3   # 组合空间(≈2.6e-5)+时间(BE dt=0.25 ≈2e-3)误差


def test_bdf_matches_hiprec(cfg):
    """BDF（生产容差）与高精度参考解在 100 s 一致。"""
    N = 200
    op = _const_heat_op(N)
    y0 = np.concatenate([np.full(N + 1, 2.55), np.full(N + 1, 301.15)])
    ref = V.hi_precision_reference(op, y0, [100.0])[0]
    res = SBDF.integrate_bdf(op, y0, 0.0, 100.0, cfg.bdf, breakpoints=(), threshold=None)
    got = res.eval([100.0])[0]
    # BDF rtol=1e-8：T~307K 处绝对差 ~1e-6 K（相对 ~4e-9），C 差更小
    assert np.max(np.abs(got - ref)) < 1e-5


def test_bdf_dense_output_rejects_out_of_interval(cfg):
    op = _const_heat_op(20)
    y0 = np.concatenate([np.full(21, 2.55), np.full(21, 301.15)])
    res = SBDF.integrate_bdf(op, y0, 0.0, 1.0, cfg.bdf,
                             breakpoints=(), threshold=None)
    res.require_reached(1.0)
    with pytest.raises(ValueError, match="超出已积分区间"):
        res.eval([-1.0])
    with pytest.raises(ValueError, match="超出已积分区间"):
        res.eval([2.0])
    with pytest.raises(ValueError, match="有限"):
        res.eval([np.nan])


def test_bdf_concat_preserves_continuous_augmented_state(cfg):
    base = _const_heat_op(20)
    op = FVMOperator(
        base.grid, base.props, base.env, h=base.h, hm=base.hm, R0=base.R0,
        interface="harmonic", augmented=True,
    )
    y0 = np.concatenate([np.full(21, 2.55), np.full(21, 301.15), [0.0]])
    first = SBDF.integrate_bdf(op, y0, 0.0, 1.0, cfg.bdf,
                               breakpoints=(), threshold=None)
    second = SBDF.continue_bdf(op, first.y_end, 1.0, 1.0, cfg.bdf)
    joined = first.concat(second)
    joined.require_reached(2.0)
    assert joined.eval([1.0])[0, -1] == pytest.approx(first.y_end[-1])
    assert joined.eval([2.0])[0, -1] == pytest.approx(second.y_end[-1])


def test_bdf_concat_rejects_time_gap(cfg):
    op = _const_heat_op(20)
    y0 = np.concatenate([np.full(21, 2.55), np.full(21, 301.15)])
    first = SBDF.integrate_bdf(op, y0, 0.0, 1.0, cfg.bdf, breakpoints=())
    second = SBDF.integrate_bdf(op, first.y_end, 1.1, 2.0, cfg.bdf, breakpoints=())
    with pytest.raises(ValueError, match="时间不连续"):
        first.concat(second)
