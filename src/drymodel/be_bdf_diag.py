"""be_bdf_diag.py —— 辅助热问题的时间积分对照诊断（第五轮返修，仅诊断，不触发生产）。

在常物性、恒环境的一维径向辅助热问题（附录 2 常物性、$T_{air}\\equiv50°C$、$N=400$、$0$--$100$ s）
上，比较三种时间积分：
    - 向后 Euler，$\\Delta t=1,0.5,0.25$ s（本模型正式计算另用的验证基线格式）；
    - 自适应 BDF，当前计算容差（config 的 bdf.rtol，默认 1e-8）；
    - 紧容差 BDF 参考（rtol 1e-11、atol 1e-13，见 verify.hi_precision_reference）。
全部使用同一空间网格与同一半离散 ODE，因此差异只来自时间积分。

只写诊断 CSV（exports/be_bdf_diag.csv 端点汇总、exports/be_bdf_history.csv 表面温度时程），
不改动 result1--4，也不重跑生产或整套测试。运行：``python -m drymodel.be_bdf_diag``。
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import config as cfgmod
from . import verify as V
from . import solver_be as SBE
from . import solver_bdf as SBDF
from .grid import RadialGrid
from .operators import FVMOperator

ROOT = cfgmod.PROJECT_ROOT
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

# 辅助热问题设定（与 tests/test_solvers.py 的 _const_heat_op 一致）
N = 400
R0 = 0.02
T0_K = 301.15            # 28 °C
TAIR_K = 323.15          # 50 °C
CENV = 0.05
H, HM = 25.0, 8e-7
T_END = 100.0
DTS = (1.0, 0.5, 0.25)


def _const_heat_op(N=N, R0=R0):
    props = V.ConstProps(820.0, 2600.0, 0.36, D_v=7e-9 * np.exp(-0.89 / 2.55))
    env = V.ConstEnv(TAIR_K, CENV)
    grid = RadialGrid(N, R0)
    return FVMOperator(grid, props, env, h=H, hm=HM, R0=R0,
                       interface="harmonic", decoupled=True)


def run(cfg=None):
    cfg = cfg or cfgmod.load_config()
    op = _const_heat_op()
    y0 = np.concatenate([np.full(N + 1, 2.55), np.full(N + 1, T0_K)])
    out_t = list(range(0, int(T_END) + 1, 2))          # 公共输出时刻（每 2 s）
    surf = op.N                                         # 表面温度节点在温度段的最后一个

    # --- 紧容差 BDF 参考（rtol 1e-11、atol 1e-13）---
    ref_states = V.hi_precision_reference(op, y0, out_t)          # (len, 2(N+1))
    T_ref = ref_states[:, N + 1:][:, -1]                         # 表面温度时程
    T_ref_end = float(T_ref[-1])

    hist = {"t": list(out_t), "T_ref_tightBDF": [float(v) for v in T_ref]}

    # --- 向后 Euler：三种步长 ---
    be_rows = []
    be_errs = []
    for dt in DTS:
        r = SBE.integrate_be(op, y0, T_END, dt, C0_ref=cfg.C0, picard=cfg.picard,
                             retry=cfg.retry, record_times=set(out_t))
        if not r.ok:
            raise RuntimeError(f"BE 积分失败 (dt={dt}): {r.message}")
        tmap = {int(round(tt)): i for i, tt in enumerate(r.t)}
        Ts = [float(r.T[tmap[tt]][-1]) for tt in out_t]
        hist[f"T_be_dt{dt}"] = Ts
        err_end = abs(Ts[-1] - T_ref_end)
        be_errs.append(err_end)
        be_rows.append({"method": "向后Euler", "N": N, "control": f"dt={dt}s",
                        "T_surf_100s_K": Ts[-1], "T_surf_100s_C": Ts[-1] - 273.15,
                        "diff_vs_tightBDF_K": err_end})
    orders = V.order_from_errors(be_errs)               # dt 减半观测阶
    # 把观测阶写回 BE 行（相邻两步之间）
    for k in range(1, len(be_rows)):
        be_rows[k]["obs_order"] = float(orders[k - 1])

    # --- 自适应 BDF：当前计算容差 ---
    resb = SBDF.integrate_bdf(op, y0, 0.0, T_END, cfg.bdf, breakpoints=(), threshold=None)
    Tb = [float(resb.eval([float(tt)])[0][N + 1:][-1]) for tt in out_t]
    hist["T_bdf_prod"] = Tb
    bdf_err_end = abs(Tb[-1] - T_ref_end)
    bdf_row = {"method": "自适应BDF", "N": N, "control": f"rtol={cfg.bdf['rtol']:g}",
               "T_surf_100s_K": Tb[-1], "T_surf_100s_C": Tb[-1] - 273.15,
               "diff_vs_tightBDF_K": bdf_err_end}

    ref_row = {"method": "紧容差BDF参考", "N": N, "control": "rtol=1e-11,atol=1e-13",
               "T_surf_100s_K": T_ref_end, "T_surf_100s_C": T_ref_end - 273.15,
               "diff_vs_tightBDF_K": 0.0}

    # --- 写端点汇总 CSV ---
    rows = be_rows + [bdf_row, ref_row]
    cols = ["method", "N", "control", "T_surf_100s_K", "T_surf_100s_C",
            "diff_vs_tightBDF_K", "obs_order"]
    with (EXPORTS / "be_bdf_diag.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rr in rows:
            w.writerow({c: rr.get(c, "") for c in cols})

    # --- 写时程 CSV ---
    hist_cols = ["t", "T_ref_tightBDF"] + [f"T_be_dt{dt}" for dt in DTS] + ["T_bdf_prod"]
    with (EXPORTS / "be_bdf_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hist_cols)
        for i, tt in enumerate(out_t):
            w.writerow([tt] + [f"{hist[c][i]:.10g}" for c in hist_cols[1:]])

    summary = {
        "aux_problem": {"N": N, "R0_m": R0, "T0_C": 28.0, "Tair_C": 50.0,
                        "t_end_s": T_END, "interface": "harmonic", "decoupled": True},
        "be": be_rows, "bdf_prod": bdf_row, "ref": ref_row,
        "be_orders_dt_halving": [float(o) for o in orders],
        "bdf_prod_diff_vs_ref_K": bdf_err_end,
    }
    (EXPORTS / "be_bdf_diag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 控制台报告 ---
    print("辅助热问题时间积分对照（N=400, 0-100 s, 表面温度 @100 s）")
    print(f"  紧容差BDF参考 T_surf(100s) = {T_ref_end:.10f} K = {T_ref_end-273.15:.6f} °C")
    for rr in be_rows:
        print(f"  BE {rr['control']:>10}: T={rr['T_surf_100s_K']:.10f} K, "
              f"|Δ|={rr['diff_vs_tightBDF_K']:.3e} K, order={rr.get('obs_order','-')}")
    print(f"  BDF {bdf_row['control']:>10}: T={bdf_row['T_surf_100s_K']:.10f} K, "
          f"|Δ|={bdf_err_end:.3e} K")
    print("  写出：exports/be_bdf_diag.csv, be_bdf_history.csv, be_bdf_diag_summary.json")
    return summary


if __name__ == "__main__":
    run()
