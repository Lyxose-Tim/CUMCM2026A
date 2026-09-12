"""production.py —— D12 授权后的正式续算与官方导出（步 4–5）。

- 分问最终配置（final_N + integral 界面 + BDF）；**事件、严格合格采样、续算检查、表值共用同一条轨迹**
  （merge_bdf 拼接；禁区间外稠密输出）。
- 导出 result1–4 与表 1–6；导出后跑 V-8（跨文件）/V-9（工作簿结构）。
- **实现与测试不必等待授权**：test_mode 可用缩比配置写入临时目录，正式 outputs/ 不受影响。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import criterion as CR
from . import data_io
from . import postprocess as PP
from . import runners
from . import solver_bdf as SBDF
from . import writers as W

A1 = "时间\\到药材中心的距离"
COLS21 = runners.RESULT_COLS_CM                 # 0.0..2.0
COLS20 = [round(0.1 * j, 4) for j in range(20)]  # 0.0..1.9
KELVIN = 273.15


def _final_config(cfg, question, override=None):
    """分问最终生效配置：final_N（授权后登记）+ 候选界面/求积点。override 供测试注入。"""
    if override and question in override:
        return override[question]
    pq = cfg.raw["numerics"]["per_question"][question]
    fn = pq["final_N"]
    if fn is None:
        raise RuntimeError(f"{question}.final_N 未登记（D12 未授权）；不得用探针值代填")
    rc = cfg.resolved(question, "candidate")
    return {"N": int(fn), "interface": rc["interface"], "npts": rc["integral_npts"],
            "t_cap_h": rc["t_cap_h"]}


# --------------------------------------------------------------------------
def produce_q1(cfg, fc, outdir):
    """Q1：result1（1 s×0.1 cm，温度/水分浓度）+ 表 1/2。仅 1800 s，无达标事件。"""
    N, interface, npts = fc["N"], fc["interface"], fc["npts"]
    op, _ = runners.build_fixed_operator(cfg, "q1", N, interface=interface,
                                         integral_npts=npts, decoupled=True)
    n = N + 1
    res = SBDF.integrate_bdf(op, runners.initial_state(cfg, N), 0.0, 1800.0, cfg.bdf,
                             breakpoints=(), threshold=None)
    if not res.ok:
        raise RuntimeError(f"Q1 生产积分失败：{res.message}")
    node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
    ts = np.arange(1, 1801)
    Y = res.eval(ts)
    C = Y[:, node_idx]
    T = Y[:, node_idx + n] - KELVIN
    W.write_result12(outdir / "result1.xlsx", ts, C, T, COLS21, A1)

    # 表 1/2：7 时刻 × 5 位置
    tt = np.array([100, 300, 600, 900, 1200, 1500, 1800])
    idx = np.array([op.grid.output_index(rc) for rc in (0.0, 0.5, 1.0, 1.5, 2.0)])
    Yt = res.eval(tt)
    tbl1_T = Yt[:, idx + n] - KELVIN
    tbl2_C = Yt[:, idx]
    _write_table(outdir / "table1_temp.csv", tt, ["r0", "r0.5", "r1.0", "r1.5", "r2.0"], tbl1_T)
    _write_table(outdir / "table2_moist.csv", tt, ["r0", "r0.5", "r1.0", "r1.5", "r2.0"], tbl2_C)
    return {"N": N, "result1_rows": len(ts)}


def _single_trajectory(cfg, question, fc, moving):
    """事件轨迹 + 续算拼成同一条轨迹（task 4）。返回 (full, op, det)。"""
    det, res, cont, op = runners.q23_detect(
        cfg, N=fc["N"], interface=fc["interface"], npts=fc["npts"],
        question=question, moving=moving, t_cap_h=fc["t_cap_h"])
    full = SBDF.merge_bdf(res, cont)
    return full, op, det


def _sample_cols(full, ts, node_idx, n, chunk=8000):
    """分块求值，仅提取输出列的 C、T（控制内存；避免全场 (len,2n) 巨阵）。"""
    ts = np.asarray(ts, float)
    Cc, Tc = [], []
    for i in range(0, len(ts), chunk):
        Y = full.eval(ts[i:i + chunk])
        Cc.append(Y[:, node_idx].copy())
        Tc.append(Y[:, node_idx + n].copy())
    return np.vstack(Cc), np.vstack(Tc)


def produce_q23(cfg, fc, outdir, *, result2_mode="until_dry_1s"):
    """Q2/Q3：result2（1 s，达标终点）、result3（60 s，至 t_sample）+ 表 3/4/5。同一轨迹。"""
    full, op, det = _single_trajectory(cfg, "q23", fc, moving=False)
    n = op.N + 1
    thr = cfg.threshold
    t_cross = det.t_cross
    t_sample = det.t_sample

    # 失败阻断：续算回穿检查、严格合格末行
    if not det.post_ok:
        raise RuntimeError(f"Q23 生产失败：续算回穿检查未过（post_max_cmax={det.post_max_cmax}）")

    def ev(t):
        if t > full.t_end + 1e-6:
            raise ValueError(f"Q23 生产采样越界 t={t}")
        return full.eval([float(t)])[0]

    if t_sample is None or CR.cmax(ev(float(t_sample))[:n]) >= thr:
        raise RuntimeError(f"Q23 生产失败：严格合格末行 t_sample={t_sample} 实测 C_max≥{thr}")

    # t_end_1s（同一轨迹求值）
    t_end_1s = int(np.ceil(t_cross))
    while CR.cmax(ev(float(t_end_1s))[:n]) >= thr:
        t_end_1s += 1
    if result2_mode == "3h":
        t2_end = 10800
    elif result2_mode == "72h":
        t2_end = 259200
    else:
        t2_end = t_end_1s
    t2_end = min(t2_end, int(full.t_end))

    node_idx = np.array([op.grid.output_index(rc) for rc in COLS21])
    # result2（1 s）——分块采样避免内存爆
    ts2 = np.arange(1, t2_end + 1)
    C2, T2 = _sample_cols(full, ts2, node_idx, n)
    W.write_result12(outdir / "result2.xlsx", ts2, C2, T2 - KELVIN, COLS21, A1)
    # result3（60 s 至 t_sample）
    ts3 = np.arange(60, int(t_sample) + 1, 60)
    C3, _ = _sample_cols(full, ts3, node_idx, n)
    W.write_result34(outdir / "result3.xlsx", ts3, C3, COLS21, A1, sheet_name="Sheet1")

    # 表 3/4/5
    t34 = np.arange(0.5, 3.01, 0.5) * 3600.0
    idx5 = np.array([op.grid.output_index(rc) for rc in (0.0, 0.5, 1.0, 1.5, 2.0)])
    Y34 = full.eval(t34)
    _write_table(outdir / "table3_temp.csv", (t34 / 3600), ["r0", "r0.5", "r1.0", "r1.5", "r2.0"],
                 Y34[:, idx5 + n] - KELVIN)
    _write_table(outdir / "table4_moist.csv", (t34 / 3600), ["r0", "r0.5", "r1.0", "r1.5", "r2.0"],
                 Y34[:, idx5])
    t5 = runners._dedup_hours_last(t_cross, 6.0)
    tbl5 = [[tt / 3600] + [float(ev(tt)[i]) for i in idx5] for tt in t5]
    _write_rows(outdir / "table5_moist.csv", ["t_h", "r0", "r0.5", "r1.0", "r1.5", "r2.0"], tbl5)
    return {"N": op.N, "t_star_s": float(t_cross), "t_star_h": t_cross / 3600, "t_end_1s": t_end_1s,
            "t_sample_s": int(t_sample), "post_max_cmax": float(det.post_max_cmax),
            "result2_rows": len(ts2), "result3_rows": len(ts3)}


def produce_q4(cfg, fc, outdir):
    """Q4：result4（60 s，20 列+表面，域外留空）+ 表 6 + 半径小表。同一轨迹。"""
    full, op, det = _single_trajectory(cfg, "q4", fc, moving=True)
    n = op.N + 1
    thr = cfg.threshold
    t_cross = det.t_cross
    t_sample = det.t_sample
    radius = data_io.make_radius_function(cfg)

    def ev(t):
        if t > full.t_end + 1e-6:
            raise ValueError(f"Q4 生产采样越界 t={t}")
        return full.eval([float(t)])[0]

    if not det.post_ok:
        raise RuntimeError(f"Q4 生产失败：续算回穿检查未过（post_max_cmax={det.post_max_cmax}）")
    if t_sample is None or CR.cmax(ev(float(t_sample))[:n]) >= thr:
        raise RuntimeError(f"Q4 生产失败：严格合格末行 t_sample={t_sample} 实测 C_max≥{thr}")

    ts4 = np.arange(60, int(t_sample) + 1, 60)
    grid_rows, surf_vals, mask = [], [], []
    for t in ts4:
        c = ev(float(t))[:n]
        R_t = float(radius.R(float(t)))
        row = PP.sample_q4_row(c, op.grid, R_t, COLS20)
        grid_rows.append(row)
        surf_vals.append(float(c[-1]))
        mask.append([v is not None for v in row])
    W.write_result34(outdir / "result4.xlsx", ts4, grid_rows, COLS20, A1, sheet_name="Sheet1",
                     surface_header="药材表面", surface_values=surf_vals)

    t6 = runners._dedup_hours_last(t_cross, 6.0)
    tbl6, rad = [], []
    for tt in t6:
        c = ev(tt)[:n]
        R_t = float(radius.R(tt))
        r0 = PP.sample_q4_row(c, op.grid, R_t, [0.0, 0.5, 1.0])
        tbl6.append([tt / 3600, r0[0], r0[1], r0[2], float(c[-1])])
        rad.append([tt / 3600, R_t * 100, int(radius.is_extrapolated(tt))])
    _write_rows(outdir / "table6_moist.csv", ["t_h", "r0", "r0.5", "r1.0", "surface"], tbl6)
    _write_rows(outdir / "table6_radius.csv", ["t_h", "R_cm", "extrapolated"], rad)
    return {"N": op.N, "t_star_s": float(t_cross), "t_star_h": t_cross / 3600,
            "t_sample_s": int(t_sample), "post_max_cmax": float(det.post_max_cmax),
            "result4_rows": len(ts4), "mask": mask}


# --------------------------------------------------------------------------
def run_production(cfg, *, outputs_dir=None, override=None, result2_mode="until_dry_1s",
                   require_approved=True):
    """正式续算 + 官方导出 + V-8/V-9 终检。override/require_approved=False 供缩比测试。"""
    if require_approved and not cfg.raw["production"]["approved"]:
        return {"ok": False, "reason": "production.approved=false（D12 未授权），不生成官方 result1–4"}
    outdir = Path(outputs_dir) if outputs_dir else (cfgmod.PROJECT_ROOT / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    fc = {"q1": _final_config(cfg, "q1", override),
          "q23": _final_config(cfg, "q23", override),
          "q4": _final_config(cfg, "q4", override)}
    # 续算与导出（任一失败 → 阻断，返回 ok=False，不标产物为验收通过）
    try:
        r1 = produce_q1(cfg, fc["q1"], outdir)
        r23 = produce_q23(cfg, fc["q23"], outdir, result2_mode=result2_mode)
        r4 = produce_q4(cfg, fc["q4"], outdir)
    except Exception as e:
        return {"ok": False, "reason": f"生产续算/导出失败：{e}", "fc": fc}

    # V-9 工作簿结构（生产：全部数据行格式检查 full_format_check=True）
    ff = True
    v9 = {}
    v9["result1"] = W.verify_workbook(outdir / "result1.xlsx", expected_sheets=["温度", "水分浓度"],
                                      a1_text=A1, expected_cols=COLS21, n_data_rows=r1["result1_rows"],
                                      t_start=1, t_step=1, full_format_check=ff)
    v9["result2"] = W.verify_workbook(outdir / "result2.xlsx", expected_sheets=["温度", "水分浓度"],
                                      a1_text=A1, expected_cols=COLS21, n_data_rows=r23["result2_rows"],
                                      t_start=1, t_step=1, full_format_check=ff)
    v9["result3"] = W.verify_workbook(outdir / "result3.xlsx", expected_sheets=["Sheet1"],
                                      a1_text=A1, expected_cols=COLS21, n_data_rows=r23["result3_rows"],
                                      t_start=60, t_step=60, full_format_check=ff)
    v9["result4"] = W.verify_workbook(outdir / "result4.xlsx", expected_sheets=["Sheet1"],
                                      a1_text=A1, expected_cols=COLS20, n_data_rows=r4["result4_rows"],
                                      t_start=60, t_step=60, surface_header="药材表面", mask=r4["mask"],
                                      require_surface_nonempty=True, full_format_check=ff)
    # V-8 跨文件（零共同时间不能判通过）
    v8 = W.cross_file_check(outdir / "result2.xlsx", outdir / "result3.xlsx")
    v8_ok = v8["ok"] and v8["n_common_times"] > 0

    ok = all(v["ok"] for v in v9.values()) and v8_ok
    hashes = {name: _sha256_file(outdir / name) for name in
              ("result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx")}
    return {"ok": ok, "q1": r1, "q23": r23, "q4": r4, "V9": v9, "V8": v8, "v8_ok": v8_ok,
            "fc": fc, "file_sha256": hashes, "outdir": str(outdir)}


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
def _write_table(path, times, colnames, values):
    with open(path, "w", encoding="utf-8") as f:
        f.write("t," + ",".join(colnames) + "\n")
        for i, t in enumerate(np.asarray(times)):
            f.write(f"{t:g}," + ",".join(f"{values[i][j]:.4f}" for j in range(len(colnames))) + "\n")


def _write_rows(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
                             for v in r) + "\n")
