"""config_check.py（增强版）—— 校验 A题_config.yaml。

对齐《A题_建模方案.md》§9.3 的检查器规格：
- **显式异常**（`ConfigError`），不只用 `assert`——Python `-O` 优化模式会移除 `assert`，
  本检查器在 `-O` 下仍生效。
- 校验：物理常数、公式编号（字符串与冻结规格逐字一致）、数值有穷性、正范围、
  整数网格数、每问物性组、固定长度主线、事件类型、正式输出终点与单位。
- **拒绝八类非法变异**（§9.3；作负例单元测试）：
  1) Q1 密度改 −820；2) Q2/Q3 的 D 前因子改 2.4e3；3) Q4 的 D 改 0；
  4) N_default 改 −200；5) BDF rtol 改负；6) result2 终点改 3h；
  7) energy_form 改 enthalpy_advective；8) q4.axial 改 shrinking_L。
- 主线题给值（h、hm 等）保持不变；灵敏度走单独 scenario 倍率层（不在此改基础配置）。

注意：**配置解析通过 ≠ 模型验证通过**（生产配置须经 V-1~V-13 验收）。
"""
from __future__ import annotations

import math
import sys
from typing import Any

REQ_TOP = [
    "version", "geometry", "initial", "bc", "air", "radius", "props",
    "numerics", "q4", "criterion", "output", "acceptance", "production", "storage",
]

VALID_SOURCE_LITERALS = {"题面Q1", "附录2", "附录3", "附录4", "附件1", "附件2"}
VALID_SOURCE_PREFIXES = ("D", "B", "H")

# 冻结的公式字符串（与题面附录 2/3/4 逐字一致）
EXPECTED_FORMULAS = {
    "props.q1.D_formula": "7e-9*exp(-0.89/C)",
    "props.q23.rho_formula": "650+128*C",
    "props.q23.cp_formula": "1450+2736*C/(C+1)",
    "props.q23.k_formula": "0.21+0.38*C/(C+1)",
    "props.q23.D_formula": "2.4e-3*exp(-0.45/C)*exp(-3850/T_K)",
    "props.q4.rho_formula": "760+90*C",
    "props.q4.cp_formula": "1850+2150*C/(C+1)",
    "props.q4.k_formula": "0.12+0.20*C/(C+1)",
    "props.q4.D_formula": "4.2e-4*exp(-0.30/C)*exp(-3850/T_K)",
}

A1_TEXT_EXPECTED = "时间\\到药材中心的距离"  # 解析后含单个反斜杠


class ConfigError(ValueError):
    """配置校验失败。"""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _get(d: dict, path: str) -> Any:
    v: Any = d
    for p in path.split("."):
        if not isinstance(v, dict) or p not in v:
            raise ConfigError(f"缺少配置键 {path}（在 {p} 处中断）")
        v = v[p]
    return v


def num(d: dict, path: str, unit: str | None = None, *, source: bool = True,
        positive: bool = False, finite: bool = True) -> float:
    v = _get(d, path)
    require(isinstance(v, dict) and isinstance(v.get("value"), (int, float))
            and not isinstance(v.get("value"), bool),
            f"{path}: 需 {{value, unit, source}} 且 value 为数值")
    val = float(v["value"])
    if finite:
        require(math.isfinite(val), f"{path}: value 非有限")
    if positive:
        require(val > 0.0, f"{path}: value 须为正，得 {val}")
    if unit is not None:
        require(v.get("unit") == unit, f"{path}: unit 应为 {unit!r}，得 {v.get('unit')!r}")
    if source:
        s = v.get("source", "")
        ok = s in VALID_SOURCE_LITERALS or str(s).startswith(VALID_SOURCE_PREFIXES)
        require(ok, f"{path}: source 标签缺失或非法（得 {s!r}）")
    return val


def check(cfg: dict) -> bool:
    """逐项校验；任一不满足抛 ConfigError。全部通过返回 True。"""
    require(isinstance(cfg, dict), "配置根须为映射")
    for k in REQ_TOP:
        require(k in cfg, f"缺少顶层键 {k}")

    # ---- 几何 / 初值（正范围 + 固定值）----
    R0 = num(cfg, "geometry.R0", "m", positive=True)
    L = num(cfg, "geometry.L", "m", positive=True)
    require(R0 == 0.02, f"geometry.R0 应为 0.02 m，得 {R0}")
    require(L == 0.25, f"geometry.L 应为 0.25 m，得 {L}")
    require(num(cfg, "initial.T0", "degC") == 28.0, "initial.T0 应为 28.0 degC")
    require(num(cfg, "initial.C0", "kg/kg", positive=True) == 2.55, "initial.C0 应为 2.55")

    # ---- 边界（主线题给值保持不变）----
    require(num(cfg, "bc.h", "W/(m^2 K)", positive=True) == 25.0, "bc.h 应固定为 25.0")
    require(num(cfg, "bc.hm", "m/s", positive=True) == 8.0e-7, "bc.hm 应固定为 8e-7")
    require(cfg["bc"]["mass_basis"] == "effective_C",
            f"bc.mass_basis 应为 effective_C（D02-A），得 {cfg['bc'].get('mass_basis')!r}")
    require(cfg["bc"]["heat_latent"] is False, "bc.heat_latent 主线应为 False（D05-A）")

    # ---- 空气数据 ----
    a = cfg["air"]
    require(a.get("read_only") is True, "air.read_only 应为 True（原始附件只读）")
    require(a.get("interp") in ("linear", "smooth121"), "air.interp 应为 linear/smooth121")
    require(a.get("extrapolation") in ("hold_window_mean", "hold_last"),
            "air.extrapolation 应为 hold_window_mean/hold_last")
    require(a.get("window_s") == [10800, 14400], "air.window_s 应为 [10800,14400]")
    require(a.get("breakpoints_s") == [14400], "air.breakpoints_s 应为 [14400]")

    # ---- 半径 ----
    r = cfg["radius"]
    require(r.get("interp") == "linear", "radius.interp 应为 linear（B12a）")
    require(r.get("after_72h") == "hold_last", "radius.after_72h 应为 hold_last（B12b）")
    inside_tol = float(r.get("inside_tol", -1))
    require(math.isfinite(inside_tol) and inside_tol > 0, "radius.inside_tol 须为正有限值")

    # ---- 物性：常数正范围 + 公式字符串逐字一致（拒绝变异 1/2/3）----
    require(num(cfg, "props.q1.rho", "kg/m^3", positive=True) == 820.0,
            "props.q1.rho 应为正的 820（变异 −820 被拒）")   # 变异 1
    require(num(cfg, "props.q1.cp", "J/(kg K)", positive=True) == 2600.0, "props.q1.cp 应为 2600")
    require(num(cfg, "props.q1.k", "W/(m K)", positive=True) == 0.36, "props.q1.k 应为 0.36")
    for key, expected in EXPECTED_FORMULAS.items():
        got = _get(cfg, key)
        require(got == expected,
                f"{key}: 公式字符串与冻结规格不一致（期望 {expected!r}，得 {got!r}）")  # 变异 2/3
    require(_get(cfg, "props.q23.source") == "附录3", "props.q23.source 应为 附录3")
    require(_get(cfg, "props.q4.source") == "附录4", "props.q4.source 应为 附录4")
    require(cfg["props"]["rho_role"] == "effective_heat_capacity",
            "props.rho_role 应为 effective_heat_capacity（D06 Ⅰ-A）")
    require(cfg["props"]["energy_form"] == "effective",
            f"props.energy_form 主线应为 effective（拒绝 enthalpy_advective），得 "
            f"{cfg['props'].get('energy_form')!r}")  # 变异 7
    guards = cfg["props"].get("guards", {})
    require(float(guards.get("exp_underflow_arg", 1)) < 0, "props.guards.exp_underflow_arg 应为负")

    # ---- 数值 ----
    n = cfg["numerics"]
    require(n.get("interface") in ("harmonic", "integral"), "numerics.interface 应为 harmonic/integral")
    require(n.get("scheme_baseline") == "backward_euler", "numerics.scheme_baseline 应为 backward_euler")
    Nd = n.get("N_default")
    require(isinstance(Nd, int) and not isinstance(Nd, bool) and Nd > 0,
            f"numerics.N_default 须为正整数（拒绝 −200），得 {Nd!r}")  # 变异 4
    require(isinstance(n.get("N_verify"), list) and len(n["N_verify"]) > 0
            and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in n["N_verify"]),
            "numerics.N_verify 须为正整数列表")
    rtol = n["bdf"].get("rtol")
    require(isinstance(rtol, (int, float)) and math.isfinite(rtol) and rtol > 0 and rtol <= 1e-8,
            f"numerics.bdf.rtol 须为 (0, 1e-8]（拒绝负值），得 {rtol!r}")  # 变异 5
    require(float(n["bdf"].get("atol_C", -1)) > 0 and float(n["bdf"].get("atol_T_K", -1)) > 0,
            "numerics.bdf.atol_C / atol_T_K 须为正")
    require(int(n["picard"].get("max_iter", 0)) >= 1, "numerics.picard.max_iter 须 >=1")
    require(float(n["retry"].get("dt_min_s", -1)) > 0, "numerics.retry.dt_min_s 须为正")
    require(float(n["dt_be_s"]) > 0, "numerics.dt_be_s 须为正")

    # ---- Q4 运动学（拒绝变异 8）----
    q = cfg["q4"]
    require(q.get("kinematics") == "affine", "q4.kinematics 应为 affine（D10 K-1）")
    require(q.get("frame") == "reference_x", "q4.frame 应为 reference_x（D10 N-1）")
    require(q.get("mesh_velocity") == "solid", "q4.mesh_velocity 应为 solid")
    require(q.get("axial") == "fixed_L",
            f"q4.axial 应为 fixed_L（长度固定；拒绝 shrinking_L），得 {q.get('axial')!r}")  # 变异 8

    # ---- 判据 ----
    c = cfg["criterion"]
    require(c.get("threshold") == 0.15, "criterion.threshold 应为 0.15")
    require(c.get("domain") == "full_unrounded", "criterion.domain 应为 full_unrounded")
    require(c.get("event") == "continuous_root_on_reconstructed_field",
            "criterion.event 应为 continuous_root_on_reconstructed_field")
    require(float(c.get("post_margin_s", -1)) > 0, "criterion.post_margin_s 须为正")

    # ---- 输出（单位/终点；拒绝变异 6）----
    o = cfg["output"]
    require(o.get("decimals") == 4, "output.decimals 应为 4")
    require(int(o.get("excel_max_rows_including_header", 0)) == 1048576,
            "output.excel_max_rows_including_header 应为 1048576")
    require(o["result4"].get("outside_fill") == "blank", "output.result4.outside_fill 应为 blank")
    require(o["result4"].get("surface_col") == "药材表面", "output.result4.surface_col 应为 药材表面")
    require(o.get("a1_text") == A1_TEXT_EXPECTED, "output.a1_text 应为 时间\\到药材中心的距离")
    require(o["result2"].get("t_end") == "until_dry_1s",
            f"output.result2.t_end 应为 until_dry_1s（拒绝 3h/72h），得 "
            f"{o['result2'].get('t_end')!r}")  # 变异 6

    # ---- 验收阈值 ----
    acc = cfg["acceptance"]
    require(acc.get("t_star_h") == 0.02, "acceptance.t_star_h 应为 0.02")
    require(float(acc["table_points"].get("dC")) == 5.0e-4, "acceptance.table_points.dC 应为 5e-4")

    # ---- 生产配置门控 ----
    prod = cfg["production"]
    require(prod.get("approved") is False or prod.get("config_id"),
            "production.approved=True 时须给 config_id")

    # ---- 定点返修新增字段（PATCH-04/05）----
    _check_returrepair_fields(cfg)
    return True


def _check_returrepair_fields(cfg: dict) -> None:
    """校验定点返修新增字段：分问数值配置、8/16 点求积、W/m 参考尺度、result2 测试模式。"""
    n = cfg["numerics"]

    # 8/16 点求积（独立数值键）
    q = n.get("quadrature", {})
    ip = q.get("interface_points")
    ipc = q.get("interface_points_check")
    require(isinstance(ip, int) and not isinstance(ip, bool) and ip > 0,
            "numerics.quadrature.interface_points 须为正整数")
    require(isinstance(ipc, int) and not isinstance(ipc, bool) and ipc > ip,
            "numerics.quadrature.interface_points_check 须为大于生产点数的正整数（如 16）")

    # 分问数值配置
    pq = n.get("per_question", {})
    for qkey, expect_event in (("q1", False), ("q23", True), ("q4", True)):
        require(qkey in pq, f"numerics.per_question 缺少 {qkey}")
        blk = pq[qkey]
        cand = blk.get("candidate_N")
        require(isinstance(cand, list) and len(cand) > 0
                and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in cand),
                f"per_question.{qkey}.candidate_N 须为正整数列表")
        fn = blk.get("final_N", None)
        require(fn is None or (isinstance(fn, int) and not isinstance(fn, bool) and fn > 0),
                f"per_question.{qkey}.final_N 须为 null（待定）或正整数，不得填历史探针值")
        require(blk.get("applies_event") is expect_event,
                f"per_question.{qkey}.applies_event 应为 {expect_event}（Q1 不适用达标事件）")

    # 未授权时最终网格保持待定（不填探针值）
    if cfg["production"].get("approved") is False:
        for qkey in ("q1", "q23", "q4"):
            require(pq[qkey].get("final_N") is None,
                    f"未授权（approved=false）时 per_question.{qkey}.final_N 须为 null")

    # 达标事件仅 Q23/Q4
    require(cfg["criterion"].get("applies_to") == ["q23", "q4"],
            "criterion.applies_to 应为 ['q23','q4']（Q1 不适用达标事件）")

    # 能量残差 W/m 口径，参考尺度不含 L
    er = cfg["acceptance"]["energy_residual"]
    require(er.get("unit") == "W/m", "acceptance.energy_residual.unit 应为 W/m")
    ref = str(er.get("ref_scale", ""))
    require("L" not in ref, f"能量残差参考尺度不得含长度 L（W/m 口径）：{ref!r}")
    require("abs_W_per_m" in er, "acceptance.energy_residual 须给 abs_W_per_m（W/m 口径）")

    # result2 正式终点仅达标；3h/72h 仅测试模式
    require(cfg["output"]["result2"].get("t_end") == "until_dry_1s",
            "output.result2.t_end 正式导出仅允许 until_dry_1s")
    tm = cfg["output"].get("result2_test_modes", [])
    require(isinstance(tm, list) and set(tm) <= {"3h", "72h"},
            "output.result2_test_modes 仅可含 '3h'/'72h'（测试模式，不导出正式）")

    # 候选阶段生效配置（本轮：配置真正接入运行）
    cand = n.get("candidate")
    if cand is not None:
        require(cand.get("interface") in ("harmonic", "integral"),
                "numerics.candidate.interface 应为 harmonic/integral")
        cnp = cand.get("integral_npts")
        require(isinstance(cnp, int) and not isinstance(cnp, bool) and cnp > 0,
                "numerics.candidate.integral_npts 须为正整数")
        sN = cand.get("stage_N", {})
        for qkey in ("q1", "q23", "q4"):
            require(qkey in sN and isinstance(sN[qkey], int) and not isinstance(sN[qkey], bool)
                    and sN[qkey] in pq[qkey]["candidate_N"],
                    f"numerics.candidate.stage_N.{qkey} 须为该问 candidate_N 中的整数")
        tc = cand.get("t_cap_h", {})
        for qkey in ("q23", "q4"):
            require(_pos_finite(tc.get(qkey)), f"numerics.candidate.t_cap_h.{qkey} 须为正有限数")

    # 补齐：picard / retry / bdf.max_step / envelope_tol / air.sensitivity 的有限性·类型·正范围
    pic = n["picard"]
    for key in ("tol_dC", "tol_dT_K", "tol_res_C_rel", "tol_res_T_K_per_s"):
        require(_pos_finite(pic.get(key)), f"numerics.picard.{key} 须为正有限数")
    require(_pos_finite(n["retry"].get("halvings_max")) and int(n["retry"]["halvings_max"]) >= 1,
            "numerics.retry.halvings_max 须为正整数")
    for key in ("max_step_data_s", "max_step_after_s", "atol_I"):
        if key in n["bdf"]:
            require(_pos_finite(n["bdf"][key]), f"numerics.bdf.{key} 须为正有限数")
    et = n["envelope_tol"]
    require(_pos_finite(et.get("C")) and _pos_finite(et.get("T_K")),
            "numerics.envelope_tol.C / T_K 须为正有限数")
    sens = cfg["air"].get("sensitivity", {})
    for key in ("dT_degC", "dC"):
        require(_pos_finite(sens.get(key)), f"air.sensitivity.{key} 须为正有限数")


def _pos_finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def check_props_against_formulas(rel_tol: float = 1e-12) -> bool:
    """核对 props.py 具名函数在若干 (C, T_K) 点复现题面公式（不使用 eval）。"""
    import numpy as np
    from . import props as P

    Cs = np.array([0.05, 0.5, 1.0, 2.55])
    Ts = np.array([301.15, 310.0, 323.1489344262])

    q1 = P.PropsQ1()
    for C in Cs:
        assert_close(float(q1.rho(C)), 820.0, rel_tol, f"q1.rho({C})")
        assert_close(float(q1.cp(C)), 2600.0, rel_tol, f"q1.cp({C})")
        assert_close(float(q1.k(C)), 0.36, rel_tol, f"q1.k({C})")
        assert_close(float(q1.D(C)), 7e-9 * math.exp(-0.89 / C), rel_tol, f"q1.D({C})")

    q23 = P.PropsQ23()
    q4 = P.PropsQ4()
    for C in Cs:
        assert_close(float(q23.rho(C)), 650 + 128 * C, rel_tol, f"q23.rho({C})")
        assert_close(float(q23.cp(C)), 1450 + 2736 * C / (C + 1), rel_tol, f"q23.cp({C})")
        assert_close(float(q23.k(C)), 0.21 + 0.38 * C / (C + 1), rel_tol, f"q23.k({C})")
        assert_close(float(q4.rho(C)), 760 + 90 * C, rel_tol, f"q4.rho({C})")
        assert_close(float(q4.cp(C)), 1850 + 2150 * C / (C + 1), rel_tol, f"q4.cp({C})")
        assert_close(float(q4.k(C)), 0.12 + 0.20 * C / (C + 1), rel_tol, f"q4.k({C})")
        for T in Ts:
            assert_close(float(q23.D(C, T)),
                         2.4e-3 * math.exp(-0.45 / C) * math.exp(-3850 / T), rel_tol,
                         f"q23.D({C},{T})")
            assert_close(float(q4.D(C, T)),
                         4.2e-4 * math.exp(-0.30 / C) * math.exp(-3850 / T), rel_tol,
                         f"q4.D({C},{T})")
    return True


def assert_close(got: float, expected: float, rel_tol: float, what: str) -> None:
    denom = max(abs(expected), 1e-300)
    if abs(got - expected) / denom > rel_tol:
        raise ConfigError(f"物性一致性失败 {what}: got {got!r} expected {expected!r}")


def main(argv: list[str]) -> int:
    import yaml
    path = argv[1] if len(argv) > 1 else "config/A题_config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check(cfg)
    check_props_against_formulas()
    print(f"config OK: {path} | version {cfg['version']} | interface "
          f"{cfg['numerics']['interface']} | production.approved {cfg['production']['approved']}")
    print("props consistency OK（附录 2/3/4 公式复现通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
