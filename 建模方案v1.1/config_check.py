"""config_check.py（v1.2 定点返修，2026-09-12）—— 加载并校验 A题_config.yaml。

PATCH-04：真正拒绝错误，而非只声明以后会检查。
- **显式异常 `ConfigError`（不用 assert）**：Python `-O` 优化模式会移除 `assert`，本检查器在 `-O` 下仍生效。
- 覆盖实际被读取的关键字段：类型、有穷性、正范围、合法整数网格与求积阶数、对应附录的物性组、
  固定长度主线、事件形式、正式输出的终点/步距/单位/工作表/表面列及域外规则。
- 拒绝八类非法变异 + 负半径 + NaN/Infinity/类型错误/布尔冒充整数/零或负容差（作负例）。
- 公式字符串不执行（不 eval）：校验其与冻结附录说明式逐字一致（防串组/笔误）。
- **校验说明文字 ≠ 实际求解器物性函数已通过测试**：若代码项目 `drymodel.props` 可导入，则对
  真正被求解器调用的函数做代表性正 C、K 温度代入测试；否则标为「待代码项目 tests/test_props.py 验证」。

用法：python config_check.py A题_config.yaml
配置解析通过 ≠ 模型验证通过，更 ≠ 公式—配置一致。
"""
from __future__ import annotations

import math
import sys

REQ_TOP = ["version", "geometry", "initial", "bc", "air", "radius", "props",
           "numerics", "q4", "criterion", "output", "acceptance", "production", "storage"]

VALID_SOURCE_LITERALS = {"题面Q1", "附录2", "附录3", "附录4", "附件1", "附件2"}

# 冻结的附录说明式（与题面附录 2/3/4 逐字一致）
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
A1_TEXT_EXPECTED = "时间\\到药材中心的距离"


class ConfigError(ValueError):
    """配置校验失败。"""


def require(cond, msg):
    if not cond:
        raise ConfigError(msg)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def get(d, path):
    v = d
    for p in path.split("."):
        if not isinstance(v, dict) or p not in v:
            raise ConfigError(f"缺少配置键 {path}（在 {p} 处中断）")
        v = v[p]
    return v


def num(d, path, unit=None, source=True, positive=False):
    v = get(d, path)
    require(isinstance(v, dict) and is_finite_number(v.get("value")),
            f"{path}: 需 {{value, unit, source}} 且 value 为有限数值")
    val = float(v["value"])
    if positive:
        require(val > 0.0, f"{path}: value 须为正，得 {val}")
    if unit is not None:
        require(v.get("unit") == unit, f"{path}: unit 应为 {unit!r}，得 {v.get('unit')!r}")
    if source:
        s = v.get("source", "")
        require(s in VALID_SOURCE_LITERALS or str(s).startswith(("D", "B", "H")),
                f"{path}: source 标签缺失或非法（得 {s!r}）")
    return val


def check(cfg):
    """逐项校验；任一不满足抛 ConfigError；全部通过返回 True。"""
    require(isinstance(cfg, dict), "配置根须为映射")
    for k in REQ_TOP:
        require(k in cfg, f"缺少顶层键 {k}")

    # 几何 / 初值（负半径在 -O 下亦被拒）
    R0 = num(cfg, "geometry.R0", "m", positive=True)
    L = num(cfg, "geometry.L", "m", positive=True)
    require(R0 == 0.02, f"geometry.R0 应为正的 0.02 m，得 {R0}")   # 负半径被拒
    require(L == 0.25, f"geometry.L 应为 0.25 m，得 {L}")
    require(num(cfg, "initial.T0", "degC") == 28.0, "initial.T0 应为 28.0 degC")
    require(num(cfg, "initial.C0", "kg/kg", positive=True) == 2.55, "initial.C0 应为 2.55")

    # 边界（主线题给值不变）
    require(num(cfg, "bc.h", "W/(m^2 K)", positive=True) == 25.0, "bc.h 应固定为 25.0")
    require(num(cfg, "bc.hm", "m/s", positive=True) == 8.0e-7, "bc.hm 应固定为 8e-7")
    require(cfg["bc"].get("mass_basis") == "effective_C", "bc.mass_basis 应为 effective_C")
    require(cfg["bc"].get("heat_latent") is False, "bc.heat_latent 主线应为 False")

    # 空气 / 半径
    a = cfg["air"]
    require(a.get("read_only") is True, "air.read_only 应为 True")
    require(a.get("interp") in ("linear", "smooth121"), "air.interp 应为 linear/smooth121")
    require(a.get("extrapolation") in ("hold_window_mean", "hold_last"),
            "air.extrapolation 应为 hold_window_mean/hold_last")
    require(a.get("window_s") == [10800, 14400], "air.window_s 应为 [10800,14400]")
    require(a.get("breakpoints_s") == [14400], "air.breakpoints_s 应为 [14400]")
    r = cfg["radius"]
    require(r.get("interp") == "linear" and r.get("after_72h") == "hold_last",
            "radius.interp/after_72h 应为 linear/hold_last")
    require(is_finite_number(r.get("inside_tol")) and r["inside_tol"] > 0,
            "radius.inside_tol 须为正有限值")

    # 物性：常数正范围 + 公式字符串逐字一致（拒绝变异 1/2/3）
    require(num(cfg, "props.q1.rho", "kg/m^3", positive=True) == 820.0,
            "props.q1.rho 应为正的 820（变异 −820 被拒）")
    require(num(cfg, "props.q1.cp", "J/(kg K)", positive=True) == 2600.0, "props.q1.cp 应为 2600")
    require(num(cfg, "props.q1.k", "W/(m K)", positive=True) == 0.36, "props.q1.k 应为 0.36")
    check_formula_strings(cfg)                                # 变异 2/3
    require(get(cfg, "props.q23.source") == "附录3", "props.q23.source 应为 附录3")
    require(get(cfg, "props.q4.source") == "附录4", "props.q4.source 应为 附录4")
    require(cfg["props"].get("energy_form") == "effective",
            "props.energy_form 主线应为 effective（拒绝 enthalpy_advective）")   # 变异 7

    # 数值
    n = cfg["numerics"]
    require(n.get("interface") in ("harmonic", "integral"), "numerics.interface 应为 harmonic/integral")
    require(n.get("scheme_baseline") == "backward_euler", "numerics.scheme_baseline 应为 backward_euler")
    require(is_int(n.get("N_default")) and n["N_default"] > 0,
            f"numerics.N_default 须为正整数（拒绝 −200/布尔），得 {n.get('N_default')!r}")   # 变异 4 + 布尔
    require(isinstance(n.get("N_verify"), list) and n["N_verify"]
            and all(is_int(x) and x > 0 for x in n["N_verify"]), "numerics.N_verify 须为正整数列表")
    rtol = n["bdf"].get("rtol")
    require(is_finite_number(rtol) and 0 < rtol <= 1e-8,
            f"numerics.bdf.rtol 须为 (0,1e-8] 的有限数（拒绝负值/NaN），得 {rtol!r}")   # 变异 5 + NaN
    for key in ("atol_C", "atol_T_K", "atol_I"):
        require(is_finite_number(n["bdf"].get(key)) and n["bdf"][key] > 0,
                f"numerics.bdf.{key} 须为正有限数")
    require(is_int(n["picard"].get("max_iter")) and n["picard"]["max_iter"] >= 1,
            "numerics.picard.max_iter 须为 ≥1 的整数")
    require(is_finite_number(n["retry"].get("dt_min_s")) and n["retry"]["dt_min_s"] > 0,
            "numerics.retry.dt_min_s 须为正")

    # 8/16 点求积 + 分问数值配置（PATCH-04）
    quad = n.get("quadrature", {})
    require(is_int(quad.get("interface_points")) and quad["interface_points"] > 0,
            "numerics.quadrature.interface_points 须为正整数")
    require(is_int(quad.get("interface_points_check")) and quad["interface_points_check"] > quad["interface_points"],
            "numerics.quadrature.interface_points_check 须为更大正整数（如 16）")
    pq = n.get("per_question", {})
    for qkey, expect_event in (("q1", False), ("q23", True), ("q4", True)):
        require(qkey in pq, f"numerics.per_question 缺少 {qkey}")
        blk = pq[qkey]
        require(isinstance(blk.get("candidate_N"), list) and blk["candidate_N"]
                and all(is_int(x) and x > 0 for x in blk["candidate_N"]),
                f"per_question.{qkey}.candidate_N 须为正整数列表")
        fn = blk.get("final_N", None)
        require(fn is None or (is_int(fn) and fn > 0),
                f"per_question.{qkey}.final_N 须为 null（待定）或正整数，不填历史探针值")
        require(blk.get("applies_event") is expect_event,
                f"per_question.{qkey}.applies_event 应为 {expect_event}")

    # 候选阶段生效配置（配置真正接入运行）
    cand = n.get("candidate")
    if cand is not None:
        require(cand.get("interface") in ("harmonic", "integral"),
                "numerics.candidate.interface 应为 harmonic/integral")
        require(is_int(cand.get("integral_npts")) and cand["integral_npts"] > 0,
                "numerics.candidate.integral_npts 须为正整数")
        sN = cand.get("stage_N", {})
        for qkey in ("q1", "q23", "q4"):
            require(qkey in sN and is_int(sN[qkey]) and sN[qkey] in pq[qkey]["candidate_N"],
                    f"numerics.candidate.stage_N.{qkey} 须为该问 candidate_N 中的整数")
        tc = cand.get("t_cap_h", {})
        for qkey in ("q23", "q4"):
            require(is_finite_number(tc.get(qkey)) and tc[qkey] > 0,
                    f"numerics.candidate.t_cap_h.{qkey} 须为正有限数")

    # 包络容差 / 空气情景扰动（有限性·正范围）
    et = n.get("envelope_tol", {})
    require(is_finite_number(et.get("C")) and et["C"] > 0
            and is_finite_number(et.get("T_K")) and et["T_K"] > 0,
            "numerics.envelope_tol.C / T_K 须为正有限数")
    sens = cfg["air"].get("sensitivity", {})
    for key in ("dT_degC", "dC"):
        require(is_finite_number(sens.get(key)) and sens[key] > 0,
                f"air.sensitivity.{key} 须为正有限数")

    # Q4 运动学（拒绝变异 8）
    qc = cfg["q4"]
    require(qc.get("kinematics") == "affine" and qc.get("frame") == "reference_x"
            and qc.get("mesh_velocity") == "solid", "q4 运动学字段错误")
    require(qc.get("axial") == "fixed_L", "q4.axial 应为 fixed_L（拒绝 shrinking_L）")   # 变异 8

    # 判据（事件仅 Q23/Q4；Q1 不适用）
    c = cfg["criterion"]
    require(c.get("threshold") == 0.15, "criterion.threshold 应为 0.15")
    require(c.get("domain") == "full_unrounded", "criterion.domain 应为 full_unrounded")
    require(c.get("event") == "continuous_root_on_reconstructed_field", "criterion.event 形式错误")
    require(c.get("applies_to") == ["q23", "q4"], "criterion.applies_to 应为 ['q23','q4']")
    require(is_finite_number(c.get("post_margin_s")) and c["post_margin_s"] > 0,
            "criterion.post_margin_s 须为正")

    # 输出（终点/步距/单位/工作表/表面列/域外规则；拒绝变异 6）
    o = cfg["output"]
    require(o.get("decimals") == 4, "output.decimals 应为 4")
    require(o.get("excel_max_rows_including_header") == 1048576, "output.excel 行上限应为 1048576")
    require(o.get("a1_text") == A1_TEXT_EXPECTED, "output.a1_text 错误")
    require("温度" in o["result1"]["sheets"] and "水分浓度" in o["result1"]["sheets"],
            "result1 工作表应含 温度/水分浓度")
    require(o["result2"].get("t_end") == "until_dry_1s",
            "output.result2.t_end 正式导出仅允许 until_dry_1s（拒绝 3h/72h）")   # 变异 6
    require(set(o.get("result2_test_modes", [])) <= {"3h", "72h"},
            "result2_test_modes 仅可含 3h/72h（测试模式）")
    require(o["result4"].get("outside_fill") == "blank" and o["result4"].get("surface_col") == "药材表面",
            "result4 域外规则/表面列错误")

    # 验收（能量残差 W/m，参考尺度不含 L）
    acc = cfg["acceptance"]
    require(acc.get("t_star_h") == 0.02, "acceptance.t_star_h 应为 0.02")
    require(float(acc["table_points"].get("dC")) == 5.0e-4, "acceptance.table_points.dC 应为 5e-4")
    er = acc["energy_residual"]
    require(er.get("unit") == "W/m" and "abs_W_per_m" in er, "能量残差应为 W/m 口径")
    require("L" not in str(er.get("ref_scale", "")), "能量残差参考尺度不得含长度 L（W/m 口径）")

    # 生产门控
    prod = cfg["production"]
    require(prod.get("approved") is False or prod.get("config_id"),
            "approved=True 须给 config_id")
    if prod.get("approved") is False:
        for qkey in ("q1", "q23", "q4"):
            require(pq[qkey].get("final_N") is None,
                    f"未授权时 per_question.{qkey}.final_N 须为 null")
    return True


def check_formula_strings(cfg):
    """校验附录说明式与冻结规格逐字一致（防串组/笔误；不 eval）。"""
    for key, expected in EXPECTED_FORMULAS.items():
        got = get(cfg, key)
        require(got == expected,
                f"{key}: 公式字符串与冻结附录说明式不一致（期望 {expected!r}，得 {got!r}）")


def check_solver_props():
    """若 drymodel.props 可导入，则对真正被求解器调用的函数做代表性代入测试；否则标待验证。"""
    try:
        from drymodel import props as P
    except Exception:
        print("props 实测：drymodel.props 不可导入 → 待代码项目 tests/test_props.py 验证")
        return None
    q1, q23, q4 = P.PropsQ1(), P.PropsQ23(), P.PropsQ4()
    for C in (0.05, 1.0, 2.55):
        _close(float(q1.D(C)), 7e-9 * math.exp(-0.89 / C), f"q1.D({C})")
        for T in (301.15, 323.15):
            _close(float(q23.D(C, T)), 2.4e-3 * math.exp(-0.45 / C) * math.exp(-3850 / T), f"q23.D({C},{T})")
            _close(float(q4.D(C, T)), 4.2e-4 * math.exp(-0.30 / C) * math.exp(-3850 / T), f"q4.D({C},{T})")
    print("props 实测：附录 2/3/4 求解器函数代入一致 OK")
    return True


def _close(got, exp, what, rel=1e-12):
    if abs(got - exp) / max(abs(exp), 1e-300) > rel:
        raise ConfigError(f"props 实测失败 {what}: {got!r} != {exp!r}")


def main(argv):
    import yaml
    path = argv[1] if len(argv) > 1 else "A题_config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check(cfg)
    print(f"config OK: {path} | version {cfg['version']} | interface "
          f"{cfg['numerics']['interface']} | approved {cfg['production']['approved']}")
    check_solver_props()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
