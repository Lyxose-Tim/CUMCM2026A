"""灵敏度分析（S1–S6/S10）。

情景范围、验收门槛和实测数值误差分别记录。只有基线与情景的加密求解均完成后，
才判断情景差是否可分辨；绝不把 ``acceptance.t_star_h`` 当成误差测量值。
"""
from __future__ import annotations

from . import runners


def scenario_definitions(cfg):
    """从配置构造当前实际启用的 Q23 情景。"""
    dT = float(cfg.raw["air"]["sensitivity"]["dT_degC"])
    dC = float(cfg.raw["air"]["sensitivity"]["dC"])
    return [
        ("S1 h×0.5", {"h_mult": 0.5}),
        ("S1 h×2", {"h_mult": 2.0}),
        ("S2 hm×0.5", {"hm_mult": 0.5}),
        ("S2 hm×2", {"hm_mult": 2.0}),
        (f"S3 T_air+{dT:g}", {"scenario": "dT_air+"}),
        (f"S3 T_air-{dT:g}", {"scenario": "dT_air-"}),
        (f"S4 C_env+{dC:g}", {"scenario": "dC_env+"}),
        (f"S4 C_env-{dC:g}", {"scenario": "dC_env-"}),
        ("S5 hold_last", {"scenario": "hold_last"}),
        ("S6 smooth121", {"scenario": "smooth121"}),
    ]


def _t_star_h(cfg, *, question, N, interface, t_cap_h, purpose,
              bdf_overrides=None, **scenario_kwargs):
    det, _, _, op = runners.q23_detect(
        cfg, question=question, N=N, interface=interface, t_cap_h=t_cap_h,
        purpose=purpose, bdf_overrides=bdf_overrides, **scenario_kwargs,
    )
    run = None if op is None else op.run_config
    config = None if run is None else {"digest": run.digest, **run.snapshot()}
    return float(det.t_cross / 3600.0), config


def run_scenarios(cfg, *, question="q23", N=400, interface=None,
                  t_cap_h=90.0, scenarios=None, refine=True):
    """逐情景重积分，并用独立加密结果估计情景差不确定性。

    ``refine=False`` 仅运行候选配置，所有 ``resolved`` 均为 ``None``，状态为
    ``not_run``；这比使用验收门槛冒充误差更保守且可追溯。
    """
    if scenarios is None:
        scenarios = scenario_definitions(cfg)
    interface = interface or cfg.raw["numerics"]["per_question"][question]["candidate_interface"]
    acceptance_threshold = float(cfg.raw["acceptance"]["t_star_h"])
    refined = dict(cfg.raw["numerics"]["bdf_refined"])

    base_h, base_config = _t_star_h(
        cfg, question=question, N=N, interface=interface, t_cap_h=t_cap_h,
        purpose="sensitivity-base",
    )
    base_ref_h = None
    base_ref_config = None
    base_error_h = None
    if refine:
        base_ref_h, base_ref_config = _t_star_h(
            cfg, question=question, N=N, interface=interface, t_cap_h=t_cap_h,
            purpose="sensitivity-base-time-refined", bdf_overrides=refined,
        )
        base_error_h = abs(base_h - base_ref_h)

    rows = []
    for label, kwargs in scenarios:
        value_h, scenario_config = _t_star_h(
            cfg, question=question, N=N, interface=interface, t_cap_h=t_cap_h,
            purpose=f"sensitivity-{label}", **kwargs,
        )
        delta_h = value_h - base_h
        row = {
            "label": label,
            "t_star_h": value_h,
            "dt_star_h": delta_h,
            "acceptance_threshold_h": acceptance_threshold,
            "refined_t_star_h": None,
            "refined_dt_star_h": None,
            "base_numeric_error_h": base_error_h,
            "scenario_numeric_error_h": None,
            "combined_uncertainty_h": None,
            "delta_refinement_difference_h": None,
            "resolved": None,
            "resolution_status": "not_run",
            "candidate_config": scenario_config,
            "refined_config": None,
        }
        if refine:
            refined_h, refined_config = _t_star_h(
                cfg, question=question, N=N, interface=interface, t_cap_h=t_cap_h,
                purpose=f"sensitivity-{label}-time-refined",
                bdf_overrides=refined, **kwargs,
            )
            refined_delta_h = refined_h - base_ref_h
            scenario_error_h = abs(value_h - refined_h)
            combined = float(base_error_h + scenario_error_h)
            delta_refinement_difference = abs(delta_h - refined_delta_h)
            same_direction = (
                delta_h == 0.0 and refined_delta_h == 0.0
            ) or (delta_h * refined_delta_h > 0.0)
            row.update(
                refined_t_star_h=refined_h,
                refined_dt_star_h=refined_delta_h,
                scenario_numeric_error_h=scenario_error_h,
                combined_uncertainty_h=combined,
                delta_refinement_difference_h=delta_refinement_difference,
                resolved=bool(same_direction and abs(refined_delta_h) > combined),
                resolution_status="measured",
                refined_config=refined_config,
            )
        rows.append(row)

    return {
        "question": question,
        "enabled_scope": [question],
        "not_run_scope": [q for q in ("q1", "q4") if q != question],
        "N": N,
        "interface": interface,
        "base_t_star_h": base_h,
        "base_refined_t_star_h": base_ref_h,
        "base_numeric_error_h": base_error_h,
        "base_config": base_config,
        "base_refined_config": base_ref_config,
        "acceptance_threshold_h": acceptance_threshold,
        "numeric_error_source": "time_refinement" if refine else "not_run",
        "rows": rows,
    }
