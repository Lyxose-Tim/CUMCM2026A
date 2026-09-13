"""sensitivity.py —— 灵敏度分析（§8.4，S1–S6/S10）。

- 情景 = 设定范围；**每情景重新积分**。
- 筛选尺度：0.02 h（取自验收配置 config.acceptance.t_star_h，用以突出主要变化，非数值误差量级）。
- 主线题给值不改；h/h_m 走单独倍率层（h_mult/hm_mult）。
"""
from __future__ import annotations

import numpy as np

from . import runners


# (标签, 关键字参数) —— 每项相对基线改变一处输入
SCENARIOS = [
    ("S1 h×0.5", {"h_mult": 0.5}),
    ("S1 h×2", {"h_mult": 2.0}),
    ("S2 hm×0.5", {"hm_mult": 0.5}),
    ("S2 hm×2", {"hm_mult": 2.0}),
    ("S3 T_air+0.39", {"scenario": "dT_air+"}),
    ("S3 T_air-0.39", {"scenario": "dT_air-"}),
    ("S4 C_env+0.0011", {"scenario": "dC_env+"}),
    ("S4 C_env-0.0011", {"scenario": "dC_env-"}),
    ("S5 hold_last", {"scenario": "hold_last"}),
    ("S6 smooth121", {"scenario": "smooth121"}),
]


def run_scenarios(cfg, *, question="q23", N=400, interface="integral",
                  t_cap_h=90.0, numeric_dt_star_h=0.02, scenarios=None):
    """对每个情景重新积分求 t*，返回相对基线的 Δt*（h）与分辨标记。

    numeric_dt_star_h：数值误差量级（默认 0.02 h 验收阈值），用于分辨判据。
    """
    if scenarios is None:
        scenarios = SCENARIOS
    base_det, *_ = runners.q23_detect(cfg, N=N, interface=interface, question=question,
                                      t_cap_h=t_cap_h)
    base_h = base_det.t_cross / 3600.0
    rows = []
    for label, kw in scenarios:
        det, *_ = runners.q23_detect(cfg, N=N, interface=interface, question=question,
                                     t_cap_h=t_cap_h, **kw)
        th = det.t_cross / 3600.0
        dstar = th - base_h
        resolved = abs(dstar) > numeric_dt_star_h
        rows.append({"label": label, "t_star_h": th, "dt_star_h": dstar,
                     "resolved": resolved})
    return {"question": question, "N": N, "interface": interface,
            "base_t_star_h": base_h, "numeric_dt_star_h": numeric_dt_star_h,
            "rows": rows}
