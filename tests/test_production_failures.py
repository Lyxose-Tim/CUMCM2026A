"""生产失败阻断回归（item 3）：门检、异常阻断、失败回执、V-8 零共同时刻。"""
import json

import pytest

from drymodel import config as cfgmod
from drymodel import criterion as CR
from drymodel import production as PROD
from drymodel import writers as W


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def _det(post_ok=True, t_sample=100.0, post_max=0.15):
    return CR.Detection(t_cross=90.0, t_star=90.0, t_sample=t_sample, t_safe=None,
                        delta=float("nan"), dt_star=float("nan"), argmax_node=0,
                        post_ok=post_ok, post_max_cmax=post_max, threshold=0.15)


def test_gate_rejects_post_ok_false():
    with pytest.raises(RuntimeError, match="续算回穿"):
        PROD.gate_production("Q23", _det(post_ok=False), 0.14, 0.15)


def test_gate_rejects_bad_last_row():
    # t_sample 实测 C_max ≥ 阈值 → 严格合格末行不过
    with pytest.raises(RuntimeError, match="严格合格末行"):
        PROD.gate_production("Q4", _det(t_sample=100.0), 0.151, 0.15)
    # t_sample=None
    with pytest.raises(RuntimeError, match="严格合格末行"):
        PROD.gate_production("Q4", _det(t_sample=None), 0.14, 0.15)


def test_gate_passes_valid():
    PROD.gate_production("Q23", _det(post_ok=True, t_sample=100.0), 0.149, 0.15)


def test_run_production_blocks_on_exception(cfg, tmp_path, monkeypatch):
    """任一 produce 抛错 → run_production ok=False + 失败回执，不标产物通过。"""
    def boom(*a, **k):
        raise RuntimeError("模拟续算失败")
    monkeypatch.setattr(PROD, "produce_q1", boom)
    override = {q: {"N": 200, "interface": "integral", "npts": 8, "t_cap_h": 90.0}
               for q in ("q1", "q23", "q4")}
    r = PROD.run_production(cfg, outputs_dir=tmp_path, override=override, require_approved=False)
    assert r["ok"] is False and "失败" in r["reason"]
    # 失败回执已记录
    rec = json.loads((tmp_path / "production_receipt.json").read_text(encoding="utf-8"))
    assert rec["ok"] is False
    assert (tmp_path / "production_failures.log").exists()


def test_cross_file_zero_common_not_pass(tmp_path):
    """零共同时刻不能判通过（result2/3 无 60 s 公共时刻）。"""
    import numpy as np
    # result2：仅 1..59 s（无 60 s 倍数）
    t2 = np.arange(1, 60)
    COLS21 = [round(0.1 * j, 4) for j in range(21)]
    A1 = "时间\\到药材中心的距离"
    W.write_result12(tmp_path / "result2.xlsx", t2, np.full((59, 21), 0.3),
                     np.full((59, 21), 40.0), COLS21, A1)
    # result3：仅 120,180（>result2 末，无共同）
    W.write_result34(tmp_path / "result3.xlsx", np.array([120, 180]),
                     np.full((2, 21), 0.3), COLS21, A1, sheet_name="Sheet1")
    res = W.cross_file_check(tmp_path / "result2.xlsx", tmp_path / "result3.xlsx")
    assert res["n_common_times"] == 0 and res["ok"] is False
