"""run_production 缩比端到端测试（不动正式 outputs/；require_approved=False）。

证明：分问续算→result1–4 + 表 1–6 导出→V-8/V-9 通过。用 N=200 + result2 "3h" 测试模式。
"""
import pytest

from drymodel import config as cfgmod
from drymodel import production


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config()


def test_run_production_scaled(cfg, tmp_path):
    override = {q: {"N": 200, "interface": "integral", "npts": 8, "t_cap_h": (90.0 if q != "q4" else 200.0)}
                for q in ("q1", "q23", "q4")}
    r = production.run_production(cfg, outputs_dir=tmp_path, override=override,
                                 result2_mode="3h", require_approved=False)
    assert r["ok"], (r["V8"], {k: v["issues"] for k, v in r["V9"].items() if not v["ok"]})
    # 文件确实生成于临时目录
    for name in ("result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx"):
        assert (tmp_path / name).exists()
    # V-9 逐文件通过；V-8 跨文件通过
    assert all(v["ok"] for v in r["V9"].values())
    assert r["V8"]["ok"] and r["V8"]["n_common_times"] > 0
    # result2 "3h" 测试模式：10800 行
    assert r["q23"]["result2_rows"] == 10800


def test_run_production_refuses_unauthorized(cfg, tmp_path):
    """未授权（approved=false）且 require_approved=True → 拒绝，不生成文件。"""
    import copy
    unauth = copy.deepcopy(cfg)
    unauth.raw = copy.deepcopy(cfg.raw)
    unauth.raw["production"]["approved"] = False        # 强制未授权
    r = production.run_production(unauth, outputs_dir=tmp_path, require_approved=True)
    assert r["ok"] is False and "未授权" in r["reason"]
    assert not (tmp_path / "result1.xlsx").exists()
