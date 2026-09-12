"""config.py —— 加载并类型化 A题_config.yaml（唯一参数来源）。

- 公式字符串仅作说明，运行时映射到 props.py 具名函数（**不使用 eval**）。
- 物理常数校验交由 config_check.check（增强版，显式异常/schema）。
- 提供便捷访问器与单位换算（°C→K、cm→m 由 data_io 处理数据；此处给出常量）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import props as props_mod

# 项目根：src/drymodel/config.py → parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "A题_config.yaml"
DATA_ROOT = PROJECT_ROOT / "附件"

KELVIN = 273.15


class Config:
    """已校验配置的类型化包装。原始字典见 `.raw`。"""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        self.raw = raw
        self.source_path = source_path

    # ---- 几何 / 初值 ----
    @property
    def R0(self) -> float:
        return float(self.raw["geometry"]["R0"]["value"])

    @property
    def L(self) -> float:
        return float(self.raw["geometry"]["L"]["value"])

    @property
    def T0_C(self) -> float:
        return float(self.raw["initial"]["T0"]["value"])

    @property
    def T0_K(self) -> float:
        return self.T0_C + KELVIN

    @property
    def C0(self) -> float:
        return float(self.raw["initial"]["C0"]["value"])

    # ---- 边界 ----
    @property
    def h(self) -> float:
        return float(self.raw["bc"]["h"]["value"])

    @property
    def hm(self) -> float:
        return float(self.raw["bc"]["hm"]["value"])

    # ---- 数值 ----
    @property
    def numerics(self) -> dict:
        return self.raw["numerics"]

    @property
    def interface(self) -> str:
        return self.raw["numerics"]["interface"]

    @property
    def N_default(self) -> int:
        return int(self.raw["numerics"]["N_default"])

    @property
    def dt_be_s(self) -> float:
        return float(self.raw["numerics"]["dt_be_s"])

    @property
    def picard(self) -> dict:
        return self.raw["numerics"]["picard"]

    @property
    def retry(self) -> dict:
        return self.raw["numerics"]["retry"]

    @property
    def bdf(self) -> dict:
        return self.raw["numerics"]["bdf"]

    @property
    def envelope_tol(self) -> dict:
        return self.raw["numerics"]["envelope_tol"]

    # ---- 判据 / 输出 ----
    @property
    def threshold(self) -> float:
        return float(self.raw["criterion"]["threshold"])

    @property
    def post_margin_s(self) -> float:
        return float(self.raw["criterion"]["post_margin_s"])

    @property
    def decimals(self) -> int:
        return int(self.raw["output"]["decimals"])

    @property
    def number_format(self) -> str:
        return self.raw["output"]["number_format"]

    @property
    def excel_max_rows(self) -> int:
        return int(self.raw["output"]["excel_max_rows_including_header"])

    @property
    def a1_text(self) -> str:
        return self.raw["output"]["a1_text"]

    # ---- 附件 / 半径 ----
    @property
    def air_window_s(self) -> list[int]:
        return list(self.raw["air"]["window_s"])

    @property
    def breakpoints_s(self) -> list[int]:
        return list(self.raw["air"]["breakpoints_s"])

    @property
    def inside_tol(self) -> float:
        return float(self.raw["radius"]["inside_tol"])

    @property
    def after_72h(self) -> str:
        return self.raw["radius"]["after_72h"]

    # ---- 数据文件路径 ----
    def air_file(self) -> Path:
        return DATA_ROOT / self.raw["air"]["file"]

    def radius_file(self) -> Path:
        return DATA_ROOT / self.raw["radius"]["file"]

    # ---- 物性 ----
    def props(self, question: str):
        """返回对应问题物性对象（q1/q23/q4）。"""
        return props_mod.make_props(question)

    def rho_s0(self, question: str = "q23") -> float:
        """干物质基准密度 rho_s0 = rho(C0)/(1+C0)（H12，仅用于质量换算）。"""
        p = self.props(question)
        return float(p.rho(self.C0)) / (1.0 + self.C0)

    # ---- 生效数值配置解析（本轮：配置真正接入运行）----
    def resolved(self, question: str, stage: str = "candidate") -> dict:
        """解析某问在某阶段（candidate/default）的实际生效数值配置。

        runners/verify 应从这里取 interface/N/npts/tol/step/情景，而非写死默认。
        """
        n = self.raw["numerics"]
        pq = n["per_question"][question]
        cand = n.get("candidate", {})
        if stage == "candidate":
            interface = cand.get("interface", n["interface"])
            npts = int(cand.get("integral_npts", n["quadrature"]["interface_points"]))
            N = int(cand.get("stage_N", {}).get(question, pq["candidate_N"][-1]))
            t_cap_h = float(cand.get("t_cap_h", {}).get(question, 200.0))
        else:  # default / baseline
            interface = n["interface"]
            npts = int(n["quadrature"]["interface_points"])
            N = int(n["N_default"])
            t_cap_h = 200.0
        return {
            "question": question, "stage": stage,
            "interface": interface, "integral_npts": npts, "N": N,
            "candidate_N": list(pq["candidate_N"]), "final_N": pq["final_N"],
            "applies_event": bool(pq["applies_event"]), "t_cap_h": t_cap_h,
            "scenario": "base",
            "bdf": dict(self.bdf), "picard": dict(self.picard), "retry": dict(self.retry),
            "envelope_tol": dict(self.envelope_tol),
        }

    def snapshot(self) -> dict:
        """可追溯配置快照：版本、内容 sha256、各问生效参数、断点、软件版本。"""
        import hashlib
        import platform
        raw_bytes = self.source_path.read_bytes() if self.source_path else b""
        sha = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None
        return {
            "config_version": self.raw.get("version"),
            "config_path": str(self.source_path) if self.source_path else None,
            "config_sha256": sha,
            "production_approved": self.raw["production"]["approved"],
            "default_interface": self.interface,
            "breakpoints_s": self.breakpoints_s,
            "resolved": {q: self.resolved(q, "candidate")
                         for q in ("q1", "q23", "q4")},
            "python": platform.python_version(),
        }


def load_config(path: str | Path | None = None, *, validate: bool = True) -> Config:
    """加载 YAML 并（默认）校验，返回 Config。"""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw, source_path=p)
    if validate:
        check_config(cfg)
    return cfg


def check_config(cfg: Config) -> bool:
    """委托给增强版 config_check.check（显式异常/schema，`-O` 下不失效）。"""
    from . import config_check
    return config_check.check(cfg.raw)
