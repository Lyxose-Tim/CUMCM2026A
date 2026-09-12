"""config.py —— 加载并类型化 A题_config.yaml（唯一参数来源）。

- 公式字符串仅作说明，运行时映射到 props.py 具名函数（**不使用 eval**）。
- 物理常数校验交由 config_check.check（增强版，显式异常/schema）。
- 提供便捷访问器与单位换算（°C→K、cm→m 由 data_io 处理数据；此处给出常量）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
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

    def resolve_run(
        self,
        question: str,
        *,
        purpose: str = "candidate",
        N: int | None = None,
        interface: str | None = None,
        integral_npts: int | None = None,
        scheme: str | None = None,
        scenario: str = "base",
        h_mult: float = 1.0,
        hm_mult: float = 1.0,
        augmented: bool = False,
        bdf_overrides: dict[str, Any] | None = None,
        dt_be_s: float | None = None,
    ) -> "RunConfig":
        """解析一次求解真正生效的完整配置。

        所有研究性覆盖都必须显式传入并进入快照，求解器不得另藏默认常数。
        """
        if question not in ("q1", "q23", "q4"):
            raise ValueError(f"未知问题 {question!r}")
        numerics = self.raw["numerics"]
        per_q = numerics["per_question"][question]
        use_final = N is None
        if use_final:
            N = per_q.get("final_N")
        if N is None:
            raise ValueError(f"{question}: N 必须显式给出（最终配置尚未授权）")

        default_interface = (
            per_q.get("final_interface") if use_final else per_q["candidate_interface"]
        )
        default_scheme = (
            per_q.get("final_scheme") if use_final else per_q["candidate_scheme"]
        )
        default_npts = (
            per_q.get("final_quadrature_points")
            if use_final else numerics["quadrature"]["interface_points"]
        )

        bdf = dict(numerics["bdf"])
        if bdf_overrides:
            unknown = set(bdf_overrides) - {
                "rtol", "atol_C", "atol_T_K", "atol_I",
                "max_step_data_s", "max_step_after_s",
            }
            if unknown:
                raise ValueError(f"未知 BDF 覆盖项：{sorted(unknown)}")
            bdf.update(bdf_overrides)

        return RunConfig(
            question=question,
            purpose=str(purpose),
            N=int(N),
            interface=str(interface or default_interface),
            integral_npts=int(
                integral_npts
                if integral_npts is not None
                else default_npts
            ),
            scheme=str(scheme or default_scheme),
            rtol=float(bdf["rtol"]),
            atol_C=float(bdf["atol_C"]),
            atol_T_K=float(bdf["atol_T_K"]),
            atol_I=float(bdf["atol_I"]),
            max_step_data_s=float(bdf["max_step_data_s"]),
            max_step_after_s=float(bdf["max_step_after_s"]),
            restart_at_breakpoints=bool(bdf["restart_at_breakpoints"]),
            breakpoints_s=tuple(float(value) for value in self.breakpoints_s),
            air_data_end_s=float(max(self.breakpoints_s)),
            dt_be_s=float(numerics["dt_be_s"] if dt_be_s is None else dt_be_s),
            picard=tuple(sorted(numerics["picard"].items())),
            retry=tuple(sorted(numerics["retry"].items())),
            scenario=str(scenario),
            h_mult=float(h_mult),
            hm_mult=float(hm_mult),
            sensitivity_dT_degC=float(self.raw["air"]["sensitivity"]["dT_degC"]),
            sensitivity_dC=float(self.raw["air"]["sensitivity"]["dC"]),
            augmented=bool(augmented),
        )


@dataclass(frozen=True)
class RunConfig:
    """单次运行的不可变、可追溯数值配置。"""

    question: str
    purpose: str
    N: int
    interface: str
    integral_npts: int
    scheme: str
    rtol: float
    atol_C: float
    atol_T_K: float
    atol_I: float
    max_step_data_s: float
    max_step_after_s: float
    restart_at_breakpoints: bool
    breakpoints_s: tuple[float, ...]
    air_data_end_s: float
    dt_be_s: float
    picard: tuple[tuple[str, Any], ...]
    retry: tuple[tuple[str, Any], ...]
    scenario: str
    h_mult: float
    hm_mult: float
    sensitivity_dT_degC: float
    sensitivity_dC: float
    augmented: bool = False

    @property
    def bdf(self) -> dict[str, Any]:
        return {
            "rtol": self.rtol,
            "atol_C": self.atol_C,
            "atol_T_K": self.atol_T_K,
            "atol_I": self.atol_I,
            "max_step_data_s": self.max_step_data_s,
            "max_step_after_s": self.max_step_after_s,
            "restart_at_breakpoints": self.restart_at_breakpoints,
        }

    @property
    def picard_options(self) -> dict[str, Any]:
        return dict(self.picard)

    @property
    def retry_options(self) -> dict[str, Any]:
        return dict(self.retry)

    def derive(self, *, purpose: str | None = None, **changes: Any) -> "RunConfig":
        """显式派生研究配置；所有变化仍会进入快照和摘要。"""
        if purpose is not None:
            changes["purpose"] = purpose
        return replace(self, **changes)

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["picard"] = dict(self.picard)
        data["retry"] = dict(self.retry)
        return data

    @property
    def digest(self) -> str:
        raw = json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
