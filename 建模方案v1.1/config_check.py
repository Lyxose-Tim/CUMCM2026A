"""方案目录配置检查入口。

校验规则的唯一权威实现位于 ``src/drymodel/config_check.py``。本文件只负责让
``python 建模方案v1.1/config_check.py`` 仍可独立运行，禁止复制第二份 schema。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drymodel.config_check import (  # noqa: E402,F401
    ConfigError,
    check,
    check_props_against_formulas,
    main as _authoritative_main,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) == 1:
        argv.append(str(Path(__file__).with_name("A题_config.yaml")))
    return _authoritative_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
