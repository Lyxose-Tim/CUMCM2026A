"""pytest 配置：将 src/ 加入 sys.path，使 `import drymodel` 可用（无需安装）。"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
