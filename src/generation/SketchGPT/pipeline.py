from __future__ import annotations

from pathlib import Path

_MODULE_ORDER = [
    "config.py",
    "data.py",
    "models.py",
    "training.py",
    "visualization.py",
    "pipeline_runtime.py",
]

_MODULE_DIR = Path(__file__).parent
_NS = globals()

for _name in _MODULE_ORDER:
    _path = _MODULE_DIR / _name
    _code = _path.read_text(encoding="utf-8")
    exec(compile(_code, str(_path), "exec"), _NS, _NS)
