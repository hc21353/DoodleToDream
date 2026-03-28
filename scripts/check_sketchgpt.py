#!/usr/bin/env python3
from pathlib import Path
import py_compile


TARGETS = [
    "src/sketchgpt/__init__.py",
    "src/sketchgpt/__main__.py",
    "src/sketchgpt/core.py",
    "src/sketchgpt/data.py",
    "src/sketchgpt/model.py",
    "src/sketchgpt/visualization.py",
    "src/sketchgpt/pipeline.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    project = root / "SketchGPT"
    for rel in TARGETS:
        py_compile.compile(str(project / rel), doraise=True)
    print("SketchGPT syntax check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
