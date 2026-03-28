#!/usr/bin/env python3
from pathlib import Path
import py_compile


TARGETS = [
    "src/vq_sgen/__init__.py",
    "src/vq_sgen/__main__.py",
    "src/vq_sgen/base.py",
    "src/vq_sgen/data.py",
    "src/vq_sgen/models.py",
    "src/vq_sgen/pipeline.py",
    "src/vq_sgen/runner.py",
    "src/vq_sgen/artifacts.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    project = root / "VQ-SGen"
    for rel in TARGETS:
        py_compile.compile(str(project / rel), doraise=True)
    print("VQ-SGen syntax check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
