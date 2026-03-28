#!/usr/bin/env python3
import os
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cache_dir = root / "data" / "quickdraw"
    if "QUICKDRAW_CACHE_DIR" not in os.environ and cache_dir.is_dir():
        os.environ["QUICKDRAW_CACHE_DIR"] = str(cache_dir)
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    project = root / "SketchGPT"
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(project)

    from sketchgpt.__main__ import cli

    return int(cli())


if __name__ == "__main__":
    raise SystemExit(main())
