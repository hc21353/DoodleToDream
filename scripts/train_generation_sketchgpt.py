#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.SketchGPT.train import train_sketchgpt


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_runtime_env() -> None:
    cache_root = ROOT / "scripts" / "artifacts" / "_cache" / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/configs/smoke_test.json")
    args = ap.parse_args()

    ensure_runtime_env()
    cfg = load_config(args.config)
    result = train_sketchgpt(cfg["generation"]["sketchgpt_train"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
