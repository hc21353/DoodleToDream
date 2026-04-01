#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classification.infer import predict_confidence


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/configs/smoke_test.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    image_dir = Path(eval_cfg["input_image_dir"]) if Path(eval_cfg["input_image_dir"]).is_absolute() else ROOT / eval_cfg["input_image_dir"]
    checkpoint = eval_cfg["classification_checkpoint_path"]

    out = {}
    for p in sorted(image_dir.glob("*.png")):
        out[p.name] = predict_confidence(str(p), checkpoint)

    out_path = Path(eval_cfg["output_json_path"]) if Path(eval_cfg["output_json_path"]).is_absolute() else ROOT / eval_cfg["output_json_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_path), "num_images": len(out)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
