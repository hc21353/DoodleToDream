from __future__ import annotations

import json
from pathlib import Path

from classification.infer import predict_confidence


def compute_confidence_for_directory(config: dict) -> dict:
    image_dir = Path(config["evaluation"]["input_image_dir"])
    checkpoint_path = config["classification"]["checkpoint_path"]

    results = {}
    for image_path in sorted(image_dir.glob("*.png")):
        results[image_path.name] = predict_confidence(str(image_path), checkpoint_path)

    output_path = Path(config["evaluation"]["output_json_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {"output_json": str(output_path), "num_images": len(results)}
