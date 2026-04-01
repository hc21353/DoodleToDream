from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import numpy as np

from .model import SimpleMobileNet


def _preprocess(image_path: str, image_size: int):
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array(
        [0.229, 0.224, 0.225], dtype="float32"
    )
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def predict_confidence(image_path: str, checkpoint_path: str, top_k: int = 3) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    image_size = int(ckpt.get("image_size", 256))
    dropout = float(ckpt.get("dropout_rate", 0.3))

    model = SimpleMobileNet(num_classes=len(classes), dropout_rate=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = _preprocess(image_path, image_size)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_vals, top_idx = torch.topk(probs, k=min(top_k, len(classes)))
    return {
        "predicted_class": classes[int(top_idx[0])],
        "confidence": float(top_vals[0]),
        "top_k": [
            {"class": classes[int(idx)], "prob": float(val)}
            for idx, val in zip(top_idx, top_vals)
        ],
    }
