from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .train import _apply_config, _ensure_runtime_env, _load_runtime_module, _repo_root


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (_repo_root() / p).resolve()


def _copy_preview_images(runtime_mod, class_names: list[str], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for cls_name in class_names:
        cls_key = cls_name.replace(" ", "_")
        sample_dir = Path(runtime_mod.SEQ_DIR) / cls_key / "sample_00"
        stroke_all = sample_dir / "stroke_all.png"
        if not stroke_all.exists():
            continue
        out_path = output_dir / f"{cls_key}.png"
        Image.open(stroke_all).convert("L").save(out_path)
        saved.append(str(out_path))
    return saved


def generate_sketchgpt_images(config: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_runtime_env()
    runtime_mod = _load_runtime_module()

    train_summary = {}
    checkpoint_path = config.get("checkpoint_path", "scripts/artifacts/generation/sketchgpt.pt")
    if checkpoint_path:
        manifest_path = _resolve(checkpoint_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_path = manifest.get("results_json")
            if summary_path and _resolve(summary_path).exists():
                train_summary = json.loads(_resolve(summary_path).read_text(encoding="utf-8"))
            base_dir = manifest.get("base_dir")
            if base_dir:
                config = {**dict(config), "base_dir": base_dir}

    if train_summary:
        for key in (
            "class_names",
            "n_train_per_class",
            "n_val_per_class",
            "n_test_per_class",
            "n_pretrain_per_class",
            "max_seq",
            "batch_size",
        ):
            if key not in config:
                if key == "class_names" and train_summary.get("classes"):
                    config = {**dict(config), "class_names": list(train_summary["classes"])}
                elif key in train_summary:
                    config = {**dict(config), key: train_summary[key]}

    if "class_names" not in config:
        config = {**dict(config), "class_names": ["airplane", "bus"]}
    if "n_train_per_class" not in config:
        config = {**dict(config), "n_train_per_class": 20}
    if "n_val_per_class" not in config:
        config = {**dict(config), "n_val_per_class": 8}
    if "n_test_per_class" not in config:
        config = {**dict(config), "n_test_per_class": 8}
    if "n_pretrain_per_class" not in config:
        config = {**dict(config), "n_pretrain_per_class": 20}
    if "batch_size" not in config:
        config = {**dict(config), "batch_size": 4}

    _apply_config(runtime_mod, config)
    runtime_mod.show_raw_samples = lambda n=4: None
    runtime_mod.show_generated = lambda cls_models, cls_datasets, device, n=4: None
    if not hasattr(runtime_mod, "_orig_save_sequential_strokes"):
        runtime_mod._orig_save_sequential_strokes = runtime_mod.save_sequential_strokes
    seq_per_class = int(config.get("n_generate_per_class", 1))
    runtime_mod.save_sequential_strokes = (
        lambda cls_models, cls_datasets, device, n_per_class=10, img_size=256: runtime_mod._orig_save_sequential_strokes(
            cls_models,
            cls_datasets,
            device,
            n_per_class=seq_per_class,
            img_size=img_size,
        )
    )
    runtime_mod.main(skip_eda=True, skip_pretrain=True, skip_finetune=True)

    class_names = list(config.get("class_names", runtime_mod.CLASSES))
    output_dir = _resolve(config.get("output_dir", "scripts/artifacts/generated/sketchgpt"))
    saved = _copy_preview_images(runtime_mod, class_names, output_dir)

    return {
        "saved_images": saved,
        "output_dir": str(output_dir),
        "sequential_dir": str(runtime_mod.SEQ_DIR),
    }
