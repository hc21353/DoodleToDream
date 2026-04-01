from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def _load_full_module():
    path = Path(__file__).with_name("pipeline.py")
    spec = importlib.util.spec_from_file_location("vq_sgen_full_runtime_generate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to import pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vq_sgen_full_runtime_generate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_runtime_env() -> None:
    cache_root = Path(__file__).resolve().parents[3] / "scripts" / "artifacts" / "_cache" / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[3] / p).resolve()


def _copy_local_quickdraw_if_available(cfg_root: Dict[str, Any], classes: list[str]) -> None:
    root = Path(__file__).resolve().parents[3]
    local_quickdraw = root / "scripts" / "artifacts" / "sketchgpt_full" / "quickdraw_data"
    if not local_quickdraw.exists():
        return
    raw_root = Path(cfg_root["project"]["workspace_root"]).resolve() / "target_quickdraw" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        src = local_quickdraw / f"{cls}.ndjson"
        dst = raw_root / f"{cls.replace(' ', '_').replace('/', '_')}.ndjson"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def generate_vq_sgen_images(config: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_runtime_env()
    full_mod = _load_full_module()
    manifest = json.loads(_resolve(config["checkpoint_path"]).read_text(encoding="utf-8"))
    train_summary = {}
    summary_path = manifest.get("results_json")
    if summary_path and Path(_resolve(summary_path)).exists():
        train_summary = json.loads(_resolve(summary_path).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(train_summary.get("cfg", full_mod.cfg))
    workspace_root = _resolve(manifest["workspace_root"])
    class_names = list(config.get("class_names", ["airplane", "bus"]))

    cfg["project"]["workspace_root"] = str(workspace_root)
    cfg["project"]["download_root"] = str(workspace_root / "_downloads")
    cfg["project"]["num_workers"] = int(config.get("num_workers", 0))
    cfg["project"]["device"] = str(config.get("device", "cpu"))
    cfg["project"]["mixed_precision"] = bool(config.get("mixed_precision", False))
    cfg["target"]["classes"] = class_names
    cfg["target"]["representation_max_drawings_per_class"] = max(12, int(config.get("samples_per_class", 1)))
    cfg["target"]["generator_max_drawings_per_class"] = max(12, int(config.get("samples_per_class", 1)))
    cfg["dataset_defaults"]["max_strokes"] = int(config.get("max_strokes", 16))
    _copy_local_quickdraw_if_available(cfg, class_names)

    stage_cfg = full_mod.build_stage_cfg(cfg, "target_quickdraw", "quickdraw")
    full_mod.ensure_workspace(stage_cfg)
    full_mod.prepare_subset(stage_cfg, force_download=False)

    device = full_mod.get_device(stage_cfg["project"].get("device"))
    shape_ae = full_mod.build_shape_ae(stage_cfg).to(device)
    location_ae = full_mod.build_location_ae(stage_cfg).to(device)
    shape_tokenizer = full_mod.build_shape_tokenizer(stage_cfg).to(device)
    location_tokenizer = full_mod.build_location_tokenizer(stage_cfg).to(device)
    generator = full_mod.build_generator(stage_cfg, shape_tokenizer, location_tokenizer).to(device)

    full_mod.load_checkpoint_weights(shape_ae, manifest["shape_ae_ckpt"], device)
    full_mod.load_checkpoint_weights(location_ae, manifest["location_ae_ckpt"], device)
    full_mod.load_checkpoint_weights(shape_tokenizer, manifest["shape_tokenizer_ckpt"], device)
    full_mod.load_checkpoint_weights(location_tokenizer, manifest["location_tokenizer_ckpt"], device)
    full_mod.load_checkpoint_weights(generator, manifest["generator_ckpt"], device)

    output_dir = _resolve(config.get("output_dir", "scripts/artifacts/generated/vq_sgen"))
    output_dir.mkdir(parents=True, exist_ok=True)
    temperature = float(config.get("temperature", 1.0))
    top_p = float(config.get("top_p", 0.9))
    per_class = int(config.get("samples_per_class", 1))
    max_steps = int(config.get("max_strokes", stage_cfg["dataset"]["max_strokes"]))

    saved = []
    for class_id, class_name in enumerate(class_names):
        samples = full_mod.sample_class_conditioned_sketches(
            stage_cfg,
            generator,
            shape_ae,
            location_ae,
            shape_tokenizer,
            location_tokenizer,
            class_id=class_id,
            num_samples=per_class,
            device=device,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
        )
        if not samples:
            continue
        canvas = np.asarray(samples[0]["canvas"], dtype=np.uint8)
        if canvas.max() <= 1:
            canvas = (canvas * 255).astype(np.uint8)
        out_path = output_dir / f"{class_name.replace(' ', '_')}.png"
        Image.fromarray(canvas, mode="L").save(out_path)
        saved.append(str(out_path))

    return {"saved_images": saved, "output_dir": str(output_dir)}
