from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_runtime_env() -> None:
    cache_root = _repo_root() / "scripts" / "artifacts" / "_cache" / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))


def _load_runtime_module():
    path = Path(__file__).with_name("pipeline.py")
    spec = importlib.util.spec_from_file_location("vq_sgen_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to import pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vq_sgen_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


def _copy_local_quickdraw_if_available(cfg_root: Dict[str, Any], classes: List[str]) -> None:
    local_quickdraw = _repo_root() / "scripts" / "artifacts" / "sketchgpt_full" / "quickdraw_data"
    if not local_quickdraw.exists():
        return
    raw_root = Path(cfg_root["project"]["workspace_root"]).resolve() / "target_quickdraw" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        src = local_quickdraw / f"{cls}.ndjson"
        dst = raw_root / f"{cls.replace(' ', '_').replace('/', '_')}.ndjson"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _build_cfg(runtime_mod, config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(runtime_mod.cfg)
    workspace = (_repo_root() / config.get("base_dir", "scripts/artifacts/vq_sgen_full")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    class_names = list(config.get("class_names", cfg["target"]["classes"]))
    if not class_names:
        raise ValueError("vq_sgen class_names must not be empty")

    cfg["project"]["workspace_root"] = str(workspace)
    cfg["project"]["download_root"] = str(workspace / "_downloads")
    cfg["project"]["gdrive_model_dir"] = str(workspace / "_model_store")
    cfg["project"]["num_workers"] = int(config.get("num_workers", 0))
    cfg["project"]["device"] = str(config.get("device", "cpu"))
    cfg["project"]["mixed_precision"] = bool(config.get("mixed_precision", False))
    cfg["project"]["seed"] = int(config.get("seed", cfg["project"]["seed"]))

    cfg["target"]["classes"] = class_names
    max_draw = int(config.get("representation_max_drawings_per_class", config.get("samples_per_class", 40)))
    cfg["target"]["representation_max_drawings_per_class"] = max_draw
    cfg["target"]["generator_max_drawings_per_class"] = int(config.get("generator_max_drawings_per_class", max_draw))
    cfg["target"]["dataset_fraction"] = float(config.get("dataset_fraction", 1.0))
    cfg["target"]["filter_recognized"] = bool(config.get("filter_recognized", True))

    cfg["dataset_defaults"]["max_strokes"] = int(config.get("max_strokes", cfg["dataset_defaults"]["max_strokes"]))
    cfg["dataset_defaults"]["image_size"] = int(config.get("image_size", cfg["dataset_defaults"]["image_size"]))
    cfg["dataset_defaults"]["canvas_size"] = int(config.get("canvas_size", cfg["dataset_defaults"]["canvas_size"]))
    cfg["dataset_defaults"]["source_canvas_size"] = int(
        config.get("source_canvas_size", cfg["dataset_defaults"]["source_canvas_size"])
    )

    cfg["shape_tokenizer"]["num_embeddings"] = int(config.get("shape_vocab_size", cfg["shape_tokenizer"]["num_embeddings"]))
    cfg["location_tokenizer"]["num_embeddings"] = int(config.get("loc_vocab_size", cfg["location_tokenizer"]["num_embeddings"]))
    cfg["shape_tokenizer"]["model_dim"] = int(config.get("shape_tokenizer_model_dim", cfg["shape_tokenizer"]["model_dim"]))
    cfg["shape_tokenizer"]["num_layers"] = int(config.get("shape_tokenizer_num_layers", cfg["shape_tokenizer"]["num_layers"]))
    cfg["shape_tokenizer"]["kernel_size"] = int(config.get("shape_tokenizer_kernel_size", cfg["shape_tokenizer"]["kernel_size"]))
    cfg["location_tokenizer"]["model_dim"] = int(
        config.get("location_tokenizer_model_dim", cfg["location_tokenizer"]["model_dim"])
    )
    cfg["location_tokenizer"]["num_layers"] = int(
        config.get("location_tokenizer_num_layers", cfg["location_tokenizer"]["num_layers"])
    )
    cfg["location_tokenizer"]["kernel_size"] = int(
        config.get("location_tokenizer_kernel_size", cfg["location_tokenizer"]["kernel_size"])
    )

    cfg["generator"]["model_dim"] = int(config.get("model_dim", cfg["generator"]["model_dim"]))
    cfg["generator"]["num_heads"] = int(config.get("num_heads", cfg["generator"]["num_heads"]))
    cfg["generator"]["num_layers"] = int(config.get("num_layers", cfg["generator"]["num_layers"]))
    cfg["generator"]["ff_dim"] = int(config.get("ff_dim", cfg["generator"]["ff_dim"]))
    cfg["generator"]["dropout"] = float(config.get("dropout", cfg["generator"]["dropout"]))
    cfg["generator"]["token_residual_scale"] = float(config.get("token_residual_scale", cfg["generator"]["token_residual_scale"]))

    default_epochs = int(config.get("epochs", 1))
    cfg["shape_ae"]["epochs"] = int(config.get("shape_ae_epochs", default_epochs))
    cfg["location_ae"]["epochs"] = int(config.get("location_ae_epochs", default_epochs))
    cfg["shape_tokenizer"]["epochs"] = int(config.get("shape_tokenizer_epochs", default_epochs))
    cfg["location_tokenizer"]["epochs"] = int(config.get("location_tokenizer_epochs", default_epochs))
    cfg["generator"]["epochs"] = int(config.get("generator_epochs", default_epochs))

    default_batch = int(config.get("batch_size", 8))
    cfg["shape_ae"]["batch_size"] = int(config.get("shape_ae_batch_size", default_batch))
    cfg["location_ae"]["batch_size"] = int(config.get("location_ae_batch_size", default_batch))
    cfg["shape_tokenizer"]["batch_size"] = int(config.get("shape_tokenizer_batch_size", default_batch))
    cfg["location_tokenizer"]["batch_size"] = int(config.get("location_tokenizer_batch_size", default_batch))
    cfg["generator"]["batch_size"] = int(config.get("generator_batch_size", default_batch))

    default_lr = float(config.get("lr", 1e-3))
    cfg["shape_ae"]["lr"] = float(config.get("shape_ae_lr", default_lr))
    cfg["location_ae"]["lr"] = float(config.get("location_ae_lr", default_lr))
    cfg["shape_tokenizer"]["lr"] = float(config.get("shape_tokenizer_lr", default_lr))
    cfg["location_tokenizer"]["lr"] = float(config.get("location_tokenizer_lr", default_lr))
    cfg["generator"]["lr"] = float(config.get("generator_lr", default_lr))

    cfg["runtime"]["save_newly_trained_models_to_gdrive"] = False
    cfg["debug"]["class_only_samples_per_class"] = int(config.get("class_only_samples_per_class", 1))
    cfg["debug"]["class_only_temperature"] = float(config.get("temperature", cfg["debug"]["class_only_temperature"]))
    cfg["debug"]["class_only_top_p"] = float(config.get("top_p", cfg["debug"]["class_only_top_p"]))

    for stage in ("shape_ae", "location_ae", "shape_tokenizer", "location_tokenizer", "generator"):
        cfg["runtime"]["stages"][stage]["use_pretrained"] = False
        cfg["runtime"]["stages"][stage]["train"] = bool(config.get(f"train_{stage}", True))
        cfg["runtime"]["stages"][stage]["effective_mode"] = (
            "train_from_scratch" if cfg["runtime"]["stages"][stage]["train"] else "load_only"
        )

    _copy_local_quickdraw_if_available(cfg, class_names)
    return cfg


def _checkpoint_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    ckpt_root = Path(cfg["project"]["workspace_root"]) / "target_quickdraw" / "checkpoints"
    return {
        "shape_ae_ckpt": ckpt_root / "shape_ae_best.pt",
        "location_ae_ckpt": ckpt_root / "location_ae_best.pt",
        "shape_tokenizer_ckpt": ckpt_root / "shape_tokenizer_best.pt",
        "location_tokenizer_ckpt": ckpt_root / "location_tokenizer_best.pt",
        "generator_ckpt": ckpt_root / "generator_best.pt",
    }


def _all_checkpoints_exist(paths: Dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def train_vq_sgen(config: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_runtime_env()
    runtime_mod = _load_runtime_module()
    cfg = _build_cfg(runtime_mod, config)
    force_download = bool(config.get("force_download", False))
    force_retrain = bool(config.get("force_retrain", False))

    ckpt_paths = _checkpoint_paths(cfg)
    ran_training = False
    if force_retrain or not _all_checkpoints_exist(ckpt_paths):
        runtime_mod.run_v18_pipeline(cfg_root=cfg, force_download=force_download)
        ran_training = True

    missing = [str(p) for p in ckpt_paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"VQ-SGen checkpoints missing after run: {missing}")

    summary_path = Path(config.get("history_path", Path(cfg["project"]["workspace_root"]) / "vq_sgen_results.json"))
    if not summary_path.is_absolute():
        summary_path = _repo_root() / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "workspace_root": str(cfg["project"]["workspace_root"]),
        "ran_training": ran_training,
        "cfg": cfg,
        **{k: str(v) for k, v in ckpt_paths.items()},
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "workspace_root": str(cfg["project"]["workspace_root"]),
        "results_json": str(summary_path),
        "preview_paths": [],
        **{k: str(v) for k, v in ckpt_paths.items()},
        "ran_training": ran_training,
        "subprocess_returncode": 0,
    }

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        manifest = Path(checkpoint_path)
        if not manifest.is_absolute():
            manifest = _repo_root() / manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["checkpoint_manifest"] = str(manifest)

    return out
