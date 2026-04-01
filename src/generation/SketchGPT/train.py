from __future__ import annotations

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
    spec = importlib.util.spec_from_file_location("sketchgpt_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to import SketchGPT pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sketchgpt_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


def _copy_local_quickdraw_if_available(runtime_mod, classes: List[str]) -> None:
    local_quickdraw = _repo_root() / "scripts" / "artifacts" / "sketchgpt_full" / "quickdraw_data"
    if not local_quickdraw.exists():
        return
    data_dir = Path(runtime_mod.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        src = local_quickdraw / f"{cls}.ndjson"
        dst = data_dir / f"{cls}.ndjson"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _apply_config(runtime_mod, config: Dict[str, Any]) -> Dict[str, Path]:
    classes = list(config.get("class_names", runtime_mod.CLASSES))
    if not classes:
        raise ValueError("sketchgpt class_names must not be empty")

    runtime_mod.CLASSES = classes
    runtime_mod.N_TRAIN_PER_CLASS = int(config.get("n_train_per_class", runtime_mod.N_TRAIN_PER_CLASS))
    runtime_mod.N_VAL_PER_CLASS = int(config.get("n_val_per_class", runtime_mod.N_VAL_PER_CLASS))
    runtime_mod.N_TEST_PER_CLASS = int(config.get("n_test_per_class", runtime_mod.N_TEST_PER_CLASS))
    runtime_mod.N_PRETRAIN_PER_CLASS = int(config.get("n_pretrain_per_class", runtime_mod.N_PRETRAIN_PER_CLASS))
    runtime_mod.PRETRAIN_EPOCHS = int(config.get("pretrain_epochs", runtime_mod.PRETRAIN_EPOCHS))
    runtime_mod.FINETUNE_GEN_EPOCHS = int(config.get("finetune_epochs", runtime_mod.FINETUNE_GEN_EPOCHS))
    runtime_mod.PRETRAIN_BATCH = int(config.get("batch_size", runtime_mod.PRETRAIN_BATCH))
    runtime_mod.PRETRAIN_LR = float(config.get("pretrain_lr", runtime_mod.PRETRAIN_LR))
    runtime_mod.FINETUNE_GEN_LR = float(config.get("finetune_lr", runtime_mod.FINETUNE_GEN_LR))
    runtime_mod.MAX_SEQ = int(config.get("max_seq", runtime_mod.MAX_SEQ))
    runtime_mod.TOP_K = int(config.get("top_k", runtime_mod.TOP_K))
    runtime_mod.TEMPERATURE = float(config.get("temperature", runtime_mod.TEMPERATURE))
    runtime_mod.PROMPT_RATIO = float(config.get("prompt_ratio", runtime_mod.PROMPT_RATIO))
    runtime_mod.MIN_NEW_TOKENS = int(config.get("min_new_tokens", runtime_mod.MIN_NEW_TOKENS))

    requested_device = str(config.get("device", runtime_mod.DEVICE))
    if requested_device.startswith("cuda") and not runtime_mod.torch.cuda.is_available():
        runtime_mod.DEVICE = runtime_mod.torch.device("cpu")
    else:
        runtime_mod.DEVICE = runtime_mod.torch.device(requested_device)
    runtime_mod.USE_AMP = bool(config.get("use_amp", runtime_mod.DEVICE.type == "cuda"))
    runtime_mod.scaler = runtime_mod.torch.cuda.amp.GradScaler(enabled=runtime_mod.USE_AMP)

    base_dir = (_repo_root() / config.get("base_dir", "scripts/artifacts/sketchgpt_full")).resolve()
    runtime_mod.BASE_DIR = str(base_dir)
    runtime_mod.DATA_DIR = str(base_dir / "quickdraw_data")
    runtime_mod.CKPT_DIR = str(base_dir / "checkpoints")
    runtime_mod.OUTPUT_DIR = str(base_dir / "outputs")
    runtime_mod.SEQ_DIR = str(base_dir / "outputs" / "sequential")
    for d in (runtime_mod.DATA_DIR, runtime_mod.CKPT_DIR, runtime_mod.OUTPUT_DIR, runtime_mod.SEQ_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    runtime_mod.EDA_PATH = str(Path(runtime_mod.CKPT_DIR) / "eda_params.json")
    runtime_mod.PRETRAIN_PATH = str(Path(runtime_mod.CKPT_DIR) / "pt_best.pt")

    def _ft_path(cls_name: str) -> str:
        return str(Path(runtime_mod.CKPT_DIR) / f"gen_{cls_name.replace(' ', '_')}.pt")

    runtime_mod.finetune_path = _ft_path

    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", runtime_mod.DEVICE.type == "cuda"))

    def _make_loader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
        return runtime_mod.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

    runtime_mod.make_loader = _make_loader

    if not hasattr(runtime_mod, "_orig_run_eda"):
        runtime_mod._orig_run_eda = runtime_mod.run_eda
    eda_samples = int(config.get("eda_samples", 500))
    runtime_mod.run_eda = lambda classes_, n_sample=500: runtime_mod._orig_run_eda(classes_, n_sample=eda_samples)

    if not hasattr(runtime_mod, "_orig_pretrain"):
        runtime_mod._orig_pretrain = runtime_mod.pretrain
    if not hasattr(runtime_mod, "_orig_finetune_class"):
        runtime_mod._orig_finetune_class = runtime_mod.finetune_class

    def _pretrain_proxy(model, train_ds, val_ds, device, sample_weights=None, **kwargs):
        return runtime_mod._orig_pretrain(
            model,
            train_ds,
            val_ds,
            device,
            sample_weights=sample_weights,
            epochs=int(config.get("pretrain_epochs", runtime_mod.PRETRAIN_EPOCHS)),
            lr=float(config.get("pretrain_lr", runtime_mod.PRETRAIN_LR)),
            batch=int(config.get("batch_size", runtime_mod.PRETRAIN_BATCH)),
            save_path=runtime_mod.PRETRAIN_PATH,
        )

    def _finetune_proxy(cls_name, pretrain_path, device, **kwargs):
        return runtime_mod._orig_finetune_class(
            cls_name,
            pretrain_path,
            device,
            epochs=int(config.get("finetune_epochs", runtime_mod.FINETUNE_GEN_EPOCHS)),
            lr=float(config.get("finetune_lr", runtime_mod.FINETUNE_GEN_LR)),
            batch=int(config.get("batch_size", runtime_mod.PRETRAIN_BATCH)),
        )

    runtime_mod.pretrain = _pretrain_proxy
    runtime_mod.finetune_class = _finetune_proxy

    _copy_local_quickdraw_if_available(runtime_mod, classes)
    return {
        "base_dir": base_dir,
        "ckpt_dir": Path(runtime_mod.CKPT_DIR),
        "output_dir": Path(runtime_mod.OUTPUT_DIR),
    }


def train_sketchgpt(config: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_runtime_env()
    runtime_mod = _load_runtime_module()
    paths = _apply_config(runtime_mod, config)

    pretrain_ckpt = Path(runtime_mod.PRETRAIN_PATH)
    ft_ckpts = [Path(runtime_mod.finetune_path(c)) for c in runtime_mod.CLASSES]
    had_all_ckpts = pretrain_ckpt.exists() and all(p.exists() for p in ft_ckpts)

    if bool(config.get("skip_visual_steps", True)):
        runtime_mod.show_raw_samples = lambda n=4: None
        runtime_mod.show_generated = lambda cls_models, cls_datasets, device, n=4: None
        runtime_mod.save_sequential_strokes = (
            lambda cls_models, cls_datasets, device, n_per_class=10, img_size=256: None
        )

    runtime_mod.main(
        skip_eda=bool(config.get("skip_eda", False)),
        skip_pretrain=bool(config.get("skip_pretrain", False)),
        skip_finetune=bool(config.get("skip_finetune", False)),
    )

    missing = [str(p) for p in [pretrain_ckpt, *ft_ckpts] if not p.exists()]
    if missing:
        raise RuntimeError(f"SketchGPT checkpoints missing after run: {missing}")

    summary_path = Path(config.get("history_path", "scripts/artifacts/generation/sketchgpt_history.json"))
    if not summary_path.is_absolute():
        summary_path = _repo_root() / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ran_training = not had_all_ckpts or (not bool(config.get("skip_pretrain", False))) or (not bool(config.get("skip_finetune", False)))
    summary_payload = {
        "base_dir": str(paths["base_dir"]),
        "checkpoint_dir": str(paths["ckpt_dir"]),
        "output_dir": str(paths["output_dir"]),
        "classes": list(runtime_mod.CLASSES),
        "n_train_per_class": int(runtime_mod.N_TRAIN_PER_CLASS),
        "n_val_per_class": int(runtime_mod.N_VAL_PER_CLASS),
        "n_test_per_class": int(runtime_mod.N_TEST_PER_CLASS),
        "n_pretrain_per_class": int(runtime_mod.N_PRETRAIN_PER_CLASS),
        "max_seq": int(runtime_mod.MAX_SEQ),
        "pretrain_epochs": int(runtime_mod.PRETRAIN_EPOCHS),
        "finetune_epochs": int(runtime_mod.FINETUNE_GEN_EPOCHS),
        "batch_size": int(runtime_mod.PRETRAIN_BATCH),
        "pretrain_path": str(pretrain_ckpt),
        "finetune_paths": {c: str(Path(runtime_mod.finetune_path(c))) for c in runtime_mod.CLASSES},
        "ran_training": bool(ran_training),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "base_dir": str(paths["base_dir"]),
        "checkpoint_dir": str(paths["ckpt_dir"]),
        "output_dir": str(paths["output_dir"]),
        "pretrain_path": str(pretrain_ckpt),
        "finetune_paths": {c: str(Path(runtime_mod.finetune_path(c))) for c in runtime_mod.CLASSES},
        "results_json": str(summary_path),
        "ran_training": bool(ran_training),
        "subprocess_returncode": 0,
    }

    checkpoint_path = config.get("checkpoint_path", "scripts/artifacts/generation/sketchgpt.pt")
    manifest = Path(checkpoint_path)
    if not manifest.is_absolute():
        manifest = _repo_root() / manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out["checkpoint_manifest"] = str(manifest)

    return out
