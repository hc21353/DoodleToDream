from __future__ import annotations


from __future__ import annotations

import os, sys, platform, warnings, torch

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)

if not torch.cuda.is_available():
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Platform:", platform.platform())






NOTEBOOK_VER = 27
NOTEBOOK_NAME = f"VQ_SGen_ver{NOTEBOOK_VER}"
RUN_MODE = "train"
RUN_SUFFIX = "main"
RESET_WORKSPACE = False
FORCE_DOWNLOAD = False
NUM_WORKERS = 4
PROJECT_SEED = 42


SAVE_NEWLY_TRAINED_MODELS_TO_GDRIVE = True




USER_CLASSES = [
    "hot air balloon",
    "motorbike",
    "sailboat",
    "airplane",
    "helicopter",
    "submarine",
    "canoe",
    "bus",
    "car",
    "train",
]








USE_PRETRAINED_WEIGHTS = {
    "shape_ae": True,
    "location_ae": True,
    "shape_tokenizer": True,
    "location_tokenizer": True,
    "generator": False,
}

TRAIN_OR_FINETUNE = {
    "shape_ae": False,
    "location_ae": False,
    "shape_tokenizer": False,
    "location_tokenizer": False,
    "generator": True,
}






MODEL_DATASET_ASSIGNMENTS = {
    "shape_ae": "target_quickdraw",
    "location_ae": "target_quickdraw",
    "shape_tokenizer": "target_quickdraw",
    "location_tokenizer": "target_quickdraw",
    "generator": "target_quickdraw",
}




DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/Current/빅데이터AI핀테크/딥러닝"
DRIVE_MODEL_ROOT = f"{DRIVE_PROJECT_ROOT}/모델"
DRIVE_MODEL_VER_DIR = f"{DRIVE_MODEL_ROOT}/ver{NOTEBOOK_VER}"

WORKSPACE_ROOT = f"/content/{NOTEBOOK_NAME}_workspace"
RUN_TAG = f"{NOTEBOOK_NAME}_{RUN_MODE}_{RUN_SUFFIX}"


CREATIVESKETCH_AUTO_DOWNLOAD = False
CREATIVESKETCH_LOCAL_ROOT = ""
CREATIVESKETCH_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/14ZywlSE-khagmSz23KKFbLCQLoMOxPzl"
CREATIVESKETCH_EXTRACT_ARCHIVES = False

USE_PREPROCESSED_CS_CACHE = True
PREPROCESSED_CS_PARSED_ROOT = f"{DRIVE_PROJECT_ROOT}/코드2/creativesketch/parsed/creativesketch"
PREPROCESSED_CS_SUMMARY_PATH = f"{DRIVE_PROJECT_ROOT}/코드2/preprocess_summary.json"




SOURCE_SHAPE_AE_DOMAINS = ["cb", "cc"]
SOURCE_TOKENIZER_DOMAINS = ["cb"]
SOURCE_USE_DOMAIN_AS_CLASS_LABEL = False

SOURCE_SHAPE_AE_DATASET_FRACTION = 0.1
SOURCE_TOKENIZER_DATASET_FRACTION = 0.1
TARGET_DATASET_FRACTION = 1.0

TARGET_REPRESENTATION_MAX_DRAWINGS_PER_CLASS = 1000
TARGET_GENERATOR_MAX_DRAWINGS_PER_CLASS = TARGET_REPRESENTATION_MAX_DRAWINGS_PER_CLASS
MAX_STROKES = 32





QUICKDRAW_CANONICAL_STROKE_ORDER = "bbox_area_desc"






SHAPE_AE_EPOCHS = 80
SHAPE_AE_BATCH_SIZE = 16

LOCATION_AE_EPOCHS = 40
LOCATION_AE_BATCH_SIZE = 128

SHAPE_STROKE_WIDTH = 4




SHAPE_CODEBOOK_SIZE = 8192
LOCATION_CODEBOOK_SIZE = 8192

SHAPE_TOKENIZER_EPOCHS = 100
SHAPE_TOKENIZER_BATCH_SIZE = 32
SHAPE_TOKENIZER_SCHEDULER_FACTOR = 0.5
SHAPE_TOKENIZER_SCHEDULER_PATIENCE = 6
SHAPE_TOKENIZER_MIN_LR = 1e-5
SHAPE_TOKENIZER_EARLY_STOPPING_PATIENCE = 20

LOCATION_TOKENIZER_EPOCHS = 80
LOCATION_TOKENIZER_BATCH_SIZE = 64
LOCATION_TOKENIZER_SCHEDULER_FACTOR = 0.5
LOCATION_TOKENIZER_SCHEDULER_PATIENCE = 6
LOCATION_TOKENIZER_MIN_LR = 1e-5
LOCATION_TOKENIZER_EARLY_STOPPING_PATIENCE = 15




GENERATOR_EPOCHS = 50
GENERATOR_BATCH_SIZE = 64
GENERATOR_TOKEN_RESIDUAL_SCALE = 0.10
GENERATOR_INPUT_MODE = "codebook_residual"

GENERATOR_SCHEDULED_SAMPLING_START = 1.00
GENERATOR_SCHEDULED_SAMPLING_END = 0.60
GENERATOR_SCHEDULED_SAMPLING_WARMUP_EPOCHS = 8
GENERATOR_SCHEDULED_SAMPLING_DECAY_EPOCHS = 32

GENERATOR_EARLY_STROKE_LOSS_MIN_WEIGHT = 1.0
GENERATOR_EARLY_STROKE_LOSS_MAX_WEIGHT = 2.5
GENERATOR_EARLY_STROKE_LOSS_POWER = 1.0




SOURCE_CANVAS_SIZE = 256
CANVAS_SIZE = 256
IMAGE_SIZE = 256
SHAPE_CENTER_SCALE_MARGIN = 1.00
SHAPE_MAX_CANVAS_COVERAGE = 1.00
SHAPE_DECODE_THRESHOLD = 0.42
DECODE_USE_BBOX_SIZE = False

CLASS_ONLY_SAMPLES_PER_CLASS = 5
CLASS_ONLY_TEMPERATURE = 1.0
CLASS_ONLY_TOP_P = 0.90

MIN_STROKE_POINTS = 2
MIN_STROKE_LENGTH = 5.0
MIN_STROKE_SPAN = 2.5
MIN_STROKE_BBOX_SIZE = 3.0
DROP_LONG_STROKE_FACTOR = 4.0
MERGE_SHORT_STROKES = True
MERGE_SHORT_LENGTH = 10.0
MERGE_ENDPOINT_GAP = 4.0

TRAIN_AUG_ENABLE = True
TRAIN_AUG_ROT_DEG = 10.0
TRAIN_AUG_SCALE = 0.10
TRAIN_AUG_TRANSLATE = 0.04




SHAPE_BITMAP_BCE_WEIGHT = 0.8
SHAPE_BITMAP_L1_WEIGHT = 0.8
SHAPE_DISTANCE_WEIGHT = 0.5
SHAPE_DICE_WEIGHT = 0.5
SHAPE_DISTANCE_DECAY = 2.5

EMBEDDING_STANDARDIZE = True
TOKENIZER_INPUT_CLAMP = 6.0

SHAPE_TOKENIZER_SEQ_WEIGHT = 0.5
SHAPE_TOKENIZER_LOCAL_WEIGHT = 0.25
SHAPE_TOKENIZER_COSINE_WEIGHT = 0.05
SHAPE_TOKENIZER_VQ_WEIGHT = 0.05
SHAPE_TOKENIZER_IMAGE_RECON_WEIGHT = 2.0
SHAPE_TOKENIZER_IMAGE_LOCAL_WEIGHT = 1.0
SHAPE_TOKENIZER_IMAGE_IOU_WEIGHT = 1.0
SHAPE_TOKENIZER_IMAGE_BCE_WEIGHT = 1.0
SHAPE_SMALL_STROKE_GAMMA = 0.50
SHAPE_SMALL_STROKE_WEIGHT_MIN = 0.75
SHAPE_SMALL_STROKE_WEIGHT_MAX = 3.00

LOCATION_TOKENIZER_SEQ_WEIGHT = 0.50
LOCATION_TOKENIZER_LOCAL_WEIGHT = 0.25
LOCATION_TOKENIZER_COSINE_WEIGHT = 0.00
LOCATION_TOKENIZER_VQ_WEIGHT = 0.10
LOCATION_TOKENIZER_BBOX_WEIGHT = 2.00
LOCATION_TOKENIZER_IOU_WEIGHT = 1.00

TOKENIZER_EMA_DECAY = 0.99
TOKENIZER_EMA_EPS = 1e-5
TOKENIZER_DEAD_CODE_THRESHOLD = 0.5
TOKENIZER_RESET_INTERVAL = 200
TOKENIZER_COMMITMENT_COST = 0.25
TOKENIZER_COMMITMENT_WARMUP_STEPS = 500
TOKENIZER_INIT_BATCH_SAMPLES = 4096

SHAPE_TOKENIZER_MODEL_DIM = 512
SHAPE_TOKENIZER_NUM_LAYERS = 4
SHAPE_TOKENIZER_KERNEL_SIZE = 5

LOCATION_TOKENIZER_MODEL_DIM = 256
LOCATION_TOKENIZER_NUM_LAYERS = 4
LOCATION_TOKENIZER_KERNEL_SIZE = 3

GENERATOR_MODEL_DIM = 384
GENERATOR_NUM_HEADS = 6
GENERATOR_NUM_LAYERS = 8
GENERATOR_FF_DIM = 1024
GENERATOR_DROPOUT = 0.10

GENERATOR_TEMPERATURE = 1.0
GENERATOR_TOP_P = 0.90

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_KEYS = [
    "shape_ae",
    "location_ae",
    "shape_tokenizer",
    "location_tokenizer",
    "generator",
]
MODEL_FILE_NAMES = {
    "shape_ae": "shape_ae.pt",
    "location_ae": "location_ae.pt",
    "shape_tokenizer": "shape_tokenizer.pt",
    "location_tokenizer": "location_tokenizer.pt",
    "generator": "generator.pt",
}
ALLOWED_MODEL_DATASET_PRESETS = {
    "target_quickdraw",
}

def _normalize_run_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"train", "test"}:
        raise ValueError(f"RUN_MODE must be 'train' or 'test', got: {mode}")
    return mode

def _normalize_model_dataset_assignments(assignments: dict) -> dict:
    defaults = {
        "shape_ae": "target_quickdraw",
        "location_ae": "target_quickdraw",
        "shape_tokenizer": "target_quickdraw",
        "location_tokenizer": "target_quickdraw",
        "generator": "target_quickdraw",
    }
    if assignments is None:
        assignments = {}
    merged = {}
    for key in MODEL_KEYS:
        value = assignments.get(key, defaults[key])
        value = str(value).strip()
        if value not in ALLOWED_MODEL_DATASET_PRESETS:
            raise ValueError(
                f"MODEL_DATASET_ASSIGNMENTS[{key!r}]={value!r} 는 허용되지 않습니다. "
                f"허용값: {sorted(ALLOWED_MODEL_DATASET_PRESETS)}"
            )
        merged[key] = value

    unknown = sorted(set(assignments.keys()) - set(MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown MODEL_DATASET_ASSIGNMENTS keys: {unknown}")
    return merged

def _auto_pretrained_paths(model_dir: str) -> dict:
    model_dir = str(model_dir).strip()
    return {stage: str(Path(model_dir) / fname) for stage, fname in MODEL_FILE_NAMES.items()}

def _build_effective_stage_plan(run_mode: str, use_pretrained: dict, train_or_finetune: dict) -> dict:
    run_mode = _normalize_run_mode(run_mode)
    if run_mode == "test":
        return {
            stage: {
                "use_pretrained": True,
                "train": False,
                "effective_mode": "load_only",
            }
            for stage in MODEL_KEYS
        }

    plan = {}
    for stage in MODEL_KEYS:
        use_pt = bool(use_pretrained.get(stage, False))
        do_train = bool(train_or_finetune.get(stage, False))

        if (not use_pt) and (not do_train):
            raise ValueError(
                f"{stage}: USE_PRETRAINED_WEIGHTS=False and TRAIN_OR_FINETUNE=False 는 허용되지 않습니다. "
                "최소한 pretrained load 또는 학습 하나는 켜야 합니다."
            )

        if do_train and use_pt:
            eff = "finetune"
        elif do_train and (not use_pt):
            eff = "train_from_scratch"
        elif (not do_train) and use_pt:
            eff = "load_only"
        else:
            raise RuntimeError(f"Unexpected stage state for {stage}")

        plan[stage] = {
            "use_pretrained": use_pt,
            "train": do_train,
            "effective_mode": eff,
        }
    return plan

def _validate_pretrained_paths(stage_plan: dict, pretrained_paths: dict) -> None:
    missing = []
    for stage, info in stage_plan.items():
        if not bool(info["use_pretrained"]):
            continue
        ckpt = str(pretrained_paths.get(stage, "")).strip()
        if not ckpt:
            missing.append((stage, "(empty path)"))
            continue
        if not Path(ckpt).exists():
            missing.append((stage, ckpt))
    if missing:
        msg = "\n".join([f"- {stage}: {path}" for stage, path in missing])
        raise FileNotFoundError(
            "아래 pretrained checkpoint 경로를 찾을 수 없습니다. "
            "NOTEBOOK_VER / DRIVE_MODEL_VER_DIR / 파일명을 확인하세요.\n"
            f"{msg}"
        )

def _print_stage_plan(stage_plan: dict, pretrained_paths: dict, dataset_assignments: dict) -> None:
    print("=" * 100)
    print("Effective model plan")
    print("=" * 100)
    for stage in MODEL_KEYS:
        info = stage_plan[stage]
        ckpt = str(pretrained_paths.get(stage, "")).strip()
        ckpt_msg = ckpt if info["use_pretrained"] else "(not used)"
        dataset_preset = str(dataset_assignments.get(stage, ""))
        print(
            f"{stage:20s} | dataset={dataset_preset:16s} | "
            f"mode={info['effective_mode']:18s} | "
            f"use_pretrained={str(info['use_pretrained']):5s} | "
            f"train={str(info['train']):5s} | ckpt={ckpt_msg}"
        )

RUN_MODE = _normalize_run_mode(RUN_MODE)
MODEL_DATASET_ASSIGNMENTS = _normalize_model_dataset_assignments(MODEL_DATASET_ASSIGNMENTS)
PRETRAINED_MODEL_PATHS = _auto_pretrained_paths(DRIVE_MODEL_VER_DIR)
EFFECTIVE_STAGE_PLAN = _build_effective_stage_plan(
    RUN_MODE,
    USE_PRETRAINED_WEIGHTS,
    TRAIN_OR_FINETUNE,
)
try:
    _validate_pretrained_paths(EFFECTIVE_STAGE_PLAN, PRETRAINED_MODEL_PATHS)
except Exception as _e:
    print("[WARN] pretrained path validation skipped:", _e)
_print_stage_plan(EFFECTIVE_STAGE_PLAN, PRETRAINED_MODEL_PATHS, MODEL_DATASET_ASSIGNMENTS)

workspace_signature_payload = {
    "notebook_ver": int(NOTEBOOK_VER),
    "run_mode": str(RUN_MODE),
    "run_suffix": str(RUN_SUFFIX),
    "user_classes": list(USER_CLASSES),
    "stage_plan": copy.deepcopy(EFFECTIVE_STAGE_PLAN),
    "model_dataset_assignments": copy.deepcopy(MODEL_DATASET_ASSIGNMENTS),
    "source_shape_domains": list(SOURCE_SHAPE_AE_DOMAINS),
    "source_tokenizer_domains": list(SOURCE_TOKENIZER_DOMAINS),
    "source_shape_fraction": float(SOURCE_SHAPE_AE_DATASET_FRACTION),
    "source_tokenizer_fraction": float(SOURCE_TOKENIZER_DATASET_FRACTION),
    "target_fraction": float(TARGET_DATASET_FRACTION),
    "target_rep_cap": int(TARGET_REPRESENTATION_MAX_DRAWINGS_PER_CLASS),
    "target_gen_cap": int(TARGET_GENERATOR_MAX_DRAWINGS_PER_CLASS),
    "max_strokes": int(MAX_STROKES),
    "shape_codebook": int(SHAPE_CODEBOOK_SIZE),
    "location_codebook": int(LOCATION_CODEBOOK_SIZE),
    "generator_dim": int(GENERATOR_MODEL_DIM),
    "seed": int(PROJECT_SEED),
}
config_sig = hashlib.md5(
    json.dumps(workspace_signature_payload, sort_keys=True).encode("utf-8")
).hexdigest()[:10]

workspace_dir = str(Path(WORKSPACE_ROOT) / f"{RUN_TAG}_{config_sig}")
download_dir = str(Path(workspace_dir) / "_downloads")

if RESET_WORKSPACE and os.path.isdir(workspace_dir):
    shutil.rmtree(workspace_dir)

cfg = {
    "project": {
        "notebook_ver": int(NOTEBOOK_VER),
        "notebook_name": str(NOTEBOOK_NAME),
        "seed": int(PROJECT_SEED),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": int(NUM_WORKERS),
        "mixed_precision": bool(torch.cuda.is_available()),
        "run_tag": str(RUN_TAG),
        "run_mode": str(RUN_MODE),
        "config_signature": str(config_sig),
        "workspace_root": str(workspace_dir),
        "download_root": str(download_dir),
        "gdrive_model_dir": str(DRIVE_MODEL_VER_DIR),
        "results_json_name": f"{NOTEBOOK_NAME}_results.json",
        "token_npz_template": "{split}_tokens_ver{ver}.npz",
        "generator_token_npz_template": "{split}_tokens_generator_ver{ver}.npz",
        "preview_grid_name": f"{NOTEBOOK_NAME}_class_conditioned_grid.png",
    },
    "runtime": {
        "save_newly_trained_models_to_gdrive": bool(SAVE_NEWLY_TRAINED_MODELS_TO_GDRIVE),
        "stages": copy.deepcopy(EFFECTIVE_STAGE_PLAN),
        "model_dataset_assignments": copy.deepcopy(MODEL_DATASET_ASSIGNMENTS),
    },
    "source": {
        "auto_download": bool(CREATIVESKETCH_AUTO_DOWNLOAD),
        "local_root": str(CREATIVESKETCH_LOCAL_ROOT).strip(),
        "gdrive_folder_url": str(CREATIVESKETCH_GDRIVE_FOLDER_URL),
        "extract_archives": bool(CREATIVESKETCH_EXTRACT_ARCHIVES),
        "shape_ae_domains": list(SOURCE_SHAPE_AE_DOMAINS),
        "tokenizer_domains": list(SOURCE_TOKENIZER_DOMAINS),
        "use_domain_as_class_label": bool(SOURCE_USE_DOMAIN_AS_CLASS_LABEL),
        "shape_ae_fraction": float(SOURCE_SHAPE_AE_DATASET_FRACTION),
        "tokenizer_fraction": float(SOURCE_TOKENIZER_DATASET_FRACTION),
        "train_ratio": 0.90,
        "val_ratio": 0.05,
        "test_ratio": 0.05,
        "use_preprocessed_cache": bool(USE_PREPROCESSED_CS_CACHE),
        "preprocessed_parsed_root": str(PREPROCESSED_CS_PARSED_ROOT).strip(),
        "preprocess_summary_path": str(PREPROCESSED_CS_SUMMARY_PATH).strip(),
    },
    "target": {
        "classes": list(USER_CLASSES),
        "dataset_fraction": float(TARGET_DATASET_FRACTION),
        "representation_max_drawings_per_class": int(TARGET_REPRESENTATION_MAX_DRAWINGS_PER_CLASS),
        "generator_max_drawings_per_class": int(TARGET_GENERATOR_MAX_DRAWINGS_PER_CLASS),
        "train_ratio": 0.90,
        "val_ratio": 0.05,
        "test_ratio": 0.05,
        "variant": "simplified",
        "raw_url_template": "https://storage.googleapis.com/quickdraw_dataset/full/raw/{class_name}.ndjson",
        "simplified_url_template": "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{class_name}.ndjson",
        "filter_recognized": True,
    },
    "pretrained": {
        "shape_ae_ckpt": str(PRETRAINED_MODEL_PATHS["shape_ae"]),
        "location_ae_ckpt": str(PRETRAINED_MODEL_PATHS["location_ae"]),
        "shape_tokenizer_ckpt": str(PRETRAINED_MODEL_PATHS["shape_tokenizer"]),
        "location_tokenizer_ckpt": str(PRETRAINED_MODEL_PATHS["location_tokenizer"]),
        "generator_ckpt": str(PRETRAINED_MODEL_PATHS["generator"]),
    },
    "shape_ae": {
        "epochs": int(SHAPE_AE_EPOCHS),
        "batch_size": int(SHAPE_AE_BATCH_SIZE),
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "embedding_dim": 256,
        "hidden_dims": [32, 64, 128, 256],
        "bitmap_bce_weight": float(SHAPE_BITMAP_BCE_WEIGHT),
        "bitmap_l1_weight": float(SHAPE_BITMAP_L1_WEIGHT),
        "distance_weight": float(SHAPE_DISTANCE_WEIGHT),
        "dice_weight": float(SHAPE_DICE_WEIGHT),
    },
    "location_ae": {
        "epochs": int(LOCATION_AE_EPOCHS),
        "batch_size": int(LOCATION_AE_BATCH_SIZE),
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "embedding_dim": 64,
        "hidden_dims": [64, 128, 128],
    },
    "shape_tokenizer": {
        "epochs": int(SHAPE_TOKENIZER_EPOCHS),
        "batch_size": int(SHAPE_TOKENIZER_BATCH_SIZE),
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "model_dim": int(SHAPE_TOKENIZER_MODEL_DIM),
        "num_layers": int(SHAPE_TOKENIZER_NUM_LAYERS),
        "kernel_size": int(SHAPE_TOKENIZER_KERNEL_SIZE),
        "dropout": 0.05,
        "num_embeddings": int(SHAPE_CODEBOOK_SIZE),
        "commitment_cost": float(TOKENIZER_COMMITMENT_COST),
        "commitment_warmup_steps": int(TOKENIZER_COMMITMENT_WARMUP_STEPS),
        "ema_decay": float(TOKENIZER_EMA_DECAY),
        "ema_eps": float(TOKENIZER_EMA_EPS),
        "dead_code_threshold": float(TOKENIZER_DEAD_CODE_THRESHOLD),
        "reset_interval": int(TOKENIZER_RESET_INTERVAL),
        "init_batch_samples": int(TOKENIZER_INIT_BATCH_SAMPLES),
        "seq_recon_weight": float(SHAPE_TOKENIZER_SEQ_WEIGHT),
        "local_recon_weight": float(SHAPE_TOKENIZER_LOCAL_WEIGHT),
        "cosine_weight": float(SHAPE_TOKENIZER_COSINE_WEIGHT),
        "vq_weight": float(SHAPE_TOKENIZER_VQ_WEIGHT),
        "image_recon_weight": float(SHAPE_TOKENIZER_IMAGE_RECON_WEIGHT),
        "image_local_weight": float(SHAPE_TOKENIZER_IMAGE_LOCAL_WEIGHT),
        "image_iou_weight": float(SHAPE_TOKENIZER_IMAGE_IOU_WEIGHT),
        "image_bce_weight": float(SHAPE_TOKENIZER_IMAGE_BCE_WEIGHT),
        "scheduler": "plateau",
        "scheduler_factor": float(SHAPE_TOKENIZER_SCHEDULER_FACTOR),
        "scheduler_patience": int(SHAPE_TOKENIZER_SCHEDULER_PATIENCE),
        "min_lr": float(SHAPE_TOKENIZER_MIN_LR),
        "early_stopping_patience": int(SHAPE_TOKENIZER_EARLY_STOPPING_PATIENCE),
        "small_stroke_gamma": float(SHAPE_SMALL_STROKE_GAMMA),
        "small_stroke_weight_min": float(SHAPE_SMALL_STROKE_WEIGHT_MIN),
        "small_stroke_weight_max": float(SHAPE_SMALL_STROKE_WEIGHT_MAX),
    },
    "location_tokenizer": {
        "type": "sequence_vq",
        "epochs": int(LOCATION_TOKENIZER_EPOCHS),
        "batch_size": int(LOCATION_TOKENIZER_BATCH_SIZE),
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "grad_clip_norm": 1.0,
        "model_dim": int(LOCATION_TOKENIZER_MODEL_DIM),
        "num_layers": int(LOCATION_TOKENIZER_NUM_LAYERS),
        "kernel_size": int(LOCATION_TOKENIZER_KERNEL_SIZE),
        "dropout": 0.05,
        "num_embeddings": int(LOCATION_CODEBOOK_SIZE),
        "codebook_dim": int(LOCATION_TOKENIZER_MODEL_DIM),
        "commitment_cost": float(TOKENIZER_COMMITMENT_COST),
        "commitment_warmup_steps": int(TOKENIZER_COMMITMENT_WARMUP_STEPS),
        "ema_decay": float(TOKENIZER_EMA_DECAY),
        "ema_eps": float(TOKENIZER_EMA_EPS),
        "dead_code_threshold": float(TOKENIZER_DEAD_CODE_THRESHOLD),
        "reset_interval": int(TOKENIZER_RESET_INTERVAL),
        "init_batch_samples": int(TOKENIZER_INIT_BATCH_SAMPLES),
        "seq_recon_weight": float(LOCATION_TOKENIZER_SEQ_WEIGHT),
        "local_recon_weight": float(LOCATION_TOKENIZER_LOCAL_WEIGHT),
        "cosine_weight": float(LOCATION_TOKENIZER_COSINE_WEIGHT),
        "vq_weight": float(LOCATION_TOKENIZER_VQ_WEIGHT),
        "bbox_weight": float(LOCATION_TOKENIZER_BBOX_WEIGHT),
        "iou_weight": float(LOCATION_TOKENIZER_IOU_WEIGHT),
        "scheduler": "plateau",
        "scheduler_factor": float(LOCATION_TOKENIZER_SCHEDULER_FACTOR),
        "scheduler_patience": int(LOCATION_TOKENIZER_SCHEDULER_PATIENCE),
        "min_lr": float(LOCATION_TOKENIZER_MIN_LR),
        "early_stopping_patience": int(LOCATION_TOKENIZER_EARLY_STOPPING_PATIENCE),
    },
    "generator": {
        "epochs": int(GENERATOR_EPOCHS),
        "batch_size": int(GENERATOR_BATCH_SIZE),
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "model_dim": int(GENERATOR_MODEL_DIM),
        "num_heads": int(GENERATOR_NUM_HEADS),
        "num_layers": int(GENERATOR_NUM_LAYERS),
        "ff_dim": int(GENERATOR_FF_DIM),
        "dropout": float(GENERATOR_DROPOUT),
        "temperature": float(GENERATOR_TEMPERATURE),
        "top_p": float(GENERATOR_TOP_P),
        "input_mode": str(GENERATOR_INPUT_MODE),
        "token_residual_scale": float(GENERATOR_TOKEN_RESIDUAL_SCALE),
        "scheduled_sampling_start": float(GENERATOR_SCHEDULED_SAMPLING_START),
        "scheduled_sampling_end": float(GENERATOR_SCHEDULED_SAMPLING_END),
        "scheduled_sampling_warmup_epochs": int(GENERATOR_SCHEDULED_SAMPLING_WARMUP_EPOCHS),
        "scheduled_sampling_decay_epochs": int(GENERATOR_SCHEDULED_SAMPLING_DECAY_EPOCHS),
        "early_stroke_loss_min_weight": float(GENERATOR_EARLY_STROKE_LOSS_MIN_WEIGHT),
        "early_stroke_loss_max_weight": float(GENERATOR_EARLY_STROKE_LOSS_MAX_WEIGHT),
        "early_stroke_loss_power": float(GENERATOR_EARLY_STROKE_LOSS_POWER),
    },
    "debug": {
        "class_only_samples_per_class": int(CLASS_ONLY_SAMPLES_PER_CLASS),
        "class_only_temperature": float(CLASS_ONLY_TEMPERATURE),
        "class_only_top_p": float(CLASS_ONLY_TOP_P),
    },
    "dataset_defaults": {
        "source_canvas_size": int(SOURCE_CANVAS_SIZE),
        "canvas_size": int(CANVAS_SIZE),
        "image_size": int(IMAGE_SIZE),
        "shape_stroke_width": int(SHAPE_STROKE_WIDTH),
        "shape_center_scale_margin": float(SHAPE_CENTER_SCALE_MARGIN),
        "shape_max_canvas_coverage": float(SHAPE_MAX_CANVAS_COVERAGE),
        "shape_decode_threshold": float(SHAPE_DECODE_THRESHOLD),
        "min_strokes": 1,
        "max_strokes": int(MAX_STROKES),
        "cache_parsed_drawings": 4096,
        "min_stroke_points": int(MIN_STROKE_POINTS),
        "min_stroke_length": float(MIN_STROKE_LENGTH),
        "min_stroke_span": float(MIN_STROKE_SPAN),
        "min_stroke_bbox_size": float(MIN_STROKE_BBOX_SIZE),
        "drop_long_stroke_factor": float(DROP_LONG_STROKE_FACTOR),
        "merge_short_strokes": bool(MERGE_SHORT_STROKES),
        "merge_short_length": float(MERGE_SHORT_LENGTH),
        "merge_endpoint_gap": float(MERGE_ENDPOINT_GAP),
        "train_aug_enable": bool(TRAIN_AUG_ENABLE),
        "train_aug_rot_deg": float(TRAIN_AUG_ROT_DEG),
        "train_aug_scale": float(TRAIN_AUG_SCALE),
        "train_aug_translate": float(TRAIN_AUG_TRANSLATE),
        "distance_decay": float(SHAPE_DISTANCE_DECAY),
        "embedding_standardize": bool(EMBEDDING_STANDARDIZE),
        "tokenizer_input_clamp": float(TOKENIZER_INPUT_CLAMP),
                "quickdraw_canonical_stroke_order": str(QUICKDRAW_CANONICAL_STROKE_ORDER),
"decode_use_bbox_size": bool(DECODE_USE_BBOX_SIZE),
    },
}

print("workspace:", cfg["project"]["workspace_root"])
print("download_root:", cfg["project"]["download_root"])
print("config_signature:", cfg["project"]["config_signature"])
print("run_mode:", cfg["project"]["run_mode"])
print("gdrive_model_dir:", cfg["project"]["gdrive_model_dir"])
print("target classes:", cfg["target"]["classes"])
print("model dataset assignments:", cfg["runtime"]["model_dataset_assignments"])
print("shape / location codebooks:", cfg["shape_tokenizer"]["num_embeddings"], cfg["location_tokenizer"]["num_embeddings"])

