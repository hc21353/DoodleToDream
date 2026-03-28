"""Base utilities, IO helpers, and preprocessing primitives."""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def workspace_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["dataset"]["base_dir"]).resolve()

def raw_data_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "raw"

def subset_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "subset"

def index_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "index"

def embedding_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "embeddings"

def token_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "tokens"

def checkpoint_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "checkpoints"

def artifact_root(cfg: Dict[str, Any]) -> Path:
    return workspace_root(cfg) / "artifacts"


def config_snapshot_path(cfg: Dict[str, Any]) -> Path:
    return artifact_root(cfg) / "config_used.json"

def write_config_snapshot(cfg: Dict[str, Any]) -> Path:
    path = config_snapshot_path(cfg)
    write_json(path, cfg)
    return path

def ensure_workspace(cfg: Dict[str, Any]) -> None:
    for p in [
        workspace_root(cfg),
        raw_data_root(cfg),
        subset_root(cfg),
        index_root(cfg),
        embedding_root(cfg),
        token_root(cfg),
        checkpoint_root(cfg),
        artifact_root(cfg),
    ]:
        p.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_class_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def denormalize_class_name(safe_name: str) -> str:
    return safe_name.replace("_", " ")

def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_results_json_path(cfg: Dict[str, Any], preferred_names: Sequence[str] | None = None) -> Path:
    candidates = list(preferred_names or [
        "seq_only_results_ver13.json",
        "seq_only_results_ver13_revised.json",
        "seq_only_results_ver12.json",
        "seq_only_results_ver12_revised.json",
    ])
    art_root = artifact_root(cfg)
    for name in candidates:
        path = art_root / name
        if path.exists():
            return path
    matches = sorted(
        list(art_root.glob("seq_only_results_ver13*.json")) + list(art_root.glob("seq_only_results_ver12*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No results json found under {art_root}. Run the full pipeline cell first."
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(requested: str | None = None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.float()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum() / mask.sum().clamp_min(eps)

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    return masked_mean(diff, mask)

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return masked_mean(diff, mask)

def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return masked_mean(diff, mask)

def masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_n = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = 1.0 - (pred_n * target_n).sum(dim=-1)
    return masked_mean(cos, mask)

def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(max(1, union))

class CheckpointIO:
    @staticmethod
    def save(path: str | Path, payload: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
        return torch.load(Path(path), map_location=map_location)


# ===== Root / Stage Helpers =====

import copy
import re
import shutil
from pathlib import Path

def root_workspace(cfg_root: Dict[str, Any]) -> Path:
    return Path(cfg_root["project"]["workspace_root"]).resolve()

def root_artifact_root(cfg_root: Dict[str, Any]) -> Path:
    path = root_workspace(cfg_root) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path

def root_results_json_path(cfg_root: Dict[str, Any]) -> Path:
    return root_artifact_root(cfg_root) / "seq_only_results_ver15.json"

def shared_download_root(cfg_root: Dict[str, Any]) -> Path:
    path = Path(cfg_root["project"]["download_root"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def _domain_display_name(domain: str) -> str:
    domain = str(domain).lower().strip()
    mapping = {
        "cb": "creative birds",
        "cc": "creative creatures",
    }
    if domain not in mapping:
        raise ValueError(f"Unknown CreativeSketch domain: {domain}")
    return mapping[domain]

def _scaled_cap(base_cap: int, fraction: float) -> int:
    base_cap = int(base_cap)
    fraction = float(fraction)
    return max(1, int(round(base_cap * max(0.0, min(1.0, fraction)))))

def build_stage_cfg(cfg_root: Dict[str, Any], stage_name: str, stage_kind: str, domains: Sequence[str] | None = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg_root)
    defaults = dict(out["dataset_defaults"])
    base_dir = root_workspace(cfg_root) / stage_name

    if stage_kind == "quickdraw":
        frac = float(out["target"]["dataset_fraction"])
        rep_cap = _scaled_cap(int(out["target"]["representation_max_drawings_per_class"]), frac)
        gen_cap = _scaled_cap(int(out["target"]["generator_max_drawings_per_class"]), frac)
        out["dataset"] = {
            "base_dir": str(base_dir),
            "dataset_kind": "quickdraw",
            "variant": str(out["target"]["variant"]),
            "raw_url_template": str(out["target"]["raw_url_template"]),
            "simplified_url_template": str(out["target"]["simplified_url_template"]),
            "classes": list(out["target"]["classes"]),
            "representation_max_drawings_per_class": int(rep_cap),
            "generator_max_drawings_per_class": int(gen_cap),
            "train_ratio": float(out["target"]["train_ratio"]),
            "val_ratio": float(out["target"]["val_ratio"]),
            "test_ratio": float(out["target"]["test_ratio"]),
            "filter_recognized": bool(out["target"]["filter_recognized"]),
            "quickdraw_canonical_stroke_order": str(out["dataset_defaults"].get("quickdraw_canonical_stroke_order", "none")),
            **defaults,
        }
    elif stage_kind == "creativesketch":
        domains = [str(x).lower().strip() for x in (domains or [])]
        if not domains:
            raise ValueError(f"{stage_name}: domains must be non-empty for CreativeSketch stage")
        use_domain_as_class = bool(out["source"]["use_domain_as_class_label"])
        classes = [_domain_display_name(d) for d in domains] if use_domain_as_class else ["source"]
        frac = 1.0
        if "shape_ae" in stage_name:
            frac = float(out["source"]["shape_ae_fraction"])
        elif "tokenizer" in stage_name or "location" in stage_name:
            frac = float(out["source"]["tokenizer_fraction"])
        out["dataset"] = {
            "base_dir": str(base_dir),
            "dataset_kind": "creativesketch",
            "variant": "simplified",
            "classes": list(classes),
            "source_domains": list(domains),
            "source_fraction": float(frac),
            "use_domain_as_class_label": bool(use_domain_as_class),
            "representation_max_drawings_per_class": 10**9,
            "generator_max_drawings_per_class": 10**9,
            "train_ratio": float(out["source"]["train_ratio"]),
            "val_ratio": float(out["source"]["val_ratio"]),
            "test_ratio": float(out["source"]["test_ratio"]),
            "filter_recognized": False,
            **defaults,
        }
    else:
        raise ValueError(f"Unknown stage_kind: {stage_kind}")

    out["project"]["stage_name"] = str(stage_name)
    out["project"]["config_signature"] = str(cfg_root["project"]["config_signature"])
    return out

def describe_stage_cfg(cfg_stage: Dict[str, Any]) -> None:
    print(
        f"[stage] {cfg_stage['project'].get('stage_name', '?')} | "
        f"kind={cfg_stage['dataset']['dataset_kind']} | "
        f"classes={cfg_stage['dataset']['classes']} | "
        f"max_strokes={cfg_stage['dataset']['max_strokes']}"
    )

def copy_embedding_stats_between_cfgs(src_cfg: Dict[str, Any], dst_cfg: Dict[str, Any]) -> Path:
    ensure_workspace(dst_cfg)
    src = embedding_stats_path(src_cfg)
    dst = embedding_stats_path(dst_cfg)
    if not src.exists():
        raise FileNotFoundError(f"Source embedding stats not found: {src}")
    shutil.copy2(src, dst)
    _STATS_CACHE.pop(str(dst), None)
    return dst

# ===== Render / Preprocess / QuickDraw =====

import json
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
PIL_NEAREST = _PIL_RESAMPLING.NEAREST
PIL_BILINEAR = _PIL_RESAMPLING.BILINEAR
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

@dataclass
class StrokeRenderResult:
    image: np.ndarray         # uint8, HxW
    dist_map: np.ndarray      # float32, HxW in [0, 1]
    bbox: np.ndarray          # [w, h, cx, cy], normalized to [0, 1]



def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())

def stroke_span(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(max(maxs[0] - mins[0], maxs[1] - mins[1]))

def endpoint_gap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1e9
    return float(np.linalg.norm(a[-1] - b[0]))

def merge_strokes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0:
        return b.copy()
    if b.shape[0] == 0:
        return a.copy()
    if np.allclose(a[-1], b[0]):
        return np.concatenate([a, b[1:]], axis=0)
    return np.concatenate([a, b], axis=0)



def _point_list_to_points(stroke):
    if stroke is None:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(stroke, np.ndarray):
        arr = np.asarray(stroke, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float32, copy=False)

    if not isinstance(stroke, (list, tuple)):
        return np.zeros((0, 2), dtype=np.float32)

    if len(stroke) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for p in stroke:
        if not isinstance(p, (list, tuple, np.ndarray)):
            return np.zeros((0, 2), dtype=np.float32)
        if len(p) < 2:
            return np.zeros((0, 2), dtype=np.float32)
        if isinstance(p[0], (list, tuple, np.ndarray)) or isinstance(p[1], (list, tuple, np.ndarray)):
            return np.zeros((0, 2), dtype=np.float32)
        pts.append([float(p[0]), float(p[1])])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)


def _quickdraw_xy_to_points(stroke):
    if stroke is None:
        return np.zeros((0, 2), dtype=np.float32)

    if not isinstance(stroke, (list, tuple, np.ndarray)):
        return np.zeros((0, 2), dtype=np.float32)

    if len(stroke) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    xs_raw = stroke[0]
    ys_raw = stroke[1]

    if not isinstance(xs_raw, (list, tuple, np.ndarray)):
        return np.zeros((0, 2), dtype=np.float32)
    if not isinstance(ys_raw, (list, tuple, np.ndarray)):
        return np.zeros((0, 2), dtype=np.float32)

    xs = np.asarray(xs_raw, dtype=np.float32).reshape(-1)
    ys = np.asarray(ys_raw, dtype=np.float32).reshape(-1)

    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    n = min(xs.size, ys.size)
    return np.stack([xs[:n], ys[:n]], axis=1).astype(np.float32, copy=False)


def _to_points(stroke, stroke_format: str = "auto"):
    """
    stroke_format:
      - "auto"         : 일반 point-list 우선, 실패 시 quickdraw xy fallback
      - "point_list"   : [[x,y], [x,y], ...]
      - "quickdraw_xy" : [xs, ys]
    """
    if stroke_format == "point_list":
        return _point_list_to_points(stroke)

    if stroke_format == "quickdraw_xy":
        return _quickdraw_xy_to_points(stroke)

    # Auto-detect format from the input structure.
    pts = _point_list_to_points(stroke)
    if pts.shape[0] > 0:
        return pts

    return _quickdraw_xy_to_points(stroke)


def normalize_raw_drawing(
    drawing: Sequence[Sequence[Sequence[float]]],
    out_canvas_size: int = 256,
    stroke_format: str = "auto",
) -> List[List[List[float]]]:
    point_list = []
    converted = []

    for stroke in drawing:
        pts = _to_points(stroke, stroke_format=stroke_format)
        if pts.shape[0] == 0:
            continue
        point_list.append(pts)
        converted.append(stroke)

    if not point_list:
        return []

    all_points = np.concatenate(point_list, axis=0)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)
    scale = float(out_canvas_size - 1) / float(max(size[0], size[1], 1.0))

    norm = []
    for stroke in converted:
        pts = _to_points(stroke, stroke_format=stroke_format)
        pts = (pts - min_xy[None, :]) * scale
        norm.append([pts[:, 0].tolist(), pts[:, 1].tolist()])

    return norm


def drawing_to_point_strokes(
    drawing: Sequence[Sequence[Sequence[float]]],
    max_strokes: int | None = None,
    stroke_format: str = "auto",
) -> List[np.ndarray]:
    strokes: List[np.ndarray] = []
    for stroke in drawing:
        pts = _to_points(stroke, stroke_format=stroke_format)
        if pts.shape[0] >= 2:
            strokes.append(pts.astype(np.float32))
    if max_strokes is not None:
        strokes = strokes[:max_strokes]
    return strokes


def clean_point_strokes(point_strokes: Sequence[np.ndarray], cfg: Dict) -> List[np.ndarray]:
    min_points = int(cfg["dataset"]["min_stroke_points"])
    min_len = float(cfg["dataset"]["min_stroke_length"])
    min_span = float(cfg["dataset"]["min_stroke_span"])
    min_bbox_size = float(cfg["dataset"].get("min_stroke_bbox_size", min_span))
    drop_long_factor = float(cfg["dataset"]["drop_long_stroke_factor"])
    merge_enable = bool(cfg["dataset"]["merge_short_strokes"])
    merge_short_length = float(cfg["dataset"]["merge_short_length"])
    merge_endpoint_gap = float(cfg["dataset"]["merge_endpoint_gap"])

    filtered: List[np.ndarray] = []
    all_points = [s for s in point_strokes if s.shape[0] >= min_points]
    if not all_points:
        return []

    concat = np.concatenate(all_points, axis=0)
    mins = concat.min(axis=0)
    maxs = concat.max(axis=0)
    sketch_diag = float(np.linalg.norm(maxs - mins))
    max_allowed_length = max(32.0, drop_long_factor * max(sketch_diag, 1.0))

    for pts in point_strokes:
        if pts.shape[0] < min_points:
            continue
        length = polyline_length(pts)
        span = stroke_span(pts)
        bbox_span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        if length < min_len or span < min_span or bbox_span < min_bbox_size:
            continue
        if length > max_allowed_length:
            continue
        filtered.append(pts.astype(np.float32))

    if not merge_enable or not filtered:
        return filtered

    merged: List[np.ndarray] = []
    cursor = 0
    while cursor < len(filtered):
        cur = filtered[cursor]
        cur_len = polyline_length(cur)
        if cursor + 1 < len(filtered):
            nxt = filtered[cursor + 1]
            nxt_len = polyline_length(nxt)
            if cur_len <= merge_short_length and nxt_len <= merge_short_length and endpoint_gap(cur, nxt) <= merge_endpoint_gap:
                merged.append(merge_strokes(cur, nxt))
                cursor += 2
                continue
        merged.append(cur)
        cursor += 1
    return merged



def reorder_point_strokes_top_to_bottom(point_strokes: Sequence[np.ndarray]) -> List[np.ndarray]:
    ordered: List[np.ndarray] = []
    for pts in point_strokes:
        arr = np.asarray(pts, dtype=np.float32)
        if arr.shape[0] >= 2:
            ordered.append(arr.copy())

    ordered.sort(
        key=lambda pts: (
            float(pts[:, 1].mean()),
            float(pts[:, 0].mean()),
        )
    )
    return ordered


def reorder_point_strokes_bbox_area_desc(point_strokes: Sequence[np.ndarray]) -> List[np.ndarray]:
    decorated: List[tuple[tuple[float, float, float], np.ndarray]] = []
    for pts in point_strokes:
        arr = np.asarray(pts, dtype=np.float32)
        if arr.shape[0] < 2:
            continue

        xs = arr[:, 0]
        ys = arr[:, 1]
        width = float(max(xs.max() - xs.min(), 0.0))
        height = float(max(ys.max() - ys.min(), 0.0))
        bbox_area = width * height

        decorated.append((
            (
                -bbox_area,              # sort larger bounding boxes first
                float(ys.mean()),        # tie-break: top to bottom
                float(xs.mean()),        # tie-break: left to right
            ),
            arr.copy(),
        ))

    decorated.sort(key=lambda item: item[0])
    return [arr for _, arr in decorated]


def _transform_points(points: np.ndarray, center: np.ndarray, angle_rad: float, scale: float, translate_xy: np.ndarray) -> np.ndarray:
    rot = np.array(
        [[math.cos(angle_rad), -math.sin(angle_rad)],
         [math.sin(angle_rad),  math.cos(angle_rad)]],
        dtype=np.float32,
    )
    pts = (points - center[None, :]) * scale
    pts = pts @ rot.T
    pts = pts + center[None, :] + translate_xy[None, :]
    return pts

def augment_point_strokes(point_strokes: Sequence[np.ndarray], cfg: Dict) -> List[np.ndarray]:
    if not point_strokes:
        return []
    if not bool(cfg["dataset"]["train_aug_enable"]):
        return [s.copy() for s in point_strokes]

    all_points = np.concatenate(point_strokes, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    size = np.maximum(maxs - mins, 1.0)
    diag = float(np.linalg.norm(size))

    rot_deg = float(cfg["dataset"]["train_aug_rot_deg"])
    scale_amt = float(cfg["dataset"]["train_aug_scale"])
    trans_amt = float(cfg["dataset"]["train_aug_translate"])

    angle = math.radians(random.uniform(-rot_deg, rot_deg))
    scale = 1.0 + random.uniform(-scale_amt, scale_amt)
    translate = np.array(
        [
            random.uniform(-trans_amt, trans_amt) * diag,
            random.uniform(-trans_amt, trans_amt) * diag,
        ],
        dtype=np.float32,
    )

    out: List[np.ndarray] = []
    for pts in point_strokes:
        aug = _transform_points(pts, center=center, angle_rad=angle, scale=scale, translate_xy=translate)
        aug[:, 0] = np.clip(aug[:, 0], 0.0, float(cfg["dataset"]["source_canvas_size"] - 1))
        aug[:, 1] = np.clip(aug[:, 1], 0.0, float(cfg["dataset"]["source_canvas_size"] - 1))
        out.append(aug.astype(np.float32))
    return out

def preprocess_point_strokes(point_strokes: Sequence[np.ndarray], cfg: Dict, split: str, apply_augment: bool = False) -> List[np.ndarray]:
    cleaned = clean_point_strokes(point_strokes, cfg)

    order_mode = str(cfg["dataset"].get("quickdraw_canonical_stroke_order", "none")).strip().lower()
    if order_mode == "top_to_bottom":
        cleaned = reorder_point_strokes_top_to_bottom(cleaned)
    elif order_mode == "bbox_area_desc":
        cleaned = reorder_point_strokes_bbox_area_desc(cleaned)

    if apply_augment and split == "train":
        cleaned = augment_point_strokes(cleaned, cfg)

    max_strokes = int(cfg["dataset"]["max_strokes"])
    return cleaned[:max_strokes]


def render_point_strokes(
    point_strokes: Sequence[np.ndarray],
    canvas_size: int = 64,
    line_width: int = 2,
    upto: int | None = None,
    source_canvas_size: int = 256,
) -> np.ndarray:
    if upto is None:
        upto = len(point_strokes)
    canvas = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(canvas)
    scale = (canvas_size - 1) / float(max(source_canvas_size - 1, 1))
    for pts in point_strokes[:upto]:
        if pts.shape[0] < 2:
            continue
        seq = [(float(x * scale), float(y * scale)) for x, y in pts]
        draw.line(seq, fill=255, width=line_width, joint="curve")
    return np.asarray(canvas, dtype=np.uint8)

def make_soft_distance_map(image: np.ndarray, decay: float = 2.5) -> np.ndarray:
    mask = np.asarray(image, dtype=np.uint8) > 0
    dist = distance_transform_edt(~mask)
    soft = np.exp(-dist / max(decay, 1e-6))
    soft[mask] = 1.0
    return soft.astype(np.float32)



def render_single_stroke_to_normalized_bbox(
    stroke_points: np.ndarray,
    image_size: int = 256,
    source_canvas_size: int = 256,
    line_width: int = 2,
    distance_decay: float = 2.5,
    center_scale_margin: float = 1.0,
    max_canvas_coverage: float = 1.0,
) -> StrokeRenderResult:
    _ = center_scale_margin, max_canvas_coverage  # ver12: paper-style centered render without scaling

    if stroke_points.shape[0] < 2:
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        return StrokeRenderResult(
            image=image,
            dist_map=make_soft_distance_map(image, decay=distance_decay),
            bbox=np.array([1.0 / source_canvas_size, 1.0 / source_canvas_size, 0.5, 0.5], dtype=np.float32),
        )

    xs = stroke_points[:, 0].astype(np.float32)
    ys = stroke_points[:, 1].astype(np.float32)

    xmin = float(xs.min())
    ymin = float(ys.min())
    xmax = float(xs.max())
    ymax = float(ys.max())

    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    render_canvas_size = int(source_canvas_size)
    canvas = Image.new("L", (render_canvas_size, render_canvas_size), color=0)
    draw = ImageDraw.Draw(canvas)

    centered = stroke_points.astype(np.float32).copy()
    centered[:, 0] -= cx
    centered[:, 1] -= cy

    center_x = (render_canvas_size - 1) / 2.0
    center_y = (render_canvas_size - 1) / 2.0
    translated = [(float(x + center_x), float(y + center_y)) for x, y in centered]

    draw.line(translated, fill=255, width=int(line_width), joint="curve")
    image = np.asarray(canvas, dtype=np.uint8)
    image = (image > 0).astype(np.uint8) * 255

    if int(image_size) != render_canvas_size:
        pil = Image.fromarray(image).resize((int(image_size), int(image_size)), resample=PIL_BILINEAR)
        image = np.asarray(pil, dtype=np.uint8)
        image = (image >= 127).astype(np.uint8) * 255

    dist_map = make_soft_distance_map(image, decay=distance_decay)

    bbox = np.array([
        w / max(source_canvas_size - 1, 1),
        h / max(source_canvas_size - 1, 1),
        cx / max(source_canvas_size - 1, 1),
        cy / max(source_canvas_size - 1, 1),
    ], dtype=np.float32)
    bbox = np.clip(bbox, 0.0, 1.0)
    return StrokeRenderResult(image=image, dist_map=dist_map, bbox=bbox)

def _tight_foreground_crop(arr: np.ndarray) -> np.ndarray:
    mask = np.asarray(arr, dtype=np.uint8) > 0
    if not np.any(mask):
        return np.zeros((1, 1), dtype=np.uint8)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return np.asarray(arr, dtype=np.uint8)[y0:y1, x0:x1]

def raster_to_canvas(
    shape_img: np.ndarray,
    bbox: np.ndarray,
    canvas_size: int = 128,
    decode_threshold: float = 0.42,
    use_bbox_size: bool = False,
) -> np.ndarray:
    arr = np.asarray(shape_img, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.size == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 255).astype(np.uint8)

    thresh = int(round(float(decode_threshold) * 255.0))
    arr = (arr >= thresh).astype(np.uint8) * 255
    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4:
        bbox = np.pad(bbox, (0, max(0, 4 - bbox.size)), constant_values=0.0)
    bbox = np.nan_to_num(bbox[:4], nan=0.0, posinf=1.0, neginf=0.0)
    bbox = np.clip(bbox, 0.0, 1.0)

    layer = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    if use_bbox_size:
        crop = _tight_foreground_crop(arr)
        target_w = int(round(float(bbox[0]) * float(canvas_size - 1)))
        target_h = int(round(float(bbox[1]) * float(canvas_size - 1)))
        target_w = max(1, min(canvas_size, target_w))
        target_h = max(1, min(canvas_size, target_h))

        if crop.shape[1] != target_w or crop.shape[0] != target_h:
            crop = np.asarray(
                Image.fromarray(crop).resize((target_w, target_h), resample=PIL_NEAREST),
                dtype=np.uint8,
            )
        crop = (crop > 0).astype(np.uint8) * 255

        cx = int(round(float(bbox[2]) * float(canvas_size - 1)))
        cy = int(round(float(bbox[3]) * float(canvas_size - 1)))
        x0 = int(round(cx - (target_w - 1) / 2.0))
        y0 = int(round(cy - (target_h - 1) / 2.0))
        x1 = x0 + target_w
        y1 = y0 + target_h

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        src_x1 = target_w - max(0, x1 - canvas_size)
        src_y1 = target_h - max(0, y1 - canvas_size)

        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
        dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            layer[dst_y0:dst_y1, dst_x0:dst_x1] = crop[src_y0:src_y1, src_x0:src_x1]
        return layer

    if arr.shape[0] != canvas_size or arr.shape[1] != canvas_size:
        pil = Image.fromarray(arr).resize((canvas_size, canvas_size), resample=PIL_BILINEAR)
        arr = np.asarray(pil, dtype=np.uint8)
    arr = (arr >= thresh).astype(np.uint8) * 255

    cx = float(bbox[2]) * float(canvas_size - 1)
    cy = float(bbox[3]) * float(canvas_size - 1)
    center = (canvas_size - 1) / 2.0
    dx = int(round(cx - center))
    dy = int(round(cy - center))

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(canvas_size, canvas_size - dx)
    src_y1 = min(canvas_size, canvas_size - dy)

    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        layer[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return layer

def compose_strokes_from_shape_and_location(
    decoded_shape_images: Sequence[np.ndarray],
    decoded_bboxes: Sequence[np.ndarray],
    canvas_size: int = 128,
    decode_threshold: float = 0.42,
    use_bbox_size: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    frames: List[np.ndarray] = []
    for shape_img, bbox in zip(decoded_shape_images, decoded_bboxes):
        stroke_layer = raster_to_canvas(
            shape_img,
            bbox,
            canvas_size=canvas_size,
            decode_threshold=decode_threshold,
            use_bbox_size=use_bbox_size,
        )
        canvas = np.maximum(canvas, stroke_layer)
        frames.append(canvas.copy())
    return canvas, frames

def to_float_tensor_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

class DownloadError(RuntimeError):
    pass

def _class_url(class_name: str, variant: str, cfg: Dict) -> str:
    encoded = urllib.parse.quote(class_name, safe="")
    if variant == "raw":
        template = cfg["dataset"]["raw_url_template"]
    else:
        template = cfg["dataset"]["simplified_url_template"]
    return template.format(class_name=encoded)

def download_class_file(class_name: str, cfg: Dict, force: bool = False) -> Path:
    ensure_workspace(cfg)
    variant = cfg["dataset"]["variant"]
    out_path = raw_data_root(cfg) / f"{safe_class_name(class_name)}.ndjson"
    if out_path.exists() and not force:
        return out_path

    # Reuse a shared QuickDraw cache before attempting network download.
    cache_candidates: List[Path] = []
    env_cache = os.environ.get("QUICKDRAW_CACHE_DIR", "").strip()
    if env_cache:
        cache_candidates.append(Path(env_cache).expanduser())
    proj_cache = str(cfg.get("project", {}).get("download_root", "")).strip()
    if proj_cache:
        cache_candidates.append(Path(proj_cache).expanduser())
    # Monorepo-level conventional cache location: <repo>/data/quickdraw
    cache_candidates.append(Path.cwd().parent / "data" / "quickdraw")

    cache_name = f"{safe_class_name(class_name)}.ndjson"
    for cache_dir in cache_candidates:
        candidate = cache_dir / cache_name
        if candidate.exists() and not force:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, out_path)
            return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _class_url(class_name, variant=variant, cfg=cfg)
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as exc:
        raise DownloadError(f"Failed to download {class_name!r} from {url}: {exc}") from exc
    return out_path

def reservoir_sample_drawings(path: str | Path, sample_size: int, seed: int, filter_recognized: bool = True) -> List[str]:
    rng = random.Random(seed)
    reservoir: List[str] = []
    total_seen = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Sampling {Path(path).name}", leave=False):
            if not line.strip():
                continue
            if filter_recognized:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not obj.get("recognized", True):
                    continue
                line = json.dumps(obj, ensure_ascii=False)
            total_seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                j = rng.randrange(total_seen)
                if j < sample_size:
                    reservoir[j] = line
    rng.shuffle(reservoir)
    return reservoir

def split_lines(lines: Sequence[str], train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    n = len(lines)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = list(lines[:n_train])
    val = list(lines[n_train:n_train + n_val])
    test = list(lines[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def _iter_json_objects_from_text(text: str) -> Iterator[str]:
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        if text.startswith("\\n", idx):
            idx += 2
            continue
        obj, end = decoder.raw_decode(text, idx)
        yield json.dumps(obj, ensure_ascii=False)
        idx = end

def repair_jsonl_file(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if not text:
        return
    try:
        bad = False
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
        return
    except json.JSONDecodeError:
        bad = True
    if bad:
        repaired = list(_iter_json_objects_from_text(text))
        with path.open("w", encoding="utf-8") as f:
            for rec in repaired:
                f.write(rec.rstrip("\n") + "\n")

def validate_jsonl_file(path: str | Path) -> None:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path} line {i}: {exc}") from exc

def prepare_subset(cfg: Dict, force_download: bool = False) -> Path:
    ensure_workspace(cfg)
    root = subset_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["seed"])
    classes = list(cfg["dataset"]["classes"])
    rep_max = int(cfg["dataset"]["representation_max_drawings_per_class"])
    gen_max = int(cfg["dataset"]["generator_max_drawings_per_class"])
    meta = {
        "classes": classes,
        "class_to_id": {name: idx for idx, name in enumerate(classes)},
        "variant": cfg["dataset"]["variant"],
        "representation_max_drawings_per_class": rep_max,
        "generator_max_drawings_per_class": gen_max,
    }
    write_json(root / "dataset_meta.json", meta)

    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(classes):
        raw_path = download_class_file(class_name, cfg, force=force_download)
        sampled = reservoir_sample_drawings(
            raw_path,
            sample_size=max(rep_max, gen_max),
            seed=seed + class_idx,
            filter_recognized=bool(cfg["dataset"]["filter_recognized"]),
        )
        splits = split_lines(
            sampled,
            train_ratio=float(cfg["dataset"]["train_ratio"]),
            val_ratio=float(cfg["dataset"]["val_ratio"]),
        )
        safe_name = safe_class_name(class_name)
        for split, records in splits.items():
            out_path = root / split / f"{safe_name}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for line in records:
                    f.write(line.rstrip("\n") + "\n")
            repair_jsonl_file(out_path)
            validate_jsonl_file(out_path)
            print(f"[prepare] {class_name:>12s} -> {split:<5s}: {len(records):5d} drawings")
    return root

# ===== Dataset Indexing =====

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SketchMeta:
    file_idx: int
    offset: int
    class_id: int
    n_strokes: int


def _iter_json_objects_from_file_bytes(path: str | Path) -> Iterator[tuple[int, str]]:
    path = Path(path)
    decoder = json.JSONDecoder()
    data = path.read_text(encoding="utf-8")
    idx = 0
    n = len(data)
    while idx < n:
        while idx < n and data[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        if data.startswith("\\n", idx):
            idx += 2
            continue
        start = idx
        obj, end = decoder.raw_decode(data, idx)
        yield start, json.dumps(obj, ensure_ascii=False)
        idx = end

class JsonlSketchIndex:
    def __init__(self, cfg: Dict, split: str) -> None:
        self.cfg = cfg
        self.split = split
        self.subset_dir = subset_root(cfg)
        self.index_dir = index_root(cfg)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.subset_dir / "dataset_meta.json"
        meta = read_json(self.meta_path)
        self.classes: List[str] = meta["classes"]
        self.class_to_id: Dict[str, int] = meta["class_to_id"]
        self.file_paths: List[Path] = [
            self.subset_dir / split / f"{safe_class_name(class_name)}.jsonl"
            for class_name in self.classes
        ]
        self.cache_path = self.index_dir / f"{split}_{cfg['dataset']['variant']}_max{cfg['dataset']['max_strokes']}_ver13_jsonl.pkl"
        if self.cache_path.exists():
            self._load()
        else:
            self._build_and_save()

    def _drawing_stroke_format(self) -> str:
        dataset_kind = str(self.cfg["dataset"].get("dataset_kind", "")).lower()
        if dataset_kind == "quickdraw":
            return "quickdraw_xy"
        return "auto"

    def _build_and_save(self) -> None:
        metas: List[SketchMeta] = []

        def _process_line_obj(obj: Dict, offset: int, file_idx: int, class_id: int):
            drawing = obj.get("drawing", [])
            stroke_format = self._drawing_stroke_format()

            if self.cfg["dataset"]["variant"] == "raw":
                drawing = normalize_raw_drawing(
                    drawing,
                    out_canvas_size=self.cfg["dataset"]["source_canvas_size"],
                    stroke_format=stroke_format,
                )

            point_strokes = drawing_to_point_strokes(
                drawing,
                max_strokes=None,
                stroke_format=stroke_format,
            )
            point_strokes = preprocess_point_strokes(
                point_strokes,
                self.cfg,
                split=self.split,
                apply_augment=False,
            )

            n_strokes = len(point_strokes)
            if n_strokes < int(self.cfg["dataset"]["min_strokes"]):
                return

            metas.append(
                SketchMeta(
                    file_idx=file_idx,
                    offset=offset,
                    class_id=class_id,
                    n_strokes=n_strokes,
                )
            )

        for file_idx, path in enumerate(self.file_paths):
            class_name = denormalize_class_name(path.stem)
            class_id = self.class_to_id[class_name]
            if not path.exists():
                continue

            try:
                with path.open("rb") as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        _process_line_obj(obj, offset, file_idx, class_id)

            except json.JSONDecodeError:
                repair_jsonl_file(path)
                with path.open("rb") as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        _process_line_obj(obj, offset, file_idx, class_id)

        payload = {
            "file_paths": [str(p) for p in self.file_paths],
            "classes": self.classes,
            "metas": [(m.file_idx, m.offset, m.class_id, m.n_strokes) for m in metas],
        }
        with self.cache_path.open("wb") as f:
            pickle.dump(payload, f)
        self._set_payload(payload)

    def _load(self) -> None:
        with self.cache_path.open("rb") as f:
            payload = pickle.load(f)
        self._set_payload(payload)

    def _set_payload(self, payload: Dict) -> None:
        self.file_paths = [Path(p) for p in payload["file_paths"]]
        self.classes = list(payload["classes"])
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        raw_metas = payload["metas"]
        self.metas = [SketchMeta(*item) for item in raw_metas]
        self.stroke_to_sketch, self.stroke_local_index = self._build_stroke_index(self.metas)

    @staticmethod
    def _build_stroke_index(metas: Sequence[SketchMeta]) -> tuple[np.ndarray, np.ndarray]:
        total_strokes = int(sum(m.n_strokes for m in metas))
        stroke_to_sketch = np.zeros((total_strokes,), dtype=np.int32)
        stroke_local_index = np.zeros((total_strokes,), dtype=np.int16)
        cursor = 0
        for sketch_idx, meta in enumerate(metas):
            n = meta.n_strokes
            stroke_to_sketch[cursor:cursor+n] = sketch_idx
            stroke_local_index[cursor:cursor+n] = np.arange(n, dtype=np.int16)
            cursor += n
        return stroke_to_sketch, stroke_local_index

    def __len__(self) -> int:
        return len(self.metas)

    def num_strokes(self) -> int:
        return int(self.stroke_to_sketch.shape[0])

    @lru_cache(maxsize=8)
    def _reader(self, file_path: str) -> object:
        return open(file_path, "rb")

    def _read_line(self, meta: SketchMeta) -> Dict:
        f = self._reader(str(self.file_paths[meta.file_idx]))
        f.seek(meta.offset)
        line = f.readline().decode("utf-8").strip()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            repair_jsonl_file(self.file_paths[meta.file_idx])
            self._reader.cache_clear()
            f = self._reader(str(self.file_paths[meta.file_idx]))
            f.seek(meta.offset)
            line = f.readline().decode("utf-8").strip()
            return json.loads(line)

    def _get_clean_drawing(self, sketch_idx: int) -> Dict:
        meta = self.metas[sketch_idx]
        obj = self._read_line(meta)

        stroke_format = self._drawing_stroke_format()

        drawing = obj.get("drawing", [])
        if self.cfg["dataset"]["variant"] == "raw":
            drawing = normalize_raw_drawing(
                drawing,
                out_canvas_size=self.cfg["dataset"]["source_canvas_size"],
                stroke_format=stroke_format,
            )

        point_strokes = drawing_to_point_strokes(
            drawing,
            max_strokes=None,
            stroke_format=stroke_format,
        )
        point_strokes = preprocess_point_strokes(
            point_strokes,
            self.cfg,
            split=self.split,
            apply_augment=False,
        )

        return {
            "class_id": meta.class_id,
            "class_name": self.classes[meta.class_id],
            "point_strokes": point_strokes,
            "n_strokes": len(point_strokes),
        }

    def get_drawing(self, sketch_idx: int, apply_augment: bool = False) -> Dict:
        base = self._get_clean_drawing(sketch_idx)
        point_strokes = [s.copy() for s in base["point_strokes"]]
        if apply_augment:
            point_strokes = preprocess_point_strokes(
                point_strokes,
                self.cfg,
                split=self.split,
                apply_augment=True,
            )
        return {
            "class_id": int(base["class_id"]),
            "class_name": base["class_name"],
            "point_strokes": point_strokes,
            "n_strokes": len(point_strokes),
        }
