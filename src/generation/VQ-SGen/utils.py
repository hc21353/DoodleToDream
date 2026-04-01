from __future__ import annotations



import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


