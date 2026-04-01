from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import ClassificationDataConfig, DEFAULT_CLASSES, create_dataloaders
from .model import SimpleMobileNet


@dataclass
class ClassificationTrainConfig:
    classes: list[str] = None
    samples_per_class: int = 10000
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    image_size: int = 256
    batch_size: int = 128
    num_workers: int = 2
    data_dir: str = "quickdraw_data"
    use_partial_strokes_train: bool = True
    partial_stroke_ratio_min: float = 0.5
    partial_stroke_ratio_max: float = 1.0
    dropout_rate: float = 0.3
    num_epochs: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    checkpoint_path: str = "scripts/artifacts/classification/best_model.pth"
    history_path: str = "scripts/artifacts/classification/history.json"
    seed: int = 42
    dataset_mode: str = "quickdraw"



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train_classifier(config: dict) -> dict:
    cfg = ClassificationTrainConfig(**config)
    if cfg.classes is None:
        cfg.classes = list(DEFAULT_CLASSES)

    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = ClassificationDataConfig(
        classes=cfg.classes,
        samples_per_class=cfg.samples_per_class,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_dir=cfg.data_dir,
        use_partial_strokes_train=cfg.use_partial_strokes_train,
        partial_stroke_ratio_min=cfg.partial_stroke_ratio_min,
        partial_stroke_ratio_max=cfg.partial_stroke_ratio_max,
        dataset_mode=cfg.dataset_mode,
    )
    train_loader, val_loader, test_loader = create_dataloaders(data_cfg)

    model = SimpleMobileNet(num_classes=len(cfg.classes), dropout_rate=cfg.dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    ckpt_path = Path(cfg.checkpoint_path)
    hist_path = Path(cfg.history_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(cfg.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": cfg.classes,
                "image_size": cfg.image_size,
                "dropout_rate": cfg.dropout_rate,
            }, ckpt_path)

        scheduler.step(val_loss)

    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "checkpoint_path": str(ckpt_path),
        "history_path": str(hist_path),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
