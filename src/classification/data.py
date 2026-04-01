from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms as tv_transforms
except Exception:
    tv_transforms = None

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
DEFAULT_CLASSES = [
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


@dataclass
class ClassificationDataConfig:
    classes: list[str]
    samples_per_class: int
    train_ratio: float
    val_ratio: float
    image_size: int
    batch_size: int
    num_workers: int
    data_dir: str
    use_partial_strokes_train: bool
    partial_stroke_ratio_min: float
    partial_stroke_ratio_max: float
    dataset_mode: str = "quickdraw"


def convert_raw_to_simplified(raw_strokes: list[list[list[int]]]) -> tuple[list[list[int]], list[list[int]]]:
    x_list = [stroke[0] for stroke in raw_strokes]
    y_list = [stroke[1] for stroke in raw_strokes]
    return x_list, y_list


def render_strokes_to_image(
    x_list: list[list[int]],
    y_list: list[list[int]],
    num_strokes_to_use: int | None = None,
    img_size: int = 256,
) -> Image.Image:
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)

    if num_strokes_to_use is None:
        num_strokes_to_use = len(x_list)

    all_x, all_y = [], []
    for i in range(num_strokes_to_use):
        all_x.extend(x_list[i])
        all_y.extend(y_list[i])

    if len(all_x) == 0:
        return img

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)
    padding = 20
    scale = (img_size - 2 * padding) / max(range_x, range_y)

    for i in range(num_strokes_to_use):
        x_coords, y_coords = x_list[i], y_list[i]
        for j in range(len(x_coords) - 1):
            x1 = int((x_coords[j] - min_x) * scale + padding)
            y1 = int((y_coords[j] - min_y) * scale + padding)
            x2 = int((x_coords[j + 1] - min_x) * scale + padding)
            y2 = int((y_coords[j + 1] - min_y) * scale + padding)
            draw.line([(x1, y1), (x2, y2)], fill=0, width=2)

    return img


class QuickDrawDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        image_size: int,
        transform=None,
        use_partial_strokes: bool = False,
        partial_min: float = 0.5,
        partial_max: float = 1.0,
    ) -> None:
        self.data = data
        self.image_size = image_size
        self.transform = transform
        self.use_partial_strokes = use_partial_strokes
        self.partial_min = partial_min
        self.partial_max = partial_max

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        x_strokes, y_strokes = convert_raw_to_simplified(sample["strokes"])

        if self.use_partial_strokes and len(x_strokes) > 0:
            ratio = random.uniform(self.partial_min, self.partial_max)
            num_strokes = max(1, int(len(x_strokes) * ratio))
        else:
            num_strokes = None

        img = render_strokes_to_image(x_strokes, y_strokes, num_strokes, self.image_size).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(sample["label"])


def download_quickdraw_data(classes: list[str], data_dir: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        path = Path(data_dir) / f"{class_name.replace(' ', '_')}.ndjson"
        if path.exists():
            continue
        url = f"{BASE_URL}/{class_name.replace(' ', '%20')}.ndjson"
        urlretrieve(url, str(path))


def _make_synthetic_drawings(classes: list[str], samples_per_class: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rng = random.Random(42)
    for label, cls in enumerate(classes):
        for _ in range(samples_per_class):
            n_strokes = rng.randint(4, 10)
            drawing = []
            x, y = 128, 128
            for _ in range(n_strokes):
                pts = rng.randint(3, 10)
                xs, ys = [], []
                for _ in range(pts):
                    x = max(0, min(255, x + rng.randint(-30, 30)))
                    y = max(0, min(255, y + rng.randint(-30, 30)))
                    xs.append(x)
                    ys.append(y)
                drawing.append([xs, ys])
            out.append({"strokes": drawing, "label": label, "class_name": cls})
    rng.shuffle(out)
    return out


def load_and_split_data(cfg: ClassificationDataConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if cfg.dataset_mode == "synthetic":
        all_data = _make_synthetic_drawings(cfg.classes, cfg.samples_per_class)
    else:
        download_quickdraw_data(cfg.classes, cfg.data_dir)
        all_data: list[dict[str, Any]] = []
        for label_idx, class_name in enumerate(cfg.classes):
            path = Path(cfg.data_dir) / f"{class_name.replace(' ', '_')}.ndjson"
            class_samples = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    if sample.get("recognized", False):
                        class_samples.append({"strokes": sample["drawing"], "label": label_idx, "class_name": class_name})
                    if len(class_samples) >= cfg.samples_per_class:
                        break
            all_data.extend(class_samples)
        random.shuffle(all_data)

    total = len(all_data)
    train_end = int(total * cfg.train_ratio)
    val_end = train_end + int(total * cfg.val_ratio)
    return all_data[:train_end], all_data[train_end:val_end], all_data[val_end:]


def create_dataloaders(cfg: ClassificationDataConfig):
    train_data, val_data, test_data = load_and_split_data(cfg)
    if tv_transforms is not None:
        train_transform = tv_transforms.Compose([
            tv_transforms.RandomRotation(15),
            tv_transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_transform = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        def _to_tensor(img):
            arr = np.array(img).astype("float32") / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array(
                [0.229, 0.224, 0.225], dtype="float32"
            )
            return torch.from_numpy(arr.transpose(2, 0, 1))

        train_transform = _to_tensor
        eval_transform = _to_tensor

    train_ds = QuickDrawDataset(
        train_data,
        image_size=cfg.image_size,
        transform=train_transform,
        use_partial_strokes=cfg.use_partial_strokes_train,
        partial_min=cfg.partial_stroke_ratio_min,
        partial_max=cfg.partial_stroke_ratio_max,
    )
    val_ds = QuickDrawDataset(val_data, image_size=cfg.image_size, transform=eval_transform)
    test_ds = QuickDrawDataset(test_data, image_size=cfg.image_size, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader
