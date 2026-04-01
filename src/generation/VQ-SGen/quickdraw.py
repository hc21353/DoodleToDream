from __future__ import annotations



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
    image: np.ndarray
    dist_map: np.ndarray
    bbox: np.ndarray



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
    if stroke_format == "point_list":
        return _point_list_to_points(stroke)

    if stroke_format == "quickdraw_xy":
        return _quickdraw_xy_to_points(stroke)


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
                -bbox_area,
                float(ys.mean()),
                float(xs.mean()),
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
    _ = center_scale_margin, max_canvas_coverage

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


