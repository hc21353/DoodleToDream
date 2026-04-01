from __future__ import annotations



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

class QuickDrawStrokeDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, mode: str = "shape_ae") -> None:
        if mode not in {"shape_ae", "location_ae"}:
            raise ValueError(f"Unexpected mode: {mode}")
        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.index = JsonlSketchIndex(cfg, split)
        self.image_size = int(cfg["dataset"]["image_size"])
        self.canvas_size = int(cfg["dataset"]["canvas_size"])
        self.source_canvas_size = int(cfg["dataset"]["source_canvas_size"])
        self.line_width = int(cfg["dataset"]["shape_stroke_width"])
        self.center_scale_margin = float(cfg["dataset"]["shape_center_scale_margin"])
        self.max_canvas_coverage = float(cfg["dataset"]["shape_max_canvas_coverage"])
        self.distance_decay = float(cfg["dataset"]["distance_decay"])

    def __len__(self) -> int:
        return self.index.num_strokes()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        sketch_idx = int(self.index.stroke_to_sketch[idx])
        stroke_idx = int(self.index.stroke_local_index[idx])
        apply_aug = (self.split == "train")
        record = self.index.get_drawing(sketch_idx, apply_augment=apply_aug)
        if stroke_idx >= len(record["point_strokes"]):
            record = self.index.get_drawing(sketch_idx, apply_augment=False)
        stroke_points = record["point_strokes"][stroke_idx]
        rendered = render_single_stroke_to_normalized_bbox(
            stroke_points,
            image_size=self.image_size,
            source_canvas_size=self.source_canvas_size,
            line_width=self.line_width,
            distance_decay=self.distance_decay,
            center_scale_margin=self.center_scale_margin,
            max_canvas_coverage=self.max_canvas_coverage,
        )
        result = {
            "stroke_index": idx,
            "sketch_index": sketch_idx,
            "class_id": int(record["class_id"]),
            "bbox": torch.from_numpy(rendered.bbox.astype(np.float32)),
        }
        if self.mode == "shape_ae":
            result["image"] = torch.from_numpy(to_float_tensor_image(rendered.image)).unsqueeze(0)
            result["dist_map"] = torch.from_numpy(rendered.dist_map).unsqueeze(0)
        else:
            result["vector"] = torch.from_numpy(rendered.bbox.astype(np.float32))
        return result

class EmbeddingSequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path, feature_key: str, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        npz = np.load(npz_path)
        self.class_ids = npz["class_ids"].astype(np.int64)
        self.lengths = npz["lengths"].astype(np.int64)
        self.valid_mask = npz["valid_mask"].astype(np.float32)
        self.features = npz[feature_key].astype(np.float32)
        self.raw_bboxes = npz["raw_bboxes"].astype(np.float32) if "raw_bboxes" in npz else None
        self.raw_shape_images = npz["raw_shape_images"].astype(np.float32) if "raw_shape_images" in npz else None
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.class_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features = self.features[idx]
        if self.mean is not None and self.std is not None:
            safe_std = np.maximum(self.std, 1e-4)
            features = (features - self.mean[None, :]) / safe_std[None, :]
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -8.0, 8.0)

        out = {
            "class_id": torch.tensor(int(self.class_ids[idx]), dtype=torch.long),
            "features": torch.from_numpy(features.astype(np.float32)),
            "valid_mask": torch.from_numpy(self.valid_mask[idx]),
            "length": torch.tensor(int(self.lengths[idx]), dtype=torch.long),
        }

        if self.raw_bboxes is not None:
            out["raw_bboxes"] = torch.from_numpy(self.raw_bboxes[idx].astype(np.float32))

        if self.raw_shape_images is not None:

            out["raw_shape_images"] = torch.from_numpy(self.raw_shape_images[idx].astype(np.float32)).unsqueeze(1)

        return out


class TokenSequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path, shape_vocab_size: int, loc_vocab_size: int) -> None:
        npz = np.load(npz_path)
        self.class_ids = npz["class_ids"].astype(np.int64)
        self.lengths = npz["lengths"].astype(np.int64)
        self.shape_tokens = npz["shape_tokens"].astype(np.int64)
        self.loc_tokens = npz["loc_tokens"].astype(np.int64)
        self.shape_vocab_size = int(shape_vocab_size)
        self.loc_vocab_size = int(loc_vocab_size)






        self.shape_end = self.shape_vocab_size
        self.shape_start = self.shape_vocab_size + 1
        self.shape_pad = self.shape_vocab_size + 2

        self.loc_end = self.loc_vocab_size
        self.loc_start = self.loc_vocab_size + 1
        self.loc_pad = self.loc_vocab_size + 2

    def __len__(self) -> int:
        return int(self.class_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shape = self.shape_tokens[idx]
        loc = self.loc_tokens[idx]
        length = int(self.lengths[idx])
        seq_len = shape.shape[0] + 1

        target_shape = np.full((seq_len,), fill_value=self.shape_pad, dtype=np.int64)
        target_loc = np.full((seq_len,), fill_value=self.loc_pad, dtype=np.int64)
        if length > 0:
            target_shape[:length] = shape[:length]
            target_loc[:length] = loc[:length]
        target_shape[length] = self.shape_end
        target_loc[length] = self.loc_end

        input_shape = np.full((seq_len,), fill_value=self.shape_pad, dtype=np.int64)
        input_loc = np.full((seq_len,), fill_value=self.loc_pad, dtype=np.int64)
        input_shape[0] = self.shape_start
        input_loc[0] = self.loc_start
        input_shape[1:] = target_shape[:-1]
        input_loc[1:] = target_loc[:-1]

        valid_mask = np.zeros((seq_len,), dtype=np.float32)
        valid_mask[:length + 1] = 1.0
        return {
            "class_id": torch.tensor(int(self.class_ids[idx]), dtype=torch.long),
            "input_shape": torch.tensor(input_shape, dtype=torch.long),
            "input_loc": torch.tensor(input_loc, dtype=torch.long),
            "target_shape": torch.tensor(target_shape, dtype=torch.long),
            "target_loc": torch.tensor(target_loc, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.long),
        }

