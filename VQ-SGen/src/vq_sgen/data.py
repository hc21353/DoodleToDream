"""Dataset and sequence dataset implementations."""
from __future__ import annotations

from .base import *  # noqa: F401,F403

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
            # [T, H, W] -> [T, 1, H, W]
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

        # Token layout:
        # normal tokens: [0, vocab-1]
        # END          : vocab
        # START        : vocab+1 (input only)
        # PAD          : vocab+2 (ignored in loss)
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
