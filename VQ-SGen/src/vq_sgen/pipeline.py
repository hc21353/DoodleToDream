"""Generator epoch logic, evaluation, and full pipeline runtime."""
from __future__ import annotations

from .base import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .models import _loader

def run_generator_epoch(
    model,
    loader,
    device,
    cfg,
    optimizer=None,
    mixed_precision=True,
    grad_clip_norm=None,
    teacher_force_ratio: float = 1.0,
):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = GradScaler("cuda", enabled=mixed_precision and device.type == "cuda") if is_train else None
    total_meter, shape_meter, loc_meter = AverageMeter(), AverageMeter(), AverageMeter()

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        class_ids = batch["class_id"].to(device)
        input_shape = batch["input_shape"].to(device)
        input_loc = batch["input_loc"].to(device)
        target_shape = batch["target_shape"].to(device)
        target_loc = batch["target_loc"].to(device)

        if is_train:
            input_shape, input_loc = build_mixed_teacher_inputs(
                model=model,
                class_ids=class_ids,
                input_shape=input_shape,
                input_loc=input_loc,
                target_shape=target_shape,
                target_loc=target_loc,
                teacher_force_ratio=teacher_force_ratio,
            )
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
            hidden = model._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
            shape_logits_full = model.shape_head(hidden)
            pred_shape = shape_logits_full.detach().argmax(dim=-1)

            if is_train:
                loc_condition_shape = build_mixed_loc_condition_shapes(
                    model=model,
                    target_shape=target_shape,
                    pred_shape=pred_shape,
                    teacher_force_ratio=teacher_force_ratio,
                )
            else:
                loc_condition_shape = pred_shape

            loc_logits_full = model._loc_logits_from_hidden(hidden, loc_condition_shape.long())

            position_weights = build_generator_position_weights(
                target_tokens=target_shape,
                pad_idx=model.shape_pad,
                min_weight=float(cfg["generator"].get("early_stroke_loss_min_weight", 1.0)),
                max_weight=float(cfg["generator"].get("early_stroke_loss_max_weight", 1.0)),
                power=float(cfg["generator"].get("early_stroke_loss_power", 1.0)),
            )

            shape_loss = weighted_token_cross_entropy(
                logits=shape_logits_full,
                targets=target_shape,
                pad_idx=model.shape_pad,
                position_weights=position_weights,
            )
            loc_loss = weighted_token_cross_entropy(
                logits=loc_logits_full,
                targets=target_loc,
                pad_idx=model.loc_pad,
                position_weights=position_weights,
            )
            loss = shape_loss + loc_loss

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            maybe_clip_grad(model, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        bsz = int(class_ids.size(0))
        total_meter.update(float(loss.item()), bsz)
        shape_meter.update(float(shape_loss.item()), bsz)
        loc_meter.update(float(loc_loss.item()), bsz)
        iterator.set_description(f"loss={total_meter.avg:.4f} shape={shape_meter.avg:.4f} loc={loc_meter.avg:.4f}")

    return {"loss": total_meter.avg, "shape_loss": shape_meter.avg, "loc_loss": loc_meter.avg}

def evaluate_generator_teacher_forced(model, loader, device, mixed_precision=True):
    model.eval()
    shape_correct = 0
    shape_total = 0
    loc_correct = 0
    loc_total = 0

    with torch.no_grad():
        for batch in loader:
            class_ids = batch["class_id"].to(device)
            input_shape = batch["input_shape"].to(device)
            input_loc = batch["input_loc"].to(device)
            target_shape = batch["target_shape"].to(device)
            target_loc = batch["target_loc"].to(device)

            with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                hidden = model._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
                shape_logits = model.shape_head(hidden)
                shape_pred = shape_logits.argmax(dim=-1)
                loc_pred = model._predict_loc_tokens_from_hidden(hidden, shape_pred.long())

            shape_mask = target_shape != model.shape_pad
            loc_mask = target_loc != model.loc_pad

            shape_correct += int(((shape_pred == target_shape) & shape_mask).sum().item())
            shape_total += int(shape_mask.sum().item())
            loc_correct += int(((loc_pred == target_loc) & loc_mask).sum().item())
            loc_total += int(loc_mask.sum().item())

    return {
        "shape_token_acc": shape_correct / max(1, shape_total),
        "location_token_acc": loc_correct / max(1, loc_total),
    }


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    return img

def save_gif(frames: Sequence[np.ndarray], out_path: str | Path, fps: int = 4) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [_to_rgb(frame.astype(np.uint8)) for frame in frames]
    imageio.mimsave(out_path, rgb_frames, duration=1.0 / max(1, fps))
    return out_path

def save_triptych(prefix_canvas: np.ndarray, pred_canvas: np.ndarray, gt_canvas: np.ndarray, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["prefix", "pred full", "gt full"]
    for ax, img, title in zip(axes, [prefix_canvas, pred_canvas, gt_canvas], titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def evaluate_continuations(cfg, generator, shape_ae, location_ae, shape_tokenizer, location_tokenizer, test_npz_path, device):
    npz = np.load(test_npz_path)
    class_ids = npz["class_ids"].astype(np.int64)
    lengths = npz["lengths"].astype(np.int64)
    shape_tokens = npz["shape_tokens"].astype(np.int64)
    loc_tokens = npz["loc_tokens"].astype(np.int64)

    primer = int(cfg["continuation_eval"]["primer_strokes"])
    metric_samples = int(cfg["continuation_eval"]["metric_samples"])
    preview_samples = int(cfg["continuation_eval"]["preview_samples"])
    temperature = float(cfg["continuation_eval"]["temperature"])
    top_p = float(cfg["continuation_eval"]["top_p"])

    valid_indices = np.where(lengths >= max(2, primer + 1))[0]
    if valid_indices.size == 0:
        return {"exact_match_ratio": 0.0, "mean_completion_iou": 0.0, "mean_full_canvas_iou": 0.0}, []

    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    metric_indices = rng.choice(valid_indices, size=min(metric_samples, valid_indices.size), replace=False)

    exact_matches = []
    completion_ious = []
    full_ious = []
    preview_paths = []

    for rank, idx in enumerate(metric_indices):
        class_id = int(class_ids[idx])
        length = int(lengths[idx])
        gt_shape = shape_tokens[idx, :length]
        gt_loc = loc_tokens[idx, :length]
        prefix_shape = gt_shape[:primer]
        prefix_loc = gt_loc[:primer]

        pred_shape, pred_loc = sample_continuation(
            generator=generator,
            class_id=class_id,
            prefix_shape_tokens=prefix_shape,
            prefix_loc_tokens=prefix_loc,
            device=device,
            max_steps=int(cfg["dataset"]["max_strokes"]),
            temperature=temperature,
            top_p=top_p,
        )

        gt_canvas, gt_frames = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, gt_shape, gt_loc)
        prefix_canvas, _ = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, prefix_shape, prefix_loc)
        pred_canvas, pred_frames = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, pred_shape, pred_loc)

        gt_completion = np.maximum(gt_canvas.astype(np.int16) - prefix_canvas.astype(np.int16), 0).astype(np.uint8)
        pred_completion = np.maximum(pred_canvas.astype(np.int16) - prefix_canvas.astype(np.int16), 0).astype(np.uint8)

        exact_matches.append(float(np.array_equal(pred_shape, gt_shape) and np.array_equal(pred_loc, gt_loc)))
        completion_ious.append(binary_iou(pred_completion, gt_completion))
        full_ious.append(binary_iou(pred_canvas, gt_canvas))

        if rank < preview_samples:
            triptych_path = artifact_root(cfg) / f"continuation_preview_{rank:03d}.png"
            gif_path = artifact_root(cfg) / f"continuation_pred_{rank:03d}.gif"
            save_triptych(prefix_canvas, pred_canvas, gt_canvas, triptych_path)
            save_gif(pred_frames if len(pred_frames) > 0 else [pred_canvas], gif_path)
            preview_paths.append(str(triptych_path))
            preview_paths.append(str(gif_path))

    metrics = {
        "exact_match_ratio": float(np.mean(exact_matches)),
        "mean_completion_iou": float(np.mean(completion_ious)),
        "mean_full_canvas_iou": float(np.mean(full_ious)),
    }
    return metrics, preview_paths

def decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, shape_tokens, loc_tokens):
    device = next(shape_ae.parameters()).device
    stats = load_embedding_stats(cfg)
    shape_arr = np.asarray(shape_tokens, dtype=np.int64)
    loc_arr = np.asarray(loc_tokens, dtype=np.int64)
    max_shape = int(cfg["shape_tokenizer"]["num_embeddings"])
    max_loc = int(cfg["location_tokenizer"]["num_embeddings"])

    valid_len = 0
    for s, l in zip(shape_arr, loc_arr):
        if s < 0 or l < 0 or s >= max_shape or l >= max_loc:
            break
        valid_len += 1

    if valid_len == 0:
        return np.zeros((int(cfg["dataset"]["canvas_size"]), int(cfg["dataset"]["canvas_size"])), dtype=np.uint8), []

    with torch.no_grad():
        shape_idx = torch.from_numpy(shape_arr[:valid_len]).long().unsqueeze(0).to(device)
        loc_idx = torch.from_numpy(loc_arr[:valid_len]).long().unsqueeze(0).to(device)
        valid_mask = torch.ones((1, valid_len), device=device, dtype=torch.float32)

        shape_emb_norm = shape_tokenizer.decode_sequence_indices(shape_idx, valid_mask)["recon_seq"].squeeze(0)
        shape_emb = denormalize_feature_batch(shape_emb_norm, stats["shape_mean"], stats["shape_std"])
        decoded_shapes = shape_ae.decode_from_embedding(shape_emb)["bitmap"].squeeze(1).cpu().numpy()

        loc_emb_norm = location_tokenizer.decode_sequence_indices(loc_idx, valid_mask)["recon_seq"].squeeze(0)
        loc_emb = denormalize_feature_batch(loc_emb_norm, stats["loc_mean"], stats["loc_std"])
        decoded_bboxes = location_ae.decode_from_embedding(loc_emb).cpu().numpy()

    decoded_shapes = np.clip(decoded_shapes * 255.0, 0, 255).astype(np.uint8)
    decoded_bboxes = np.clip(decoded_bboxes, 0.0, 1.0)
    canvas, frames = compose_strokes_from_shape_and_location(
        decoded_shapes,
        decoded_bboxes,
        canvas_size=int(cfg["dataset"]["canvas_size"]),
        decode_threshold=float(cfg["dataset"]["shape_decode_threshold"]),
        use_bbox_size=bool(cfg["dataset"].get("decode_use_bbox_size", False)),
    )
    return canvas, frames

def sample_continuation(generator, class_id, prefix_shape_tokens, prefix_loc_tokens, device, max_steps, temperature=1.0, top_p=0.9):
    generator.eval()
    input_shape = [generator.shape_start] + [int(x) for x in np.asarray(prefix_shape_tokens, dtype=np.int64).tolist()]
    input_loc = [generator.loc_start] + [int(x) for x in np.asarray(prefix_loc_tokens, dtype=np.int64).tolist()]

    with torch.no_grad():
        for _ in range(max_steps):
            class_ids = torch.tensor([class_id], dtype=torch.long, device=device)
            shape_t = torch.tensor([input_shape], dtype=torch.long, device=device)
            loc_t = torch.tensor([input_loc], dtype=torch.long, device=device)

            hidden = generator._encode_prefix(class_ids=class_ids, input_shape=shape_t, input_loc=loc_t)
            shape_logits = generator.shape_head(hidden)
            next_shape = int(nucleus_sample(shape_logits[:, -1, :], temperature=temperature, top_p=top_p).item())
            if next_shape == generator.shape_end:
                break

            shape_cond = torch.tensor([[next_shape]], dtype=torch.long, device=device)
            loc_logits = generator._loc_logits_from_hidden(hidden[:, -1:, :], shape_cond)
            next_loc = int(nucleus_sample(loc_logits[:, -1, :], temperature=temperature, top_p=top_p).item())
            if next_loc == generator.loc_end:
                break

            input_shape.append(next_shape)
            input_loc.append(next_loc)

            if len(input_shape) - 1 >= int(generator.max_strokes):
                break

    shape_seq = np.array(input_shape[1:], dtype=np.int64)
    loc_seq = np.array(input_loc[1:], dtype=np.int64)
    return shape_seq, loc_seq

def sample_class_conditioned_sketches(
    cfg,
    generator,
    shape_ae,
    location_ae,
    shape_tokenizer,
    location_tokenizer,
    class_id: int,
    num_samples: int,
    device,
    max_steps: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    generator.eval()
    num_samples = int(max(1, num_samples))
    max_steps = int(max_steps or cfg["dataset"]["max_strokes"])
    temperature = float(cfg["generator"]["temperature"] if temperature is None else temperature)
    top_p = float(cfg["generator"]["top_p"] if top_p is None else top_p)

    class_ids = torch.full((num_samples,), fill_value=int(class_id), dtype=torch.long, device=device)
    shape_seq, loc_seq, lengths = generator.sample(
        class_ids=class_ids,
        max_steps=max_steps,
        temperature=temperature,
        top_p=top_p,
    )

    shape_np = shape_seq.detach().cpu().numpy()
    loc_np = loc_seq.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)

    samples: List[Dict[str, Any]] = []
    for i in range(num_samples):
        seq_cap = shape_np.shape[1] if shape_np.ndim == 2 else 0
        L = int(np.clip(lengths_np[i], 0, seq_cap))
        shape_tokens_i = shape_np[i, :L] if L > 0 else np.zeros((0,), dtype=np.int64)
        loc_tokens_i = loc_np[i, :L] if L > 0 else np.zeros((0,), dtype=np.int64)
        canvas_i, frames_i = decode_token_sequence(
            cfg,
            shape_ae,
            location_ae,
            shape_tokenizer,
            location_tokenizer,
            shape_tokens_i,
            loc_tokens_i,
        )
        samples.append(
            {
                "canvas": canvas_i,
                "frames": frames_i,
                "shape_tokens": shape_tokens_i,
                "loc_tokens": loc_tokens_i,
                "length": L,
            }
        )
    return samples


# ===== Runtime Helpers / v17 Pipeline =====

from pathlib import Path
from typing import Any, Dict, Optional

def root_results_json_path(cfg_root: Dict[str, Any]) -> Path:
    name = str(cfg_root["project"].get("results_json_name", "results.json"))
    return root_artifact_root(cfg_root) / name

def encode_split_to_token_npz(cfg, split: str, emb_npz_path, shape_tokenizer, location_tokenizer, device):
    token_name = str(cfg["project"].get("token_npz_template", "{split}_tokens.npz")).format(
        split=split,
        ver=int(cfg["project"].get("notebook_ver", 0)),
    )
    out_path = token_root(cfg) / token_name
    if out_path.exists():
        return out_path

    npz = np.load(emb_npz_path)
    shape_embeddings = torch.from_numpy(npz["shape_embeddings"]).float()
    loc_embeddings = torch.from_numpy(npz["loc_embeddings"]).float()
    valid_mask = torch.from_numpy(npz["valid_mask"]).float()
    lengths = npz["lengths"].astype(np.int64)
    class_ids = npz["class_ids"].astype(np.int64)
    source_indices = npz["source_indices"].astype(np.int64) if "source_indices" in npz else np.arange(len(class_ids), dtype=np.int64)

    if shape_embeddings.size(0) == 0:
        empty = np.zeros((0, int(cfg["dataset"]["max_strokes"])), dtype=np.int64)
        np.savez_compressed(
            out_path,
            class_ids=class_ids,
            lengths=lengths,
            shape_tokens=empty,
            loc_tokens=empty.copy(),
            source_indices=source_indices,
        )
        return out_path

    stats = load_embedding_stats(cfg)
    shape_mean = stats["shape_mean"]
    shape_std = stats["shape_std"]
    loc_mean = stats["loc_mean"]
    loc_std = stats["loc_std"]

    shape_tokenizer.eval()
    location_tokenizer.eval()

    batch_size = 128
    all_shape_tokens = []
    all_loc_tokens = []

    with torch.no_grad():
        for start in tqdm(range(0, shape_embeddings.size(0), batch_size), desc=f"Encoding tokens [{split}]"):
            end = min(start + batch_size, shape_embeddings.size(0))
            shape_batch = normalize_feature_batch(shape_embeddings[start:end].to(device), shape_mean, shape_std)
            loc_batch = normalize_feature_batch(loc_embeddings[start:end].to(device), loc_mean, loc_std)
            mask_batch = valid_mask[start:end].to(device)

            shape_out = shape_tokenizer(shape_batch, mask_batch)
            loc_out = location_tokenizer(loc_batch, mask_batch)

            all_shape_tokens.append(shape_out["indices"].detach().cpu().numpy().astype(np.int64))
            all_loc_tokens.append(loc_out["indices"].detach().cpu().numpy().astype(np.int64))

    shape_tokens = np.concatenate(all_shape_tokens, axis=0)
    loc_tokens = np.concatenate(all_loc_tokens, axis=0)

    np.savez_compressed(
        out_path,
        class_ids=class_ids,
        lengths=lengths,
        shape_tokens=shape_tokens,
        loc_tokens=loc_tokens,
        source_indices=source_indices,
    )
    return out_path

def make_generator_token_npz(cfg: Dict, split: str, token_npz_path: str | Path) -> Path:
    rep_total = int(cfg["dataset"]["representation_max_drawings_per_class"])
    gen_total = int(cfg["dataset"]["generator_max_drawings_per_class"])
    if gen_total >= rep_total:
        return Path(token_npz_path)

    out_name = str(cfg["project"].get("generator_token_npz_template", "{split}_tokens_generator.npz")).format(
        split=split,
        ver=int(cfg["project"].get("notebook_ver", 0)),
    )
    out_path = token_root(cfg) / out_name
    if out_path.exists():
        return out_path

    npz = np.load(token_npz_path)
    class_ids = npz["class_ids"].astype(np.int64)
    lengths = npz["lengths"].astype(np.int64)
    shape_tokens = npz["shape_tokens"].astype(np.int64)
    loc_tokens = npz["loc_tokens"].astype(np.int64)
    source_indices = npz["source_indices"].astype(np.int64) if "source_indices" in npz else np.arange(len(class_ids), dtype=np.int64)

    limit = _generator_split_limit(cfg, split)
    seed = int(cfg["project"]["seed"]) + {"train": 11, "val": 23, "test": 37}[split]
    rng = np.random.default_rng(seed)

    keep_indices: list[int] = []
    for class_id in sorted(np.unique(class_ids).tolist()):
        cls_idx = np.where(class_ids == class_id)[0]
        if cls_idx.size <= limit:
            keep_indices.extend(cls_idx.tolist())
        else:
            chosen = rng.choice(cls_idx, size=limit, replace=False)
            keep_indices.extend(chosen.tolist())

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    np.savez_compressed(
        out_path,
        class_ids=class_ids[keep_indices],
        lengths=lengths[keep_indices],
        shape_tokens=shape_tokens[keep_indices],
        loc_tokens=loc_tokens[keep_indices],
        source_indices=source_indices[keep_indices],
    )
    return out_path

def maybe_train_or_load_simple(
    model,
    train_loader,
    val_loader,
    cfg_root,
    cfg_section,
    device,
    kind,
    epoch_fn,
    override_ckpt=None,
    init_ckpt=None,
):
    if override_ckpt is not None:
        print(f"[{kind}] loading pretrained checkpoint (skip training): {override_ckpt}")
        load_checkpoint_weights(model, override_ckpt, device)
        return Path(override_ckpt), []

    if init_ckpt is not None:
        print(f"[{kind}] loading init checkpoint (continue training): {init_ckpt}")
        load_checkpoint_weights(model, init_ckpt, device)

    if train_loader is None or val_loader is None:
        raise ValueError(f"[{kind}] train_loader/val_loader must be provided when training is enabled")

    ckpt_path, history = train_simple_model(
        model,
        train_loader,
        val_loader,
        cfg_root,
        cfg_section,
        device,
        kind,
        epoch_fn,
    )
    load_checkpoint_weights(model, ckpt_path, device)
    return Path(ckpt_path), history

_STAGE_TO_PRETRAINED_KEY = {
    "shape_ae": "shape_ae_ckpt",
    "location_ae": "location_ae_ckpt",
    "shape_tokenizer": "shape_tokenizer_ckpt",
    "location_tokenizer": "location_tokenizer_ckpt",
    "generator": "generator_ckpt",
}

def _resolve_stage_checkpoint_strategy(cfg_root: Dict[str, Any], stage_key: str) -> tuple[Optional[Path], Optional[Path]]:
    stage_cfg = dict(cfg_root["runtime"]["stages"][stage_key])
    pretrained_key = _STAGE_TO_PRETRAINED_KEY[stage_key]
    ckpt_path = None
    if bool(stage_cfg.get("use_pretrained", False)):
        ckpt_path = resolve_optional_ckpt(cfg_root["pretrained"].get(pretrained_key), stage_key)

    if bool(stage_cfg.get("train", False)):
        return None, ckpt_path
    return ckpt_path, None

def _prepare_stage_datasets(cfg_root: Dict[str, Any], force_download: bool = False):
    dataset_cfgs = {
        "target_quickdraw": build_stage_cfg(cfg_root, "target_quickdraw", "quickdraw"),
    }

    print("\n[1/9] Preparing target QuickDraw preset")
    describe_stage_cfg(dataset_cfgs["target_quickdraw"])
    prepare_subset(dataset_cfgs["target_quickdraw"], force_download=force_download)

    return dataset_cfgs

def _clear_stage_token_outputs(cfg_stage: Dict[str, Any]) -> None:
    token_template = str(cfg_stage["project"].get("token_npz_template", "{split}_tokens.npz"))
    gen_token_template = str(cfg_stage["project"].get("generator_token_npz_template", "{split}_tokens_generator.npz"))
    for split in ["train", "val", "test"]:
        for template in [token_template, gen_token_template]:
            stale_path = token_root(cfg_stage) / template.format(
                split=split,
                ver=int(cfg_stage["project"].get("notebook_ver", 0)),
            )
            if stale_path.exists():
                stale_path.unlink()

def run_v18_pipeline(cfg_root: Dict[str, Any], force_download: bool = False):
    root_artifact_root(cfg_root).mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg_root["project"]["seed"]))
    device = get_device(cfg_root["project"].get("device"))

    shape_ae_override, shape_ae_init = _resolve_stage_checkpoint_strategy(cfg_root, "shape_ae")
    location_ae_override, location_ae_init = _resolve_stage_checkpoint_strategy(cfg_root, "location_ae")
    shape_tok_override, shape_tok_init = _resolve_stage_checkpoint_strategy(cfg_root, "shape_tokenizer")
    location_tok_override, location_tok_init = _resolve_stage_checkpoint_strategy(cfg_root, "location_tokenizer")
    generator_override, generator_init = _resolve_stage_checkpoint_strategy(cfg_root, "generator")

    dataset_cfgs = _prepare_stage_datasets(cfg_root, force_download=force_download)
    dataset_map = dict(cfg_root["runtime"]["model_dataset_assignments"])

    trained_this_run = {
        stage: bool(cfg_root["runtime"]["stages"][stage]["train"])
        for stage in cfg_root["runtime"]["stages"]
    }

    results = {
        "root_workspace": str(root_workspace(cfg_root)),
        "root_results_json": str(root_results_json_path(cfg_root)),
        "target_workspace": str(workspace_root(dataset_cfgs["target_quickdraw"])),
        "dataset_workspaces": {k: str(workspace_root(v)) for k, v in dataset_cfgs.items()},
        "config_signature": str(cfg_root["project"].get("config_signature", "")),
        "effective_stage_plan": copy.deepcopy(cfg_root["runtime"]["stages"]),
        "trained_this_run": copy.deepcopy(trained_this_run),
        "pretrained_paths": copy.deepcopy(cfg_root.get("pretrained", {})),
        "model_dataset_assignments": copy.deepcopy(dataset_map),
    }

    prepared_workspaces = set()
    emb_cache: Dict[str, Dict[str, Any]] = {}
    token_cache: Dict[str, Dict[str, Any]] = {}

    def _ensure_stage_workspace(cfg_stage: Dict[str, Any]) -> None:
        stage_name = str(cfg_stage["project"].get("stage_name", ""))
        if stage_name not in prepared_workspaces:
            ensure_workspace(cfg_stage)
            write_config_snapshot(cfg_stage)
            prepared_workspaces.add(stage_name)

    def _ensure_embeddings(preset_name: str) -> Dict[str, Any]:
        if preset_name in emb_cache:
            return emb_cache[preset_name]

        cfg_stage = dataset_cfgs[preset_name]
        _ensure_stage_workspace(cfg_stage)

        train_emb = encode_embedding_sequences(cfg_stage, "train", shape_ae, location_ae, device)
        val_emb = encode_embedding_sequences(cfg_stage, "val", shape_ae, location_ae, device)
        test_emb = encode_embedding_sequences(cfg_stage, "test", shape_ae, location_ae, device)
        stats_npz = compute_embedding_stats(cfg_stage, train_emb)
        stats = load_embedding_stats(cfg_stage)

        payload = {
            "cfg": cfg_stage,
            "train_emb": train_emb,
            "val_emb": val_emb,
            "test_emb": test_emb,
            "stats_npz": stats_npz,
            "stats": stats,
        }
        emb_cache[preset_name] = payload

        results[f"{preset_name}_train_emb_npz"] = str(train_emb)
        results[f"{preset_name}_val_emb_npz"] = str(val_emb)
        results[f"{preset_name}_test_emb_npz"] = str(test_emb)
        results[f"{preset_name}_embedding_stats_npz"] = str(stats_npz)
        return payload

    def _ensure_tokenized_dataset(
        preset_name: str,
        shape_stats: Dict[str, Any],
        loc_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        if preset_name in token_cache:
            return token_cache[preset_name]

        emb_info = _ensure_embeddings(preset_name)
        cfg_stage = emb_info["cfg"]
        _ensure_stage_workspace(cfg_stage)

        mixed_stats_path = write_mixed_embedding_stats(
            cfg_stage,
            shape_stats=shape_stats,
            loc_stats=loc_stats,
        )
        _clear_stage_token_outputs(cfg_stage)

        train_npz_full = encode_split_to_token_npz(
            cfg_stage,
            "train",
            emb_info["train_emb"],
            shape_tokenizer,
            location_tokenizer,
            device,
        )
        val_npz_full = encode_split_to_token_npz(
            cfg_stage,
            "val",
            emb_info["val_emb"],
            shape_tokenizer,
            location_tokenizer,
            device,
        )
        test_npz_full = encode_split_to_token_npz(
            cfg_stage,
            "test",
            emb_info["test_emb"],
            shape_tokenizer,
            location_tokenizer,
            device,
        )

        train_npz = make_generator_token_npz(cfg_stage, "train", train_npz_full)
        val_npz = make_generator_token_npz(cfg_stage, "val", val_npz_full)
        test_npz = make_generator_token_npz(cfg_stage, "test", test_npz_full)

        payload = {
            "cfg": cfg_stage,
            "mixed_stats_npz": mixed_stats_path,
            "train_npz_full": train_npz_full,
            "val_npz_full": val_npz_full,
            "test_npz_full": test_npz_full,
            "train_npz": train_npz,
            "val_npz": val_npz,
            "test_npz": test_npz,
        }
        token_cache[preset_name] = payload

        results[f"{preset_name}_mixed_embedding_stats_npz"] = str(mixed_stats_path)
        results[f"{preset_name}_train_npz_full"] = str(train_npz_full)
        results[f"{preset_name}_val_npz_full"] = str(val_npz_full)
        results[f"{preset_name}_test_npz_full"] = str(test_npz_full)
        results[f"{preset_name}_train_npz"] = str(train_npz)
        results[f"{preset_name}_val_npz"] = str(val_npz)
        results[f"{preset_name}_test_npz"] = str(test_npz)
        return payload

    shape_ae_preset = dataset_map["shape_ae"]
    location_ae_preset = dataset_map["location_ae"]
    shape_tok_preset = dataset_map["shape_tokenizer"]
    location_tok_preset = dataset_map["location_tokenizer"]
    generator_preset = dataset_map["generator"]

    shape_ae_cfg = dataset_cfgs[shape_ae_preset]
    location_ae_cfg = dataset_cfgs[location_ae_preset]
    shape_tok_cfg = dataset_cfgs[shape_tok_preset]
    location_tok_cfg = dataset_cfgs[location_tok_preset]
    generator_cfg = dataset_cfgs[generator_preset]

    print(f"\n[2/9] Training / loading shape_ae on preset='{shape_ae_preset}'")
    _ensure_stage_workspace(shape_ae_cfg)
    if trained_this_run["shape_ae"]:
        train_shape_ds = QuickDrawStrokeDataset(shape_ae_cfg, split="train", mode="shape_ae")
        val_shape_ds = QuickDrawStrokeDataset(shape_ae_cfg, split="val", mode="shape_ae")
        train_shape_loader = _loader(train_shape_ds, batch_size=int(shape_ae_cfg["shape_ae"]["batch_size"]), shuffle=True, cfg=shape_ae_cfg)
        val_shape_loader = _loader(val_shape_ds, batch_size=int(shape_ae_cfg["shape_ae"]["batch_size"]), shuffle=False, cfg=shape_ae_cfg)
    else:
        train_shape_loader = None
        val_shape_loader = None
    shape_ae = build_shape_ae(shape_ae_cfg).to(device)
    print("shape_ae params:", count_parameters(shape_ae))
    shape_ae_ckpt, _ = maybe_train_or_load_simple(
        shape_ae,
        train_shape_loader,
        val_shape_loader,
        shape_ae_cfg,
        shape_ae_cfg["shape_ae"],
        device,
        "shape_ae",
        run_shape_ae_epoch,
        override_ckpt=shape_ae_override,
        init_ckpt=shape_ae_init,
    )
    results["shape_ae_ckpt"] = str(shape_ae_ckpt)

    print(f"\n[3/9] Training / loading location_ae on preset='{location_ae_preset}'")
    _ensure_stage_workspace(location_ae_cfg)
    if trained_this_run["location_ae"]:
        train_loc_ds = QuickDrawStrokeDataset(location_ae_cfg, split="train", mode="location_ae")
        val_loc_ds = QuickDrawStrokeDataset(location_ae_cfg, split="val", mode="location_ae")
        train_loc_loader = _loader(train_loc_ds, batch_size=int(location_ae_cfg["location_ae"]["batch_size"]), shuffle=True, cfg=location_ae_cfg)
        val_loc_loader = _loader(val_loc_ds, batch_size=int(location_ae_cfg["location_ae"]["batch_size"]), shuffle=False, cfg=location_ae_cfg)
    else:
        train_loc_loader = None
        val_loc_loader = None
    location_ae = build_location_ae(location_ae_cfg).to(device)
    print("location_ae params:", count_parameters(location_ae))
    location_ae_ckpt, _ = maybe_train_or_load_simple(
        location_ae,
        train_loc_loader,
        val_loc_loader,
        location_ae_cfg,
        location_ae_cfg["location_ae"],
        device,
        "location_ae",
        run_location_ae_epoch,
        override_ckpt=location_ae_override,
        init_ckpt=location_ae_init,
    )
    results["location_ae_ckpt"] = str(location_ae_ckpt)

    print(f"\n[4/9] Encoding embeddings for location_tokenizer preset='{location_tok_preset}'")
    location_tok_emb = _ensure_embeddings(location_tok_preset)

    print(f"\n[5/9] Training / loading location_tokenizer on preset='{location_tok_preset}'")
    if trained_this_run["location_tokenizer"]:
        train_loc_seq_ds = EmbeddingSequenceDataset(
            location_tok_emb["train_emb"],
            feature_key="loc_embeddings",
            mean=location_tok_emb["stats"]["loc_mean"],
            std=location_tok_emb["stats"]["loc_std"],
        )
        val_loc_seq_ds = EmbeddingSequenceDataset(
            location_tok_emb["val_emb"],
            feature_key="loc_embeddings",
            mean=location_tok_emb["stats"]["loc_mean"],
            std=location_tok_emb["stats"]["loc_std"],
        )
        train_loc_seq_loader = _loader(train_loc_seq_ds, batch_size=int(location_tok_cfg["location_tokenizer"]["batch_size"]), shuffle=True, cfg=location_tok_cfg)
        val_loc_seq_loader = _loader(val_loc_seq_ds, batch_size=int(location_tok_cfg["location_tokenizer"]["batch_size"]), shuffle=False, cfg=location_tok_cfg)
    else:
        train_loc_seq_loader = None
        val_loc_seq_loader = None

    location_tokenizer = build_location_tokenizer(location_tok_cfg).to(device)
    print("location_tokenizer params:", count_parameters(location_tokenizer))
    location_tok_ckpt, _ = maybe_train_or_load_tokenizer(
        location_tokenizer,
        train_loc_seq_loader,
        val_loc_seq_loader,
        location_tok_cfg,
        location_tok_cfg["location_tokenizer"],
        device,
        "location_tokenizer",
        override_ckpt=location_tok_override,
        init_ckpt=location_tok_init,
        aux_decoder_model=location_ae,
        stats=location_tok_emb["stats"],
    )
    results["location_tokenizer_ckpt"] = str(location_tok_ckpt)

    print(f"\n[6/9] Encoding embeddings for shape_tokenizer preset='{shape_tok_preset}'")
    shape_tok_emb = _ensure_embeddings(shape_tok_preset)

    print(f"\n[7/9] Training / loading shape_tokenizer on preset='{shape_tok_preset}'")
    if trained_this_run["shape_tokenizer"]:
        train_shape_seq_ds = EmbeddingSequenceDataset(
            shape_tok_emb["train_emb"],
            feature_key="shape_embeddings",
            mean=shape_tok_emb["stats"]["shape_mean"],
            std=shape_tok_emb["stats"]["shape_std"],
        )
        val_shape_seq_ds = EmbeddingSequenceDataset(
            shape_tok_emb["val_emb"],
            feature_key="shape_embeddings",
            mean=shape_tok_emb["stats"]["shape_mean"],
            std=shape_tok_emb["stats"]["shape_std"],
        )
        train_shape_seq_loader = _loader(train_shape_seq_ds, batch_size=int(shape_tok_cfg["shape_tokenizer"]["batch_size"]), shuffle=True, cfg=shape_tok_cfg)
        val_shape_seq_loader = _loader(val_shape_seq_ds, batch_size=int(shape_tok_cfg["shape_tokenizer"]["batch_size"]), shuffle=False, cfg=shape_tok_cfg)
    else:
        train_shape_seq_loader = None
        val_shape_seq_loader = None

    shape_tokenizer = build_shape_tokenizer(shape_tok_cfg).to(device)
    print("shape_tokenizer params:", count_parameters(shape_tokenizer))
    shape_tok_ckpt, _ = maybe_train_or_load_tokenizer(
        shape_tokenizer,
        train_shape_seq_loader,
        val_shape_seq_loader,
        shape_tok_cfg,
        shape_tok_cfg["shape_tokenizer"],
        device,
        "shape_tokenizer",
        override_ckpt=shape_tok_override,
        init_ckpt=shape_tok_init,
        aux_decoder_model=shape_ae,
        stats=shape_tok_emb["stats"],
    )
    results["shape_tokenizer_ckpt"] = str(shape_tok_ckpt)

    print(f"\n[8/9] Writing mixed stats + tokenizing generator preset='{generator_preset}'")
    generator_tokens = _ensure_tokenized_dataset(
        generator_preset,
        shape_stats=shape_tok_emb["stats"],
        loc_stats=location_tok_emb["stats"],
    )

    print(f"\n[9/9] Training / loading generator on preset='{generator_preset}'")
    train_token_loader = None
    val_token_loader = None
    if trained_this_run["generator"]:
        train_token_ds = TokenSequenceDataset(
            generator_tokens["train_npz"],
            shape_vocab_size=int(generator_cfg["shape_tokenizer"]["num_embeddings"]),
            loc_vocab_size=int(generator_cfg["location_tokenizer"]["num_embeddings"]),
        )
        val_token_ds = TokenSequenceDataset(
            generator_tokens["val_npz"],
            shape_vocab_size=int(generator_cfg["shape_tokenizer"]["num_embeddings"]),
            loc_vocab_size=int(generator_cfg["location_tokenizer"]["num_embeddings"]),
        )
        train_token_loader = _loader(train_token_ds, batch_size=int(generator_cfg["generator"]["batch_size"]), shuffle=True, cfg=generator_cfg)
        val_token_loader = _loader(val_token_ds, batch_size=int(generator_cfg["generator"]["batch_size"]), shuffle=False, cfg=generator_cfg)

    test_token_ds = TokenSequenceDataset(
        generator_tokens["test_npz"],
        shape_vocab_size=int(generator_cfg["shape_tokenizer"]["num_embeddings"]),
        loc_vocab_size=int(generator_cfg["location_tokenizer"]["num_embeddings"]),
    )
    test_token_loader = _loader(test_token_ds, batch_size=int(generator_cfg["generator"]["batch_size"]), shuffle=False, cfg=generator_cfg)

    generator = build_generator(generator_cfg, shape_tokenizer=shape_tokenizer, location_tokenizer=location_tokenizer).to(device)
    print("generator params:", count_parameters(generator))
    generator_ckpt, _ = maybe_train_or_load_generator(
        generator,
        train_token_loader,
        val_token_loader,
        generator_cfg,
        device,
        override_ckpt=generator_override,
        init_ckpt=generator_init,
    )
    results["generator_ckpt"] = str(generator_ckpt)

    teacher_metrics = evaluate_generator_teacher_forced(
        generator,
        test_token_loader,
        device,
        mixed_precision=bool(generator_cfg["project"]["mixed_precision"]),
    )
    results["teacher_forced_metrics"] = teacher_metrics
    results["generator_dataset_preset"] = str(generator_preset)

    sample_paths = []
    class_samples = []
    for class_id, class_name in enumerate(generator_cfg["dataset"]["classes"]):
        samples = sample_class_conditioned_sketches(
            generator_cfg,
            generator,
            shape_ae,
            location_ae,
            shape_tokenizer,
            location_tokenizer,
            class_id=class_id,
            num_samples=int(generator_cfg["debug"]["class_only_samples_per_class"]),
            device=device,
            max_steps=int(generator_cfg["dataset"]["max_strokes"]),
            temperature=float(generator_cfg["debug"]["class_only_temperature"]),
            top_p=float(generator_cfg["debug"]["class_only_top_p"]),
        )
        class_samples.append((class_name, samples))

    if class_samples:
        n_cols = max(len(s[1]) for s in class_samples)
        fig, axes = plt.subplots(len(class_samples), n_cols, figsize=(3.2 * n_cols, 3.2 * len(class_samples)))
        if len(class_samples) == 1:
            axes = np.array([axes])
        if axes.ndim == 1:
            axes = axes.reshape(len(class_samples), -1)

        for row, (class_name, samples) in enumerate(class_samples):
            for col in range(axes.shape[1]):
                ax = axes[row, col]
                ax.axis("off")
                if col < len(samples):
                    sample = samples[col]
                    ax.imshow(sample["canvas"], cmap="gray")
                    ax.set_title(f"{class_name}\nlen={sample['length']}", fontsize=10)

        plt.tight_layout()
        grid_name = str(generator_cfg["project"].get("preview_grid_name", "class_conditioned_grid.png"))
        grid_path = artifact_root(generator_cfg) / grid_name
        plt.savefig(grid_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        sample_paths.append(str(grid_path))

    results["preview_paths"] = sample_paths

    # --- debug artifacts kept for existing QuickDraw visualization flows ---
    target_debug_emb = _ensure_embeddings("target_quickdraw")
    results["target_train_emb_npz"] = str(target_debug_emb["train_emb"])
    results["target_val_emb_npz"] = str(target_debug_emb["val_emb"])
    results["target_test_emb_npz"] = str(target_debug_emb["test_emb"])
    results["target_shape_embedding_stats_npz"] = str(target_debug_emb["stats_npz"])

    target_debug_tokens = _ensure_tokenized_dataset(
        "target_quickdraw",
        shape_stats=shape_tok_emb["stats"],
        loc_stats=location_tok_emb["stats"],
    )
    results["target_mixed_embedding_stats_npz"] = str(target_debug_tokens["mixed_stats_npz"])
    results["target_train_npz_full"] = str(target_debug_tokens["train_npz_full"])
    results["target_val_npz_full"] = str(target_debug_tokens["val_npz_full"])
    results["target_test_npz_full"] = str(target_debug_tokens["test_npz_full"])
    results["target_train_npz"] = str(target_debug_tokens["train_npz"])
    results["target_val_npz"] = str(target_debug_tokens["val_npz"])
    results["target_test_npz"] = str(target_debug_tokens["test_npz"])

    write_json(root_results_json_path(cfg_root), results)
    write_json(artifact_root(generator_cfg) / f"{cfg_root['project']['notebook_name']}_generator_results.json", results)

    print("\nDone.")
    print("root results json:", root_results_json_path(cfg_root))
    if sample_paths:
        print("generator preview:", sample_paths[0])

    models = {
        "shape_ae": shape_ae,
        "location_ae": location_ae,
        "shape_tokenizer": shape_tokenizer,
        "location_tokenizer": location_tokenizer,
        "generator": generator,
        "generator_cfg": generator_cfg,
        "target_cfg": dataset_cfgs["target_quickdraw"],
        "dataset_cfgs": dataset_cfgs,
        "model_dataset_assignments": copy.deepcopy(dataset_map),
    }
    return results, models
