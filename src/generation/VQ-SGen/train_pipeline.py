from __future__ import annotations



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
