from pathlib import Path
import copy
import json
import shutil


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_trained_artifacts(cfg, results):
    export_dir = Path(cfg["project"]["gdrive_model_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    if not bool(cfg["runtime"].get("save_newly_trained_models_to_gdrive", False)):
        print("cfg['runtime']['save_newly_trained_models_to_gdrive']=False 이므로 저장을 건너뜁니다.")
        return None

    config_path = export_dir / "config_used.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(copy.deepcopy(cfg)), f, ensure_ascii=False, indent=2)

    results_json_src = Path(results["root_results_json"])
    results_json_dst = export_dir / results_json_src.name
    if results_json_src.exists() and results_json_src.resolve() != results_json_dst.resolve():
        shutil.copy2(results_json_src, results_json_dst)

    model_specs = [
        {"stage_key": "shape_ae", "out_name": "shape_ae.pt", "result_key": "shape_ae_ckpt"},
        {"stage_key": "location_ae", "out_name": "location_ae.pt", "result_key": "location_ae_ckpt"},
        {"stage_key": "shape_tokenizer", "out_name": "shape_tokenizer.pt", "result_key": "shape_tokenizer_ckpt"},
        {"stage_key": "location_tokenizer", "out_name": "location_tokenizer.pt", "result_key": "location_tokenizer_ckpt"},
        {"stage_key": "generator", "out_name": "generator.pt", "result_key": "generator_ckpt"},
    ]

    trained_flags = dict(results.get("trained_this_run", {}))
    copied = {"config_used.json": str(config_path)}
    if results_json_dst.exists():
        copied[results_json_dst.name] = str(results_json_dst)

    skipped = {}
    missing = {}

    for spec in model_specs:
        stage_key = spec["stage_key"]
        out_name = spec["out_name"]
        result_key = spec["result_key"]

        if not bool(trained_flags.get(stage_key, False)):
            skipped[stage_key] = {"reason": "not_trained_this_run"}
            continue

        src = results.get(result_key, "")
        src_path = Path(str(src)).expanduser()

        if not str(src).strip():
            missing[stage_key] = f"RESULTS['{result_key}']가 비어 있음"
            continue
        if not src_path.exists():
            missing[stage_key] = f"파일 없음: {src_path}"
            continue

        dst_path = export_dir / out_name
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        copied[out_name] = str(dst_path)

    inventory = {
        "export_dir": str(export_dir),
        "trained_this_run": trained_flags,
        "model_dataset_assignments": results.get("model_dataset_assignments", {}),
        "files": copied,
        "skipped": skipped,
        "missing": missing,
        "run_tag": cfg["project"].get("run_tag", ""),
        "run_mode": cfg["project"].get("run_mode", ""),
    }
    inventory_path = export_dir / "artifact_inventory.json"
    with inventory_path.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("Saved artifacts")
    print("=" * 100)
    for name, path in copied.items():
        print(f"[SAVED]   {name:35s} -> {path}")

    if skipped:
        print("\n" + "-" * 100)
        print("Skipped artifacts")
        print("-" * 100)
        for name, info in skipped.items():
            print(f"[SKIPPED] {name:35s} -> {info}")

    if missing:
        print("\n" + "-" * 100)
        print("Missing artifacts")
        print("-" * 100)
        for name, msg in missing.items():
            print(f"[MISSING] {name:35s} -> {msg}")

    print("\n" + "=" * 100)
    print(f"[OK] artifact_inventory.json 저장 완료: {inventory_path}")
    print("이번 런에서 train / finetune 된 모델만 저장했고, 같은 이름의 기존 가중치는 덮어썼습니다.")
    return inventory_path
