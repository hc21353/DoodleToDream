import argparse
import json
from pathlib import Path

def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    skip_eda = bool(cfg.get("skip_eda", False)) or args.skip_eda
    skip_pretrain = bool(cfg.get("skip_pretrain", False)) or args.skip_pretrain
    skip_finetune = bool(cfg.get("skip_finetune", False)) or args.skip_finetune

    from .core import apply_config_overrides, main

    apply_config_overrides(cfg)
    main(
        skip_eda=skip_eda,
        skip_pretrain=skip_pretrain,
        skip_finetune=skip_finetune,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
