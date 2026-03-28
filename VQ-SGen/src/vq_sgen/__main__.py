import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JSON config file (defaults to configs/config.json if present).",
    )
    args = parser.parse_args()
    if args.config:
        os.environ["VQ_SGEN_CONFIG"] = args.config

    from .runner import main as runner_main

    runner_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
