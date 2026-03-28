# VQ-SGen + SketchGPT Monorepo

This repository contains two modular projects:

- `VQ-SGen/`: VQ-SGen training/generation pipeline
- `SketchGPT/`: SketchGPT training/generation pipeline

## Repository Layout

- `VQ-SGen/src/vq_sgen`: VQ-SGen package
- `VQ-SGen/configs`: VQ-SGen configs
- `SketchGPT/src/sketchgpt`: SketchGPT package
- `SketchGPT/configs`: SketchGPT configs
- `scripts/`: shared setup/run/check scripts

## Requirements

- Python 3.9+
- macOS/Linux shell examples below

## Quick Start (Clone-and-Run)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Then run:

```bash
.venv/bin/python scripts/check_vq_sgen.py
.venv/bin/python scripts/check_sketchgpt.py

.venv/bin/python scripts/run_vq_sgen.py --config configs/config.json
.venv/bin/python scripts/run_sketchgpt.py --config configs/config.json
```

## Direct Run with Current Interpreter

If your current interpreter already has dependencies installed:

```bash
python3 scripts/run_vq_sgen.py --config configs/config.json
python3 scripts/run_sketchgpt.py --config configs/config.json
```
