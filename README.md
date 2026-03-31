# QuickDraw README

# Project Overview

This project is inspired by **Quick, Draw!**, a game where one player draws as quickly as possible and the other tries to guess the category as early as possible. We implement both roles directly:

- **player1 - classification**: predicts the class from an incomplete or completed sketch
- **player2 - generation**: generates a sketch stroke by stroke from a class condition

The project focuses not only on final sketch quality, but also on the **drawing process itself**: how a sketch gradually becomes recognizable, and how the classifier confidence changes as more strokes are added. We compare two stroke-based generation approaches on 10 QuickDraw classes: **SketchGPT** and **VQ-SGen**.

---

## player1 - classification

### Confidence score over time

<table>
  <tr>
    <td align="center"><strong>airplane</strong></td>
    <td align="center"><strong>bus</strong></td>
    <td align="center"><strong>canoe</strong></td>
    <td align="center"><strong>car</strong></td>
    <td align="center"><strong>helicopter</strong></td>
  </tr>
  <tr>
    <td><img src="assets/classification/confidence_curve/airplane.png" width="180" alt="airplane confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/bus.png" width="180" alt="bus confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/canoe.png" width="180" alt="canoe confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/car.png" width="180" alt="car confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/helicopter.png" width="180" alt="helicopter confidence curve" /></td>
  </tr>
  <tr>
    <td align="center"><strong>hot air balloon</strong></td>
    <td align="center"><strong>motorbike</strong></td>
    <td align="center"><strong>sailboat</strong></td>
    <td align="center"><strong>submarine</strong></td>
    <td align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td><img src="assets/classification/confidence_curve/hot_air_balloon.png" width="180" alt="hot air balloon confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/motorbike.png" width="180" alt="motorbike confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/sailboat.png" width="180" alt="sailboat confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/submarine.png" width="180" alt="submarine confidence curve" /></td>
    <td><img src="assets/classification/confidence_curve/train.png" width="180" alt="train confidence curve" /></td>
  </tr>
</table>

---

## player2 - generation

### SketchGPT stroke-by-stroke generation

<table>
  <tr>
    <td align="center"><strong>airplane</strong></td>
    <td align="center"><strong>bus</strong></td>
    <td align="center"><strong>canoe</strong></td>
    <td align="center"><strong>car</strong></td>
    <td align="center"><strong>helicopter</strong></td>
  </tr>
  <tr>
    <td><img src="assets/generation/sketchgpt/airplane.gif" width="180" alt="SketchGPT airplane gif" /></td>
    <td><img src="assets/generation/sketchgpt/bus.gif" width="180" alt="SketchGPT bus gif" /></td>
    <td><img src="assets/generation/sketchgpt/canoe.gif" width="180" alt="SketchGPT canoe gif" /></td>
    <td><img src="assets/generation/sketchgpt/car.gif" width="180" alt="SketchGPT car gif" /></td>
    <td><img src="assets/generation/sketchgpt/helicopter.gif" width="180" alt="SketchGPT helicopter gif" /></td>
  </tr>
  <tr>
    <td align="center"><strong>hot air balloon</strong></td>
    <td align="center"><strong>motorbike</strong></td>
    <td align="center"><strong>sailboat</strong></td>
    <td align="center"><strong>submarine</strong></td>
    <td align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td><img src="assets/generation/sketchgpt/hot_air_balloon.gif" width="180" alt="SketchGPT hot air balloon gif" /></td>
    <td><img src="assets/generation/sketchgpt/motorbike.gif" width="180" alt="SketchGPT motorbike gif" /></td>
    <td><img src="assets/generation/sketchgpt/sailboat.gif" width="180" alt="SketchGPT sailboat gif" /></td>
    <td><img src="assets/generation/sketchgpt/submarine.gif" width="180" alt="SketchGPT submarine gif" /></td>
    <td><img src="assets/generation/sketchgpt/train.gif" width="180" alt="SketchGPT train gif" /></td>
  </tr>
</table>

### VQ-SGen stroke-by-stroke generation

<table>
  <tr>
    <td align="center"><strong>airplane</strong></td>
    <td align="center"><strong>bus</strong></td>
    <td align="center"><strong>canoe</strong></td>
    <td align="center"><strong>car</strong></td>
    <td align="center"><strong>helicopter</strong></td>
  </tr>
  <tr>
    <td><img src="assets/generation/vq_sgen/airplane.gif" width="180" alt="VQ-SGen airplane gif" /></td>
    <td><img src="assets/generation/vq_sgen/bus.gif" width="180" alt="VQ-SGen bus gif" /></td>
    <td><img src="assets/generation/vq_sgen/canoe.gif" width="180" alt="VQ-SGen canoe gif" /></td>
    <td><img src="assets/generation/vq_sgen/car.gif" width="180" alt="VQ-SGen car gif" /></td>
    <td><img src="assets/generation/vq_sgen/helicopter.gif" width="180" alt="VQ-SGen helicopter gif" /></td>
  </tr>
  <tr>
    <td align="center"><strong>hot air balloon</strong></td>
    <td align="center"><strong>motorbike</strong></td>
    <td align="center"><strong>sailboat</strong></td>
    <td align="center"><strong>submarine</strong></td>
    <td align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td><img src="assets/generation/vq_sgen/hot_air_balloon.gif" width="180" alt="VQ-SGen hot air balloon gif" /></td>
    <td><img src="assets/generation/vq_sgen/motorbike.gif" width="180" alt="VQ-SGen motorbike gif" /></td>
    <td><img src="assets/generation/vq_sgen/sailboat.gif" width="180" alt="VQ-SGen sailboat gif" /></td>
    <td><img src="assets/generation/vq_sgen/submarine.gif" width="180" alt="VQ-SGen submarine gif" /></td>
    <td><img src="assets/generation/vq_sgen/train.gif" width="180" alt="VQ-SGen train gif" /></td>
  </tr>
</table>

---

# Implementation Detail

## Classification

## Generation

### SketchGPT

### VQ-SGen

---

# Environment Setup

```bash
conda create -n quickdraw python=3.10
conda activate quickdraw
pip install -r requirements.txt
```

# Repository Structure

```text
DoodleToDream-main/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── check_sketchgpt.py
│   ├── check_vq_sgen.py
│   ├── run_sketchgpt.py
│   └── run_vq_sgen.py
├── SketchGPT/
│   ├── configs/
│   └── src/sketchgpt/
├── VQ-SGen/
│   ├── configs/
│   └── src/vq_sgen/
├── evaluation_metrics/
│   ├── classification_metrics.py
│   ├── evaluation_KID.py
│   └── KID/
└── sketchGPT.py
```
