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
    <td align="center">
      <img src="assets/classification/confidence_curve/sample1_step1.png" width="180" alt="step 1" /><br/>
      <strong>confidence = 0.12</strong>
    </td>
    <td align="center">
      <img src="assets/classification/confidence_curve/sample1_step2.png" width="180" alt="step 2" /><br/>
      <strong>confidence = 0.34</strong>
    </td>
    <td align="center">
      <img src="assets/classification/confidence_curve/sample1_step3.png" width="180" alt="step 3" /><br/>
      <strong>confidence = 0.57</strong>
    </td>
    <td align="center">
      <img src="assets/classification/confidence_curve/sample1_step4.png" width="180" alt="step 4" /><br/>
      <strong>confidence = 0.81</strong>
    </td>
    <td align="center">
      <img src="assets/classification/confidence_curve/sample1_step5.png" width="180" alt="step 5" /><br/>
      <strong>confidence = 0.93</strong>
    </td>
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
    <td><img src="https://github.com/user-attachments/assets/526a579e-233f-41f1-8317-b8d76f75e7a3" width="180" alt="VQ-SGen airplane gif" /></td>
    <td><img src="PUT_BUS_GIF_LINK_HERE" width="180" alt="VQ-SGen bus gif" /></td>
    <td><img src="PUT_CANOE_GIF_LINK_HERE" width="180" alt="VQ-SGen canoe gif" /></td>
    <td><img src="PUT_CAR_GIF_LINK_HERE" width="180" alt="VQ-SGen car gif" /></td>
    <td><img src="PUT_HELICOPTER_GIF_LINK_HERE" width="180" alt="VQ-SGen helicopter gif" /></td>
  </tr>
  <tr>
    <td align="center"><strong>hot air balloon</strong></td>
    <td align="center"><strong>motorbike</strong></td>
    <td align="center"><strong>sailboat</strong></td>
    <td align="center"><strong>submarine</strong></td>
    <td align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td><img src="PUT_HOT_AIR_BALLOON_GIF_LINK_HERE" width="180" alt="VQ-SGen hot air balloon gif" /></td>
    <td><img src="PUT_MOTORBIKE_GIF_LINK_HERE" width="180" alt="VQ-SGen motorbike gif" /></td>
    <td><img src="PUT_SAILBOAT_GIF_LINK_HERE" width="180" alt="VQ-SGen sailboat gif" /></td>
    <td><img src="PUT_SUBMARINE_GIF_LINK_HERE" width="180" alt="VQ-SGen submarine gif" /></td>
    <td><img src="PUT_TRAIN_GIF_LINK_HERE" width="180" alt="VQ-SGen train gif" /></td>
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
