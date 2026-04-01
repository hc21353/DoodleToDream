# Project Overview

This project is inspired by **Quick, Draw!**, a game where one player draws as quickly as possible and the other tries to guess the category as early as possible. We implement both roles directly:

- **player1 - classification**: predicts the class from an incomplete or completed sketch
- **player2 - generation**: generates a sketch stroke by stroke from a class condition

The project focuses not only on final sketch quality, but also on the **drawing process itself**: how a sketch gradually becomes recognizable, and how the classifier confidence changes as more strokes are added. We compare two stroke-based generation approaches on 10 QuickDraw classes: **SketchGPT** and **VQ-SGen**.

---

## player1 - classification

### Confidence score over time

<table width="100%">
  <tr>
    <td width="20%" align="center">
      <img src="assets/classification/confidence_curve/sample1_step1.png" width="140" alt="step 1" /><br/>
      <strong>confidence = 0.12</strong>
    </td>
    <td width="20%" align="center">
      <img src="assets/classification/confidence_curve/sample1_step2.png" width="140" alt="step 2" /><br/>
      <strong>confidence = 0.34</strong>
    </td>
    <td width="20%" align="center">
      <img src="assets/classification/confidence_curve/sample1_step3.png" width="140" alt="step 3" /><br/>
      <strong>confidence = 0.57</strong>
    </td>
    <td width="20%" align="center">
      <img src="assets/classification/confidence_curve/sample1_step4.png" width="140" alt="step 4" /><br/>
      <strong>confidence = 0.81</strong>
    </td>
    <td width="20%" align="center">
      <img src="assets/classification/confidence_curve/sample1_step5.png" width="140" alt="step 5" /><br/>
      <strong>confidence = 0.93</strong>
    </td>
  </tr>
</table>

---

## player2 - generation

### SketchGPT stroke-by-stroke generation

<table width="100%">
  <tr>
    <td width="20%" align="center"><strong>airplane</strong></td>
    <td width="20%" align="center"><strong>bus</strong></td>
    <td width="20%" align="center"><strong>canoe</strong></td>
    <td width="20%" align="center"><strong>car</strong></td>
    <td width="20%" align="center"><strong>helicopter</strong></td>
  </tr>
  <tr>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/airplane.gif" width="140" alt="SketchGPT airplane gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/bus.gif" width="140" alt="SketchGPT bus gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/canoe.gif" width="140" alt="SketchGPT canoe gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/car.gif" width="140" alt="SketchGPT car gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/helicopter.gif" width="140" alt="SketchGPT helicopter gif" /></td>
  </tr>
  <tr>
    <td width="20%" align="center"><strong>hot air balloon</strong></td>
    <td width="20%" align="center"><strong>motorbike</strong></td>
    <td width="20%" align="center"><strong>sailboat</strong></td>
    <td width="20%" align="center"><strong>submarine</strong></td>
    <td width="20%" align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/hot_air_balloon.gif" width="140" alt="SketchGPT hot air balloon gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/motorbike.gif" width="140" alt="SketchGPT motorbike gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/sailboat.gif" width="140" alt="SketchGPT sailboat gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/submarine.gif" width="140" alt="SketchGPT submarine gif" /></td>
    <td width="20%" align="center"><img src="assets/generation/sketchgpt/train.gif" width="140" alt="SketchGPT train gif" /></td>
  </tr>
</table>

### VQ-SGen stroke-by-stroke generation

<table width="100%">
  <tr>
    <td width="20%" align="center"><strong>airplane</strong></td>
    <td width="20%" align="center"><strong>bus</strong></td>
    <td width="20%" align="center"><strong>canoe</strong></td>
    <td width="20%" align="center"><strong>car</strong></td>
    <td width="20%" align="center"><strong>helicopter</strong></td>
  </tr>
  <tr>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/526a579e-233f-41f1-8317-b8d76f75e7a3" width="140" alt="VQ-SGen airplane gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/dc4bf3dc-f31f-45d7-81bf-c4dd11e244f3" width="140" alt="VQ-SGen bus gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/f462eaaa-4736-4620-a8c3-dc042f8164af" width="140" alt="VQ-SGen canoe gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/f84e2735-e821-428b-844e-f787f2372903" width="140" alt="VQ-SGen car gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/bbd12eca-1d06-4652-8bb8-8d2542a53a42" width="140" alt="VQ-SGen helicopter gif" /></td>
  </tr>
  <tr>
    <td width="20%" align="center"><strong>hot air balloon</strong></td>
    <td width="20%" align="center"><strong>motorbike</strong></td>
    <td width="20%" align="center"><strong>sailboat</strong></td>
    <td width="20%" align="center"><strong>submarine</strong></td>
    <td width="20%" align="center"><strong>train</strong></td>
  </tr>
  <tr>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/7784982f-fa75-49f6-89f3-7e0d5aa4485f" width="140" alt="VQ-SGen hot air balloon gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/06ec801b-fd15-4a46-a36f-4542c7db876f" width="140" alt="VQ-SGen motorbike gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/765b7534-d348-4b29-bc62-3c8722b38550" width="140" alt="VQ-SGen sailboat gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/f8fcb9fd-6f36-4e43-8bb3-b19408c5e85e" width="140" alt="VQ-SGen submarine gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/ae53a3f6-85a3-4753-b10a-68b421ea3526" width="140" alt="VQ-SGen train gif" /></td>
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
├── classification_mobilenetv2.ipynb
├── classification_final.ipynb
├── scripts/
│   ├── check_sketchgpt.py
│   ├── check_vq_sgen.py
│   ├── run_classification.py
│   ├── run_sketchgpt.py
│   └── run_vq_sgen.py
├── Classification/
│   ├── configs/
│   │   └── config.json
│   └── src/quickdraw_classifier/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── model.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       ├── thresholds.py
│       └── train.py
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
