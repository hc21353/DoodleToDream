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
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/bd1ed43f-650d-4a55-9459-3104d120555c" width="140" alt="SketchGPT car gif" /></td>
    <td width="20%" align="center"><img src="https://github.com/user-attachments/assets/df386b8f-6f28-484c-9f57-534680db6b86" width="140" alt="SketchGPT helicopter gif" /></td>
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

The classifier is a stroke-to-raster baseline built for the "guess as the sketch unfolds" setting. Each QuickDraw sample is loaded from the simplified `.ndjson` format, filtered to recognized drawings, rasterized on-the-fly into a square image, and then fed to a MobileNetV2 classifier initialized with ImageNet weights. During training, the model does not always see the full sketch: a random prefix of strokes is sampled so that the classifier learns to predict from incomplete drawings as well as completed ones. Standard image-space augmentation (rotation, affine transform) is applied after rasterization, and the model is trained with cross-entropy on the 10 target classes. At inference time, the same checkpoint is used to report both the top prediction and its confidence, which is later used to visualize how recognizability changes as more strokes are added.

## Generation

### SketchGPT

Our SketchGPT implementation keeps the paper’s primitive-token autoregressive idea, but adapts it into a lightweight project pipeline centered on 10 QuickDraw classes. Raw drawings are converted to stroke-3 format, normalized, and discretized into direction primitives; longer movements are represented by repeating the same primitive token multiple times, and stroke boundaries are marked with a separator token. Before training, we run a small EDA step on the target classes to estimate a suitable primitive length and sequence limit, so the tokenizer is tuned to the actual class subset instead of being fully fixed beforehand.

Compared with the original paper-style setup, this repository uses a more practical two-stage workflow: one shared language-model pretraining stage on the mixed class set, followed by separate class-wise fine-tuning and separate checkpoints for each class. So, instead of one unified conditional generator used at runtime, generation is operationalized as "one pretrained base + one finetuned generator per class." Sampling is also simplified for project use: top-k sampling with temperature is used, a minimum number of new tokens is enforced before EOS, and the outputs are exported not only as final sketches but also as sequential stroke accumulation images for confidence analysis.

### VQ-SGen

Our VQ-SGen pipeline keeps the core decomposition of the original paper—separating **shape** and **location** representations and generating their discrete codes autoregressively—but the actual implementation is intentionally more task-specific. In this repository, the active pipeline is QuickDraw-only, with the generator trained on the 10 target classes and the representation modules reused from pretrained checkpoints by default. In other words, the code is set up so that shape AE, location AE, and both tokenizers are usually treated as reusable representation modules, while the main project-side adaptation happens in the generator.

Relative to the original VQ-SGen formulation, the generator here contains several practical modifications. First, QuickDraw strokes are canonically reordered before training (currently by descending stroke bounding-box area), which makes generation more stable but changes the sequential target itself. Second, the generator does not use only discrete code IDs: it adds a small residual token embedding on top of projected codebook features (`codebook_residual` mode), so the autoregressive model can retain token-specific flexibility beyond pure codebook lookup. Third, training includes scheduled sampling and an explicit early-stroke loss upweighting scheme, reflecting the project goal that the beginning of the drawing should already become recognizable. Finally, the output path is evaluation-oriented: generated shape/location token sequences are decoded back into cumulative stroke frames and final canvases, so the model can be compared directly with the classifier’s confidence-over-time analysis.

---

# Environment Setup

```bash
conda create -n quickdraw python=3.10
conda activate quickdraw
pip install -r requirements.txt
```

# Repository Structure

```text
```
