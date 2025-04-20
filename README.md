# SemColorNet
**A Single-Image Depth-Aware Colorization Framework Integrating Semantic Segmentation**

---

## 📌 Overview

**SemColorNet** is a lightweight, single-image colorization framework that reconstructs stylistically diverse RGB outputs from sparse architectural inputs—primarily a sketch and an estimated depth map. It optionally incorporates semantic masks and palette guidance to improve structural consistency and regional color modulation.

The model is designed for data-scarce applications such as architectural visualization and heritage restoration, where ground-truth color supervision is limited or absent.

---

## 🧠 Method Summary

### 🔶 Input:
- 🖋️ Sketch (1-channel grayscale, 512×512 or patch)
- 🌐 Depth map (ZoeDepth-estimated)
- 🧩 Semantic mask (optional, pre-segmented or SAM-based)
- 🎨 Optional palette configuration (HSV, Lab, fixed or learned)

### 🔶 Output:
- 💡 Stylized RGB image (diverse and semantically coherent)
- Evaluation metrics: SSIM, PSNR

### 🔶 Architecture:
- Backbone: **U-Net** with 3 downsampling and 3 upsampling blocks
- Bottleneck: **Self-Attention**
- Style modulation: Optional **AdaIN** or **palette-guided**
- Loss: **L1 + optional VGG perceptual loss**
- Color space: HSV or Lab (configurable)
- Semantic-aware concatenation (when mask is used)

---

## 📁 Directory Structure

```
project/
├── config/
│   └── color_palettes.py       # Defines color palette configurations
├── models/
│   ├── attention.py            # Self-attention module
│   └── unet.py                 # U-Net model architecture
├── utils/
│   ├── color_utils.py          # Functions for color processing
│   ├── image_utils.py          # General image utility functions
│   ├── lab_processor.py        # LAB color space processing
│   └── visualization.py        # Visualization utilities
├── data/
│   └── dataset.py              # Dataset class definition
├── loss/
│   └── combined_loss.py        # Custom loss function definitions
├── process/
│   ├── high_res.py             # High-resolution image inference
│   └── sketch_depth.py         # Sketch and depth map generation
├── train.py                    # Model training script
├── process.py                  # Inference pipeline script
├── batch_process.py            # Batch image processing script
└── generate_styles.py          # Stylized image generation script
```

---

## 🚀 How to Run

### ✅ Environment Setup

```bash
pip install -r requirements.txt
```

### ✅ Training

Run the training pipeline using the provided script:

```bash
chmod +x train.sh
bash train.sh
```

You can monitor training in terminal using:

```bash
tmux attach -t sketch_train
cat metrics_train/training.log
```

### ✅ Inference (High-Res)

Use the inference script:

```bash
chmod +x process.sh
bash process.sh
```

---

## 📊 Evaluation

Metrics are computed during training and logged:
- **Structural Similarity (SSIM)**
- **Peak Signal-to-Noise Ratio (PSNR)**

Final model achieved:
```
SSIM  = 0.5345
PSNR  = 17.29 dB
```

Trained from a **single image**, with consistent stylization.

---

## 🔍 Training Strategy

- Stage 1: Patch-based training (100 epochs @ 256×256)
- Stage 2: Full-image fine-tuning (50 epochs @ 512×512)
- Loss: L1 + optional VGG (ablation showed VGG hurt PSNR)
- Optimizer: Adam
- Learning rate: Adaptive decay (0.0002 → 0.00001)

Augmentation includes:
- Random crop
- Horizontal/vertical flip
- Rotation
- Color jitter (all applied synchronously on sketch + depth + target canvas)

---

## 🔬 Ablation Summary

| Configuration             | SSIM   | PSNR (dB) |
|--------------------------|--------|-----------|
| Base (Finetuned)         | 0.5091 | 14.10     |
| + Style Encoder (AdaIN)  | 0.5086 | 14.06     |
| + Adaptive LR Decay      | 0.5236 | 14.76     |
| + VGG Perceptual Loss    | 0.5373 | 14.46     |
| ✅ Final (w/o VGG)        | **0.5345** | **17.29**     |

---

## 📷 Sample Results

![inputs](figures/combined_inputs_row.png)
> Input: Sketch + Depth + Semantic (optional)

![outputs](figures/stylized_outputs_grid.png)
> Stylized outputs under different configurations

---

## ⚠️ Ethical Considerations

As a generative method, outputs may be misinterpreted as factual. Style bias and visual artifacts may distort structural perception. Use cases should clearly disclose AI generation and treat the model as an assistive tool—not a source of objective truth.

---

## 📚 References

[1] Wang et al., SDE-Net: Sketch-guided Depth-aware Edge-enhanced Network, ECCV 2022  
[2] Shaham et al., SinGAN: Learning a Generative Model from a Single Image, ICCV 2019  
[3] Mildenhall et al., NeRF, ECCV 2020  
[4] Yang et al., NeRFCodec, CVPR 2023

---

## 🏁 Project Author

Rui Luo | u8076655 | [ENGN6528 - Computer Vision, ANU]