# FG-MSCNet: Frequency-Guided Multi-Scale Composite Network for Document Tampering Detection

A deep learning framework for **document image tampering detection**, featuring frequency-guided attention, wavelet-based multi-scale encoding/decoding, and multi-scale deep supervision.

---

## Overview

FG-MSCNet is a U-Net-style network designed for pixel-level document tampering localization. It integrates:

- **SRM (Steganalysis Rich Model)** noise residual extraction as a secondary input stream
- **DWT/IDWT** (Haar wavelet) for lossless downsampling and upsampling
- **FG-SRA** (Frequency-Guided Spatial Reduction Attention) for multi-scale attention
- **DDG Bridge** (Dual-Domain Gate): spatial MHSA + learnable complex spectral filtering
- **Multi-scale deep supervision** with 4 output scales (Full / 1/2 / 1/4 / 1/8)
- **Adaptive Bias Head** driven by bridge features

### Architecture

```
Input [B, 3, H, W]
  │
  ├─ InputStem (RGB + SRM dual-stream fusion)         → [B, 64, H, W]
  │
  ├─ EncoderStage × 4  (DWT ↓ + FG-SRA attention)
  │     enc1: [B,  64, H/2]    enc2: [B, 128, H/4]
  │     enc3: [B, 256, H/8]    enc4: [B, 512, H/16]
  │
  ├─ DDG_Bridge  (Spatial MHSA + Learnable Spectral Filter)
  │                                                   → [B, 512, H/16]
  │
  ├─ DecoderStage × 4  (IDWT ↑ + skip + mask guidance)
  │     dec1: [B, 256, H/8]    dec2: [B, 128, H/4]
  │     dec3: [B,  64, H/2]    dec4: [B,  32, H]
  │
  └─ FinalHead (1×1 Conv)                             → [B, 1, H, W]

Outputs: [Final(H), Fine(H/2), Mid(H/4), Coarse(H/8)]
```

---

## Dataset

This project is built on the **DocTamper** dataset. Organize your data as:

```
<root_dir>/
  DocTamperV1-TrainingSet/    # LMDB format, used for training
  DocTamperV1-TestingSet/     # LMDB format, used for validation / testing
  DocTamperV1-FCD/            # Forgery by Copy-pasting Detection
  DocTamperV1-SCD/            # Forgery by Splicing and Copy-move Detection
```

Each LMDB database contains key-value pairs:
- `image-XXXXXX` → JPEG-encoded RGB image bytes
- `label-XXXXXX` → JPEG-encoded grayscale mask bytes

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/FG-MSCNet.git
cd FG-MSCNet

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **PyTorch**: Install a CUDA-compatible version from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command if not already installed.

---

## Training

Single-node multi-GPU training via `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py \
    --data_root <path/to/dataset_root> \
    --img_size 512 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir runs/exp1
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_root` | required | Root directory of the DocTamper dataset |
| `--img_size` | `512` | Input resolution |
| `--batch_size` | `8` | Per-GPU batch size |
| `--epochs` | `100` | Total training epochs |
| `--lr` | `1e-4` | Initial learning rate (AdamW) |
| `--train_ratio` | `1.0` | Fraction of training data to use |
| `--output_dir` | `runs/` | Directory for checkpoints and TensorBoard logs |
| `--resume` | `None` | Path to checkpoint to fully resume training |
| `--finetune` | `None` | Path to checkpoint to load weights only |

Training uses **BFloat16 AMP**, `SyncBatchNorm`, gradient clipping (`max_norm=5.0`), and a **Warmup + Cosine LR** schedule.

---

## Evaluation

```bash
python eval.py \
    --data_root <path/to/dataset_root> \
    --checkpoint runs/exp1/best_model.pth \
    --img_size 512 \
    --output_dir eval_results/
```

The evaluation script:
- Scans thresholds in [0.1, 0.9] to find the **best F1 threshold** per dataset
- Reports **F1 / Precision / Recall / mIoU / Accuracy / AUC-ROC / AP**
- Saves visualizations: original image, GT mask, prediction heatmap, binary prediction, error map (red = FN, blue = FP), FG-SRA attention map
- Outputs a **Markdown comparison table** across all test datasets

---

## Loss Function

`MultiScaleCompositeLoss` combines three terms across 4 prediction scales:

$$\mathcal{L} = \sum_{s=1}^{4} w_s \left( \lambda_{bce} \cdot \mathcal{L}_{BCE}^{(s)} + \lambda_{dice} \cdot \mathcal{L}_{Dice}^{(s)} + \lambda_{edge} \cdot \mathcal{L}_{Edge}^{(s)} \right)$$

Scale weights: $w = [1.0,\ 0.75,\ 0.5,\ 0.25]$ (coarse-to-fine).  
Ground-truth downsampling uses `MaxPool` to preserve tampered regions.

---

## Model Summary

```bash
python look.py
```

Prints a full layer-by-layer parameter summary via `torchinfo`.

---

## Project Structure

```
FG-MSCNet/
├── layer/
│   ├── fg_mscnet.py    # Main network architecture
│   ├── modules.py      # All sub-modules (InputStem, FG-SRA, DDG_Bridge, ...)
│   ├── dct.py          # Orthogonal DCT-II transform layer
│   ├── srm.py          # Fixed SRM high-pass filter bank (30 kernels)
│   └── wavelet.py      # Haar DWT downsampling / IDWT upsampling
├── dataset.py          # DocTamperDataset (LMDB, albumentations augmentation)
├── loss.py             # DiceLoss, EdgeLoss, MultiScaleCompositeLoss
├── train.py            # DDP training script
├── eval.py             # Multi-dataset evaluation & visualization
├── look.py             # Quick model summary
├── main.py             # LMDB connectivity test utility
├── requirements.txt
└── README.md
```

---

## License

This project is released under the [MIT License](LICENSE).
