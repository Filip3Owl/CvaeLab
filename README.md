# cvae-lab

A convolutional Variational Autoencoder (CVAE) trained on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset for image generation and latent space exploration.

## Overview

This project implements a CVAE from scratch using PyTorch. The model learns a compact latent representation of animal images and can generate new samples by decoding random points sampled from the latent space.

**Dataset:** Animals-10 — ~27,000 images across 10 classes (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel)

## Architecture

```
Encoder: 3×64×64 → Conv×4 → Flatten → μ (128), log σ² (128)
                                         ↓ reparameterization
Decoder:                        z (128) → FC → ConvTranspose×4 → 3×64×64
```

| Component | Details |
|---|---|
| Input resolution | 64 × 64 × 3 |
| Latent dimension | 128 |
| Encoder | 4× Conv2d + BatchNorm + LeakyReLU |
| Decoder | FC + 4× ConvTranspose2d + BatchNorm + ReLU + Tanh |
| Loss | MSE reconstruction + β · KL divergence (β = 1) |
| Optimizer | Adam (lr = 1e-3, ReduceLROnPlateau) |

## Project structure

```
cvae-lab/
├── vae_animals.ipynb   # Main notebook (exploration → training → generation)
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Getting started

**1. Clone the repository**
```bash
git clone https://github.com/Filip3Owl/CvaeLab.git
cd CvaeLab
```

**2. Download the dataset**

Download [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle and place it as:
```
archive/
└── raw-img/
    ├── cane/
    ├── cavallo/
    └── ...
```

**3. Create the virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**4. Run the notebook**
```bash
jupyter notebook vae_animals.ipynb
```

## Results

The notebook produces the following outputs after training:

| File | Description |
|---|---|
| `training_history.png` | Loss curves (total, reconstruction, KL) |
| `generated_samples.png` | 32 images sampled from z ~ N(0, I) |
| `reconstructions.png` | Original vs. reconstructed images |
| `interpolation.png` | Smooth transition between two latent vectors |

## Requirements

- Python 3.11
- torch 2.2.2
- torchvision 0.17.2
- numpy, Pillow, matplotlib, tqdm, jupyter
