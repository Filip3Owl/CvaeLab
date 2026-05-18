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
| Encoder | 4× Conv2d + BatchNorm + LeakyReLU + Dropout2d |
| Decoder | FC + 4× ConvTranspose2d + BatchNorm + ReLU + Dropout2d + Tanh |
| Loss | MSE reconstruction + β · KL divergence (β = 1) |
| Optimizer | Adam (lr = 1e-3, ReduceLROnPlateau) |
| Dropout | 0.2 (encoder and decoder) |
| Parameters | 6,561,792 |

## Training results — Run v1

Training for 50 epochs on ~27,000 images (CPU, Intel Mac).

| Epoch | Total loss | Recon loss | KL divergence |
|---|---|---|---|
| 1 | 1408.23 | 1224.72 | 183.51 |
| 10 | ~620.00 | ~468.00 | ~152.00 |
| 20 | 623.92 | 471.08 | 152.84 |
| 50 | 572.12 | 420.24 | 151.88 |

**Reconstruction loss improved 66%** — from 1,224 down to 420 over 50 epochs.  
**KL divergence stabilized at ~152** — healthy latent space organization, no posterior collapse.

![Training history](training_history.png)

## Project structure

```
cvae-lab/
├── vae_animals.ipynb   # Main notebook (exploration → training → generation)
├── model.py            # VAE architecture (Encoder, Decoder, VAE, vae_loss)
├── dataset.py          # AnimalsDataset and DataLoader
├── train.py            # Training script (CLI alternative to notebook)
├── generate.py         # Image generation and latent space visualization
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

**5. Compare experiments with MLflow**
```bash
mlflow ui   # → http://localhost:5000
```

## Outputs

| File | Description |
|---|---|
| `training_history.png` | Loss curves (total, reconstruction, KL) |
| `generated_samples.png` | 32 images sampled from z ~ N(0, I) |
| `reconstructions.png` | Original vs. reconstructed images |
| `interpolation.png` | Smooth transition between two latent vectors |

## Roadmap

- [x] Convolutional VAE baseline
- [x] MLflow experiment tracking
- [x] Dropout regularization
- [ ] KL annealing
- [ ] Data augmentation
- [ ] Perceptual loss
- [ ] Conditional VAE (class-guided generation)
- [ ] Latent space visualization (t-SNE / UMAP)

## Dataset license

Animals-10 is licensed under [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).  
The dataset is **not included** in this repository — download it separately from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

## Requirements

- Python 3.11
- torch 2.2.2
- torchvision 0.17.2
- mlflow 2.13.2
- numpy, Pillow, matplotlib, tqdm, jupyter
