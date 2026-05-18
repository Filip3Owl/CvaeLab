# cvae-lab

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13.2-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.0-11557C?style=for-the-badge&logo=python&logoColor=white)

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

## Training results

All runs: 50 epochs, ~27,000 images, CPU (Intel Mac), Adam lr=1e-3, latent dim=128.

### Run v1 — Baseline

| Epoch | Total loss | Recon loss | KL divergence |
|---|---|---|---|
| 1 | 1408.23 | 1224.72 | 183.51 |
| 10 | ~620.00 | ~468.00 | ~152.00 |
| 50 | 572.12 | 420.24 | 151.88 |

![Training history v1](results/run_v1/training_history.png)

### Run v2 — KL Annealing

Added linear β warmup (0→1 over 25 epochs) to prevent posterior collapse in early training.

| Epoch | Total loss | Recon loss | KL divergence |
|---|---|---|---|
| 1 | 1408.23 | 1224.72 | 183.51 |
| 25 | ~614.00 | ~462.00 | ~152.00 |
| 50 | ~572.00 | ~420.00 | ~152.00 |

![Training history v2](results/run_v2/training_history.png)

### Run v3 — Data Augmentation

Added `RandomHorizontalFlip` + `ColorJitter` to improve generalization.

| Epoch | Total loss | Recon loss | KL divergence |
|---|---|---|---|
| 1 | 1408.23 | 1224.72 | 183.51 |
| 25 | 616.62 | 464.85 | 151.77 |
| 50 | 572.12 | 420.24 | 151.88 |

![Training history v3](results/run_v3/training_history.png)

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
mlflow ui
```
> Opens the experiment dashboard at `http://localhost:5000` (local only — requires the command above to be running).

## Outputs

Results are organized by run under `results/`:

| File | Description |
|---|---|
| `results/run_vN/training_history.png` | Loss curves (total, reconstruction, KL, β) |
| `results/run_vN/generated_samples.png` | 32 images sampled from z ~ N(0, I) |
| `results/run_vN/reconstructions.png` | Original vs. reconstructed images |
| `results/run_vN/interpolation.png` | Smooth transition between two latent vectors |

## Roadmap

- [x] Convolutional VAE baseline
- [x] MLflow experiment tracking
- [x] Dropout regularization
- [x] KL annealing
- [x] Data augmentation (RandomHorizontalFlip, ColorJitter)
- [ ] Latent space visualization (t-SNE / UMAP)
- [ ] Perceptual loss
- [ ] Conditional VAE (class-guided generation)

## Dataset license

Animals-10 is licensed under [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).  
The dataset is **not included** in this repository — download it separately from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

## Requirements

- Python 3.11
- torch 2.2.2
- torchvision 0.17.2
- mlflow 2.13.2
- numpy, Pillow, matplotlib, tqdm, jupyter
