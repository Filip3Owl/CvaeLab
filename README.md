# cvae-lab

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13.2-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.0-11557C?style=for-the-badge&logo=python&logoColor=white)
![torchmetrics](https://img.shields.io/badge/torchmetrics-1.9.0-FF6F00?style=for-the-badge&logo=python&logoColor=white)

A convolutional Variational Autoencoder trained on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset for image generation and latent space exploration. Currently evolving towards a **Conditional VAE** for class-guided generation.

## Overview

This project implements a VAE from scratch using PyTorch, with progressive improvements across runs. The model learns a compact latent representation of animal images and can generate new samples by decoding random points sampled from the latent space. The next milestone is a Conditional VAE (cVAE) that accepts a class label as input, enabling controlled generation (e.g. "generate a cat").

**Dataset:** Animals-10 — ~27,000 images across 10 classes (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel)

## Architecture

### VAE

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
| Parameters | 2,958,659 |

### cVAE

The **Conditional VAE** conditions both encoder and decoder on the class label `y` via a learned embedding (`EMBED_DIM=64`):

```
Image x ──► ConditionalEncoder ──► μ, log σ²  ──► z = μ + σ·ε ──► ConditionalDecoder ──► x̂
                      ↑                                                       ↑
                  embed(y)                                                embed(y)
```

| Component | Details |
|---|---|
| Class embedding | `nn.Embedding(10, 64)` in both encoder and decoder |
| Encoder conditioning | `cat([conv_features, embed(y)])` before μ / log σ² heads |
| Decoder conditioning | `cat([z, embed(y)])` before FC projection |
| Parameters | 3,238,467 |

At inference, `CVAE.generate(y)` samples `z ~ N(0, I)` and decodes with the target class label.

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

### Latent space — t-SNE projection (run v3)

![Latent space t-SNE](results/run_v3/latent_space.png)

### Run v4 — Notebook run (KL collapse)

Trained directly from the notebook training cell with fixed β = 1.0 (no warmup). The KL term spiked to ~10 000 at epoch 1 then collapsed to ~0, meaning the encoder ignored the input and the decoder stopped using the latent space.

| Epoch | Total loss | Recon loss | KL divergence |
|---|---|---|---|
| 1 | ~11 000 | ~1 400 | ~9 600 |
| 10 | ~900 | ~750 | ~150 → 0 |
| 50 | ~600 | ~430 | ~0 |

> **Lesson:** always use `KL_WARMUP` to anneal β from 0 → 1. Without it, a high initial KL pushes the encoder to collapse the posterior to `N(0, I)`, making the latent code uninformative.

![Training history v4](results/run_v4/training_history.png)
![Generated samples v4](results/run_v4/generated_samples.png)

## Evaluation

IS and FID are computed in notebook section 12 using `torchmetrics[image]` with 2048 images:

| Metric | Measures | Direction |
|---|---|---|
| **Inception Score (IS)** | Quality + diversity of generated images via Inception-v3 | Higher = better |
| **Fréchet Inception Distance (FID)** | Distance between real and generated distributions in Inception-v3 feature space | Lower = better |

Scores are logged to the respective MLflow run for easy comparison between VAE and cVAE.

## Project structure

```
cvae-lab/
├── vae_animals.ipynb   # Main notebook (exploration → training → generation → evaluation)
├── model.py            # VAE + cVAE architectures (Encoder, Decoder, VAE, CVAE, vae_loss)
├── dataset.py          # AnimalsDataset with class labels, CLASS_TO_IDX, get_dataloader
├── train.py            # VAE training script
├── train_cvae.py       # cVAE training script
├── generate.py         # Image generation and latent space visualization
├── visualize.py        # t-SNE / UMAP latent space visualization
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
| `results/run_vN/latent_space.png` | t-SNE / UMAP projection of the latent space, colored by class |

## Roadmap

- [x] Convolutional VAE baseline
- [x] MLflow experiment tracking
- [x] Dropout regularization
- [x] KL annealing
- [x] Data augmentation (RandomHorizontalFlip, ColorJitter)
- [x] Latent space visualization (t-SNE / UMAP)
- [x] IS & FID evaluation metrics
- [ ] Conditional VAE (class-guided generation)
  - [x] Etapa 1 — Dataset com labels (`dataset.py`)
  - [x] Etapa 2 — Arquitetura cVAE (`model.py`)
  - [x] Etapa 3 — Script de treino (`train_cvae.py`)
  - [ ] Etapa 4 — Geração condicional (`generate.py --class dog`)
  - [ ] Etapa 5 — Visualização t-SNE com clusters separados
- [ ] Perceptual loss

## Dataset license

Animals-10 is licensed under [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).  
The dataset is **not included** in this repository — download it separately from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

## Requirements

- Python 3.11
- torch 2.2.2
- torchvision 0.17.2
- mlflow 2.13.2
- torchmetrics[image] 1.9.0
- numpy, Pillow, matplotlib, tqdm, jupyter
