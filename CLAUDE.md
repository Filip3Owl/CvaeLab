# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.11 with a local virtual environment. Always activate it before running anything:

```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Common commands

**Run the notebook:**
```bash
jupyter notebook vae_animals.ipynb
```

**Train from the terminal (alternative to notebook):**
```bash
python train.py
```

**Generate images from a saved checkpoint:**
```bash
python generate.py --checkpoint checkpoints/vae_final.pt --n 16
python generate.py --checkpoint checkpoints/vae_final.pt --interpolate
python generate.py --checkpoint checkpoints/vae_final.pt --history
```

**Open MLflow UI to compare experiment runs:**
```bash
mlflow ui   # → http://localhost:5000
```

## Architecture

The project trains a **convolutional Variational Autoencoder (CVAE)** on the Animals-10 dataset (~27k images, 10 classes).

### Data flow

```
archive/raw-img/<class>/*.jpg  →  AnimalsDataset  →  DataLoader  →  VAE  →  checkpoints/
                                                                            results/
```

Images are resized to 64×64 and normalized to [-1, 1] to match the decoder's `Tanh` output.

### Model (`model.py`)

Three classes: `Encoder`, `Decoder`, `VAE`. Two utility functions: `vae_loss`, `get_beta`.

- **Encoder**: 4× `Conv2d` (stride=2) + `BatchNorm2d` + `LeakyReLU` + `Dropout2d` → flatten → `Dropout` → two linear heads producing `μ` and `log σ²` (both `LATENT_DIM=128`).
- **Reparameterization**: `z = μ + σ·ε`, `ε ~ N(0,I)`. In `eval()` mode returns `μ` directly (no noise).
- **Decoder**: linear projection + `Dropout` → reshape to `(256, 4, 4)` → 4× `ConvTranspose2d` (stride=2) + `BatchNorm2d` + `ReLU` + `Dropout2d` → `Tanh`.
- **Loss** (`vae_loss`): `MSE(recon, x)` + `β · KL(q(z|x) ∥ p(z))`. KL has a closed form for Gaussians.
- **KL Annealing** (`get_beta`): β rises linearly from 0 to `BETA_MAX` over `KL_WARMUP` epochs, then stays at `BETA_MAX`. Prevents posterior collapse in early training by letting the model focus on reconstruction first.

### Experiment tracking

Every training run is wrapped in `mlflow.start_run()`. Parameters, per-epoch metrics (`loss_total`, `loss_recon`, `loss_kl`, `beta`, `lr`), checkpoints, and output images are all logged automatically. Run data is stored in `mlruns/` (gitignored).

### Results

All output images are saved to `results/` (committed to the repo):

| File | Description |
|---|---|
| `results/training_history.png` | 4-panel plot: total loss, recon, KL, β schedule |
| `results/generated_samples.png` | 32 images sampled from z ~ N(0, I) |
| `results/reconstructions.png` | Original vs. reconstructed images |
| `results/interpolation.png` | Latent space interpolation (z₁ → z₂) |

### Key constants (defined in `model.py`)

| Constant | Default | Description |
|---|---|---|
| `LATENT_DIM` | 128 | Latent space dimensionality |
| `DROPOUT` | 0.2 | Dropout probability in encoder and decoder |
| `BETA_MAX` | 1.0 | Maximum KL weight after warmup |
| `KL_WARMUP` | 25 | Epochs to anneal β from 0 → BETA_MAX |

Training constants (`EPOCHS`, `LR`, `BATCH_SIZE`, `IMG_SIZE`) are defined at the top of `train.py` and mirrored in notebook cell 7.

## Roadmap

- [x] Convolutional VAE baseline
- [x] MLflow experiment tracking
- [x] Dropout regularization
- [x] KL annealing
- [ ] Data augmentation (RandomHorizontalFlip, ColorJitter)
- [ ] Latent space visualization (t-SNE / UMAP)
- [ ] Conditional VAE (class-guided generation)
- [ ] Perceptual loss

## Dataset

`archive/raw-img/` is gitignored and must be downloaded separately from [Kaggle — Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Class folders are named in Italian (`cane`, `gatto`, etc.); `archive/translate.py` has the Italian → English mapping. License: GNU GPL.
