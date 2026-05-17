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
```

Images are resized to 64×64 and normalized to [-1, 1] to match the decoder's `Tanh` output.

### Model (`model.py`)

Three classes: `Encoder`, `Decoder`, `VAE`.

- **Encoder**: 4× `Conv2d` (stride=2, halving resolution each time) → flatten → two linear heads producing `μ` and `log σ²` (both `LATENT_DIM=128`).
- **Reparameterization**: `z = μ + σ·ε`, `ε ~ N(0,I)`. In `eval()` mode returns `μ` directly (no noise).
- **Decoder**: linear projection → reshape to `(256, 4, 4)` → 4× `ConvTranspose2d` (stride=2, doubling resolution) → `Tanh`.
- **Loss** (`vae_loss`): `MSE(recon, x)` + `β · KL(q(z|x) ∥ p(z))`. KL has a closed form for Gaussians. `β=1` is a standard VAE; increasing `β` trades reconstruction quality for disentanglement.

### Experiment tracking

Every training run is wrapped in `mlflow.start_run()`. Parameters, per-epoch metrics (`loss_total`, `loss_recon`, `loss_kl`, `lr`), checkpoints, and output images are all logged automatically. Run data is stored in `mlruns/` (gitignored).

### Key constants

Defined at the top of each file and mirrored in the notebook:

| Constant | Default | Where |
|---|---|---|
| `LATENT_DIM` | 128 | `model.py`, notebook cell 4 |
| `IMG_SIZE` | 64 | `dataset.py`, notebook cell 3 |
| `BATCH_SIZE` | 64 | `dataset.py`, notebook cell 3 |
| `BETA` | 1.0 | `train.py`, notebook cell 7 |

## Dataset

`archive/raw-img/` is gitignored and must be downloaded separately from [Kaggle — Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Class folders are named in Italian (`cane`, `gatto`, etc.); `translate.py` in `archive/` has the mapping.
