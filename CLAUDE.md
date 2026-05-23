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

**Train VAE from the terminal:**
```bash
python train.py
```

**Train cVAE (class-conditional):**
```bash
python train_cvae.py
```

**Generate images from a saved checkpoint:**
```bash
python generate.py --checkpoint checkpoints/vae_final.pt --n 16
python generate.py --checkpoint checkpoints/vae_final.pt --interpolate
python generate.py --checkpoint checkpoints/vae_final.pt --history
```

**Visualize the latent space (t-SNE):**
```bash
python visualize.py --checkpoint checkpoints/vae_final.pt
python visualize.py --checkpoint checkpoints/vae_final.pt --samples 300 --method umap
```

**Open MLflow UI to compare experiment runs:**
```bash
mlflow ui   # → http://localhost:5000
```

## Architecture

The project trains a **convolutional Variational Autoencoder (VAE)** on the Animals-10 dataset (~27k images, 10 classes), evolving towards a **Conditional VAE (cVAE)** for class-guided generation.

### Data flow

```
archive/raw-img/<class>/*.jpg  →  AnimalsDataset  →  DataLoader  →  VAE / CVAE  →  checkpoints/
                                  (image, label)                                     results/
```

Images are resized to 64×64 and normalized to [-1, 1] to match the decoder's `Tanh` output.

### Dataset module (`dataset.py`)

Exports: `AnimalsDataset`, `CLASS_TO_IDX`, `IDX_TO_CLASS`, `NUM_CLASSES`, `LABEL_MAP`, `train_transform`, `eval_transform`, `get_dataloader`.

- **`AnimalsDataset`**: returns `(image_tensor, class_index)` pairs. Walks `root_dir` subdirectories, filtering by Italian folder names in `CLASS_TO_IDX`.
- **`CLASS_TO_IDX`**: deterministic mapping from Italian folder name → integer index (sorted alphabetically: `cane=0 … scoiattolo=9`).
- **`IDX_TO_CLASS`**: reverse mapping from integer index → English label.
- **`LABEL_MAP`**: Italian folder name → English label.
- **`get_dataloader`**: convenience wrapper that builds a `DataLoader` with optional augmentation.

### Model (`model.py`)

VAE classes: `Encoder`, `Decoder`, `VAE`. cVAE classes: `ConditionalEncoder`, `ConditionalDecoder`, `CVAE`. Shared utilities: `vae_loss`, `get_beta`.

**VAE:**
- **Encoder**: 4× `Conv2d` (stride=2) + `BatchNorm2d` + `LeakyReLU` + `Dropout2d` → flatten → `Dropout` → two linear heads producing `μ` and `log σ²` (both `LATENT_DIM=128`).
- **Reparameterization**: `z = μ + σ·ε`, `ε ~ N(0,I)`. In `eval()` mode returns `μ` directly (no noise).
- **Decoder**: linear projection + `Dropout` → reshape to `(256, 4, 4)` → 4× `ConvTranspose2d` (stride=2) + `BatchNorm2d` + `ReLU` + `Dropout2d` → `Tanh`.

**cVAE:**
- **ConditionalEncoder**: same conv backbone as `Encoder`. After flattening, concatenates `embed(y)` (`EMBED_DIM=64`) to the conv features before the μ / log σ² heads.
- **ConditionalDecoder**: concatenates `embed(y)` to `z` before the FC projection. Same deconv backbone as `Decoder`.
- **CVAE**: combines both. `CVAE.generate(y)` samples `z ~ N(0, I)` and decodes with the target class label.

**Shared utilities:**
- **Loss** (`vae_loss`): `MSE(recon, x)` + `β · KL(q(z|x) ∥ p(z))`. KL has a closed form for Gaussians.
- **KL Annealing** (`get_beta`): β rises linearly from 0 to `BETA_MAX` over `KL_WARMUP` epochs, then stays at `BETA_MAX`. Prevents posterior collapse in early training by letting the model focus on reconstruction first.

### Experiment tracking

Every training run is wrapped in `mlflow.start_run()`. Parameters, per-epoch metrics (`loss_total`, `loss_recon`, `loss_kl`, `beta`, `lr`), checkpoints, and output images are all logged automatically. IS and FID scores are logged to the same run after evaluation. Run data is stored in `mlruns/` (gitignored).

### Evaluation metrics

IS and FID are computed in notebook section 12 using `torchmetrics[image]`:

| Metric | Measures | Direction |
|---|---|---|
| **Inception Score (IS)** | Quality + diversity of generated images via Inception-v3 | Higher = better |
| **Fréchet Inception Distance (FID)** | Distance between real and generated distributions in Inception-v3 feature space | Lower = better |

Both metrics upsample 64×64 → 299×299 for Inception-v3. Use them to compare VAE vs. cVAE runs, not as absolute quality targets.

### Results

Output images are organized by run under `results/run_vN/` (committed to the repo):

| File | Description |
|---|---|
| `results/run_vN/training_history.png` | 4-panel plot: total loss, recon, KL, β schedule |
| `results/run_vN/generated_samples.png` | 32 images sampled from z ~ N(0, I) |
| `results/run_vN/reconstructions.png` | Original vs. reconstructed images |
| `results/run_vN/interpolation.png` | Latent space interpolation (z₁ → z₂) |
| `results/run_vN/latent_space.png` | t-SNE / UMAP projection of the latent space, colored by class |

After each run, move the images from the project root to `results/run_vN/` and add the new folder to `.gitignore` exceptions.

#### Run history

| Run | Model | Notes |
|---|---|---|
| `run_v1` | VAE | Baseline — no dropout, no KL annealing |
| `run_v2` | VAE | Added dropout + data augmentation |
| `run_v3` | VAE | Added KL annealing (`KL_WARMUP=25`) + t-SNE visualization |
| `run_v4` | VAE | KL collapsed to ~0 (no annealing in notebook training cell); 3-panel history plot (before fix) |

> **KL posterior collapse**: run_v4 shows KL → 0, meaning the encoder ignores the input and the latent space is unused. Always use `get_beta` / `KL_WARMUP` to anneal β from 0 → 1 during training.

#### macOS SSL note (IS / FID cell)

Inception-v3 is downloaded on first run. On macOS with Homebrew Python the SSL handshake fails. A patch cell is already inserted before the metrics cell in the notebook:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Key constants (defined in `model.py`)

| Constant | Default | Description |
|---|---|---|
| `LATENT_DIM` | 128 | Latent space dimensionality |
| `DROPOUT` | 0.2 | Dropout probability in encoder and decoder |
| `BETA_MAX` | 1.0 | Maximum KL weight after warmup |
| `KL_WARMUP` | 25 | Epochs to anneal β from 0 → BETA_MAX |
| `NUM_CLASSES` | 10 | Number of animal classes |
| `EMBED_DIM` | 64 | Class embedding dimensionality in cVAE |

Training constants (`EPOCHS`, `LR`, `BATCH_SIZE`, `IMG_SIZE`, `AUGMENT`) are defined at the top of `train.py` and mirrored in notebook cells 3 and 7.

## Roadmap

- [x] Convolutional VAE baseline
- [x] MLflow experiment tracking
- [x] Dropout regularization
- [x] KL annealing
- [x] Data augmentation (RandomHorizontalFlip, ColorJitter)
- [x] Latent space visualization (t-SNE / UMAP)
- [x] IS & FID evaluation metrics (notebook section 12)
- [ ] Conditional VAE (class-guided generation)
  - [x] Etapa 1 — Dataset com labels (`dataset.py`)
  - [x] Etapa 2 — Arquitetura cVAE (`model.py`)
  - [x] Etapa 3 — Script de treino (`train_cvae.py`)
  - [ ] Etapa 4 — Geração condicional (`generate.py --class dog`)
  - [ ] Etapa 5 — Visualização t-SNE com clusters separados
- [ ] Perceptual loss

## Dataset

`archive/raw-img/` is gitignored and must be downloaded separately from [Kaggle — Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10). Class folders are named in Italian (`cane`, `gatto`, etc.); `archive/translate.py` has the Italian → English mapping. License: GNU GPL.
