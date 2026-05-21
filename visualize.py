"""
Latent space visualization via t-SNE (or UMAP if umap-learn is installed).
Works with both VAE and cVAE checkpoints — detected automatically.

Usage:
  python visualize.py --checkpoint checkpoints/vae_final.pt
  python visualize.py --checkpoint checkpoints/cvae_final.pt
  python visualize.py --checkpoint checkpoints/vae_final.pt --samples 300 --method umap
"""

import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import CLASS_TO_IDX, IDX_TO_CLASS, eval_transform
from generate import load_model


def collect_samples(data_dir: str, samples_per_class: int):
    """Return (path, class_idx) pairs and the ordered list of class names."""
    samples, class_names = [], []
    for cls, idx in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1]):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(samples_per_class, len(files)))
        for f in chosen:
            samples.append((os.path.join(cls_dir, f), idx))
        class_names.append(IDX_TO_CLASS[idx])
    return samples, class_names


def encode(model, samples: list, device, is_cvae: bool, batch_size: int = 128):
    """Encode images to latent μ vectors. Passes labels to encoder for cVAE."""
    model.eval()
    mus, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            eval_transform(Image.open(p).convert("RGB"))
            for p, _ in batch
        ]).to(device)
        batch_labels = torch.tensor([idx for _, idx in batch],
                                    dtype=torch.long, device=device)
        with torch.no_grad():
            if is_cvae:
                mu, _ = model.encoder(imgs, batch_labels)
            else:
                mu, _ = model.encoder(imgs)
        mus.append(mu.cpu().numpy())
        labels.extend(idx for _, idx in batch)
    return np.concatenate(mus), np.array(labels)


def reduce_tsne(mus: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    print("Running t-SNE (this may take a minute)...")
    return TSNE(n_components=2, perplexity=40, random_state=42,
                max_iter=1000).fit_transform(mus)


def reduce_umap(mus: np.ndarray) -> np.ndarray:
    import umap
    print("Running UMAP...")
    return umap.UMAP(n_components=2, random_state=42).fit_transform(mus)


def plot(z2d: np.ndarray, labels: np.ndarray, class_names: list,
         method: str, model_type: str, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == idx
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=[color], label=name,
                   alpha=0.6, s=10, linewidths=0)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=10)
    ax.set_title(f"Latent space — {method.upper()} projection  ({model_type})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the latent space of a VAE or cVAE checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt")
    parser.add_argument("--data",    default="archive/raw-img")
    parser.add_argument("--samples", type=int, default=200,
                        help="Images sampled per class (default: 200)")
    parser.add_argument("--method",  choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--output",  default=None,
                        help="Output PNG path (auto-named by model type if omitted)")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _, is_cvae = load_model(args.checkpoint, device)
    model_type = "cVAE" if is_cvae else "VAE"
    print(f"Loaded {model_type} checkpoint: {args.checkpoint}")

    output_path = args.output or f"latent_space_{model_type.lower()}.png"

    print(f"Collecting {args.samples} samples per class from {args.data}...")
    samples, class_names = collect_samples(args.data, args.samples)
    print(f"  {len(samples)} images across {len(class_names)} classes")

    print("Encoding images...")
    mus, labels = encode(model, samples, device, is_cvae)

    z2d = reduce_umap(mus) if args.method == "umap" else reduce_tsne(mus)
    plot(z2d, labels, class_names, args.method, model_type, output_path)


if __name__ == "__main__":
    main()
