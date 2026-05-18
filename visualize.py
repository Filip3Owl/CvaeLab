"""
Latent space visualization via t-SNE (or UMAP if umap-learn is installed).

Usage:
  python visualize.py --checkpoint checkpoints/vae_final.pt
  python visualize.py --checkpoint checkpoints/vae_final.pt --samples 300 --method umap
"""

import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from generate import load_model

TRANSLATE = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant",
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
    "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider",
}

_eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def collect_samples(data_dir, samples_per_class):
    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    samples, class_names = [], []
    for idx, cls in enumerate(classes):
        class_names.append(TRANSLATE.get(cls, cls))
        cls_dir = os.path.join(data_dir, cls)
        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for f in random.sample(files, min(samples_per_class, len(files))):
            samples.append((os.path.join(cls_dir, f), idx))
    return samples, class_names


def encode(model, samples, device, batch_size=128):
    model.eval()
    mus, labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([
            _eval_transform(Image.open(p).convert("RGB"))
            for p, _ in batch
        ]).to(device)
        with torch.no_grad():
            mu, _ = model.encoder(imgs)
        mus.append(mu.cpu().numpy())
        labels.extend(idx for _, idx in batch)
    return np.concatenate(mus), np.array(labels)


def reduce_tsne(mus):
    from sklearn.manifold import TSNE
    print("Running t-SNE (this may take a minute)...")
    return TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000).fit_transform(mus)


def reduce_umap(mus):
    import umap
    print("Running UMAP...")
    return umap.UMAP(n_components=2, random_state=42).fit_transform(mus)


def plot(z2d, labels, class_names, method, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels == idx
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=[color], label=name,
                   alpha=0.6, s=10, linewidths=0)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=10)
    ax.set_title(f"Latent space — {method.upper()} projection")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt")
    parser.add_argument("--data",       default="archive/raw-img")
    parser.add_argument("--samples",    type=int, default=200,
                        help="Images sampled per class (default: 200)")
    parser.add_argument("--method",     choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--output",     default="latent_space.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _ = load_model(args.checkpoint, device)

    print(f"Collecting {args.samples} samples per class from {args.data}...")
    samples, class_names = collect_samples(args.data, args.samples)
    print(f"  {len(samples)} images across {len(class_names)} classes")

    print("Encoding images...")
    mus, labels = encode(model, samples, device)

    z2d = reduce_umap(mus) if args.method == "umap" else reduce_tsne(mus)
    plot(z2d, labels, class_names, args.method, args.output)


if __name__ == "__main__":
    main()
