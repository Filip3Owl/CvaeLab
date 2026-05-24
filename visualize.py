"""
Visualização do espaço latente via t-SNE (ou UMAP se umap-learn estiver instalado).
Funciona com checkpoints de VAE e cVAE — detectado automaticamente.

Uso:
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
    """Retorna pares (caminho, índice_de_classe) e a lista ordenada de nomes de classes."""
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
    """Codifica imagens para vetores μ latentes. Passa rótulos ao encoder para cVAE."""
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
    print("Executando t-SNE (pode levar alguns minutos)...")
    return TSNE(n_components=2, perplexity=40, random_state=42,
                max_iter=1000).fit_transform(mus)


def reduce_umap(mus: np.ndarray) -> np.ndarray:
    import umap
    print("Executando UMAP...")
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
    ax.set_title(f"Espaço latente — projeção {method.upper()}  ({model_type})")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Salvo em {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza o espaço latente de um checkpoint de VAE ou cVAE."
    )
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt")
    parser.add_argument("--data",    default="archive/raw-img")
    parser.add_argument("--samples", type=int, default=200,
                        help="Imagens amostradas por classe (padrão: 200)")
    parser.add_argument("--method",  choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--output",  default=None,
                        help="Caminho PNG de saída (nomeado automaticamente por tipo de modelo se omitido)")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _, is_cvae = load_model(args.checkpoint, device)
    model_type = "cVAE" if is_cvae else "VAE"
    print(f"Checkpoint {model_type} carregado: {args.checkpoint}")

    output_path = args.output or f"latent_space_{model_type.lower()}.png"

    print(f"Coletando {args.samples} amostras por classe de {args.data}...")
    samples, class_names = collect_samples(args.data, args.samples)
    print(f"  {len(samples)} imagens em {len(class_names)} classes")

    print("Codificando imagens...")
    mus, labels = encode(model, samples, device, is_cvae)

    z2d = reduce_umap(mus) if args.method == "umap" else reduce_tsne(mus)
    plot(z2d, labels, class_names, args.method, model_type, output_path)


if __name__ == "__main__":
    main()
