"""
Generates and visualizes images from a trained VAE.

Usage:
  python generate.py --checkpoint checkpoints/vae_final.pt --n 16
  python generate.py --checkpoint checkpoints/vae_final.pt --interpolate
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import VAE, LATENT_DIM


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = VAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("history", {})


def plot_samples(model, device, n=16):
    imgs = model.generate(n, device).cpu()
    imgs = denormalize(imgs)
    grid_size = int(n ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle("Generated samples from VAE", fontsize=14)
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=150)
    plt.show()
    print("Saved generated_samples.png")


def plot_interpolation(model, device, steps=10):
    """Linearly interpolate between two random points in latent space."""
    z1 = torch.randn(1, LATENT_DIM).to(device)
    z2 = torch.randn(1, LATENT_DIM).to(device)
    alphas = torch.linspace(0, 1, steps).to(device)
    zs = torch.stack([z1 * (1 - a) + z2 * a for a in alphas]).squeeze(1)
    with torch.no_grad():
        imgs = model.decoder(zs).cpu()
    imgs = denormalize(imgs)
    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle("Latent space interpolation", fontsize=14)
    plt.tight_layout()
    plt.savefig("interpolation.png", dpi=150)
    plt.show()
    print("Saved interpolation.png")


def plot_loss_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history["total"], label="Total loss")
    plt.plot(history["recon"], label="Reconstruction loss")
    plt.plot(history["kl"],    label="KL divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved training_history.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt")
    parser.add_argument("--n", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, history = load_model(args.checkpoint, device)

    if args.history and history:
        plot_loss_history(history)

    if args.interpolate:
        plot_interpolation(model, device)
    else:
        plot_samples(model, device, n=args.n)


if __name__ == "__main__":
    main()
