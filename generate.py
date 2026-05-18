"""
Generates and visualizes images from a trained VAE.

Usage:
  python generate.py --checkpoint checkpoints/vae_final.pt --n 16
  python generate.py --checkpoint checkpoints/vae_final.pt --interpolate
  python generate.py --checkpoint checkpoints/vae_final.pt --history
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import VAE, LATENT_DIM


# ---------------------------------------------------------------------------
# Legacy model (no Dropout2d in conv blocks, plain Linear in decoder.fc).
# Used automatically when loading checkpoints saved before dropout was added.
# ---------------------------------------------------------------------------

class _LegacyEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.pre_latent = nn.Dropout(0.2)
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        h = self.pre_latent(h)
        return self.fc_mu(h), self.fc_logvar(h)


class _LegacyDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), 256, 4, 4))


class _LegacyVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = _LegacyEncoder(latent_dim)
        self.decoder = _LegacyDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def generate(self, n):
        z = torch.randn(n, self.encoder.fc_mu.out_features).to(next(self.parameters()).device)
        return self.decoder(z)


# ---------------------------------------------------------------------------

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model") or ckpt["model_state"]
    latent_dim = state["encoder.fc_mu.weight"].shape[0]

    # Detect architecture: old checkpoints have plain 'decoder.fc.weight'
    # (not wrapped in Sequential), new ones have 'decoder.fc.0.weight'.
    if "decoder.fc.weight" in state:
        model = _LegacyVAE(latent_dim).to(device)
    else:
        model = VAE(latent_dim=latent_dim).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get("history", {})


def plot_samples(model, device, n=16):
    imgs = model.generate(n).cpu()
    imgs = denormalize(imgs)
    grid_size = int(n ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle("Generated samples from VAE", fontsize=14)
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=150)
    plt.close()
    print("Saved generated_samples.png")


def plot_interpolation(model, device, steps=10):
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
    plt.close()
    print("Saved interpolation.png")


def plot_loss_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(history["total"]);  axes[0].set_title("Total loss")
    axes[1].plot(history["recon"]);  axes[1].set_title("Reconstruction loss")
    axes[2].plot(history["kl"]);     axes[2].set_title("KL divergence")
    if "beta" in history:
        axes[3].plot(history["beta"])
    axes[3].set_title("β schedule")
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()
    print("Saved training_history.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt")
    parser.add_argument("--n", type=int, default=16)
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
