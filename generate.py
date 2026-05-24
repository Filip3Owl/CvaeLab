"""
Gera e visualiza imagens a partir de um VAE ou cVAE treinado.

Uso (VAE):
  python generate.py --checkpoint checkpoints/vae_final.pt --n 16
  python generate.py --checkpoint checkpoints/vae_final.pt --interpolate
  python generate.py --checkpoint checkpoints/vae_final.pt --history

Uso (cVAE):
  python generate.py --checkpoint checkpoints/cvae_final.pt --class dog
  python generate.py --checkpoint checkpoints/cvae_final.pt --class dog --n 32
  python generate.py --checkpoint checkpoints/cvae_final.pt            # 2 por classe
  python generate.py --checkpoint checkpoints/cvae_final.pt --interpolate --class dog
  python generate.py --checkpoint checkpoints/cvae_final.pt --history
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import VAE, CVAE, LATENT_DIM
from dataset import IDX_TO_CLASS, CLASS_TO_IDX


# ── VAE legado (blocos conv sem Dropout2d) ───────────────────────────────────
# Carregado automaticamente quando um checkpoint não possui Dropout2d nos blocos conv.

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


# ── Auxiliares ───────────────────────────────────────────────────────────────

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def resolve_class(class_arg: str) -> int:
    """Converte um nome de classe em inglês ou string numérica para índice de classe."""
    try:
        idx = int(class_arg)
        if idx not in IDX_TO_CLASS:
            raise ValueError(f"Índice {idx} fora do intervalo (0–{len(IDX_TO_CLASS) - 1})")
        return idx
    except ValueError:
        pass

    name_to_idx = {v: k for k, v in IDX_TO_CLASS.items()}
    if class_arg.lower() in name_to_idx:
        return name_to_idx[class_arg.lower()]

    # Também aceita nomes das pastas em italiano
    if class_arg.lower() in CLASS_TO_IDX:
        return CLASS_TO_IDX[class_arg.lower()]

    valid = ", ".join(sorted(name_to_idx))
    raise ValueError(f"Classe desconhecida '{class_arg}'. Nomes válidos: {valid}")


def load_model(checkpoint_path: str, device):
    """Carrega VAE ou cVAE a partir do checkpoint, retornando (model, history, is_cvae)."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model") or ckpt["model_state"]
    latent_dim = state["encoder.fc_mu.weight"].shape[0]

    is_cvae = "encoder.embedding.weight" in state

    if is_cvae:
        num_classes = state["encoder.embedding.weight"].shape[0]
        embed_dim   = state["encoder.embedding.weight"].shape[1]
        model = CVAE(latent_dim=latent_dim, num_classes=num_classes,
                     embed_dim=embed_dim).to(device)
    elif "decoder.fc.weight" in state:
        model = _LegacyVAE(latent_dim).to(device)
    else:
        model = VAE(latent_dim=latent_dim).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get("history", {}), is_cvae


# ── Funções de plotagem — VAE ─────────────────────────────────────────────────

def plot_samples(model, device, n: int = 16):
    imgs = denormalize(model.generate(n).cpu())
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle("VAE — amostras geradas (z ~ N(0, I))", fontsize=13)
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=150)
    plt.close()
    print("Salvo em generated_samples.png")


def plot_interpolation(model, device, steps: int = 10):
    latent_dim = model.encoder.fc_mu.out_features
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack([z1 * (1 - a) + z2 * a for a in alphas]).squeeze(1)
    with torch.no_grad():
        imgs = denormalize(model.decoder(zs).cpu())
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.set_title(f"α={alphas[i].item():.1f}", fontsize=8)
        ax.axis("off")
    plt.suptitle("VAE — interpolação no espaço latente (z₁ → z₂)", fontsize=12)
    plt.tight_layout()
    plt.savefig("interpolation.png", dpi=150)
    plt.close()
    print("Salvo em interpolation.png")


# ── Funções de plotagem — cVAE ────────────────────────────────────────────────

def plot_samples_conditional(model, device, class_idx: int, class_name: str, n: int = 16):
    """Gera n imagens de uma única classe."""
    y = torch.full((n,), class_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        imgs = denormalize(model.generate(y).cpu())
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle(f"cVAE — amostras geradas  (classe: {class_name})", fontsize=13)
    plt.tight_layout()
    fname = f"generated_{class_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Salvo em {fname}")


def plot_samples_all_classes(model, device, per_class: int = 2):
    """Gera per_class imagens para cada classe e exibe como grade."""
    num_classes = len(IDX_TO_CLASS)
    fig, axes = plt.subplots(num_classes, per_class,
                             figsize=(per_class * 2, num_classes * 2))
    for class_idx in range(num_classes):
        y = torch.full((per_class,), class_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            imgs = denormalize(model.generate(y).cpu())
        for c in range(per_class):
            ax = axes[class_idx, c] if per_class > 1 else axes[class_idx]
            ax.imshow(imgs[c].permute(1, 2, 0).numpy())
            ax.axis("off")
        axes[class_idx, 0].set_ylabel(IDX_TO_CLASS[class_idx],
                                      fontsize=9, rotation=0,
                                      labelpad=50, va="center")
    plt.suptitle("cVAE — geração condicional (todas as classes)", fontsize=13)
    plt.tight_layout()
    plt.savefig("generated_all_classes.png", dpi=150)
    plt.close()
    print("Salvo em generated_all_classes.png")


def plot_interpolation_conditional(model, device, class_idx: int,
                                   class_name: str, steps: int = 10):
    """Interpola entre dois vetores latentes mantendo o rótulo de classe fixo."""
    latent_dim = model.encoder.fc_mu.out_features
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device)
    zs = torch.stack([z1 * (1 - a) + z2 * a for a in alphas]).squeeze(1)
    y  = torch.full((steps,), class_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        imgs = denormalize(model.decoder(zs, y).cpu())
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.set_title(f"α={alphas[i].item():.1f}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"cVAE — interpolação  (classe: {class_name})", fontsize=12)
    plt.tight_layout()
    fname = f"interpolation_{class_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Salvo em {fname}")


# ── Histórico de perda ────────────────────────────────────────────────────────

def plot_loss_history(history: dict):
    beta_data = history.get("beta", [])
    n_panels  = 4 if beta_data else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 4))

    panels = [
        ("total", "Perda total",           "royalblue"),
        ("recon", "Perda de reconstrução", "darkorange"),
        ("kl",    "Divergência KL",        "seagreen"),
    ]
    if beta_data:
        panels.append(("beta", "Escalonamento de β", "mediumpurple"))

    for ax, (key, title, color) in zip(axes, panels):
        data   = history.get(key, [])
        epochs = range(1, len(data) + 1)
        ax.plot(epochs, data, color=color, linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo em training_history.png")


# ── Principal ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gera imagens a partir de um VAE ou cVAE treinado."
    )
    parser.add_argument("--checkpoint", default="checkpoints/vae_final.pt",
                        help="Caminho para o checkpoint do modelo")
    parser.add_argument("--n", type=int, default=16,
                        help="Número de imagens a gerar")
    parser.add_argument("--interpolate", action="store_true",
                        help="Gera uma interpolação no espaço latente")
    parser.add_argument("--history", action="store_true",
                        help="Plota as curvas de perda do treinamento")
    parser.add_argument("--class", dest="cls", default=None,
                        help="Nome ou índice da classe para geração condicional (somente cVAE)")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, history, is_cvae = load_model(args.checkpoint, device)

    model_type = "cVAE" if is_cvae else "VAE"
    print(f"Checkpoint {model_type} carregado: {args.checkpoint}")

    if args.history and history:
        plot_loss_history(history)

    if args.interpolate:
        if is_cvae:
            class_idx  = resolve_class(args.cls) if args.cls else 0
            class_name = IDX_TO_CLASS[class_idx]
            print(f"Interpolando com classe: {class_name} (idx={class_idx})")
            plot_interpolation_conditional(model, device, class_idx, class_name)
        else:
            plot_interpolation(model, device)
    else:
        if is_cvae:
            if args.cls:
                class_idx  = resolve_class(args.cls)
                class_name = IDX_TO_CLASS[class_idx]
                print(f"Gerando {args.n} imagens — classe: {class_name} (idx={class_idx})")
                plot_samples_conditional(model, device, class_idx, class_name, n=args.n)
            else:
                print("--class não especificado — gerando 2 amostras por classe.")
                plot_samples_all_classes(model, device)
        else:
            plot_samples(model, device, n=args.n)


if __name__ == "__main__":
    main()
