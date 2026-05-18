import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloader
from model import VAE, vae_loss, get_beta

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "archive/raw-img"
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
LATENT_DIM = 128
DROPOUT    = 0.2
BETA_MAX   = 1.0   # maximum KL weight
KL_WARMUP  = 25    # epochs to anneal β from 0 → BETA_MAX
SAVE_EVERY = 5
CKPT_DIR   = "checkpoints"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Dataset size: {len(loader.dataset)} images")

    model     = VAE(latent_dim=LATENT_DIM, dropout=DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"total": [], "recon": [], "kl": [], "beta": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_sum = recon_sum = kl_sum = 0.0

        # Compute annealed β for this epoch
        beta = get_beta(epoch, warmup=KL_WARMUP, beta_max=BETA_MAX)

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_sum += loss.item()
            recon_sum += recon_loss.item()
            kl_sum    += kl_loss.item()

        n = len(loader)
        avg_total = total_sum / n
        avg_recon = recon_sum / n
        avg_kl    = kl_sum    / n

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        history["beta"].append(beta)

        scheduler.step(avg_total)
        print(f"Epoch {epoch:3d}/{EPOCHS} | β={beta:.2f} | "
              f"loss={avg_total:.2f}  recon={avg_recon:.2f}  kl={avg_kl:.2f}")

        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"vae_epoch{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "history": history}, path)
            print(f"  Saved {path}")

    torch.save({"epoch": EPOCHS, "model_state": model.state_dict(), "history": history},
               os.path.join(CKPT_DIR, "vae_final.pt"))
    print("Training complete. Final model saved to checkpoints/vae_final.pt")


if __name__ == "__main__":
    main()
