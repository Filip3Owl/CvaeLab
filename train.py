import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloader
from model import VAE, vae_loss

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "archive/raw-img"
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
LATENT_DIM  = 128
BETA        = 1.0          # weight on KL term (beta-VAE: try 2–4 for disentanglement)
SAVE_EVERY  = 5            # save checkpoint every N epochs
CHECKPOINT  = "checkpoints"
# ────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(CHECKPOINT, exist_ok=True)

    loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Dataset size: {len(loader.dataset)} images")

    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    history = {"total": [], "recon": [], "kl": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = recon_sum = kl_sum = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_loss.item()
            kl_sum     += kl_loss.item()

        n = len(loader)
        avg_total = total_loss / n
        avg_recon = recon_sum  / n
        avg_kl    = kl_sum     / n

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        scheduler.step(avg_total)
        print(f"Epoch {epoch:3d} | loss={avg_total:.2f}  recon={avg_recon:.2f}  kl={avg_kl:.2f}")

        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CHECKPOINT, f"vae_epoch{epoch:03d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "history": history}, path)
            print(f"  Saved {path}")

    # final checkpoint
    torch.save({"epoch": EPOCHS, "model": model.state_dict(), "history": history},
               os.path.join(CHECKPOINT, "vae_final.pt"))
    print("Training complete. Final model saved to checkpoints/vae_final.pt")


if __name__ == "__main__":
    main()
