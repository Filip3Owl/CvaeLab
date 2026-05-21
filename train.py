import argparse
import os
import torch
import mlflow
import mlflow.pytorch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloader
from model import VAE, vae_loss, get_beta

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR        = "archive/raw-img"
BATCH_SIZE      = 64
EPOCHS          = 50
LR              = 1e-3
LATENT_DIM      = 128
DROPOUT         = 0.2
BETA_MAX        = 1.0   # maximum KL weight
KL_WARMUP       = 25    # epochs to anneal β from 0 → BETA_MAX
AUGMENT         = True  # data augmentation (flip + color jitter)
SAVE_EVERY      = 5
CKPT_DIR        = "checkpoints"
EXPERIMENT_NAME = "cvae-animals10"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a vae_epochXXX.pt checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE, augment=AUGMENT)
    print(f"Dataset size: {len(loader.dataset)} images")

    mlflow.set_experiment(EXPERIMENT_NAME)

    model     = VAE(latent_dim=LATENT_DIM, dropout=DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history     = {"total": [], "recon": [], "kl": [], "beta": []}
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history     = ckpt.get("history", history)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} — continuing from epoch {start_epoch}")

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params({
            "epochs":      EPOCHS,
            "lr":          LR,
            "beta_max":    BETA_MAX,
            "kl_warmup":   KL_WARMUP,
            "dropout":     DROPOUT,
            "latent_dim":  LATENT_DIM,
            "batch_size":  BATCH_SIZE,
            "augment":     AUGMENT,
            "optimizer":   "Adam",
            "device":      str(device),
        })

        for epoch in range(start_epoch, EPOCHS + 1):
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

            mlflow.log_metrics({
                "loss_total": avg_total,
                "loss_recon": avg_recon,
                "loss_kl":    avg_kl,
                "beta":       beta,
                "lr":         optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(f"Epoch {epoch:3d}/{EPOCHS} | β={beta:.2f} | "
                  f"loss={avg_total:.2f}  recon={avg_recon:.2f}  kl={avg_kl:.2f}")

            if epoch % SAVE_EVERY == 0:
                path = os.path.join(CKPT_DIR, f"vae_epoch{epoch:03d}.pt")
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "history": history}, path)
                mlflow.log_artifact(path, artifact_path="checkpoints")
                print(f"  Checkpoint saved & logged: {path}")

        final_path = os.path.join(CKPT_DIR, "vae_final.pt")
        torch.save({"epoch": EPOCHS, "model_state": model.state_dict(),
                    "history": history}, final_path)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(final_path, artifact_path="checkpoints")

        print(f"\nTraining complete. Run ID: {run.info.run_id}")
        print(f"View results: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    main()
