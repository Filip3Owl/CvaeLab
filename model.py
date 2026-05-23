import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM  = 128
DROPOUT     = 0.2
BETA_MAX    = 1.0
KL_WARMUP   = 25
NORM        = "batch"   # "batch" | "group"
GROUP_SIZE  = 8         # channels per group when NORM="group"


def _norm(norm: str, num_channels: int) -> nn.Module:
    """Return BatchNorm2d or GroupNorm depending on `norm`."""
    if norm == "group":
        return nn.GroupNorm(num_channels // GROUP_SIZE, num_channels)
    return nn.BatchNorm2d(num_channels)


class Encoder(nn.Module):
    """
    Convolutional encoder: maps a 3×64×64 image to the parameters
    (μ, log σ²) of a Gaussian distribution in latent space.

    Each conv block uses stride=2 to halve the spatial resolution
    (learnable downsampling, equivalent to Conv + MaxPool).
    Dropout2d drops entire feature maps to regularize spatial features.
    A final Dropout is applied before the linear heads to regularize
    the transition from spatial to latent representation.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = DROPOUT, norm: str = NORM):
        super().__init__()
        self.conv = nn.Sequential(
            # 3 × 64 × 64  ->  32 × 32 × 32
            nn.Conv2d(3,   32,  4, stride=2, padding=1),
            _norm(norm, 32),
            nn.LeakyReLU(0.2),

            # 32 × 32 × 32  ->  64 × 16 × 16
            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            _norm(norm, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),

            # 64 × 16 × 16  ->  128 × 8 × 8
            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            _norm(norm, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),

            # 128 × 8 × 8  ->  256 × 4 × 4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            _norm(norm, 256),
            nn.LeakyReLU(0.2),
        )
        flat = 256 * 4 * 4
        self.pre_latent = nn.Dropout(dropout)
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).view(x.size(0), -1)  # flatten: (B, 256*4*4)
        h = self.pre_latent(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Convolutional decoder: maps a latent vector z ∈ ℝ^{latent_dim}
    back to a 3×64×64 image in the range [-1, 1].

    ConvTranspose2d with stride=2 doubles the spatial resolution at
    each step — the mirror image of the encoder.
    Dropout is applied after the initial FC projection to prevent
    the decoder from over-relying on specific latent dimensions.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = DROPOUT, norm: str = NORM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Dropout(dropout),
        )
        self.deconv = nn.Sequential(
            # 256 × 4 × 4  ->  128 × 8 × 8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            _norm(norm, 128),
            nn.ReLU(),

            # 128 × 8 × 8  ->  64 × 16 × 16
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            _norm(norm, 64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            # 64 × 16 × 16  ->  32 × 32 × 32
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            _norm(norm, 32),
            nn.ReLU(),

            # 32 × 32 × 32  ->  3 × 64 × 64
            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 256, 4, 4)  # reshape: (B, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Full Variational Autoencoder.

    `reparameterize` implements the reparameterization trick:
        z = μ + σ·ε,  ε ~ N(0, I)
    This keeps gradients flowing through the encoder during backprop
    because ε is sampled independently from the model parameters.
    During eval mode, returns μ directly (no noise).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = DROPOUT, norm: str = NORM):
        super().__init__()
        self.encoder = Encoder(latent_dim, dropout, norm)
        self.decoder = Decoder(latent_dim, dropout, norm)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)  # σ = exp(0.5 · log σ²)
            eps = torch.randn_like(std)     # ε ~ N(0, I)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, n: int) -> torch.Tensor:
        """Generate `n` images by sampling z ~ N(0, I) directly."""
        z = torch.randn(n, self.encoder.fc_mu.out_features).to(
            next(self.parameters()).device
        )
        return self.decoder(z)


NUM_CLASSES = 10
EMBED_DIM   = 64


class ConditionalEncoder(nn.Module):
    """
    Conditional encoder: maps (image, class label) → (μ, log σ²).

    The class label is projected to an embedding vector and concatenated
    with the flattened conv features before the linear heads.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
        norm: str = NORM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),
            _norm(norm, 32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,  64,  4, stride=2, padding=1),
            _norm(norm, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),

            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            _norm(norm, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            _norm(norm, 256),
            nn.LeakyReLU(0.2),
        )
        flat = 256 * 4 * 4
        self.pre_latent = nn.Dropout(dropout)
        self.fc_mu     = nn.Linear(flat + embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat + embed_dim, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        h = self.conv(x).view(x.size(0), -1)
        h = self.pre_latent(h)
        c = self.embedding(y)                   # (B, embed_dim)
        h = torch.cat([h, c], dim=1)            # (B, flat + embed_dim)
        return self.fc_mu(h), self.fc_logvar(h)


class ConditionalDecoder(nn.Module):
    """
    Conditional decoder: maps (z, class label) → 3×64×64 image.

    The class embedding is concatenated to z before the FC projection,
    allowing the decoder to produce class-specific textures and shapes.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
        norm: str = NORM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * 4 * 4),
            nn.Dropout(dropout),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            _norm(norm, 128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            _norm(norm, 64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            _norm(norm, 32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,  3,   4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.embedding(y)                           # (B, embed_dim)
        h = self.fc(torch.cat([z, c], dim=1))           # (B, 256*4*4)
        return self.deconv(h.view(z.size(0), 256, 4, 4))


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.

    Both encoder and decoder receive the class label y alongside the
    image / latent vector, enabling class-guided generation at inference.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
        norm: str = NORM,
    ):
        super().__init__()
        self.encoder = ConditionalEncoder(latent_dim, num_classes, embed_dim, dropout, norm)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, embed_dim, dropout, norm)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, y: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on class labels y."""
        z = torch.randn(y.size(0), self.encoder.fc_mu.out_features).to(y.device)
        return self.decoder(z, y)


def get_beta(epoch: int, warmup: int = KL_WARMUP, beta_max: float = BETA_MAX) -> float:
    """
    Compute the KL weight β for the current epoch using linear annealing.

    β rises linearly from 0 to beta_max over `warmup` epochs,
    then stays at beta_max for the remainder of training.

    Args:
        epoch    : Current epoch (1-indexed).
        warmup   : Number of epochs for the linear ramp.
        beta_max : Target KL weight after warmup.

    Returns:
        β value for this epoch.
    """
    return min(epoch / warmup, 1.0) * beta_max


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE ELBO loss.

    Args:
        recon   : Reconstructed image from the decoder, shape (B, C, H, W).
        x       : Original image, shape (B, C, H, W).
        mu      : Latent distribution means, shape (B, latent_dim).
        logvar  : Latent distribution log-variances, shape (B, latent_dim).
        beta    : KL weight (default=1). Increase for beta-VAE behavior.

    Returns:
        total_loss : Combined scalar loss.
        recon_loss : Reconstruction term (MSE per sample).
        kl_loss    : KL divergence term per sample.
    """
    # MSE summed over pixels, averaged over the batch
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    # Closed-form KL for Gaussians: summed over latent dims, averaged over batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss
