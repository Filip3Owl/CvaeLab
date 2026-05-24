import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM  = 128
DROPOUT     = 0.2
BETA_MAX    = 1.0
KL_WARMUP   = 25
NORM        = "batch"   # "batch" | "group"
GROUP_SIZE  = 8         # canais por grupo quando NORM="group"


def _norm(norm: str, num_channels: int) -> nn.Module:
    """Retorna BatchNorm2d ou GroupNorm conforme o valor de `norm`."""
    if norm == "group":
        return nn.GroupNorm(num_channels // GROUP_SIZE, num_channels)
    return nn.BatchNorm2d(num_channels)


class Encoder(nn.Module):
    """
    Encoder convolucional: mapeia uma imagem 3×64×64 para os parâmetros
    (μ, log σ²) de uma distribuição Gaussiana no espaço latente.

    Cada bloco conv usa stride=2 para reduzir a resolução espacial à metade
    (downsampling aprendível, equivalente a Conv + MaxPool).
    Dropout2d descarta mapas de features inteiros para regularizar features espaciais.
    Um Dropout final é aplicado antes das cabeças lineares para regularizar
    a transição da representação espacial para a latente.
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
        h = self.conv(x).view(x.size(0), -1)  # achatamento: (B, 256*4*4)
        h = self.pre_latent(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Decoder convolucional: mapeia um vetor latente z ∈ ℝ^{latent_dim}
    de volta para uma imagem 3×64×64 no intervalo [-1, 1].

    ConvTranspose2d com stride=2 dobra a resolução espacial a cada
    passo — imagem espelhada do encoder.
    Dropout é aplicado após a projeção FC inicial para evitar que o
    decoder dependa excessivamente de dimensões latentes específicas.
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
            nn.Tanh(),  # saída em [-1, 1], correspondendo à normalização dos dados
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 256, 4, 4)  # reshape: (B, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Autoencoder Variacional completo.

    `reparameterize` implementa o truque da reparametrização:
        z = μ + σ·ε,  ε ~ N(0, I)
    Isso mantém o fluxo de gradientes pelo encoder durante a retropropagação,
    pois ε é amostrado independentemente dos parâmetros do modelo.
    No modo eval, retorna μ diretamente (sem ruído).
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
        """Gera `n` imagens amostrando z ~ N(0, I) diretamente."""
        z = torch.randn(n, self.encoder.fc_mu.out_features).to(
            next(self.parameters()).device
        )
        return self.decoder(z)


NUM_CLASSES = 10
EMBED_DIM   = 64


class ConditionalEncoder(nn.Module):
    """
    Encoder condicional: mapeia (imagem, rótulo de classe) → (μ, log σ²).

    O rótulo de classe é projetado em um vetor de embedding e concatenado
    às features conv achatadas antes das cabeças lineares.
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
    Decoder condicional: mapeia (z, rótulo de classe) → imagem 3×64×64.

    O embedding de classe é concatenado a z antes da projeção FC,
    permitindo que o decoder produza texturas e formas específicas por classe.
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
    Autoencoder Variacional Condicional.

    Tanto o encoder quanto o decoder recebem o rótulo de classe y junto com a
    imagem / vetor latente, permitindo geração guiada por classe na inferência.
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
        """Gera imagens condicionadas aos rótulos de classe y."""
        z = torch.randn(y.size(0), self.encoder.fc_mu.out_features).to(y.device)
        return self.decoder(z, y)


def get_beta(epoch: int, warmup: int = KL_WARMUP, beta_max: float = BETA_MAX) -> float:
    """
    Calcula o peso KL β para a época atual usando annealing linear.

    β sobe linearmente de 0 até beta_max ao longo de `warmup` épocas,
    depois permanece em beta_max pelo restante do treinamento.

    Args:
        epoch    : Época atual (indexada a partir de 1).
        warmup   : Número de épocas para a rampa linear.
        beta_max : Peso KL alvo após o warmup.

    Returns:
        Valor de β para esta época.
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
    Calcula a função de perda ELBO do VAE.

    Args:
        recon   : Imagem reconstruída pelo decoder, shape (B, C, H, W).
        x       : Imagem original, shape (B, C, H, W).
        mu      : Médias da distribuição latente, shape (B, latent_dim).
        logvar  : Log-variâncias da distribuição latente, shape (B, latent_dim).
        beta    : Peso da KL (padrão=1). Aumentar para comportamento β-VAE.

    Returns:
        total_loss : Perda escalar combinada.
        recon_loss : Termo de reconstrução (MSE por amostra).
        kl_loss    : Termo de divergência KL por amostra.
    """
    # MSE somado sobre pixels, com média sobre o batch
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    # KL em forma fechada para Gaussianas: somado sobre dims latentes, média sobre batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss
