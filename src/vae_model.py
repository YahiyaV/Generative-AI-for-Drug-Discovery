"""
vae_model.py — Character-level SMILES Variational Autoencoder.

Architecture:
  Encoder: Embedding → GRU → FC → (μ, log σ²)
  Decoder: z → FC → GRU → FC → softmax over vocab
  Loss:    Recon (CE) + β · KL divergence  (β warm-up)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.config import (
    VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM, VAE_LATENT_DIM,
    VAE_NUM_LAYERS, VAE_DROPOUT, MAX_SMILES_LEN
)
from src.mol_utils import TOKENIZER


class Encoder(nn.Module):
    """GRU-based encoder that maps tokenised SMILES to (μ, log σ²)."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)      # *2 for bidir
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        embedded = self.embedding(x)                          # (B, L, E)
        _, hidden = self.gru(embedded)                        # (2*layers, B, H)
        # Concatenate last forward + backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)   # (B, 2H)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class Decoder(nn.Module):
    """GRU-based decoder that reconstructs SMILES from latent z."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 latent_dim: int, num_layers: int, dropout: float,
                 max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor,
                target: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.9) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) latent vector
            target: (batch, seq_len) ground-truth tokens (for teacher forcing)
            teacher_forcing_ratio: probability of using ground truth at each step
        Returns:
            outputs: (batch, seq_len, vocab_size) logits
        """
        batch_size = z.size(0)
        seq_len = self.max_len

        # Init hidden from z
        hidden = self.latent_to_hidden(z)                    # (B, H*layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()        # (layers, B, H)

        # Start with SOS token
        input_tok = torch.full(
            (batch_size, 1), TOKENIZER.sos_idx,
            dtype=torch.long, device=z.device
        )

        outputs = []
        for t in range(seq_len):
            embedded = self.dropout(self.embedding(input_tok))  # (B, 1, E)
            output, hidden = self.gru(embedded, hidden)          # (B, 1, H)
            logits = self.fc_out(output)                         # (B, 1, V)
            outputs.append(logits)

            # Teacher forcing decision
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_tok = target[:, t:t + 1]
            else:
                input_tok = logits.argmax(dim=-1)                # (B, 1)

        return torch.cat(outputs, dim=1)  # (B, L, V)


class SmilesVAE(nn.Module):
    """Complete SMILES VAE with encode, decode, and sample methods."""

    def __init__(self, vocab_size: int = None):
        super().__init__()
        if vocab_size is None:
            vocab_size = TOKENIZER.vocab_size
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            vocab_size, VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM,
            VAE_LATENT_DIM, VAE_NUM_LAYERS, VAE_DROPOUT
        )
        self.decoder = Decoder(
            vocab_size, VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM,
            VAE_LATENT_DIM, VAE_NUM_LAYERS, VAE_DROPOUT,
            MAX_SMILES_LEN
        )

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor,
                teacher_forcing_ratio: float = 0.9):
        """
        Full forward pass: encode → reparameterize → decode.

        Returns:
            recon_logits: (B, L, V) reconstruction logits
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to logits (no teacher forcing)."""
        return self.decoder(z, target=None, teacher_forcing_ratio=0.0)

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu",
               temperature: float = 1.0) -> list:
        """
        Sample n molecules from the prior N(0, I).

        Args:
            n: number of molecules to generate
            device: 'cuda' or 'cpu'
            temperature: sampling temperature (lower = less diverse)

        Returns:
            List of SMILES strings
        """
        self.eval()
        z = torch.randn(n, VAE_LATENT_DIM, device=device) * temperature
        logits = self.decode(z)                         # (n, L, V)
        indices = logits.argmax(dim=-1).cpu().tolist()   # greedy decode

        smiles_list = []
        for seq in indices:
            smi = TOKENIZER.decode(seq)
            smiles_list.append(smi)
        return smiles_list

    @torch.no_grad()
    def sample_diverse(self, n: int = 1, device: str = "cpu",
                       temperature: float = 1.0) -> list:
        """
        Sample diverse molecules using multinomial sampling instead of greedy.
        """
        self.eval()
        z = torch.randn(n, VAE_LATENT_DIM, device=device) * temperature
        logits = self.decode(z)                         # (n, L, V)

        # Apply temperature and sample
        probs = F.softmax(logits / temperature, dim=-1)
        indices = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(n, -1).cpu().tolist()

        smiles_list = []
        for seq in indices:
            smi = TOKENIZER.decode(seq)
            smiles_list.append(smi)
        return smiles_list


def vae_loss(recon_logits: torch.Tensor, targets: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, float, float]:
    """
    Compute VAE loss = Reconstruction (CE) + β · KL divergence.

    Args:
        recon_logits: (B, L, V) predicted logits
        targets: (B, L) ground-truth token indices
        mu, logvar: latent distribution parameters
        beta: KL weight (for warm-up schedule)

    Returns:
        total_loss, recon_loss_value, kl_loss_value
    """
    # Reconstruction loss (ignore padding)
    recon_loss = F.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        targets.view(-1),
        ignore_index=TOKENIZER.pad_idx,
        reduction="mean"
    )

    # KL divergence: -0.5 * Σ(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()
