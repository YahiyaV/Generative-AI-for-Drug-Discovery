"""
train_vae.py — Train the SMILES VAE on molecular data.

Usage:
    python src/train_vae.py                    # Full training (50 epochs)
    python src/train_vae.py --epochs 5         # Quick smoke test
    python src/train_vae.py --resume           # Resume from checkpoint
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    VAE_BATCH_SIZE, VAE_LEARNING_RATE, VAE_EPOCHS,
    VAE_BETA_START, VAE_BETA_END, VAE_BETA_WARMUP_EPOCHS,
    VAE_TEACHER_FORCING, MAX_SMILES_LEN
)
from src.mol_utils import TOKENIZER
from src.vae_model import SmilesVAE, vae_loss

import pandas as pd
from rdkit import Chem


def load_data(max_len=MAX_SMILES_LEN):
    """Load and tokenise SMILES data."""
    csv_path = os.path.join(PROCESSED_DIR, "molecules.csv")
    if not os.path.exists(csv_path):
        print("❌ Dataset not found. Run 'python data/download_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    smiles_list = df["SMILES"].tolist()

    # Filter SMILES that are too long (by token count, not char count)
    from src.mol_utils import tokenize_smiles
    smiles_list = [s for s in smiles_list if len(tokenize_smiles(s)) <= max_len - 2]
    print(f"📦 Loaded {len(smiles_list):,} SMILES (max_len={max_len})")

    # Tokenise — this builds the vocabulary dynamically
    tokens = TOKENIZER.batch_encode(smiles_list, max_len=max_len)
    TOKENIZER.freeze()
    print(f"📝 Vocabulary size: {TOKENIZER.vocab_size}")
    return tokens, smiles_list


def get_beta(epoch, warmup_epochs=VAE_BETA_WARMUP_EPOCHS,
             beta_start=VAE_BETA_START, beta_end=VAE_BETA_END):
    """Compute β for KL warm-up schedule."""
    if epoch >= warmup_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)


def train_epoch(model, loader, optimizer, epoch, beta):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    # Decrease teacher forcing over training
    tf_ratio = max(0.5, VAE_TEACHER_FORCING - epoch * 0.01)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
    for (batch,) in pbar:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        recon_logits, mu, logvar = model(batch, teacher_forcing_ratio=tf_ratio)
        loss, recon_val, kl_val = vae_loss(recon_logits, batch, mu, logvar, beta=beta)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_val
        total_kl += kl_val
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "recon": f"{recon_val:.4f}",
            "kl": f"{kl_val:.4f}",
            "β": f"{beta:.3f}"
        })

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "beta": beta,
        "tf_ratio": tf_ratio,
    }


def validate(model, smiles_list, n_samples=100):
    """Generate samples and compute validity rate."""
    model.eval()

    # Suppress RDKit parse warnings during validation
    from rdkit import RDLogger
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)

    generated = model.sample(n=n_samples, device=DEVICE)

    valid = 0
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid += 1

    # Restore warnings
    logger.setLevel(RDLogger.WARNING)

    validity = valid / n_samples
    return validity, generated[:5]  # Return 5 samples for display


def save_loss_curves(history, path):
    """Plot and save training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history) + 1)

    # Total loss
    axes[0].plot(epochs, [h["loss"] for h in history], "b-", linewidth=2)
    axes[0].set_title("Total Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    # Reconstruction vs KL
    axes[1].plot(epochs, [h["recon"] for h in history], "g-", label="Recon", linewidth=2)
    axes[1].plot(epochs, [h["kl"] for h in history], "r-", label="KL", linewidth=2)
    axes[1].set_title("Recon & KL Loss", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    # Validity
    if "validity" in history[0]:
        axes[2].plot(epochs, [h.get("validity", 0) for h in history],
                     "m-o", linewidth=2, markersize=3)
        axes[2].set_title("Validity Rate", fontsize=12, fontweight="bold")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Loss curves saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SMILES VAE")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=VAE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VAE_LEARNING_RATE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print(f"🧪 SMILES VAE Training")
    print(f"   Device: {DEVICE}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print()

    # Load data (this builds the full vocabulary)
    tokens, smiles_list = load_data()
    dataset = TensorDataset(tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    # Model — use vocab size AFTER data is loaded
    model = SmilesVAE(vocab_size=TOKENIZER.vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    start_epoch = 0
    history = []

    # Resume from checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vae_best.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        history = ckpt.get("history", [])
        print(f"⏪ Resumed from epoch {start_epoch}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print()

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        beta = get_beta(epoch)

        # Train
        metrics = train_epoch(model, loader, optimizer, epoch, beta)

        # Validate (sample molecules)
        validity, samples = validate(model, smiles_list)
        metrics["validity"] = validity

        history.append(metrics)
        scheduler.step(metrics["loss"])

        # Print epoch summary
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{args.epochs} │ "
              f"Loss: {metrics['loss']:.4f} │ "
              f"Recon: {metrics['recon']:.4f} │ "
              f"KL: {metrics['kl']:.4f} │ "
              f"β: {beta:.3f} │ "
              f"Validity: {validity:.1%} │ "
              f"Time: {elapsed:.0f}s")

        # Show samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  📝 Sample molecules:")
            for s in samples:
                valid_mark = "✅" if Chem.MolFromSmiles(s) else "❌"
                print(f"     {valid_mark} {s}")

        # Save best checkpoint
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_loss": best_loss,
                "vocab_size": TOKENIZER.vocab_size,
                "tokenizer_chars": TOKENIZER.chars,  # Save tokenizer state
            }, ckpt_path)

    # Save loss curves
    save_loss_curves(history, os.path.join(RESULTS_DIR, "vae_loss.png"))

    # Save final metrics
    metrics_path = os.path.join(RESULTS_DIR, "vae_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "final_loss": history[-1]["loss"],
            "final_validity": history[-1].get("validity", 0),
            "best_loss": best_loss,
            "total_epochs": len(history),
            "total_time_seconds": time.time() - start_time,
        }, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Final validity: {history[-1].get('validity', 0):.1%}")
    print(f"   Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
