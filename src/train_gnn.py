"""
train_gnn.py — Train the GNN molecular property predictor.

Usage:
    python src/train_gnn.py                    # Full training (100 epochs)
    python src/train_gnn.py --epochs 10        # Quick smoke test
    python src/train_gnn.py --model gat        # Use GAT instead of GCN
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
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    GNN_BATCH_SIZE, GNN_LEARNING_RATE, GNN_EPOCHS, GNN_TARGET_PROPS
)
from src.mol_utils import smiles_to_graph, get_node_feature_dim
from src.gnn_model import MoleculeGNN, MoleculeGAT


def prepare_graphs(df, target_props=GNN_TARGET_PROPS):
    """Convert SMILES dataframe to list of PyG Data objects with targets."""
    graphs = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        smi = row["SMILES"]
        data = smiles_to_graph(smi)
        if data is None:
            skipped += 1
            continue

        # Attach target properties
        targets = []
        valid = True
        for prop in target_props:
            val = row.get(prop, None)
            if val is None or pd.isna(val):
                valid = False
                break
            targets.append(float(val))

        if not valid:
            skipped += 1
            continue

        data.y = torch.tensor([targets], dtype=torch.float)  # (1, T) for correct batching
        graphs.append(data)

    print(f"  ✅ {len(graphs):,} graphs built, {skipped} skipped")
    return graphs


def normalize_targets(graphs, target_props):
    """Z-score normalize targets and return stats for inverse transform."""
    all_targets = torch.stack([g.y for g in graphs]).squeeze(1)  # (N, T)
    mean = all_targets.mean(dim=0)
    std = all_targets.std(dim=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero

    for g in graphs:
        g.y = (g.y - mean) / std

    stats = {prop: {"mean": mean[i].item(), "std": std[i].item()}
             for i, prop in enumerate(target_props)}
    return graphs, stats


def denormalize(preds, stats, target_props):
    """Inverse transform predictions back to original scale."""
    result = preds.clone()
    for i, prop in enumerate(target_props):
        result[:, i] = result[:, i] * stats[prop]["std"] + stats[prop]["mean"]
    return result


def train_epoch(model, loader, optimizer, criterion):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, stats, target_props):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(DEVICE)
        preds = model(batch)
        all_preds.append(preds.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Denormalize
    preds_dn = denormalize(preds, stats, target_props)
    targets_dn = denormalize(targets, stats, target_props)

    metrics = {}
    for i, prop in enumerate(target_props):
        p = preds_dn[:, i].numpy()
        t = targets_dn[:, i].numpy()
        metrics[prop] = {
            "r2": float(r2_score(t, p)),
            "mae": float(mean_absolute_error(t, p)),
        }

    overall_r2 = np.mean([m["r2"] for m in metrics.values()])
    return metrics, overall_r2, preds_dn, targets_dn


def save_parity_plots(preds, targets, target_props, path):
    """Plot predicted vs actual for each property."""
    n = len(target_props)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, prop in enumerate(target_props):
        ax = axes[i]
        p = preds[:, i].numpy()
        t = targets[:, i].numpy()

        ax.scatter(t, p, alpha=0.3, s=10, c="#00bcd4")
        # Perfect prediction line
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "r--", alpha=0.7, linewidth=2)

        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)

        ax.set_title(f"{prop}\nR² = {r2:.3f}  MAE = {mae:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Parity plots saved → {path}")


def save_training_curve(history, path):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history) + 1)
    ax.plot(epochs, [h["train_loss"] for h in history], "b-",
            label="Train", linewidth=2)
    ax.plot(epochs, [h["val_r2"] for h in history], "g-",
            label="Val R²", linewidth=2)

    ax.set_title("GNN Training Progress", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train GNN Property Predictor")
    parser.add_argument("--epochs", type=int, default=GNN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=GNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=GNN_LEARNING_RATE)
    parser.add_argument("--model", choices=["gcn", "gat"], default="gcn")
    args = parser.parse_args()

    print(f"🔬 GNN Property Predictor Training")
    print(f"   Device: {DEVICE}")
    print(f"   Model: {args.model.upper()}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print()

    # Load data
    csv_path = os.path.join(PROCESSED_DIR, "molecules.csv")
    if not os.path.exists(csv_path):
        print("❌ Dataset not found. Run 'python data/download_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"📦 Loaded {len(df):,} molecules")

    # Build graphs
    graphs = prepare_graphs(df)
    graphs, norm_stats = normalize_targets(graphs, GNN_TARGET_PROPS)

    # Train/test split
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(train_graphs):,}  |  Test: {len(test_graphs):,}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Model
    node_dim = get_node_feature_dim()
    if args.model == "gat":
        model = MoleculeGAT(node_feature_dim=node_dim).to(DEVICE)
    else:
        model = MoleculeGNN(node_feature_dim=node_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Parameters: {n_params:,}")
    print()

    history = []
    best_r2 = -float("inf")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "gnn_best.pt")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        # Evaluate
        test_metrics, overall_r2, preds_dn, targets_dn = evaluate(
            model, test_loader, norm_stats, GNN_TARGET_PROPS
        )

        history.append({
            "train_loss": train_loss,
            "val_r2": overall_r2,
            "per_target": test_metrics,
        })

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{args.epochs} │ "
                  f"Train Loss: {train_loss:.6f} │ "
                  f"Val R²: {overall_r2:.4f} │ "
                  f"Time: {elapsed:.0f}s")
            for prop, m in test_metrics.items():
                print(f"    {prop:8s}: R²={m['r2']:.4f}  MAE={m['mae']:.3f}")

        # Save best
        if overall_r2 > best_r2:
            best_r2 = overall_r2
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "model_type": args.model,
                "node_feature_dim": node_dim,
                "norm_stats": norm_stats,
                "best_r2": best_r2,
                "history": history,
            }, ckpt_path)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"🏆 Best validation R²: {best_r2:.4f}")

    # Load best model for final plots
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, _, preds_dn, targets_dn = evaluate(
        model, test_loader, norm_stats, GNN_TARGET_PROPS
    )

    # Save plots
    save_parity_plots(preds_dn, targets_dn, GNN_TARGET_PROPS,
                      os.path.join(RESULTS_DIR, "gnn_parity.png"))
    save_training_curve(history, os.path.join(RESULTS_DIR, "gnn_training.png"))

    # Save metrics
    final_metrics = {}
    for prop, m in test_metrics.items():
        final_metrics[prop] = m
    final_metrics["overall_r2"] = best_r2
    final_metrics["total_time_seconds"] = time.time() - start_time

    with open(os.path.join(RESULTS_DIR, "gnn_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n✅ GNN training complete!")
    print(f"   Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
