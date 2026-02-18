"""
evaluate.py — Generate novel molecules and compute evaluation metrics.

Usage:
    python src/evaluate.py                      # Generate 1000 molecules
    python src/evaluate.py --n_samples 500      # Custom count
    python src/evaluate.py --temperature 0.8    # Lower diversity
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QEDModule, AllChem, DataStructs

from src.config import (
    DEVICE, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    EVAL_N_SAMPLES, NOVELTY_TANIMOTO_THRESHOLD, GNN_TARGET_PROPS
)
from src.mol_utils import (
    compute_fingerprint, compute_properties,
    lipinski_pass, draw_molecules_grid, TOKENIZER
)
from src.vae_model import SmilesVAE
from src.gnn_model import MoleculeGNN
from src.mol_utils import smiles_to_graph, get_node_feature_dim


def load_vae(device):
    """Load trained VAE from checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vae_best.pt")
    if not os.path.exists(ckpt_path):
        print("❌ VAE checkpoint not found. Run 'python src/train_vae.py' first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore tokenizer state if saved
    if "tokenizer_chars" in ckpt:
        TOKENIZER.chars = ckpt["tokenizer_chars"]
        TOKENIZER.char_to_idx = {c: i for i, c in enumerate(TOKENIZER.chars)}
        TOKENIZER.idx_to_char = {i: c for i, c in enumerate(TOKENIZER.chars)}
        TOKENIZER.vocab_size = len(TOKENIZER.chars)

    vocab_size = ckpt.get("vocab_size", TOKENIZER.vocab_size)
    model = SmilesVAE(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ✅ VAE loaded (epoch {ckpt.get('epoch', '?')}, "
          f"loss {ckpt.get('best_loss', '?'):.4f}, vocab={vocab_size})")
    return model


def load_gnn(device):
    """Load trained GNN from checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "gnn_best.pt")
    if not os.path.exists(ckpt_path):
        print("⚠️  GNN checkpoint not found. Skipping property prediction.")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    node_dim = ckpt.get("node_feature_dim", get_node_feature_dim())
    model = MoleculeGNN(node_feature_dim=node_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm_stats = ckpt.get("norm_stats", None)
    print(f"  ✅ GNN loaded (R² {ckpt.get('best_r2', '?'):.4f})")
    return model, norm_stats


def generate_molecules(vae, n_samples, device, temperature=1.0):
    """Generate molecules from the VAE."""
    print(f"\n🧬 Generating {n_samples:,} molecules (T={temperature})...")

    # Generate in batches to avoid OOM
    batch_size = 256
    all_smiles = []
    for i in tqdm(range(0, n_samples, batch_size), desc="Generating"):
        n = min(batch_size, n_samples - i)
        batch = vae.sample(n=n, device=device, temperature=temperature)
        all_smiles.extend(batch)

    return all_smiles


def compute_metrics(generated_smiles, training_smiles):
    """Compute generation quality metrics."""
    print("\n📊 Computing evaluation metrics...")

    # Validity
    valid_smiles = []
    valid_mols = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical)
            valid_mols.append(mol)

    validity = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0

    # Uniqueness (among valid)
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0

    # Novelty (not in training set, by exact match)
    training_set = set(training_smiles)
    novel_smiles = [s for s in unique_smiles if s not in training_set]
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    # Structural novelty (Tanimoto-based)
    print("  Computing structural novelty (Tanimoto)...")
    train_fps = []
    for smi in tqdm(list(training_set)[:5000], desc="  Train FPs", leave=False):
        fp = compute_fingerprint(smi)
        if fp is not None:
            train_fps.append(fp)

    structurally_novel = 0
    for smi in tqdm(novel_smiles[:500], desc="  Novelty check", leave=False):
        fp = compute_fingerprint(smi)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        if max(sims) < NOVELTY_TANIMOTO_THRESHOLD:
            structurally_novel += 1

    structural_novelty = structurally_novel / min(len(novel_smiles), 500) if novel_smiles else 0

    # Drug-likeness (Lipinski Rule of 5)
    lipinski_count = sum(1 for s in unique_smiles if lipinski_pass(s))
    lipinski_rate = lipinski_count / len(unique_smiles) if unique_smiles else 0

    # QED distribution
    qed_scores = []
    for mol in valid_mols:
        try:
            qed_scores.append(QEDModule.qed(mol))
        except Exception:
            pass

    metrics = {
        "total_generated": len(generated_smiles),
        "valid": len(valid_smiles),
        "validity": round(validity, 4),
        "unique": len(unique_smiles),
        "uniqueness": round(uniqueness, 4),
        "novel": len(novel_smiles),
        "novelty": round(novelty, 4),
        "structural_novelty": round(structural_novelty, 4),
        "lipinski_pass": lipinski_count,
        "lipinski_rate": round(lipinski_rate, 4),
        "mean_qed": round(np.mean(qed_scores), 4) if qed_scores else 0,
        "median_qed": round(np.median(qed_scores), 4) if qed_scores else 0,
    }

    return metrics, unique_smiles, novel_smiles


def predict_properties_gnn(gnn, norm_stats, smiles_list, device):
    """Use trained GNN to predict properties of generated molecules."""
    if gnn is None:
        return None

    print("\n🔬 Predicting properties with GNN...")
    predictions = []

    for smi in tqdm(smiles_list[:500], desc="GNN prediction"):
        graph = smiles_to_graph(smi)
        if graph is None:
            continue

        graph = graph.to(device)
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            pred = gnn(graph).squeeze().cpu().numpy()

        # Denormalize
        props = {}
        for i, prop in enumerate(GNN_TARGET_PROPS):
            if norm_stats and prop in norm_stats:
                val = pred[i] * norm_stats[prop]["std"] + norm_stats[prop]["mean"]
            else:
                val = pred[i]
            props[prop] = round(float(val), 3)
        props["SMILES"] = smi
        predictions.append(props)

    return pd.DataFrame(predictions)


def save_visualisations(metrics, unique_smiles, results_dir):
    """Create and save evaluation visualisation plots."""

    # 1. Metrics summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Rates
    rates = ["validity", "uniqueness", "novelty", "lipinski_rate"]
    vals = [metrics[r] for r in rates]
    colors = ["#00bcd4", "#4caf50", "#ff9800", "#e91e63"]
    axes[0].bar(["Valid", "Unique", "Novel", "Lipinski"], vals, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Generation Quality Metrics", fontweight="bold")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

    # 2. Property distributions
    props_data = {"MolWt": [], "LogP": [], "QED": []}
    for smi in unique_smiles[:2000]:
        p = compute_properties(smi)
        if p:
            for k in props_data:
                if k in p:
                    props_data[k].append(p[k])

    if props_data["MolWt"]:
        axes[1].hist(props_data["MolWt"], bins=40, color="#00bcd4", alpha=0.7, edgecolor="white")
        axes[1].set_title("Molecular Weight Distribution", fontweight="bold")
        axes[1].set_xlabel("MolWt (Da)")

    if props_data["QED"]:
        axes[2].hist(props_data["QED"], bins=40, color="#4caf50", alpha=0.7, edgecolor="white")
        axes[2].set_title("QED (Drug-likeness) Distribution", fontweight="bold")
        axes[2].set_xlabel("QED Score")
        axes[2].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="QED=0.5")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "evaluation_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Molecule grid image
    if unique_smiles:
        grid = draw_molecules_grid(unique_smiles[:12], mols_per_row=4)
        if grid:
            grid.save(os.path.join(results_dir, "generated_molecules.png"))
            print(f"  🖼️  Molecule grid saved")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("--n_samples", type=int, default=EVAL_N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Generative AI Drug Discovery — Evaluation Pipeline")
    print("=" * 60)

    # Load training data
    csv_path = os.path.join(PROCESSED_DIR, "molecules.csv")
    train_df = pd.read_csv(csv_path)
    training_smiles = train_df["SMILES"].tolist()
    print(f"📦 Training set: {len(training_smiles):,} molecules")

    # Load models
    vae = load_vae(DEVICE)
    gnn, norm_stats = load_gnn(DEVICE)

    # Generate
    generated = generate_molecules(vae, args.n_samples, DEVICE, args.temperature)

    # Evaluate
    metrics, unique_smiles, novel_smiles = compute_metrics(generated, training_smiles)

    # Print metrics
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Generated:          {metrics['total_generated']:,}")
    print(f"  Valid:              {metrics['valid']:,}  ({metrics['validity']:.1%})")
    print(f"  Unique:            {metrics['unique']:,}  ({metrics['uniqueness']:.1%})")
    print(f"  Novel:             {metrics['novel']:,}  ({metrics['novelty']:.1%})")
    print(f"  Structural Novelty: {metrics['structural_novelty']:.1%}")
    print(f"  Lipinski Pass:     {metrics['lipinski_pass']:,}  ({metrics['lipinski_rate']:.1%})")
    print(f"  Mean QED:          {metrics['mean_qed']:.4f}")
    print(f"  Median QED:        {metrics['median_qed']:.4f}")

    # GNN property predictions
    if gnn is not None and unique_smiles:
        pred_df = predict_properties_gnn(gnn, norm_stats, unique_smiles, DEVICE)
        if pred_df is not None:
            pred_path = os.path.join(RESULTS_DIR, "generated_properties.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"\n  📋 GNN predictions saved → {pred_path}")
            print(pred_df.describe().round(3).to_string())

    # Save results
    save_visualisations(metrics, unique_smiles, RESULTS_DIR)

    # Save generated molecules
    gen_df = pd.DataFrame({"SMILES": unique_smiles})
    gen_df.to_csv(os.path.join(RESULTS_DIR, "generated_molecules.csv"), index=False)

    # Save metrics JSON
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Evaluation complete! Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
