"""
config.py — Central configuration for the Drug Discovery project.
All hyperparameters and paths are defined here for easy tuning.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create dirs if missing
for d in [PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_MOLECULES = 50_000          # Target download count
MAX_SMILES_LEN = 120            # Pad / truncate SMILES to this length
MIN_HEAVY_ATOMS = 5
MAX_HEAVY_ATOMS = 50

# ─── VAE Hyper-parameters ────────────────────────────────────────────────────
VAE_EMBEDDING_DIM = 64
VAE_HIDDEN_DIM = 512
VAE_LATENT_DIM = 256
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.1
VAE_BATCH_SIZE = 128            # Fits well in 4 GB VRAM
VAE_LEARNING_RATE = 3e-4
VAE_EPOCHS = 50
VAE_BETA_START = 0.0            # KL warm-up from 0 → 1
VAE_BETA_END = 1.0
VAE_BETA_WARMUP_EPOCHS = 10
VAE_TEACHER_FORCING = 0.9

# ─── GNN Hyper-parameters ────────────────────────────────────────────────────
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.2
GNN_BATCH_SIZE = 64
GNN_LEARNING_RATE = 1e-3
GNN_EPOCHS = 100
GNN_TARGET_PROPS = ["MolWt", "LogP", "TPSA", "QED"]  # regression targets

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_N_SAMPLES = 1000           # How many molecules to generate for eval
NOVELTY_TANIMOTO_THRESHOLD = 0.4
