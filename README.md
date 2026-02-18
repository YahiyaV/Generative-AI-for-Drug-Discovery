# 🧬 Generative AI for Drug Discovery

> **Design novel drug molecules with deep generative models and predict their properties using Graph Neural Networks.**

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Chemistry-00bcd4?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=flat-square&logo=streamlit)

---

## 🎯 Overview

This project builds an end-to-end AI pipeline for computational drug discovery:

1. **Data Pipeline** — Downloads and preprocesses ~50,000 drug-like molecules from ZINC250K
2. **SMILES VAE** — A character-level Variational Autoencoder that learns the chemical latent space and generates novel molecular structures
3. **GNN Property Predictor** — A Graph Convolutional Network that predicts molecular properties (MolWt, LogP, TPSA, QED) from molecular graphs
4. **Evaluation** — Measures validity, uniqueness, novelty, structural novelty, and drug-likeness (Lipinski Rule of 5)
5. **Interactive Dashboard** — A Streamlit app to generate, evaluate, and explore molecules in real-time

## 🏗️ Architecture

```
SMILES Dataset ──→ VAE (Encoder → z → Decoder) ──→ Novel Molecules ──→ GNN Predictions
     │                       │                            │                    │
 50K drugs           Latent space              SMILES strings        MolWt, LogP,
 from ZINC           sampling                 + 2D structures        TPSA, QED
```

## 📁 Project Structure

```
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── data/
│   ├── download_data.py       # Data download & preprocessing
│   └── processed/             # Cleaned CSV datasets
├── src/
│   ├── config.py              # Central hyperparameters
│   ├── mol_utils.py           # SMILES ↔ Graph, fingerprints, tokenizer
│   ├── vae_model.py           # SMILES VAE architecture
│   ├── gnn_model.py           # GNN property predictor
│   ├── train_vae.py           # VAE training script
│   ├── train_gnn.py           # GNN training script
│   └── evaluate.py            # Generation & evaluation pipeline
├── checkpoints/               # Saved model weights
└── results/                   # Plots, metrics, generated molecules
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For PyTorch Geometric, you may need to install matching versions for your CUDA setup. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Download & Preprocess Data

```bash
python data/download_data.py
```

This downloads ~50,000 drug-like molecules from ZINC250K and computes 9 molecular descriptors.

### 3. Train the VAE

```bash
# Full training (50 epochs, ~20 min on GPU)
python src/train_vae.py

# Quick test (2 epochs)
python src/train_vae.py --epochs 2
```

### 4. Train the GNN

```bash
# Full training (100 epochs, ~15 min on GPU)
python src/train_gnn.py

# Quick test (5 epochs)
python src/train_gnn.py --epochs 5
```

### 5. Evaluate Generated Molecules

```bash
python src/evaluate.py --n_samples 1000
```

### 6. Launch Dashboard

```bash
streamlit run app.py
```

## ⚙️ Configuration

All hyperparameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_MOLECULES` | 50,000 | Dataset size |
| `VAE_LATENT_DIM` | 256 | VAE latent space dimension |
| `VAE_BATCH_SIZE` | 128 | Optimized for 4GB VRAM |
| `VAE_EPOCHS` | 50 | Training epochs |
| `GNN_HIDDEN_DIM` | 128 | GNN hidden layer size |
| `GNN_EPOCHS` | 100 | GNN training epochs |

## 📊 Metrics

The evaluation pipeline measures:
- **Validity** — % of generated SMILES parseable by RDKit
- **Uniqueness** — % unique among valid molecules
- **Novelty** — % not present in training set
- **Structural Novelty** — % with Tanimoto similarity < 0.4 to training set
- **Drug-likeness** — Lipinski Rule of 5 pass rate
- **QED Distribution** — Quantitative Estimate of Drug-likeness scores

## 🔧 Hardware

Tested on:
- **GPU**: NVIDIA RTX 3050 (4GB VRAM) — batch sizes optimized accordingly
- **CPU**: Supported but ~10× slower for training

## 📜 License

This project is for educational and research purposes.

## 🙏 Acknowledgements

- [ZINC Database](https://zinc.docking.org/) for molecular data
- [RDKit](https://www.rdkit.org/) for cheminformatics
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN layers
