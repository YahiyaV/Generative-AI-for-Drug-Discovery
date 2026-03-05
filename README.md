---
title: Generative AI for Drug Discovery
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 Generative AI for Drug Discovery

> **Design novel drug molecules with deep generative models and predict their properties using Graph Neural Networks.**

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Chemistry-00bcd4?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?style=flat-square&logo=flask&logoColor=white)

---

## 🎯 Overview

This project builds an end-to-end AI pipeline for computational drug discovery:

1. **Data Pipeline** — Downloads and preprocesses ~50,000 drug-like molecules from ZINC250K
2. **SMILES VAE** — A character-level Variational Autoencoder that learns the chemical latent space and generates novel molecular structures
3. **GNN Property Predictor** — A Graph Convolutional Network that predicts molecular properties (MolWt, LogP, TPSA, QED) from molecular graphs
4. **Evaluation** — Measures validity, uniqueness, novelty, structural novelty, and drug-likeness (Lipinski Rule of 5)
5. **Interactive Dashboard** — A Flask-powered web UI to generate, evaluate, and explore molecules in real-time

## 🏗️ Architecture

```
SMILES Dataset ──→ VAE (Encoder → z → Decoder) ──→ Novel Molecules ──→ GNN Predictions
     │                       │                            │                    │
 50K drugs           Latent space              SMILES strings        MolWt, LogP,
 from ZINC           sampling                 + 2D structures        TPSA, QED
```


## 📁 Project Structure

```text
├── server.py                  # Flask API server & entry point
├── requirements.txt           # Python dependencies
├── data/
│   ├── download_data.py       # Data download & preprocessing
│   └── processed/
│       └── molecules.csv      # Cleaned molecular dataset
├── src/
│   ├── __init__.py            # Package initializer
│   ├── config.py              # Central hyperparameters
│   ├── mol_utils.py           # SMILES ↔ Graph, fingerprints, tokenizer
│   ├── vae_model.py           # SMILES VAE architecture
│   ├── gnn_model.py           # GNN property predictor
│   ├── train_vae.py           # VAE training script
│   ├── train_gnn.py           # GNN training script
│   └── evaluate.py            # Generation & evaluation pipeline
├── static/
│   ├── index.html             # Web UI main page
│   ├── style.css              # Stylesheet
│   └── script.js              # Frontend logic & API interaction
├── checkpoints/
│   ├── vae_best.pt            # Trained VAE weights
│   └── gnn_best.pt            # Trained GNN weights
└── results/
    ├── evaluation_metrics.json
    ├── evaluation_summary.png
    ├── generated_molecules.csv
    ├── generated_properties.csv
    ├── generated_molecules.png
    ├── gnn_metrics.json
    ├── gnn_parity.png
    ├── gnn_training.png
    ├── vae_loss.png
    └── vae_metrics.json
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
python server.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### 7. Deploying to Hugging Face Spaces

This project is configured to be deployed as a Docker Space on Hugging Face.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select **Docker** as the SDK.
2. Choose a Blank template.
3. Push this repository to your Hugging Face Space repository:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push --force space main
```

4. The Space will automatically build the Docker image and launch the Flask server on port `7860`.

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
