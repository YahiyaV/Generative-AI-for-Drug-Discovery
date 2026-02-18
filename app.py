"""
app.py — Streamlit Dashboard for Generative AI Drug Discovery.

Run with:  streamlit run app.py
"""

import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem
from rdkit.Chem import Draw, QED as QEDModule, Descriptors

from src.config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, PROCESSED_DIR,
    GNN_TARGET_PROPS, VAE_LATENT_DIM
)
from src.mol_utils import (
    compute_properties, lipinski_pass, draw_molecule,
    draw_molecules_grid, smiles_to_graph, get_node_feature_dim,
    TOKENIZER
)
from src.vae_model import SmilesVAE
from src.gnn_model import MoleculeGNN


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🧬 AI Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0f1c 0%, #111827 50%, #0d1321 100%);
        color: #e0e7ff;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1a1f3a 100%);
        border-right: 1px solid rgba(0, 188, 212, 0.2);
    }

    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00bcd4, #00e5ff, #76ff03);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0,188,212,0.1), rgba(0,229,255,0.05));
        border: 1px solid rgba(0,188,212,0.3);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,188,212,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,188,212,0.2);
    }
    [data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(17, 24, 39, 0.8);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,188,212,0.2), rgba(0,229,255,0.1));
        color: #00e5ff !important;
        border: 1px solid rgba(0,188,212,0.4);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00bcd4, #0097a7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,188,212,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00e5ff, #00bcd4);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,188,212,0.4);
    }

    /* Data frames */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,188,212,0.2);
        border-radius: 8px;
    }

    /* Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(0, 188, 212, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }

    /* Molecule card */
    .mol-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(0, 188, 212, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .mol-card:hover {
        border-color: rgba(0, 229, 255, 0.5);
        box-shadow: 0 4px 20px rgba(0, 188, 212, 0.15);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background: rgba(0, 188, 212, 0.1);
    }

    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(0, 188, 212, 0.1) !important;
        border: 1px solid rgba(0, 188, 212, 0.3) !important;
        border-radius: 8px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animated gradient border */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-title {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(120deg, #00bcd4, #00e5ff, #76ff03, #00bcd4);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        margin-bottom: 0;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 1.1em;
        margin-top: 4px;
    }

    /* Glow effect for generated molecules */
    .glow {
        box-shadow: 0 0 20px rgba(0, 188, 212, 0.3),
                    0 0 40px rgba(0, 188, 212, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_vae_model():
    """Load the trained VAE model."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vae_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Restore tokenizer state if saved
    if "tokenizer_chars" in ckpt:
        TOKENIZER.chars = ckpt["tokenizer_chars"]
        TOKENIZER.char_to_idx = {c: i for i, c in enumerate(TOKENIZER.chars)}
        TOKENIZER.idx_to_char = {i: c for i, c in enumerate(TOKENIZER.chars)}
        TOKENIZER.vocab_size = len(TOKENIZER.chars)

    vocab_size = ckpt.get("vocab_size", TOKENIZER.vocab_size)
    model = SmilesVAE(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@st.cache_resource
def load_gnn_model():
    """Load the trained GNN model."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "gnn_best.pt")
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    node_dim = ckpt.get("node_feature_dim", get_node_feature_dim())
    model = MoleculeGNN(node_feature_dim=node_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("norm_stats", None)


@st.cache_data
def load_training_data():
    """Load the training dataset."""
    csv_path = os.path.join(PROCESSED_DIR, "molecules.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_eval_metrics():
    """Load pre-computed evaluation metrics."""
    path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def mol_to_base64(smiles, size=(250, 250)):
    """Convert SMILES to base64 PNG for inline display."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict_with_gnn(gnn, norm_stats, smiles):
    """Predict properties with GNN."""
    if gnn is None:
        return None
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    graph = graph.to(DEVICE)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        pred = gnn(graph).squeeze().cpu().numpy()
    props = {}
    for i, prop in enumerate(GNN_TARGET_PROPS):
        if norm_stats and prop in norm_stats:
            val = pred[i] * norm_stats[prop]["std"] + norm_stats[prop]["mean"]
        else:
            val = pred[i]
        props[prop] = round(float(val), 3)
    return props


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="hero-title">🧬</p>', unsafe_allow_html=True)
    st.markdown("### AI Drug Discovery")
    st.markdown("---")

    # Status indicators
    vae = load_vae_model()
    gnn, gnn_stats = load_gnn_model()
    train_df = load_training_data()

    st.markdown("#### 📡 System Status")

    if train_df is not None:
        st.success(f"📦 Dataset: {len(train_df):,} molecules")
    else:
        st.error("📦 Dataset: Not loaded")

    if vae is not None:
        st.success("🧠 VAE: Ready")
    else:
        st.warning("🧠 VAE: Not trained")

    if gnn is not None:
        st.success("🔬 GNN: Ready")
    else:
        st.warning("🔬 GNN: Not trained")

    st.markdown("---")
    st.markdown("#### ⚙️ Device")
    st.code(str(DEVICE).upper())

    st.markdown("---")
    st.markdown(
        "<p style='color: #475569; font-size: 0.8em; text-align: center;'>"
        "Built with PyTorch • RDKit • Streamlit</p>",
        unsafe_allow_html=True
    )


# ─── Main Content ────────────────────────────────────────────────────────────

# Hero
st.markdown(
    '<p class="hero-title">Generative AI for Drug Discovery</p>'
    '<p class="hero-sub">Design novel drug candidates with deep generative models '
    'and predict their molecular properties using Graph Neural Networks</p>',
    unsafe_allow_html=True
)

st.markdown("")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", "🧬 Generate", "📊 Evaluate", "🔬 Explore"
])


# ━━━ TAB 1: Overview ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
        <h3>🎯 Project Pipeline</h3>
        <p>This system uses a <b>two-stage AI pipeline</b> for drug molecule design:</p>
        <ol>
            <li><b>SMILES VAE</b> — A Variational Autoencoder trained on character-level SMILES
            strings learns the chemical latent space. Sampling from this space generates novel,
            chemically diverse molecules.</li>
            <li><b>GNN Property Predictor</b> — A Graph Convolutional Network takes molecular
            graphs (atoms = nodes, bonds = edges) and predicts key drug properties:
            <em>molecular weight, LogP, TPSA, and QED</em>.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        # Architecture diagram
        st.markdown("""
        <div class="glass-card">
        <h3>🏗️ Architecture</h3>
        </div>
        """, unsafe_allow_html=True)

        arch_fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.set_facecolor("#111827")
        arch_fig.patch.set_facecolor("#111827")

        # Pipeline boxes
        boxes = [
            (0.5, 1, 2.5, 1.5, "📦 SMILES\nDataset", "#1a237e"),
            (3.5, 1, 2.5, 1.5, "🧠 VAE\nEncoder→z→Decoder", "#004d40"),
            (6.5, 1, 2.5, 1.5, "🧬 Novel\nMolecules", "#b71c1c"),
            (9.5, 1, 2.5, 1.5, "🔬 GNN\nPredictions", "#e65100"),
        ]

        for x, y, w, h, text, color in boxes:
            rect = plt.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor="#00bcd4", facecolor=color + "40",
                                 zorder=2, clip_on=False)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#e0e7ff", zorder=3)

        # Arrows
        for x in [3.0, 6.0, 9.0]:
            ax.annotate("", xy=(x + 0.5, 1.75), xytext=(x, 1.75),
                        arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2))

        st.pyplot(arch_fig)
        plt.close()

    with col2:
        st.markdown("""
        <div class="glass-card">
        <h3>📊 Dataset Stats</h3>
        </div>
        """, unsafe_allow_html=True)

        if train_df is not None:
            st.metric("Total Molecules", f"{len(train_df):,}")
            st.metric("Avg Mol Weight", f"{train_df['MolWt'].mean():.1f} Da")
            st.metric("Avg QED", f"{train_df['QED'].mean():.3f}")
            st.metric("Avg LogP", f"{train_df['LogP'].mean():.2f}")

            # Mini dist plot
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#111827")
            ax.hist(train_df["QED"], bins=30, color="#00bcd4", alpha=0.7,
                    edgecolor="#111827")
            ax.set_xlabel("QED Score", color="#94a3b8")
            ax.set_ylabel("Count", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#334155")
            ax.spines["bottom"].set_color("#334155")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run `python data/download_data.py` to load data.")


# ━━━ TAB 2: Generate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.markdown("""
    <div class="glass-card">
    <h3>🧬 Generate Novel Drug Molecules</h3>
    <p>Sample from the trained VAE's latent space to create entirely new
    molecular structures. Adjust temperature to control diversity.</p>
    </div>
    """, unsafe_allow_html=True)

    if vae is None:
        st.warning("⚠️ VAE model not trained yet. Run `python src/train_vae.py` first.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            n_molecules = st.slider("Number of molecules", 1, 50, 12,
                                    help="How many molecules to generate")
        with col2:
            temperature = st.slider("Temperature", 0.3, 2.0, 1.0, 0.1,
                                    help="Lower = more similar to training data, "
                                         "Higher = more diverse/novel")
        with col3:
            sampling_mode = st.selectbox("Sampling mode",
                                         ["Greedy", "Diverse (multinomial)"])

        if st.button("🧬 Generate Molecules", use_container_width=True):
            with st.spinner("Generating molecules from latent space..."):
                if sampling_mode == "Greedy":
                    generated = vae.sample(n=n_molecules, device=DEVICE,
                                          temperature=temperature)
                else:
                    generated = vae.sample_diverse(n=n_molecules, device=DEVICE,
                                                   temperature=temperature)

            # Display results
            valid_count = 0
            cols = st.columns(4)
            for i, smi in enumerate(generated):
                mol = Chem.MolFromSmiles(smi)
                is_valid = mol is not None

                with cols[i % 4]:
                    if is_valid:
                        valid_count += 1
                        img = Draw.MolToImage(mol, size=(250, 250))
                        st.image(img, caption=f"✅ {smi[:40]}",
                                 use_container_width=True)

                        props = compute_properties(smi)
                        if props:
                            st.markdown(
                                f"**MW:** {props['MolWt']:.0f} · "
                                f"**LogP:** {props['LogP']:.1f} · "
                                f"**QED:** {props['QED']:.2f}"
                            )

                        # GNN prediction
                        if gnn is not None:
                            gnn_pred = predict_with_gnn(gnn, gnn_stats, smi)
                            if gnn_pred:
                                st.caption(
                                    f"🔬 GNN: MW={gnn_pred['MolWt']:.0f}, "
                                    f"LogP={gnn_pred['LogP']:.1f}"
                                )

                        lip = lipinski_pass(smi)
                        st.caption("💊 Lipinski: " + ("✅ Pass" if lip else "❌ Fail"))
                    else:
                        st.markdown(f"❌ Invalid: `{smi[:40]}`")

            st.markdown("---")
            st.metric("Validity Rate", f"{valid_count}/{len(generated)} "
                      f"({valid_count/len(generated):.0%})")


# ━━━ TAB 3: Evaluate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown("""
    <div class="glass-card">
    <h3>📊 Generation Quality Metrics</h3>
    <p>Comprehensive evaluation of the generated molecules' quality, novelty,
    and drug-likeness compared to the training distribution.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load pre-computed metrics
    eval_metrics = load_eval_metrics()

    if eval_metrics:
        # Key metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Validity", f"{eval_metrics['validity']:.1%}")
        m2.metric("🔷 Uniqueness", f"{eval_metrics['uniqueness']:.1%}")
        m3.metric("✨ Novelty", f"{eval_metrics['novelty']:.1%}")
        m4.metric("💊 Lipinski Pass", f"{eval_metrics['lipinski_rate']:.1%}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("🧬 Total Generated", f"{eval_metrics['total_generated']:,}")
        m6.metric("✅ Valid", f"{eval_metrics['valid']:,}")
        m7.metric("🆕 Novel", f"{eval_metrics['novel']:,}")
        m8.metric("💎 Mean QED", f"{eval_metrics['mean_qed']:.3f}")

        st.markdown("---")

        # Load result images
        col1, col2 = st.columns(2)

        eval_img_path = os.path.join(RESULTS_DIR, "evaluation_summary.png")
        if os.path.exists(eval_img_path):
            with col1:
                st.image(eval_img_path, caption="Evaluation Summary",
                         use_container_width=True)

        mol_img_path = os.path.join(RESULTS_DIR, "generated_molecules.png")
        if os.path.exists(mol_img_path):
            with col2:
                st.image(mol_img_path, caption="Sample Generated Molecules",
                         use_container_width=True)

        # VAE training curves
        vae_loss_path = os.path.join(RESULTS_DIR, "vae_loss.png")
        gnn_parity_path = os.path.join(RESULTS_DIR, "gnn_parity.png")

        if os.path.exists(vae_loss_path) or os.path.exists(gnn_parity_path):
            st.markdown("---")
            st.markdown("### 📈 Training Curves")
            col1, col2 = st.columns(2)
            if os.path.exists(vae_loss_path):
                with col1:
                    st.image(vae_loss_path, caption="VAE Training Loss",
                             use_container_width=True)
            if os.path.exists(gnn_parity_path):
                with col2:
                    st.image(gnn_parity_path, caption="GNN Predicted vs Actual",
                             use_container_width=True)
    else:
        st.info("📊 Run the evaluation pipeline first:\n\n"
                "```bash\npython src/evaluate.py\n```")

        # Show training metrics if available
        vae_metrics_path = os.path.join(RESULTS_DIR, "vae_metrics.json")
        gnn_metrics_path = os.path.join(RESULTS_DIR, "gnn_metrics.json")

        if os.path.exists(vae_metrics_path):
            with open(vae_metrics_path) as f:
                vm = json.load(f)
            st.markdown("#### 🧠 VAE Training Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Loss", f"{vm.get('best_loss', 'N/A'):.4f}")
            c2.metric("Final Validity", f"{vm.get('final_validity', 0):.1%}")
            c3.metric("Epochs", vm.get("total_epochs", "N/A"))

        if os.path.exists(gnn_metrics_path):
            with open(gnn_metrics_path) as f:
                gm = json.load(f)
            st.markdown("#### 🔬 GNN Training Summary")
            cols = st.columns(len(GNN_TARGET_PROPS))
            for i, prop in enumerate(GNN_TARGET_PROPS):
                if prop in gm:
                    cols[i].metric(f"{prop} R²", f"{gm[prop]['r2']:.3f}")


# ━━━ TAB 4: Explore ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab4:
    st.markdown("""
    <div class="glass-card">
    <h3>🔬 Explore Generated Molecules</h3>
    <p>Search, filter, and compare generated molecules by their predicted
    properties. Find optimal drug candidates.</p>
    </div>
    """, unsafe_allow_html=True)

    # Try to load generated molecules
    gen_path = os.path.join(RESULTS_DIR, "generated_molecules.csv")
    gen_props_path = os.path.join(RESULTS_DIR, "generated_properties.csv")

    if os.path.exists(gen_path):
        gen_df = pd.read_csv(gen_path)

        # If we have GNN predictions, merge them
        if os.path.exists(gen_props_path):
            props_df = pd.read_csv(gen_props_path)
            gen_df = props_df  # Use the one with properties
        else:
            # Compute properties using RDKit
            props_list = []
            for smi in gen_df["SMILES"]:
                p = compute_properties(smi)
                if p:
                    p["SMILES"] = smi
                    props_list.append(p)
            if props_list:
                gen_df = pd.DataFrame(props_list)

        st.markdown(f"**{len(gen_df):,} generated molecules available**")

        # Filters
        st.markdown("#### 🎚️ Filter by Properties")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            mw_range = st.slider(
                "Molecular Weight",
                float(gen_df["MolWt"].min()),
                float(gen_df["MolWt"].max()),
                (float(gen_df["MolWt"].min()), float(gen_df["MolWt"].max()))
            )
        with fc2:
            logp_range = st.slider(
                "LogP",
                float(gen_df["LogP"].min()),
                float(gen_df["LogP"].max()),
                (float(gen_df["LogP"].min()), float(gen_df["LogP"].max())),
                step=0.1
            )
        with fc3:
            if "QED" in gen_df.columns:
                qed_range = st.slider(
                    "QED Score",
                    0.0, 1.0,
                    (0.0, 1.0), step=0.05
                )
            else:
                qed_range = (0.0, 1.0)
        with fc4:
            if "TPSA" in gen_df.columns:
                tpsa_range = st.slider(
                    "TPSA",
                    float(gen_df["TPSA"].min()),
                    float(gen_df["TPSA"].max()),
                    (float(gen_df["TPSA"].min()), float(gen_df["TPSA"].max()))
                )
            else:
                tpsa_range = None

        # Apply filters
        mask = (
            (gen_df["MolWt"] >= mw_range[0]) & (gen_df["MolWt"] <= mw_range[1]) &
            (gen_df["LogP"] >= logp_range[0]) & (gen_df["LogP"] <= logp_range[1])
        )
        if "QED" in gen_df.columns:
            mask &= (gen_df["QED"] >= qed_range[0]) & (gen_df["QED"] <= qed_range[1])
        if tpsa_range and "TPSA" in gen_df.columns:
            mask &= (gen_df["TPSA"] >= tpsa_range[0]) & (gen_df["TPSA"] <= tpsa_range[1])

        filtered = gen_df[mask]
        st.markdown(f"**Showing {len(filtered):,} molecules** matching filters")

        # Data table
        st.dataframe(
            filtered.head(100).style.format({
                "MolWt": "{:.1f}",
                "LogP": "{:.2f}",
                "QED": "{:.3f}",
                "TPSA": "{:.1f}",
            }),
            use_container_width=True,
            height=400
        )

        # Molecule comparison
        st.markdown("---")
        st.markdown("#### 🔍 Compare Molecules")

        selected = st.multiselect(
            "Select SMILES to compare (paste or pick from table)",
            options=filtered["SMILES"].head(50).tolist(),
            max_selections=4,
            default=filtered["SMILES"].head(2).tolist() if len(filtered) >= 2 else []
        )

        if selected:
            comp_cols = st.columns(len(selected))
            for i, smi in enumerate(selected):
                with comp_cols[i]:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, use_container_width=True)
                    st.caption(f"`{smi[:50]}`")
                    props = compute_properties(smi)
                    if props:
                        for k, v in props.items():
                            st.metric(k, f"{v}")
    else:
        st.info("🔬 Generate molecules first:\n\n"
                "```bash\npython src/evaluate.py\n```")

        # Manual molecule analyzer
        st.markdown("---")
        st.markdown("#### 🧪 Analyze a Custom Molecule")
        custom_smi = st.text_input("Enter a SMILES string",
                                    value="CC(=O)OC1=CC=CC=C1C(O)=O",
                                    help="e.g., Aspirin: CC(=O)OC1=CC=CC=C1C(O)=O")

        if custom_smi:
            mol = Chem.MolFromSmiles(custom_smi)
            if mol:
                col1, col2 = st.columns([1, 2])
                with col1:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption="2D Structure", use_container_width=True)
                with col2:
                    props = compute_properties(custom_smi)
                    if props:
                        for k, v in props.items():
                            st.metric(k, f"{v}")
                    lip = lipinski_pass(custom_smi)
                    st.metric("Lipinski Rule of 5",
                              "✅ Pass" if lip else "❌ Fail")

                    if gnn is not None:
                        st.markdown("**🔬 GNN Predictions:**")
                        gnn_pred = predict_with_gnn(gnn, gnn_stats, custom_smi)
                        if gnn_pred:
                            for k, v in gnn_pred.items():
                                st.metric(f"GNN {k}", f"{v:.3f}")
            else:
                st.error(f"❌ Invalid SMILES: `{custom_smi}`")
