"""
server.py — Flask API backend for Generative AI Drug Discovery.

Run with: python server.py
"""

import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem
from rdkit.Chem import Draw

from src.config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, PROCESSED_DIR,
    GNN_TARGET_PROPS, VAE_LATENT_DIM
)
from src.mol_utils import (
    compute_properties, lipinski_pass, draw_molecule,
    smiles_to_graph, get_node_feature_dim,
    TOKENIZER
)
from src.vae_model import SmilesVAE
from src.gnn_model import MoleculeGNN

app = Flask(__name__, static_folder='static', static_url_path='')

# ─── Model Loading ────────────────────────────────────────────────────────────

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

def load_training_data():
    """Load the training dataset."""
    csv_path = os.path.join(PROCESSED_DIR, "molecules.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_eval_metrics():
    """Load pre-computed evaluation metrics."""
    path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_generated_data():
    """Load generated molecules and their properties."""
    gen_path = os.path.join(RESULTS_DIR, "generated_molecules.csv")
    gen_props_path = os.path.join(RESULTS_DIR, "generated_properties.csv")
    
    if os.path.exists(gen_path):
        gen_df = pd.read_csv(gen_path)
        if os.path.exists(gen_props_path):
            props_df = pd.read_csv(gen_props_path)
            return props_df
        else:
            # Generate basic properties if full file missing
            props_list = []
            for smi in gen_df["SMILES"]:
                p = compute_properties(smi)
                if p:
                    p["SMILES"] = smi
                    props_list.append(p)
            if props_list:
                return pd.DataFrame(props_list)
    return None

# Load models at startup
print("Loading models and data for API server...")
vae_model = load_vae_model()
gnn_model, gnn_stats = load_gnn_model()
train_df = load_training_data()
eval_metrics = load_eval_metrics()
generated_df = load_generated_data()
print("Initialization complete.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

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

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML application file."""
    if not os.path.exists(os.path.join('static', 'index.html')):
        return "Frontend not built yet. Create static/index.html.", 404
    return app.send_static_file('index.html')

@app.route('/api/status')
def status():
    """Return model load status and device information."""
    dataset_size = len(train_df) if train_df is not None else 0
    return jsonify({
        "device": str(DEVICE),
        "vae_ready": vae_model is not None,
        "gnn_ready": gnn_model is not None,
        "dataset_loaded": dataset_size > 0,
        "dataset_size": dataset_size
    })

@app.route('/api/data/stats')
def data_stats():
    """Return dataset statistics."""
    if train_df is None:
        return jsonify({"error": "Dataset not loaded"}), 404
        
    return jsonify({
        "total": len(train_df),
        "avg_mol_wt": round(float(train_df['MolWt'].mean()), 1),
        "avg_qed": round(float(train_df['QED'].mean()), 3),
        "avg_logp": round(float(train_df['LogP'].mean()), 2)
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate molecules using the VAE."""
    if vae_model is None:
        return jsonify({"error": "VAE model not trained"})
        
    data = request.json or {}
    n_molecules = min(int(data.get('n', 12)), 50)
    temperature = float(data.get('temperature', 1.0))
    mode = data.get('mode', 'greedy').lower()
    
    try:
        if mode == 'diverse':
            generated = vae_model.sample_diverse(n=n_molecules, device=DEVICE, temperature=temperature)
        else:
            generated = vae_model.sample(n=n_molecules, device=DEVICE, temperature=temperature)
            
        results = []
        valid_count = 0
        
        for smi in generated:
            mol_data = {"smiles": smi, "is_valid": False}
            mol = Chem.MolFromSmiles(smi)
            
            if mol is not None:
                mol_data["is_valid"] = True
                valid_count += 1
                mol_data["image_b64"] = mol_to_base64(smi)
                
                # RDKit Properties
                props = compute_properties(smi)
                if props:
                    mol_data["properties"] = props
                
                # Lipinski
                mol_data["lipinski_pass"] = lipinski_pass(smi)
                
                # GNN Predictions
                if gnn_model is not None:
                    gnn_pred = predict_with_gnn(gnn_model, gnn_stats, smi)
                    if gnn_pred:
                        mol_data["gnn_predictions"] = gnn_pred
                        
            results.append(mol_data)
            
        return jsonify({
            "molecules": results,
            "total_generated": len(generated),
            "valid_count": valid_count,
            "validity_rate": valid_count / len(generated) if generated else 0
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate')
def evaluate():
    """Return evaluation metrics."""
    global eval_metrics
    if not eval_metrics:
        eval_metrics = load_eval_metrics()
        
    if not eval_metrics:
        # Fallback to try loading training summary
        vae_metrics_path = os.path.join(RESULTS_DIR, "vae_metrics.json")
        gnn_metrics_path = os.path.join(RESULTS_DIR, "gnn_metrics.json")
        
        fallback = {}
        if os.path.exists(vae_metrics_path):
            with open(vae_metrics_path) as f:
                fallback["vae"] = json.load(f)
        if os.path.exists(gnn_metrics_path):
            with open(gnn_metrics_path) as f:
                fallback["gnn"] = json.load(f)
                
        if fallback:
            return jsonify({"status": "partial", "training_only": fallback})
            
        return jsonify({"error": "Evaluation metrics not found"}), 404
        
    return jsonify({
        "status": "complete",
        "metrics": eval_metrics
    })

@app.route('/api/explore')
def explore():
    """Return generated molecules with optional filtering."""
    global generated_df
    if generated_df is None:
        generated_df = load_generated_data()
        
    if generated_df is None:
        return jsonify({"error": "Generated molecules not found"}), 404
        
    # Get filters
    mol_wt_min = request.args.get('molwt_min', type=float, default=0.0)
    mol_wt_max = request.args.get('molwt_max', type=float, default=1000.0)
    logp_min = request.args.get('logp_min', type=float, default=-10.0)
    logp_max = request.args.get('logp_max', type=float, default=15.0)
    qed_min = request.args.get('qed_min', type=float, default=0.0)
    qed_max = request.args.get('qed_max', type=float, default=1.0)
    limit = request.args.get('limit', type=int, default=100)
    
    # Apply filters
    mask = (
        (generated_df["MolWt"] >= mol_wt_min) & (generated_df["MolWt"] <= mol_wt_max) &
        (generated_df["LogP"] >= logp_min) & (generated_df["LogP"] <= logp_max)
    )
    if "QED" in generated_df.columns:
        mask &= (generated_df["QED"] >= qed_min) & (generated_df["QED"] <= qed_max)
        
    filtered = generated_df[mask]
    
    # Check if images are requested
    include_images = request.args.get('images', type=str, default='false').lower() == 'true'
    
    # Take first N
    result_list = filtered.head(limit).to_dict(orient='records')
    
    if include_images:
        for item in result_list:
            item['image_b64'] = mol_to_base64(item['SMILES'])
            
    return jsonify({
        "total_available": len(generated_df),
        "total_filtered": len(filtered),
        "molecules": result_list
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a custom SMILES string."""
    data = request.json or {}
    smiles = data.get('smiles', '')
    
    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({
            "is_valid": False,
            "error_msg": "Invalid SMILES string"
        })
        
    result = {
        "is_valid": True,
        "smiles": smiles,
        "image_b64": mol_to_base64(smiles, size=(300, 300)),
        "lipinski_pass": lipinski_pass(smiles)
    }
    
    # RDKit Properties
    props = compute_properties(smiles)
    if props:
        result["properties"] = props
        
    # GNN predictions
    if gnn_model is not None:
        gnn_pred = predict_with_gnn(gnn_model, gnn_stats, smiles)
        if gnn_pred:
            result["gnn_predictions"] = gnn_pred
            
    return jsonify(result)

@app.route('/api/assets/<path:path>')
def send_asset(path):
    """Serve any images or files from the results directory."""
    return send_from_directory(RESULTS_DIR, path)

if __name__ == '__main__':
    # Ensure static dir exists
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 7860))
    print(f"Server starting on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
