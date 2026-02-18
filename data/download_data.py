"""
download_data.py — Download and preprocess drug-like molecules.

Strategy:
  1. Try to download from ZINC15 "tranches" (drug-like subset).
  2. Fallback: download from PubChem bulk FTP.
  3. Filter by heavy-atom count, validate SMILES with RDKit.
  4. Compute molecular descriptors (MolWt, LogP, TPSA, QED, etc.).
  5. Save to data/processed/molecules.csv
"""

import os
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QEDModule

from src.config import (
    PROCESSED_DIR, NUM_MOLECULES,
    MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS
)


# ─── ZINC15 Download ─────────────────────────────────────────────────────────

# Drug-like ZINC tranches (2D, SMILES format)
ZINC_TRANCHE_URLS = [
    "https://zinc15.docking.org/tranches/download?representation=smiles&"
    "tranch_type=drug-like&logp_range=a&mwt_range=c&count=all",
]

# Backup: curated ZINC250K from various ML benchmarks  
ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

PUBCHEM_COMPOUND_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    "cid/{start}-{end}/property/CanonicalSMILES,MolecularWeight,"
    "XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/CSV"
)


def download_zinc250k():
    """Download the ZINC250K benchmark dataset (most reliable source)."""
    print("📥 Downloading ZINC250K dataset...")
    try:
        resp = requests.get(ZINC250K_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        # Columns: smiles, logP, qed, SAS
        df = df.rename(columns={"smiles": "SMILES"})
        print(f"  ✅ Downloaded {len(df):,} molecules from ZINC250K")
        return df
    except Exception as e:
        print(f"  ⚠️  ZINC250K download failed: {e}")
        return None


def download_pubchem_batch(start_cid=1, batch_size=500, max_molecules=10000):
    """Download molecules from PubChem REST API in batches."""
    print(f"📥 Downloading from PubChem (CID {start_cid}+)...")
    all_dfs = []
    collected = 0
    cid = start_cid

    pbar = tqdm(total=max_molecules, desc="PubChem", unit="mol")
    while collected < max_molecules:
        end_cid = cid + batch_size - 1
        url = PUBCHEM_COMPOUND_URL.format(start=cid, end=end_cid)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                batch_df = pd.read_csv(io.StringIO(resp.text))
                batch_df = batch_df.rename(columns={"CanonicalSMILES": "SMILES"})
                all_dfs.append(batch_df)
                collected += len(batch_df)
                pbar.update(len(batch_df))
            time.sleep(0.3)  # Be polite to PubChem servers
        except Exception:
            pass
        cid += batch_size

        if cid > start_cid + max_molecules * 5:
            break

    pbar.close()
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None


def validate_smiles(smiles):
    """Check if a SMILES string is valid and within atom-count bounds."""
    if not isinstance(smiles, str) or len(smiles) < 2:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    return MIN_HEAVY_ATOMS <= n_heavy <= MAX_HEAVY_ATOMS


def compute_descriptors(smiles):
    """Compute molecular descriptors for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MolWt": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 3),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "QED": round(QEDModule.qed(mol), 4),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "NumRotBonds": Descriptors.NumRotatableBonds(mol),
            "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            "RingCount": Descriptors.RingCount(mol),
        }
    except Exception:
        return None


def process_dataset(df, target_n=NUM_MOLECULES):
    """Filter, deduplicate, compute descriptors."""
    print("\n🔬 Processing molecules...")

    # Keep only SMILES column if extras exist
    if "SMILES" not in df.columns:
        # Try to find a smiles-like column
        for col in df.columns:
            if "smiles" in col.lower():
                df = df.rename(columns={col: "SMILES"})
                break

    smiles_list = df["SMILES"].dropna().unique().tolist()
    print(f"  Total raw SMILES: {len(smiles_list):,}")

    # Validate
    valid_smiles = []
    for s in tqdm(smiles_list, desc="Validating", unit="mol"):
        if validate_smiles(s):
            valid_smiles.append(s)
        if len(valid_smiles) >= target_n:
            break

    print(f"  Valid molecules: {len(valid_smiles):,}")

    # Compute descriptors
    records = []
    for s in tqdm(valid_smiles, desc="Computing descriptors", unit="mol"):
        desc = compute_descriptors(s)
        if desc is not None:
            desc["SMILES"] = s
            records.append(desc)

    result_df = pd.DataFrame(records)
    # Reorder columns
    cols = ["SMILES", "MolWt", "LogP", "TPSA", "QED",
            "HBD", "HBA", "NumRotBonds", "NumHeavyAtoms", "RingCount"]
    result_df = result_df[cols]

    print(f"  Final dataset: {len(result_df):,} molecules with descriptors")
    return result_df


def main():
    output_path = os.path.join(PROCESSED_DIR, "molecules.csv")

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        print(f"⚡ Dataset already exists with {len(existing):,} molecules.")
        resp = input("Re-download? (y/N): ").strip().lower()
        if resp != "y":
            print("Skipping download.")
            return

    # Strategy 1: ZINC250K (fast & reliable)
    df = download_zinc250k()

    # Strategy 2: PubChem fallback
    if df is None or len(df) < 1000:
        print("Falling back to PubChem API...")
        df = download_pubchem_batch(start_cid=1, max_molecules=NUM_MOLECULES)

    if df is None or len(df) < 100:
        print("❌ Could not download sufficient data. Check internet connection.")
        sys.exit(1)

    # Process
    result = process_dataset(df, target_n=NUM_MOLECULES)

    # Save
    result.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(result):,} molecules → {output_path}")

    # Quick stats
    print("\n📊 Dataset Summary:")
    print(result.describe().round(2).to_string())


if __name__ == "__main__":
    main()
