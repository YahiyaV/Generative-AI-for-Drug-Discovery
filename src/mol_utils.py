"""
mol_utils.py — Molecular utility functions.

Provides SMILES ↔ Graph conversion, fingerprints, tokenization,
and visualisation helpers used across all modules.
"""

import os
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, DataStructs
from rdkit.Chem import QED as QEDModule
from torch_geometric.data import Data
from PIL import Image

from src.config import MAX_SMILES_LEN


# ─── Atom & Bond Feature Definitions ─────────────────────────────────────────

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se",
              "Na", "K", "Ca", "Mg", "Zn", "Fe", "Cu", "other"]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices):
    """One-hot encode a value given a list of choices."""
    vec = [0] * (len(choices) + 1)  # +1 for unknown
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


# ─── SMILES → Graph ──────────────────────────────────────────────────────────

def atom_features(atom) -> List[float]:
    """Extract feature vector for a single atom."""
    symbol = atom.GetSymbol()
    features = []

    # Atom type one-hot (20 + 1)
    features += one_hot(symbol, ATOM_TYPES)

    # Degree one-hot (0-5)
    features += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

    # Formal charge
    features.append(atom.GetFormalCharge())

    # Num Hs
    features.append(atom.GetTotalNumHs())

    # Hybridization one-hot
    features += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)

    # Aromaticity
    features.append(1 if atom.GetIsAromatic() else 0)

    return features


def bond_features(bond) -> List[float]:
    """Extract feature vector for a single bond."""
    features = one_hot(bond.GetBondType(), BOND_TYPES)
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Node features: atom type, degree, charge, Hs, hybridization, aromaticity (36-dim)
    Edge features: bond type, conjugation, ring membership (7-dim)

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge index & edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Undirected: add both directions
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_TYPES) + 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data


def get_node_feature_dim() -> int:
    """Return the dimensionality of node features."""
    mol = Chem.MolFromSmiles("C")
    return len(atom_features(mol.GetAtomWithIdx(0)))


# ─── Fingerprints & Similarity ───────────────────────────────────────────────

def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two molecules."""
    fp_a = compute_fingerprint(smiles_a)
    fp_b = compute_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def max_tanimoto_to_set(smiles: str, reference_fps: list) -> float:
    """Find the maximum Tanimoto similarity of a molecule to a reference set."""
    fp = compute_fingerprint(smiles)
    if fp is None:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return max(sims) if sims else 0.0


# ─── SMILES Tokenizer ────────────────────────────────────────────────────────

import re

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Regex pattern for SMILES tokenization — handles multi-char tokens first
SMILES_REGEX = re.compile(
    r"(\[.*?\]|"         # Bracketed atoms e.g. [NH2+], [O-]
    r"Br|Cl|Si|Se|Na|"   # Two-letter elements
    r"[A-Z][a-z]?|"      # Other elements (C, N, O, c, n, etc.)
    r"[=#\-\+\(\)\[\]\/\\@\.\%]|"  # Bonds and structural chars
    r"\d)"               # Ring closure digits
)


def tokenize_smiles(smiles: str) -> list:
    """Tokenize a SMILES string into a list of tokens using regex."""
    return SMILES_REGEX.findall(smiles)


class SmilesTokenizer:
    """Regex-based SMILES tokenizer that properly handles multi-char tokens."""

    def __init__(self):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # Build vocab from common SMILES tokens
        common_tokens = list(
            "CNOSFIPBKcnos=#()-+[]1234567890@/\\."
        ) + ["Cl", "Br", "Si", "Se", "Na", "Mg", "Zn", "Fe", "Cu",
             "[nH]", "[NH]", "[NH2+]", "[NH3+]", "[N+]", "[N-]",
             "[O-]", "[OH]", "[S-]", "[n+]", "[o]", "[se]",
             "%10", "%11", "%12"]

        # Deduplicate while preserving order
        seen = set()
        unique_tokens = []
        for t in common_tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        self.chars = self.special_tokens + unique_tokens
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_idx = self.char_to_idx[PAD_TOKEN]
        self.sos_idx = self.char_to_idx[SOS_TOKEN]
        self.eos_idx = self.char_to_idx[EOS_TOKEN]
        self.unk_idx = self.char_to_idx[UNK_TOKEN]

    def _ensure_token(self, token: str) -> int:
        """Get index for a token, adding to vocab if unseen."""
        if token not in self.char_to_idx:
            idx = len(self.chars)
            self.chars.append(token)
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
            self.vocab_size = len(self.chars)
        return self.char_to_idx[token]

    def encode(self, smiles: str, max_len: int = MAX_SMILES_LEN) -> list:
        """Encode SMILES string to list of token indices."""
        tokens = [self.sos_idx]
        smiles_tokens = tokenize_smiles(smiles)
        for tok in smiles_tokens[:max_len - 2]:  # Leave room for SOS + EOS
            tokens.append(self._ensure_token(tok))
        tokens.append(self.eos_idx)
        # Pad
        while len(tokens) < max_len:
            tokens.append(self.pad_idx)
        return tokens

    def decode(self, indices: list) -> str:
        """Decode token indices back to SMILES string."""
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            tok = self.idx_to_char.get(idx, "")
            if tok == UNK_TOKEN:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def batch_encode(self, smiles_list: list,
                     max_len: int = MAX_SMILES_LEN) -> torch.Tensor:
        """Encode a list of SMILES to a padded tensor."""
        encoded = [self.encode(s, max_len) for s in smiles_list]
        return torch.tensor(encoded, dtype=torch.long)

    def freeze(self):
        """Freeze vocab size after building from training data."""
        self.vocab_size = len(self.chars)


# Global tokenizer instance
TOKENIZER = SmilesTokenizer()


# ─── Visualisation ────────────────────────────────────────────────────────────

def draw_molecule(smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[Image.Image]:
    """Draw a 2D molecule image from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def draw_molecules_grid(smiles_list: List[str], mols_per_row: int = 4,
                        sub_img_size: Tuple[int, int] = (300, 300)) -> Optional[Image.Image]:
    """Draw a grid of molecules."""
    mols = []
    legends = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            mols.append(mol)
            legends.append(s[:30])  # Truncate long SMILES for legend

    if not mols:
        return None

    return Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row,
        subImgSize=sub_img_size, legends=legends
    )


# ─── Property Helpers ─────────────────────────────────────────────────────────

def compute_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Compute key molecular properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "QED": round(QEDModule.qed(mol), 4),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
    }


def lipinski_pass(smiles: str) -> bool:
    """Check if a molecule passes Lipinski's Rule of 5."""
    props = compute_properties(smiles)
    if props is None:
        return False
    violations = 0
    if props["MolWt"] > 500:
        violations += 1
    if props["LogP"] > 5:
        violations += 1
    if props["HBD"] > 5:
        violations += 1
    if props["HBA"] > 10:
        violations += 1
    return violations <= 1  # Allow 1 violation
