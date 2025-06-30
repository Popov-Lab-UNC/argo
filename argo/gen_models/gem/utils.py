# gen_models/gem/utils.py

import numpy as np
import pandas as pd
from typing import Literal, List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

TOKENS = (
    'X', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2',
    '5', '4', '7', '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O',
    'N', 'P', 'S', '[', ']', '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r',
    '\n'
)
CHAR_2_IDX = {token: i for i, token in enumerate(TOKENS)}

def to_item(lst):
    if isinstance(lst, list):
        return lst[0]
    if isinstance(lst, np.ndarray):
        return lst[0]
    if isinstance(lst, pd.Series):
        return lst.values[0]
    return lst

def calc_rdkit(mol: Chem.Mol, min_path=1, max_path=7, n_bits=2048, n_bits_per_hash=2) -> np.ndarray:
    mol = to_item(mol)
    if mol is None:
        return np.full(n_bits, np.nan)
    _fp = Chem.RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=n_bits, nBitsPerHash=n_bits_per_hash)
    fp = np.zeros(n_bits, dtype=np.int8)
    ConvertToNumpyArray(_fp, fp)
    return fp

def calc_morgan(mol: Chem.Mol, radius=4, n_bits=256, count=False, use_chirality=True) -> DataStructs.ExplicitBitVect:
    mol = to_item(mol)
    if mol is None:
        return np.full(n_bits, np.nan)
    if count:
        _fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)
    else:
        _fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=use_chirality)
    return _fp

def get_fps(smis: List[str], y: Optional[np.ndarray] = None, func: Literal["morgan", "rdkit"] = "morgan") -> Tuple[List[str], list, Optional[np.ndarray]]:
    """
    Calculates fingerprints for a list of SMILES.

    Returns a tuple of (valid_smiles, fingerprints, corresponding_y_values),
    ensuring that all returned lists/arrays have the same length and correspond
    to only the molecules for which fingerprint generation was successful.
    """
    mols = [Chem.MolFromSmiles(s) for s in smis]
    
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    
    if not valid_indices:
        return [], [], (np.array([]) if y is not None else None)

    valid_smiles = [smis[i] for i in valid_indices]
    valid_mols = [mols[i] for i in valid_indices]
    y_out = y[valid_indices] if y is not None else None
    
    if func == "morgan":
        fps = [calc_morgan(m) for m in valid_mols]
    elif func == "rdkit":
        fps = np.array([calc_rdkit(m) for m in valid_mols])
    else:
        raise ValueError(f"Unknown fingerprint function: {func}")
        
    return valid_smiles, fps, y_out

def canonicalize_smiles(smis: List[str], y: Optional[np.ndarray] = None) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Filters for valid SMILES and converts them to a canonical form.
    """
    mols = [Chem.MolFromSmiles(s) for s in smis]
    
    if y is not None:
        valid_pairs = [(m, y_val) for m, y_val in zip(mols, y) if m is not None]
        if not valid_pairs:
            return [], np.array([])
        valid_mols, y_out = zip(*valid_pairs)
        y_out = np.array(y_out)
    else:
        valid_mols = [m for m in mols if m is not None]
        y_out = None

    return [Chem.MolToSmiles(m, isomericSmiles=False) for m in valid_mols], y_out

def load_smiles_from_file(file_path: str, smi_col: str, label_col: Optional[str] = None, **kwargs) -> Tuple[List[str], Optional[List[float]]]:
    df = pd.read_csv(file_path, **kwargs)
    smiles = df[smi_col].tolist()
    labels = df[label_col].tolist() if label_col and label_col in df.columns else None
    return smiles, labels

def bulk_fp_tanimoto_similarity(query_fps: list, target_fps: list, pooling: Optional[Literal['max', 'mean']] = None) -> np.ndarray:
    if not target_fps:
        return np.zeros(len(query_fps))
    
    sims = np.array([DataStructs.BulkTanimotoSimilarity(q_fp, target_fps) for q_fp in query_fps])
    
    if pooling is None:
        return sims
    if pooling == "max":
        return np.max(sims, axis=1)
    if pooling == "mean":
        return np.mean(sims, axis=1)
    raise ValueError(f"Pooling must be in ['max', 'mean'], got {pooling}")

def pdist_tanimoto(smiles: List[str]) -> list:
    valid_smiles, fps, _ = get_fps(smiles, func="morgan")
    if len(fps) < 2:
        return []
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    return dists

def cluster_butina(smiles: List[str], dist_thresh: float = 0.2) -> List[tuple]:
    dists = pdist_tanimoto(smiles)
    if not dists:
        # If no distances, each SMILES is its own cluster
        return [(i,) for i in range(len(smiles))]
    return Butina.ClusterData(dists, len(smiles), distThresh=dist_thresh, isDistData=True)