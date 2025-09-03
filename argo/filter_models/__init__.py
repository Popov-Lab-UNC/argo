import os
import logging
from typing import List, Optional
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances
from rdkit import Chem
import torch

from chemprop.data import BatchMolGraph
from argo.gen_models.gem import utils

class SmilesFilterModel:
    def __init__(self, model: Optional[RandomForestClassifier] = None):
        self.model = model

    def train(self, smiles: List[str], labels: np.ndarray):
        valid_smiles, x, y = utils.get_fps(smiles, labels, func="rdkit")
        if len(x) == 0:
            logging.error("No valid molecules found to train the classifier. Aborting training.")
            return None
        nan_mask = ~np.isnan(x).any(axis=1)
        x, y = x[nan_mask], y[nan_mask]
        if len(x) == 0:
            logging.error("All valid molecules resulted in NaN fingerprints. Aborting training.")
            return None
        self.model = RandomForestClassifier(n_jobs=-1, class_weight='balanced').fit(x, y)
        return self.model

    def save(self, save_path: str):
        if self.model is not None:
            dump(self.model, save_path)
            logging.info(f"Classifier model saved to {save_path}")
        else:
            logging.warning("No model to save.")

    @classmethod
    def load(cls, load_path: str):
        model = load(load_path)
        logging.info(f"Loaded classifier model from {load_path}")
        return cls(model)

    def filter(self, smiles: List[str], conf_thresh: float = 0.6) -> List[tuple[str, float]]:
        """
        Filter molecules based on the trained classifier.
        
        Args:
            smiles: List of SMILES strings to filter
            conf_thresh: Confidence threshold for positive class prediction
            
        Returns:
            List of tuples (smiles, score) where score is the probability of positive class
        """
        if self.model is None:
            raise ValueError("No trained model available for filtering.")
        valid_smiles, fps, _ = utils.get_fps(smiles, func="rdkit")
        if len(fps) == 0:
            return []
        good_nan_mask = ~np.any(np.isnan(fps), axis=1)
        if not np.any(good_nan_mask):
            return []
        fps_no_nan = np.array(fps)[good_nan_mask]
        candidates_no_nan = np.array(valid_smiles)[good_nan_mask]
        probs = self.model.predict_proba(fps_no_nan)[:, 1]
        
        # Create list of (smiles, score) tuples
        smiles_scores = [(candidates_no_nan[i], probs[i]) for i in range(len(candidates_no_nan))]
        
        # Filter molecules based on threshold
        filtered_smiles_scores = [(smiles, score) for smiles, score in smiles_scores if score >= conf_thresh]
        
        return filtered_smiles_scores 

class ChemeleonFilterModel:
    """
    A filter model that uses CheMeleon fingerprints to compare molecules against positive controls.
    Molecules are scored based on their similarity to the positive control set.
    """
    
    def __init__(self, positive_controls: list[str], batch_size: int = 128):
        """
        Initialize the Chemeleon filter model.
        
        Args:
            positive_controls: List of SMILES strings representing positive control molecules
            batch_size: Number of molecules to process at once (to avoid memory issues)
        """
        try:
            from chemprop import featurizers, nn
            from chemprop.data import BatchMolGraph
            from chemprop.nn import RegressionFFN
            from chemprop.models import MPNN
            from pathlib import Path
            from urllib.request import urlretrieve
        except ImportError:
            raise ImportError("ChemeleonFilterModel requires chemprop>=2.2.0. Install with: pip install 'chemprop>=2.2.0'")
        
        self.positive_controls = positive_controls
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize CheMeleon fingerprint model
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        
        # Download and load the CheMeleon model
        ckpt_dir = Path().home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()
        self.model.to(device=self.device)
        
        # Generate fingerprints for positive controls
        self.positive_fingerprints = self._generate_fingerprints(positive_controls)
        
    def _generate_fingerprints(self, smiles_list: list[str]) -> np.ndarray:
        """Generate CheMeleon fingerprints for a list of SMILES strings in batches."""
        try:
            mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            mols = [mol for mol in mols if mol is not None]  # Filter out None values
            
            if not mols:
                return np.array([])
            
            all_fingerprints = []
            
            # Process in batches to avoid memory issues
            for i in range(0, len(mols), self.batch_size):
                batch_mols = mols[i:i + self.batch_size]
                
                try:
                    bmg = BatchMolGraph([self.featurizer(mol) for mol in batch_mols])
                    bmg.to(device=self.model.device)
                    batch_fingerprints = self.model.fingerprint(bmg).numpy(force=True)
                    all_fingerprints.append(batch_fingerprints)
                    
                    # Clear GPU memory after each batch
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {i//self.batch_size + 1}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
            
            if not all_fingerprints:
                return np.array([])
                
            return np.vstack(all_fingerprints)
            
        except Exception as e:
            print(f"Error generating fingerprints: {e}")
            return np.array([])
    
    def filter(self, smiles_list: list[str], distance_threshold: float = 0.2, no_identical: bool = True) -> list[tuple[str, float]]:
        """
        Filter molecules based on their distance to positive controls.
        
        Args:
            smiles_list: List of SMILES strings to filter
            distance_threshold: Maximum distance threshold
                              Distance = 1 - similarity, so 0.0 = identical, 1.0 = completely different
            no_identical: If True, exclude molecules identical to positive controls
            
        Returns:
            List of tuples (smiles, score) where score is the minimum distance to positive controls
        """
    
        if len(self.positive_fingerprints) == 0:
            print("Warning: No valid positive control fingerprints available")
            return [(smiles, 0.0) for smiles in smiles_list]
        
        # Generate fingerprints for input molecules
        # Important: _generate_fingerprints filters out invalid SMILES (None mols),
        # so we must mirror that filtering here to keep indices aligned
        parsed_mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        valid_smiles = [smiles for smiles, mol in zip(smiles_list, parsed_mols) if mol is not None]
        input_fingerprints = self._generate_fingerprints(valid_smiles)
        
        if len(input_fingerprints) == 0:
            print("Warning: No valid fingerprints generated for input molecules")
            return []
        
        # Calculate similarity to positive controls
        distances = cosine_distances(input_fingerprints, self.positive_fingerprints)
        
        min_distances = np.min(distances, axis=1)
        
        # Create list of (smiles, score) tuples aligned with valid_smiles only
        smiles_scores = [(valid_smiles[i], float(min_distances[i])) for i in range(len(valid_smiles))]
        
        # Filter molecules based on threshold
        filtered_smiles_scores = [(smiles, score) for smiles, score in smiles_scores if score <= distance_threshold]
        
        if no_identical:
            filtered_smiles_scores = [(smiles, score) for smiles, score in filtered_smiles_scores if score > 0.0]
        
        print(f"Filtered {len(smiles_list)} molecules ({len(valid_smiles)} valid): {len(filtered_smiles_scores)} passed (distance threshold: {distance_threshold:.3f})")
        
        return filtered_smiles_scores