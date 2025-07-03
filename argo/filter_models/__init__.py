import os
import logging
from typing import List, Optional
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

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

    def filter(self, smiles: List[str], conf_thresh: float = 0.6) -> List[str]:
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
        passing_indices = np.where(probs >= conf_thresh)[0]
        return candidates_no_nan[passing_indices].tolist() 