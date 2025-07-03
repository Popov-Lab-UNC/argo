# gen_models/gem/workflow.py

import os
import logging
import time
from datetime import datetime
from typing import Optional, List

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from . import utils
from .model import Transformer, SmilesDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_pretrained_transformer(model_path: str, device) -> Transformer:
    """Loads a pre-trained Transformer model."""
    n_src_vocab = len(utils.TOKENS)
    model_params = {
        'n_src_vocab': n_src_vocab, 'd_word_vec': 512, 'n_layers': 8,
        'n_head': 8, 'd_k': 64, 'd_inner': 1024
    }
    model = Transformer(**model_params)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    logging.info(f"Loaded pre-trained Transformer model from {model_path}")
    return model

def _fine_tune_model(model: Transformer, smiles_data: List[str], lr: float, n_epochs: int, device, save_path: Optional[str] = None):
    """Fine-tunes the generative model on a given set of SMILES."""
    logging.info(f"Starting fine-tuning for {n_epochs} epochs with {len(smiles_data)} SMILES.")
    dataset = SmilesDataset(smiles_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.fit(loader, optimizer, scheduler=None, n_epochs=n_epochs, device=device)
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Fine-tuned model saved to {save_path}")
    return model

# --- Deprecated Filtering Model Logic (moved to filter_models) ---
# def _train_rf_classifier(smiles: List[str], labels: np.ndarray, save_path: Optional[str] = None):
#     """Trains a RandomForest classifier for filtering."""
#     logging.info("Training RandomForest classifier...")
#     valid_smiles, x, y = utils.get_fps(smiles, labels, func="rdkit")
#     
#     if len(x) == 0:
#         logging.error("No valid molecules found to train the classifier. Aborting training.")
#         return None
# 
#     # Remove any rows with NaNs that might result from failed fingerprinting
#     nan_mask = ~np.isnan(x).any(axis=1)
#     x, y = x[nan_mask], y[nan_mask]
# 
#     if len(x) == 0:
#         logging.error("All valid molecules resulted in NaN fingerprints. Aborting training.")
#         return None
# 
#     clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced').fit(x, y)
#     
#     if save_path:
#         dump(clf, save_path)
#         logging.info(f"Classifier model saved to {save_path}")
#     return clf

def run_generation_workflow(
    mode: str,
    smiles: List[str],
    gen_model_path: str,
    use_cuda: bool,
    labels: Optional[List[float]] = None,
    clf_model_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    save_models: bool = False,
    save_files: bool = False,
    # Generation parameters
    tot_hits: int = 1000,
    batch_size: int = 100,
    # Fine-tuning parameters
    gen_model_tuned: bool = False,
    fine_tune_epochs: int = 10,
    fine_tune_lr: float = 1e-5,
    # Bias set preparation parameters
    score_quantile: float = 0.01,
    cluster_biasing_set: bool = True,
    cluster_threshold: float = 0.35,
    # Filtering parameters
    do_filter: bool = True,
    conf_thresh: float = 0.6,
    diverse_thresh: Optional[float] = None
):
    """
    Main workflow for fine-tuning a generative model and producing novel molecules.
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")

    # --- 1. Setup directories and paths ---
    if save_models or save_files:
        out_dir = out_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        prefix = prefix or str(int(time.time()))

    # --- 2. Prepare Biasing Set ---
    logging.info("Preparing biasing set for fine-tuning...")
    smiles, labels = utils.canonicalize_smiles(smiles, np.array(labels) if labels else None)
    
    if mode == 'filter':
        if labels is None:
            raise ValueError("'labels' must be provided for 'filter' mode.")
        score_thresh = np.quantile(labels, score_quantile)
        is_top_scoring = (labels <= score_thresh)
        biasing_smiles = [s for s, top in zip(smiles, is_top_scoring) if top]
        logging.info(f"Selected {len(biasing_smiles)} top-scoring SMILES using {score_quantile} quantile.")
    elif mode == 'bias_set':
        biasing_smiles = smiles
        logging.info(f"Using provided {len(biasing_smiles)} SMILES as the biasing set.")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'filter' or 'bias_set'.")
    
    if cluster_biasing_set:
        clusters = utils.cluster_butina(biasing_smiles, dist_thresh=cluster_threshold)
        # Select one representative from each cluster (the first one)
        biasing_smiles = [biasing_smiles[c[0]] for c in clusters]
        logging.info(f"Clustered biasing set to {len(biasing_smiles)} representatives.")

    # --- 3. Prepare Models (Classifier and Generator) ---
    clf = None
    # --- Filtering logic deprecated: now handled by filter_models ---
    # if do_filter:
    #     if clf_model_path and os.path.exists(clf_model_path):
    #         clf = load(clf_model_path)
    #         logging.info(f"Loaded pre-trained classifier from {clf_model_path}")
    #     elif mode == 'filter':
    #         is_active = (labels <= np.quantile(labels, score_quantile)).astype(int)
    #         clf_save_path = os.path.join(out_dir, f"{prefix}_RF.joblib") if save_models else None
    #         clf = _train_rf_classifier(smiles, is_active, save_path=clf_save_path)
    #     else:
    #         logging.warning("Filtering is enabled ('do_filter'=True) but no classifier path was provided and mode is not 'filter'. Cannot train a new one. Filtering will be skipped.")
    #         do_filter = False

    gen_model = _load_pretrained_transformer(gen_model_path, device)
    if not gen_model_tuned:
        tuned_model_save_path = os.path.join(out_dir, f"{prefix}_tuned.pt") if save_models else None
        gen_model = _fine_tune_model(gen_model, biasing_smiles, fine_tune_lr, fine_tune_epochs, device, save_path=tuned_model_save_path)

    # --- 4. Generation Loop ---
    logging.info(f"Starting generation to find {tot_hits} total hits.")
    generated_hits = set()
    initial_smiles_set = set(biasing_smiles)

    num_epochs = 0
    while len(generated_hits) < tot_hits:
        num_epochs += 1
        raw_batch = gen_model.generate(batch_size, device)
        # Canonicalize and filter out invalid/empty SMILES from generation
        batch_smiles, _ = utils.canonicalize_smiles(raw_batch)
        
        # Filter out SMILES that were in the original biasing set or already generated
        new_candidates = [s for s in batch_smiles if s not in initial_smiles_set and s not in generated_hits]
        
        if not new_candidates:
            continue
            
        # Optional Filtering: Classifier
        # --- Filtering logic deprecated: now handled by filter_models ---
        # if do_filter and clf:
        #     valid_candidates, fps, _ = utils.get_fps(new_candidates, func="rdkit")
        #     
        #     if len(fps) > 0:
        #         good_nan_mask = ~np.any(np.isnan(fps), axis=1)
        #         
        #         if np.any(good_nan_mask):
        #             fps_no_nan = fps[good_nan_mask]
        #             candidates_no_nan = np.array(valid_candidates)[good_nan_mask]
        #
        #             probs = clf.predict_proba(fps_no_nan)[:, 1]
        #             passing_indices = np.where(probs >= conf_thresh)[0]
        #             
        #             new_candidates = candidates_no_nan[passing_indices].tolist()
        #         else:
        #             new_candidates = []
        #     else:
        #         new_candidates = []

        # Optional Filtering: Diversity
        if diverse_thresh is not None and len(generated_hits) > 0 and len(new_candidates) > 0:
            query_smiles, query_fps, _ = utils.get_fps(new_candidates, func="morgan")
            
            if len(query_fps) > 0:
                # We only need fingerprints for the target set
                _, target_fps, _ = utils.get_fps(list(generated_hits), func="morgan")
                if len(target_fps) > 0:
                    sims = utils.bulk_fp_tanimoto_similarity(query_fps, target_fps, pooling="max")
                    # Filter the list of SMILES that corresponds to the query_fps
                    new_candidates = [s for s, sim in zip(query_smiles, sims) if sim < diverse_thresh]

        for hit in new_candidates:
            if len(generated_hits) < tot_hits:
                generated_hits.add(hit)
        
        logging.info(f"Progress: {len(generated_hits)} / {tot_hits} hits generated.")

        if save_files:
            with open(os.path.join(out_dir, f"{prefix}_denovo_hits.smi"), "w") as f:
                f.write("\n".join(generated_hits))
            with open(os.path.join(out_dir, f"{prefix}_status.txt"), 'w') as f:
                f.write(f"Num Epochs: {num_epochs}\nNum Hits: {len(generated_hits)}\nLast Update: {str(datetime.now())}\n")

    return list(generated_hits)