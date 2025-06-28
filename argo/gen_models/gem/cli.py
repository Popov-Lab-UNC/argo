# gen_models/gem/cli.py

import argparse
import os
import time
import logging

from .workflow import run_generation_workflow
from .utils import load_smiles_from_file

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser("GEM: Generative Fine-Tuning and Molecule Generation")
    
    # Input/Output Arguments
    parser.add_argument("--inpath", type=str, required=True, help="Path to input CSV file with SMILES and optional scores.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to store saved files.")
    parser.add_argument("--smi_col", type=str, default='SMILES', help="Name of the SMILES column in the input file.")
    parser.add_argument("--score_col", type=str, default='Score', help="Name of the score column (used in 'filter' mode).")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for all output files (defaults to timestamp).")
    parser.add_argument("--save_models", action="store_true", help="Save the fine-tuned generator and trained classifier.")
    parser.add_argument("--save_aux_files", action="store_true", help="Save auxiliary files like hit lists and status updates.")

    # Model and Execution Arguments
    parser.add_argument("--mode", type=str, required=True, choices=['filter', 'bias_set'], help="Workflow mode: 'filter' (use scores) or 'bias_set' (use all SMILES).")
    parser.add_argument("--gen_model_path", type=str, required=True, help="Path to the pre-trained generative model (.pt file).")
    parser.add_argument("--clf_model_path", type=str, default=None, help="Path to a pre-trained classifier model (.joblib file).")
    parser.add_argument("--gen_model_tuned", action="store_true", help="Flag if the generative model is already fine-tuned.")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for model training and generation if available.")
    
    # Generation and Fine-Tuning Hyperparameters
    parser.add_argument("--tot_hits", type=int, default=1000, help="Total number of unique molecules to generate.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for molecule generation.")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Number of epochs for fine-tuning.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--score_quantile", type=float, default=0.01, help="Quantile of scores to define 'top' compounds for fine-tuning in 'filter' mode.")

    # Filtering Hyperparameters
    parser.add_argument("--no_filter", action="store_true", help="Disable filtering of generated molecules with the classifier.")
    parser.add_argument("--conf_thresh", type=float, default=0.6, help="Confidence threshold for the classifier to accept a molecule.")
    parser.add_argument("--no_cluster", action="store_true", help="Disable clustering of the biasing set.")
    parser.add_argument("--cluster_thresh", type=float, default=0.35, help="Tanimoto distance threshold for Butina clustering.")
    parser.add_argument("--diverse_thresh", type=float, default=None, help="Maximum Tanimoto similarity a new molecule can have to the existing generated set.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    logging.info(f"Loading data from {args.inpath}")
    smiles, labels = load_smiles_from_file(args.inpath, args.smi_col, args.score_col, delimiter=',')

    # Run the workflow
    generated_hits = run_generation_workflow(
        mode=args.mode,
        smiles=smiles,
        labels=labels,
        gen_model_path=args.gen_model_path,
        clf_model_path=args.clf_model_path,
        use_cuda=args.use_cuda,
        out_dir=args.outdir,
        prefix=args.prefix,
        save_models=args.save_models,
        save_auxiliary_files=args.save_aux_files,
        tot_hits=args.tot_hits,
        batch_size=args.batch_size,
        gen_model_tuned=args.gen_model_tuned,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        score_quantile=args.score_quantile,
        cluster_biasing_set=(not args.no_cluster),
        cluster_threshold=args.cluster_thresh,
        do_filter=(not args.no_filter),
        conf_thresh=args.conf_thresh,
        diverse_thresh=args.diverse_thresh,
    )

    logging.info(f"Successfully generated {len(generated_hits)} molecules.")
    logging.info(f"Total time: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()