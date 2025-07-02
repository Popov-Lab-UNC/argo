import os
import re
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import logging

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from decompose_worker import decompose_row
from similarity_worker import compute_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def parallel_decompose(df, slicer_params, n_workers):
    """
    Parallel decomposition of molecules into arms and linkers using SAFE encoding.
    Returns two DataFrames: one for arms, one for linkers.
    """
    args = [(i, smiles, slicer_params) for i, smiles in enumerate(df['smiles'])]
    arm_idx_list, arm_safe_list, arm_list = [], [], []
    linker_idx_list, linker_safe_list, linker_list = [], [], []
    total = len(args)
    logging.info(f"Starting decomposition of {total} molecules with {n_workers} workers...")
    with mp.Pool(n_workers) as pool:
        for idx, result in enumerate(pool.imap(decompose_row, args)):
            for frag_type, idx_, safe, frag in result:
                if frag_type == 'arm':
                    arm_idx_list.append(idx_)
                    arm_safe_list.append(safe)
                    arm_list.append(frag)
                else:
                    linker_idx_list.append(idx_)
                    linker_safe_list.append(safe)
                    linker_list.append(frag)
            if (idx + 1) % max(1, total // 10) == 0 or (idx + 1) == total:
                logging.info(f"Decomposition progress: {idx + 1}/{total} ({100 * (idx + 1) // total}%)")
    logging.info("Decomposition complete.")
    df_arm = pd.DataFrame({'idx': arm_idx_list, 'safe': arm_safe_list, 'frag': arm_list})
    df_linker = pd.DataFrame({'idx': linker_idx_list, 'safe': linker_safe_list, 'frag': linker_list})
    return df_arm, df_linker

def parallel_similarity(df, num_retrieve, n_workers, frag_type, slicer_params):
    """
    Parallel similarity calculation for arm or linker fragments.
    Passes frag_type and slicer_params to the worker for correct molecule reconstruction.
    Returns a DataFrame with new SAFE strings and retrieved fragments.
    """
    chunks = np.array_split(df, n_workers)
    #fps_all = df['fps'].tolist()
    args = [(chunk, df, num_retrieve, frag_type, slicer_params) for chunk in chunks]
    idx_list, input_list, retrieved_list = [], [], []
    total = len(chunks)
    logging.info(f"Starting similarity calculation for {frag_type} with {n_workers} workers...")
    with mp.Pool(n_workers) as pool:
        for idx, (idxs, inputs, retrieveds) in enumerate(pool.imap(compute_similarity, args)):
            idx_list.extend(idxs)
            input_list.extend(inputs)
            retrieved_list.extend(retrieveds)
            if (idx + 1) % max(1, total // 10) == 0 or (idx + 1) == total:
                logging.info(f"Similarity progress for {frag_type}: {idx + 1}/{total} ({100 * (idx + 1) // total}%)")
    logging.info(f"Similarity calculation for {frag_type} complete.")
    return pd.DataFrame({'idx': idx_list, 'input': input_list, 'retrieved': retrieved_list})

if __name__ == '__main__':
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)

    data_dir = '../'
    train_csv = os.path.join(data_dir, 'zinc250k_train.csv')

    # Only run preprocessing if the train CSV does not exist
    if not os.path.exists(train_csv):
        num_retrieve = 10
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))

        # Load the raw ZINC250k dataset
        df = pd.read_csv(os.path.join(data_dir, 'zinc250k.csv'))
        slicer_params = {'shortest_linker': True}

        # --- Decomposition Step ---
        # Decompose molecules into arms and linkers in parallel
        df_arm, df_linker = parallel_decompose(df, slicer_params, n_workers)

        # Generate fingerprints
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)

        # Save df_arm and df_linker
        df_arm.to_csv(os.path.join(data_dir, 'zinc250k_arm.csv'), index=False)
        df_linker.to_csv(os.path.join(data_dir, 'zinc250k_linker.csv'), index=False)

        df_arm['fps'] = [mfpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in df_arm['frag']]
        df_linker['fps'] = [mfpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in df_linker['frag']]

        # Load validation indices and determine training indices
        with open(os.path.join(data_dir, 'valid_idx_zinc250k.json')) as f:
            test_idx = set(json.load(f))
        train_idx = {i for i in range(len(df)) if i not in test_idx}

        # --- Similarity Step for Arms ---
        print("Calculating similarity for arm fragments...")
        df_arm_sim = parallel_similarity(df_arm, num_retrieve, n_workers, 'arm', slicer_params)
        df_arm_sim = df_arm_sim[df_arm_sim['idx'].isin(train_idx)]
        df_arm_sim.drop(columns=['idx'], inplace=True)
        df_arm_sim.to_csv(os.path.join(data_dir, 'zinc250k_arm.csv'), index=False)

        # --- Similarity Step for Linkers ---
        print("Calculating similarity for linker fragments...")
        df_linker_sim = parallel_similarity(df_linker, num_retrieve, n_workers, 'linker', slicer_params)
        df_linker_sim = df_linker_sim[df_linker_sim['idx'].isin(train_idx)]
        df_linker_sim.drop(columns=['idx'], inplace=True)
        df_linker_sim.to_csv(os.path.join(data_dir, 'zinc250k_linker.csv'), index=False)

        # --- Merge and Save ---
        # Concatenate arm and linker results and save as training set
        df_arm = pd.read_csv(os.path.join(data_dir, 'zinc250k_arm.csv'))
        df_linker = pd.read_csv(os.path.join(data_dir, 'zinc250k_linker.csv'))
        df = pd.concat([df_arm, df_linker])
        df.to_csv(train_csv, index=False)
        print(f'{len(df)} training samples')
        os.remove(os.path.join(data_dir, 'zinc250k_arm.csv'))
        os.remove(os.path.join(data_dir, 'zinc250k_linker.csv'))

    # --- Save as HuggingFace Dataset ---
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files={'train': train_csv})
    dataset.save_to_disk(os.path.join(data_dir, 'zinc250k'))