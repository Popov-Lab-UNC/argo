import os
import glob
import pandas as pd
from meeko import PDBQTMolecule
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import tempfile
import gzip  # Import the gzip library
import argparse

# Determine cores from SLURM or default to 1
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

def process_dlg_file(root_dir, filepath) -> list:
    """
    Processes a single .dlg or .dlg.gz file to extract data for the best pose in each cluster.
    It handles gzipped files by decompressing them to a temporary file on disk.
    """
    results = []
    temp_dlg_path = None

    try:
        path_to_process = filepath

        if filepath.endswith('.gz'):
            with tempfile.NamedTemporaryFile(mode='wt', delete=False, suffix=".dlg") as tmp:
                temp_dlg_path = tmp.name
                with gzip.open(filepath, 'rt') as f_gz:
                    tmp.write(f_gz.read())
            path_to_process = temp_dlg_path
        
        pdbqt_mol = PDBQTMolecule.from_file(path_to_process, is_dlg=True, skip_typing=True)

        data = pdbqt_mol._pose_data
        
        # --- THE FIX: Robust SMILES parsing ---
        smiles_data = data.get("smiles")
        smiles = "" # Default to empty string

        if isinstance(smiles_data, str):
            # Case 1: It's already a clean string
            smiles = smiles_data
        elif isinstance(smiles_data, dict) and smiles_data:
            # Case 2: It's a dictionary like {0: '...'}. Extract the first value.
            smiles = next(iter(smiles_data.values()))
        elif isinstance(smiles_data, list) and smiles_data:
            # Case 3: It's a list of strings.
            smiles = smiles_data[0]
        # ----------------------------------------
        
        compound_id = Path(filepath).name.split('.')[0]
        ranks = data["rank_in_cluster"]
        top_pose_indices = [i for i, rank in enumerate(ranks) if rank == 1]
        
        for idx in top_pose_indices:
            results.append({
                'compound_id': compound_id,
                'smiles': smiles,
                'file_path': os.path.relpath(filepath, root_dir), 
                'pose_index': idx,
                'cluster_id': data['cluster_id'][idx],
                'free_energy': data['free_energies'][idx],
                'intermolecular_energy': data['intermolecular_energies'][idx],
                'internal_energy': data['internal_energies'][idx],
                'cluster_size': data['cluster_size'][idx],
            })

    except Exception as e:
        print(f"Could not process file {filepath}: {e}")
        return []
        
    finally:
        if temp_dlg_path:
            os.remove(temp_dlg_path)
            
    return results

# Use multiprocessing Pool to process files in parallel
def _star(args):
    return process_dlg_file(*args)

def main():
    """Main function to find files, run parallel processing, and save the CSV."""
    parser = argparse.ArgumentParser(description="Compile docking results from one or more directories.")
    parser.add_argument(
        "--root-dir",
        dest="root_dirs",
        nargs="+",
        required=True,
        help="One or more root directories containing .dlg or .dlg.gz files",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        required=True,
        help="Path to the output CSV file",
    )
    args = parser.parse_args()

    # Expand and validate root dirs
    root_dirs = [os.path.abspath(d) for d in args.root_dirs]
    for d in root_dirs:
        if not os.path.isdir(d):
            print(f"Error: Root directory does not exist or is not a directory: {d}")
            return

    print("Searching for .dlg and .dlg.gz files in:")
    for d in root_dirs:
        print(f"  - {d}")

    # Discover files across all roots
    file_root_pairs = []  # list of (root_dir, filepath)
    for root in root_dirs:
        # Non-recursive search by default; change to '**/*.dlg' with recursive=True if needed
        pattern_gz = os.path.join(root, '*.dlg.gz')
        pattern_dlg = os.path.join(root, '*.dlg')
        files = glob.glob(pattern_gz, recursive=True) + glob.glob(pattern_dlg, recursive=True)
        file_root_pairs.extend((root, f) for f in files)

    if not file_root_pairs:
        print("Error: No .dlg or .dlg.gz files found in the provided root directories.")
        return

    print(f"Found {len(file_root_pairs):,} files across {len(root_dirs)} root(s). Starting parallel processing on {num_cores} cores...")

    with Pool(num_cores) as pool:
        all_results = list(tqdm(pool.imap_unordered(_star, file_root_pairs), total=len(file_root_pairs)))

    # Flatten the list of lists into a single list of dictionaries
    flat_results = [item for sublist in all_results for item in sublist]

    if not flat_results:
        print("Processing finished, but no valid data was extracted.")
        return

    print(f"\nProcessing complete. Extracted {len(flat_results):,} top-ranked poses.")

    # Create a pandas DataFrame and save to CSV
    output_csv_path = os.path.abspath(args.output_csv)
    print(f"Saving results to {output_csv_path}...")
    df = pd.DataFrame(flat_results)

    # Reorder columns for clarity (if they exist)
    column_order = [
        'compound_id', 'smiles', 'free_energy', 'intermolecular_energy', 'internal_energy',
        'cluster_id', 'cluster_size', 'pose_index', 'file_path'
    ]
    df = df[[c for c in column_order if c in df.columns]]

    df.to_csv(output_csv_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()