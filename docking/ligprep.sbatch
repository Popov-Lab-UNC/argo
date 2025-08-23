#!/bin/bash
#SBATCH --job-name=ligprep
#SBATCH --output=logs/ligprep_%j.out
#SBATCH --error=logs/ligprep_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=general
#SBATCH --mem=128G

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate docking

# Run the ligand preparation script
python ligprep.py --n_processes 64 --input-file PptT/ligands/HGD-2825_UNC11995_HGD_analogue_ideas_20250820.csv --output-dir /work/users/s/h/shuhang/argo_docking/PptT/HGD_analogue_20aug2025 --compound-id-col Designation --smiles-col SMILES

echo "Ligand preparation completed!"
