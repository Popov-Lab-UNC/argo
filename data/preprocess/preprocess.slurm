#!/bin/bash

#SBATCH -p small
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --mem=512g

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate f-rag

python preprocess.py
