#!/bin/bash
#SBATCH --job-name=hdg
#SBATCH --output=logs/hdg.out
#SBATCH --error=logs/hdg.err
#SBATCH -p volta-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=shuhang@unc.edu

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate f-rag

module load cuda/12.5
echo "Allocated CUDA devices: $SLURM_JOB_GPUS"

python -m argo.gen_models.gem.cli \
  --inpath ~/HIDDEN-GEM/CHD1/CHD1_gem.csv \
  --outdir CHD1_gen1 \
  --smi_col "smiles" \
  --score_col "score" \
  --use_cuda \
  --mode filter \
  --gen_model_path argo/gen_models/pretrained/gem_chembl.pt \
  --save_files
