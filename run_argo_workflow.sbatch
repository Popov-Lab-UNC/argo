#!/bin/bash
#SBATCH --job-name=argo_workflow
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/argo_workflow_%j.out
#SBATCH --error=logs/argo_workflow_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=shuhang@unc.edu

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate f-rag

# Load CUDA module
module load cuda/12.5
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the argo workflow
echo "Starting ARGO workflow..."
python argo_workflow.py

echo "ARGO workflow completed!" 