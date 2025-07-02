#!/bin/bash
#SBATCH --job-name=fim_train
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fim_train_%j.out
#SBATCH --error=logs/fim_train_%j.err

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate f-rag

# Set output directory (edit as needed)
output_dir=outputs/fim_run

# Run the training script
python argo/gen_models/f_rag/fusion/trainer/train.py \
    --dataset data/zinc250k \
    --output_dir ${output_dir} \
    --per_device_train_batch_size 128 \
    --save_strategy epoch \
    --num_train_epochs 8 \
    --learning_rate 1e-4
