#!/bin/bash
#SBATCH --job-name=train_vit
#SBATCH --output=logs/vit_run_%j.out
#SBATCH --error=logs/vit_run_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00

# Load the modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load 2023 CUDA/12.4.0

# Activate the virtual environment
source venv/bin/activate

# Run the script
python main.py