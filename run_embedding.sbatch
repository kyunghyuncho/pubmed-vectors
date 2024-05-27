#!/bin/bash
#SBATCH --job-name=embedding_job
#SBATCH --output=embedding_job.out
#SBATCH --error=embedding_job.err
#SBATCH --time=120:00:00  # Maximum runtime of 72 hours
#SBATCH --partition=cilvr_a100_1   # Assuming you are using a GPU partition
#SBATCH --gres=gpu:1      # Requesting one GPU
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --mem=64G         # Memory per node

# Activate your Python environment
source /scratch/kc119/miniforge3/etc/profile.d/conda.sh
conda activate pubmed-vectors

# move to the correct directory
cd /home/kc119/repos/pubmed-vectors

# Run your Python script
python embed_pubmed_st.py


