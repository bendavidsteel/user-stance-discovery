#!/bin/bash
#SBATCH --job-name=gpfa-smoke
#SBATCH --account=def-aengusb_gpu
#SBATCH --time=2:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1
#SBATCH --output=/home/bsteel/scratch/gpfa/logs/%x-%j.out
#SBATCH --error=/home/bsteel/scratch/gpfa/logs/%x-%j.err

set -euo pipefail
module load python/3.11.5
source "$HOME/scratch/gpfa/venv/bin/activate"
cd "$HOME/scratch/gpfa"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export POLARS_MAX_THREADS=6

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('jax devices:', jax.devices())"

# One trial at the current sweep leader's configuration, so its wall clock is
# directly comparable with the same configuration on prometheus.
python -u scripts/flows/nn_potential.py \
    num_epochs=100 patience=10 eval_horizons='[90]' breakdown_scenario=null \
    latents.obs_model=channel latents.obs_temperature=1.8 \
    latents.n_fast=1 latents.fast_kind=ou latents.fast_tau=20 \
    latents.slow_kind=const latents.slow_tau=2560 \
    latents.bin_factor=8 n_dims=6
