#!/bin/bash
# One 2-hour slice of a wandb sweep on a Fir H100.
#
# The agent takes a fixed number of trials and exits, rather than running until
# the wall: a trial killed by SLURM mid-step is charged to the allocation and
# recorded as crashed, so the count is set from the measured trial time to fit
# inside the walltime with room to spare. The sweep's own per-trial timeout is
# the second guard, and sits below the walltime for the same reason.
#
#   sbatch --export=SWEEP=<entity/project/id>,COUNT=<n> gpfa_agent.sh
#
#SBATCH --job-name=gpfa-agent
#SBATCH --account=def-aengusb_gpu
#SBATCH --time=2:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=h100:1
#SBATCH --output=/home/bsteel/scratch/gpfa/logs/%x-%j.out
#SBATCH --error=/home/bsteel/scratch/gpfa/logs/%x-%j.err

set -euo pipefail
: "${SWEEP:?set SWEEP=entity/project/sweep_id}"
COUNT="${COUNT:-1}"

module load python/3.11.5
source "$HOME/venvs/gpfa/bin/activate"
cd "$HOME/scratch/gpfa"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export POLARS_MAX_THREADS=6
export PATH="$HOME/venvs/gpfa/bin:$PATH"     # the agent runs `python` via env

echo "=== $(date '+%F %T') node $(hostname) sweep $SWEEP count $COUNT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

wandb agent --count "$COUNT" "$SWEEP"
echo "=== $(date '+%F %T') agent exited ==="
