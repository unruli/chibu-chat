#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=chibuchat-rl
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/rl_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/rl_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== CHIBUCHAT RL (Reinforcement Learning) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

echo "===== GPU Info ====="
nvidia-smi || true

# Load CUDA toolkit
log_with_time "Loading CUDA module..."
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

cd /orange/ufdatastudios/c.okocha/chibu-chat

# ============================================================================
# Environment Settings
# ============================================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export NCCL_DEBUG=WARN

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY="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"
export WANDB_RUN="rl_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

log_with_time "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
log_with_time "WANDB_RUN: $WANDB_RUN"

# Activate virtual environment
log_with_time "Activating virtual environment..."
source .venv/bin/activate

# Install nanochat package
log_with_time "Installing NanoChat package..."
uv pip install -e . --quiet

# Verify environment
log_with_time "===== Environment Check ====="
python --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || true

# Verify SFT checkpoint exists (RL loads from SFT by default)
log_with_time "===== Checkpoint Check ====="
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20"
if [ -f "$SFT_DIR/model_000698.pt" ]; then
    log_with_time "✓ Found SFT checkpoint at step 698"
else
    log_with_time "✗ ERROR: SFT checkpoint not found at $SFT_DIR/model_000698.pt"
    ls -lh "$SFT_DIR/" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

# WandB setup
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

# Number of GPUs (must match --gpus above)
NPROC_PER_NODE=4

log_with_time "===== Starting RL ====="
log_with_time "RL parameters:"
log_with_time "  - Source model: SFT checkpoint (d20, step 698)"
log_with_time "  - GPUs: $NPROC_PER_NODE"
log_with_time "  - Algorithm: GRPO/REINFORCE on GSM8K"
log_with_time "  - 16 samples per example, 16 examples per step"
log_with_time "  - Eval every 60 steps, Save every 60 steps"
log_with_time "  - Saves to: chatrl_checkpoints/d20/"

# RL
log_with_time "Running RL..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_with_time "RL succeeded, running chat_eval..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl
    log_with_time "Chat eval complete (exit code: $?)"
else
    log_with_time "RL FAILED with exit code $TRAIN_EXIT_CODE, skipping eval"
fi

log_with_time "===== RL Complete ====="
log_with_time "Exit code: $TRAIN_EXIT_CODE"
log_with_time "End Time: $(date)"

# Check output checkpoint
RL_DIR="$NANOCHAT_BASE_DIR/chatrl_checkpoints/d20"
if [ -d "$RL_DIR" ]; then
    log_with_time "✓ RL checkpoint directory:"
    ls -lh "$RL_DIR/"
else
    log_with_time "✗ RL checkpoint directory not found"
fi

nvidia-smi || true

exit $TRAIN_EXIT_CODE
