#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=chibuchat-sft
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/sft_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== CHIBUCHAT SFT (Supervised Fine-Tuning) ====="
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
export WANDB_RUN="sft_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
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

# Verify mid checkpoint exists (SFT loads from mid by default)
log_with_time "===== Checkpoint Check ====="
MID_DIR="$NANOCHAT_BASE_DIR/mid_checkpoints/d20"
if [ -f "$MID_DIR/model_000811.pt" ]; then
    log_with_time "✓ Found mid checkpoint at step 811"
else
    log_with_time "✗ ERROR: Mid checkpoint not found at $MID_DIR/model_000811.pt"
    ls -lh "$MID_DIR/" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

# Verify identity conversations file
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    IDENTITY_LINES=$(wc -l < "$IDENTITY_FILE")
    log_with_time "✓ Found identity conversations: $IDENTITY_LINES entries"
else
    log_with_time "✗ WARNING: Identity file not found, copying from project..."
    cp /orange/ufdatastudios/c.okocha/chibu-chat/identity_conversations.jsonl "$IDENTITY_FILE"
fi

# WandB setup
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

# Number of GPUs (must match --gpus above)
NPROC_PER_NODE=4

log_with_time "===== Starting SFT ====="
log_with_time "SFT parameters:"
log_with_time "  - Source model: mid checkpoint (d20, step 811)"
log_with_time "  - GPUs: $NPROC_PER_NODE"
log_with_time "  - device_batch_size: 4 (default for SFT)"
log_with_time "  - Data: ARC-Easy(2.3K) + ARC-Challenge(1.1K) + GSM8K(8K) + SmolTalk(10K) + Identity(950) + Spelling(600)"
log_with_time "  - torch.compile: disabled (variable-length inputs)"
log_with_time "  - Saves to: chatsft_checkpoints/d20/"

# SFT
log_with_time "Running SFT..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_with_time "SFT succeeded, running chat_eval..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
    log_with_time "Chat eval complete (exit code: $?)"
else
    log_with_time "SFT FAILED with exit code $TRAIN_EXIT_CODE, skipping eval"
fi

log_with_time "===== SFT Complete ====="
log_with_time "Exit code: $TRAIN_EXIT_CODE"
log_with_time "End Time: $(date)"

# Check output checkpoint
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20"
if [ -d "$SFT_DIR" ]; then
    log_with_time "✓ SFT checkpoint directory:"
    ls -lh "$SFT_DIR/"
else
    log_with_time "✗ SFT checkpoint directory not found"
fi

nvidia-smi || true

exit $TRAIN_EXIT_CODE
