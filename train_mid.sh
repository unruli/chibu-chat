#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=chibuchat-midtrain
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/midtrain_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/midtrain_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=15:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== CHIBUCHAT MID-TRAINING ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Function to log with timestamp
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

# Go to project root
cd /orange/ufdatastudios/c.okocha/chibu-chat

# ============================================================================
# torch.compile Settings
# ============================================================================
# Uncomment to disable torch.compile if issues arise:
# export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_INDUCTOR_INSTALL_TRITON=0
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_compile_cache_$SLURM_JOB_ID"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# ============================================================================
# Multiprocessing Settings for SLURM Compatibility
# ============================================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ============================================================================
# CUDA Settings
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export NCCL_DEBUG=WARN

# ============================================================================
# Environment Setup
# ============================================================================
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY="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"
export WANDB_RUN="midtrain_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
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

# Verify base checkpoint exists (mid-training loads from base)
log_with_time "===== Checkpoint Check ====="
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
if [ -f "$CHECKPOINT_DIR/model_021400.pt" ]; then
    log_with_time "✓ Found base checkpoint at step 21400"
else
    log_with_time "✗ ERROR: Base checkpoint not found at $CHECKPOINT_DIR/model_021400.pt"
    ls -lh "$CHECKPOINT_DIR/" 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

# Verify identity conversations file
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    IDENTITY_LINES=$(wc -l < "$IDENTITY_FILE")
    log_with_time "✓ Found identity conversations: $IDENTITY_LINES entries"
else
    log_with_time "✗ WARNING: Identity file not found at $IDENTITY_FILE, copying from project..."
    cp /orange/ufdatastudios/c.okocha/chibu-chat/identity_conversations.jsonl "$IDENTITY_FILE"
fi

# WandB setup
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

# Number of GPUs (must match --gpus above)
NPROC_PER_NODE=4

log_with_time "===== Starting Mid-Training ====="
log_with_time "Mid-training parameters:"
log_with_time "  - Base model: d20 (step 21400)"
log_with_time "  - GPUs: $NPROC_PER_NODE"
log_with_time "  - Device batch size: 16 (same as base training)"
log_with_time "  - Data: SmolTalk(460K) + SimpleSpelling(200K) + MMLU(100K) + SpellingBee(80K) + GSM8K(8K) + Identity(2x950)"
log_with_time "  - Saves to: mid_checkpoints/d20/"

# NOTE: ensure that we use the same device_batch_size here as the base training script.
log_with_time "Running mid-training..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=16 --run=$WANDB_RUN

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_with_time "Mid-training succeeded, running chat_eval..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
    log_with_time "Chat eval complete (exit code: $?)"
else
    log_with_time "Mid-training FAILED with exit code $TRAIN_EXIT_CODE, skipping eval"
fi

log_with_time "===== Mid-Training Complete ====="
log_with_time "Exit code: $TRAIN_EXIT_CODE"
log_with_time "End Time: $(date)"

# Check output checkpoint
MID_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/mid_checkpoints/d20"
if [ -d "$MID_CHECKPOINT_DIR" ]; then
    log_with_time "✓ Mid checkpoint directory:"
    ls -lh "$MID_CHECKPOINT_DIR/"
else
    log_with_time "✗ Mid checkpoint directory not found"
fi

nvidia-smi || true

exit $TRAIN_EXIT_CODE
