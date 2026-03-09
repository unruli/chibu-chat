#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-resume-compile
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/resume_compile_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/resume_compile_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== NANOCHAT RESUME WITH TORCH.COMPILE TEST ====="
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
# TORCH.COMPILE ENABLED - Testing if it works now
# ============================================================================
# Remove the disable flag - let torch.compile run
# export TORCH_COMPILE_DISABLE=1  # COMMENTED OUT - torch.compile ENABLED

# Limit compile workers to prevent deadlock
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_INDUCTOR_INSTALL_TRITON=0

# Use a fresh compile cache for this run
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
export WANDB_RUN="resume_compile_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

log_with_time "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
log_with_time "WANDB_RUN: $WANDB_RUN"
log_with_time "TORCH.COMPILE: ENABLED (testing)"
log_with_time "TORCHINDUCTOR_COMPILE_THREADS: $TORCHINDUCTOR_COMPILE_THREADS"

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

# Verify checkpoint exists
log_with_time "===== Checkpoint Check ====="
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
if [ -f "$CHECKPOINT_DIR/model_001000.pt" ]; then
    log_with_time "✓ Found checkpoint at step 1000"
    ls -lh "$CHECKPOINT_DIR/"
else
    log_with_time "✗ ERROR: Checkpoint not found at $CHECKPOINT_DIR"
    exit 1
fi

# WandB setup
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

log_with_time "===== Resuming Training with torch.compile() ====="
log_with_time "Resume parameters:"
log_with_time "  - Resuming from step: 1000"
log_with_time "  - torch.compile: ENABLED"
log_with_time "  - Model depth: 20 layers"
log_with_time "  - Context length: 2048 tokens"
log_with_time "  - Device batch size: 16"
log_with_time "  - Total batch size: 524,288 tokens"

log_with_time "Executing training with torch.compile..."

# torch.compile() has been verified to work - run without timeout
python -u -m scripts.base_train \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=16 \
    --total_batch_size=524288 \
    --resume_from_step=1000 \
    --eval_every=250 \
    --save_every=1000 \
    --core_metric_every=2000 \
    --sample_every=2000

TRAIN_EXIT_CODE=$?

log_with_time "===== Training Complete ====="
log_with_time "Exit code: $TRAIN_EXIT_CODE"
log_with_time "End Time: $(date)"

nvidia-smi || true

exit $TRAIN_EXIT_CODE
