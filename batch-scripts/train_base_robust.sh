#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-base-robust
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/base_robust_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/base_robust_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== NANOCHAT ROBUST BASE TRAINING ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Function to log with timestamp
log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to monitor GPU and log periodically
monitor_gpu() {
    while true; do
        sleep 300  # Every 5 minutes
        log_with_time "=== GPU Status Check ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
        log_with_time "=== Process Status ==="
        ps aux | grep python | grep -v grep || true
    done &
    MONITOR_PID=$!
}

# Function to cleanup on exit
cleanup() {
    log_with_time "Cleaning up monitoring process..."
    kill $MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

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

# Environment variables for nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY="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"
export WANDB_RUN="robust_base_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

log_with_time "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
log_with_time "WANDB_RUN: $WANDB_RUN"

# Activate virtual environment
log_with_time "Activating virtual environment..."
source .venv/bin/activate

# Install nanochat package in development mode
log_with_time "Installing NanoChat package..."
uv pip install -e . --quiet

# Verify Python and PyTorch installation
log_with_time "===== Environment Check ====="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || true

# Verify wandb login
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

# Initialize report system
log_with_time "===== Initializing Report System ====="
python -m nanochat.report reset

# Check data availability
log_with_time "===== Training Data Check ====="
CURRENT_SHARDS=$(ls "$NANOCHAT_BASE_DIR/base_data"/*.parquet 2>/dev/null | wc -l)
log_with_time "Current data shards: $CURRENT_SHARDS"

if [ $CURRENT_SHARDS -lt 20 ]; then
    log_with_time "Need more data shards, downloading..."
    python -m nanochat.dataset -n 20 -w 4
else
    log_with_time "Sufficient training data available ($CURRENT_SHARDS shards)"
fi

# Verify tokenizer exists
log_with_time "===== Tokenizer Check ====="
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    log_with_time "ERROR: Tokenizer not found!"
    exit 1
else
    log_with_time "Tokenizer found: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
fi

# Start GPU monitoring
log_with_time "Starting GPU monitoring..."
monitor_gpu

log_with_time "===== Starting Robust Base Training ====="
log_with_time "Training parameters (optimized for stability):"
log_with_time "  - Model depth: 20 layers (full size)"
log_with_time "  - Context length: 2048 tokens (full context)"
log_with_time "  - Device batch size: 8 (reduced for stability)"
log_with_time "  - Total batch size: 262,144 tokens (reduced from 524,288)"
log_with_time "  - More frequent checkpointing enabled"
log_with_time "  - GPU: B200"
log_with_time "  - WANDB Run: $WANDB_RUN"

# Run base model training with modified parameters
log_with_time "Executing robust training script..."

# Create a modified training script with better parameters
python -c "
import sys
import signal
import time
import os
from contextlib import contextmanager

# Timeout context manager to detect hangs
@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError('Operation timed out after {} seconds'.format(duration))
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

# Set smaller, more stable parameters
import os
# Add timeout monitoring
last_step_time = time.time()

def step_callback():
    global last_step_time
    last_step_time = time.time()

# Modified parameters for stability
import sys
sys.path.insert(0, '/orange/ufdatastudios/c.okocha/chibu-chat')

print('[PYTHON] Starting training with robust parameters...')

# Read and modify the base training script with safer parameters
with open('/orange/ufdatastudios/c.okocha/chibu-chat/scripts/base_train.py', 'r') as f:
    script_content = f.read()

# Replace problematic parameters - keep model size but reduce batch sizes
script_content = script_content.replace('device_batch_size = 32', 'device_batch_size = 8')  # Smaller batches
script_content = script_content.replace('total_batch_size = 524288', 'total_batch_size = 262144')  # Half batch size
script_content = script_content.replace('eval_every = 250', 'eval_every = 100')  # More frequent eval

print('[PYTHON] Modified batch sizes for stability while keeping full model size')
exec(script_content)
"

log_with_time "===== Training Complete or Stopped ====="

log_with_time "===== Job Summary ====="
log_with_time "End Time: $(date)"
log_with_time "WANDB Run: $WANDB_RUN"

# Check output files
log_with_time "===== Output Files Check ====="
if [ -d "$NANOCHAT_BASE_DIR/checkpoints" ]; then
    log_with_time "Model checkpoints found:"
    ls -la "$NANOCHAT_BASE_DIR/checkpoints/" | tail -5
else
    log_with_time "No checkpoints directory found"
fi

log_with_time "===== ROBUST BASE TRAINING JOB COMPLETE ====="