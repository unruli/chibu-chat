#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-base-train
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/base_train_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/base_train_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== NANOCHAT BASE MODEL TRAINING ON B200 GPU ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

echo "===== GPU Info ====="
nvidia-smi || true

# Load CUDA toolkit
echo "Loading CUDA module..."
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Go to project root
cd /orange/ufdatastudios/c.okocha/chibu-chat

# Create logs directory
mkdir -p logs

# Environment variables for nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_API_KEY="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"
export WANDB_RUN="b200_base_train_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "WANDB_RUN: $WANDB_RUN"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install nanochat package in development mode
echo "===== Installing NanoChat Package ====="
uv pip install -e . --quiet
echo "NanoChat package installed successfully"

# Verify Python and PyTorch installation
echo "===== Environment Check ====="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || true

# Verify wandb login
echo "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || echo "WandB authentication failed"

# Ensure Rust environment is available
echo "===== Rust Environment ====="
source "$HOME/.cargo/env" || true
rustc --version || echo "Rust not available"
cargo --version || echo "Cargo not available"

# Performance optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize report system
echo "===== Initializing Report System ====="
python -m nanochat.report reset

# Download additional training data if needed
echo "===== Training Data Setup ====="
CURRENT_SHARDS=$(ls "$NANOCHAT_BASE_DIR/base_data"/*.parquet 2>/dev/null | wc -l)
echo "Current data shards: $CURRENT_SHARDS"

if [ $CURRENT_SHARDS -lt 20 ]; then
    echo "Downloading additional training data (targeting 20 shards for faster initial training)..."
    python -m nanochat.dataset -n 20 -w 4
    echo "Data download complete"
else
    echo "Sufficient training data already exists ($CURRENT_SHARDS shards)"
fi

# Verify tokenizer exists
echo "===== Tokenizer Check ====="
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found! Please run tokenizer training first."
    exit 1
else
    echo "Tokenizer found: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
fi

echo "===== Starting Base Model Training ====="
echo "Training parameters:"
echo "  - Model depth: 20 layers (configurable)"
echo "  - Context length: 2048 tokens"
echo "  - Batch size: 524,288 tokens total"
echo "  - Target param-data ratio: 20 (Chinchilla optimal)"
echo "  - GPU: B200"
echo "  - WANDB Run: $WANDB_RUN"

# Run base model training
echo "Executing base training script..."
srun python -c "
import sys
sys.path.insert(0, '/orange/ufdatastudios/c.okocha/chibu-chat')
exec(open('/orange/ufdatastudios/c.okocha/chibu-chat/scripts/base_train.py').read())
"

echo "===== Base Model Training Complete ====="

echo "===== Job Summary ====="
echo "End Time: $(date)"
echo "Model saved to: $NANOCHAT_BASE_DIR/checkpoints/"
echo "Training report: $NANOCHAT_BASE_DIR/report/"
echo "WANDB Run: $WANDB_RUN"

# Check output files
echo "===== Output Files ====="
if [ -d "$NANOCHAT_BASE_DIR/checkpoints" ]; then
    echo "Model checkpoints:"
    ls -la "$NANOCHAT_BASE_DIR/checkpoints/" | tail -10
else
    echo "WARNING: Checkpoints directory not found"
fi

echo "===== BASE MODEL TRAINING JOB COMPLETE ====="