#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-tokenizer
#SBATCH --output=/orange/ufdatastudios/c.okocha/chibu-chat/logs/tokenizer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/chibu-chat/logs/tokenizer_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200

set -euo pipefail

echo "===== NANOCHAT TOKENIZER TRAINING ON B200 GPU ====="
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
export WANDB_RUN="b200_tokenizer_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "WANDB_RUN: $WANDB_RUN"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python and PyTorch installation
echo "===== Environment Check ====="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || true

# Verify wandb login
echo "===== WANDB Setup ====="
wandb --version
echo "Checking wandb login status..."
wandb whoami || echo "WANDB not logged in - will prompt for login during training"

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

# Check if training data exists, download if needed
echo "===== Training Data Check ====="
if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ] || [ $(ls -1 "$NANOCHAT_BASE_DIR/base_data"/*.parquet 2>/dev/null | wc -l) -lt 8 ]; then
    echo "Downloading training data (8 shards for tokenizer training)..."
    python -m nanochat.dataset -n 8
else
    echo "Training data already exists"
fi

echo "===== Starting Tokenizer Training ====="
echo "Training parameters:"
echo "  - Vocabulary size: 65,536 (2^16)"
echo "  - Max characters: 2,000,000,000 (2B)"
echo "  - Document cap: 10,000 characters"
echo "  - GPU: B200"
echo "  - WANDB Run: $WANDB_RUN"

# Run tokenizer training
srun python -m scripts.tok_train \
    --max_chars=2000000000 \
    --vocab_size=65536 \
    --doc_cap=10000

echo "===== Tokenizer Training Complete ====="

# Run tokenizer evaluation
echo "===== Running Tokenizer Evaluation ====="
srun python -m scripts.tok_eval

echo "===== Job Summary ====="
echo "End Time: $(date)"
echo "Tokenizer saved to: $NANOCHAT_BASE_DIR/tokenizer"
echo "Training report: $NANOCHAT_BASE_DIR/report"
echo "WANDB Run: $WANDB_RUN"

# Check output files
echo "===== Output Files ====="
if [ -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
    echo "Tokenizer files:"
    ls -la "$NANOCHAT_BASE_DIR/tokenizer/"
else
    echo "ERROR: Tokenizer directory not found!"
fi

if [ -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "Token bytes mapping created successfully"
else
    echo "WARNING: Token bytes mapping not found"
fi

echo "===== TOKENIZER TRAINING JOB COMPLETE ====="