#!/bin/bash

# Enhanced Training Script for CellNet
# Usage: ./run_training.sh [config_file] [optional_args]

set -e  # Exit on any error

# Default configuration
CONFIG_FILE="train_config.json"
PYTHON_ENV="py38"  # Conda environment name

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -e|--env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -c, --config FILE    Configuration file (default: train_config.json)"
            echo "  -e, --env ENV        Conda environment (default: py38)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "CellNet Enhanced Training Script"
echo "=================================="
echo "Configuration file: $CONFIG_FILE"
echo "Python environment: $PYTHON_ENV"
echo "=================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one with -c option."
    exit 1
fi

# Check if conda environment exists
if ! conda info --envs | grep -q "$PYTHON_ENV"; then
    echo "Error: Conda environment '$PYTHON_ENV' not found!"
    echo "Please create the environment or specify an existing one with -e option."
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $PYTHON_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PYTHON_ENV

# Verify Python packages
echo "Verifying required packages..."
python -c "import torch, torch_geometric, numpy, matplotlib, sklearn, seaborn; print('All packages available')"

# Create output directories
echo "Creating output directories..."
python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
import os
os.makedirs(config['res_dir'], exist_ok=True)
os.makedirs('output', exist_ok=True)
print(f\"Output directory: {config['res_dir']}\")
"

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('GPU not available, will use CPU')
"

# Display configuration
echo "Training configuration:"
python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
for key, value in config.items():
    print(f'  {key}: {value}')
"

echo "=================================="
echo "Starting training..."
echo "=================================="

# Start training with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="training_${TIMESTAMP}.log"

# Run training with both console and file logging
python train.py 2>&1 | tee "$LOG_FILE"

echo "=================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "=================================="

# Optional: Send notification (uncomment if needed)
# echo "Training completed for CellNet model" | mail -s "Training Complete" your_email@example.com
