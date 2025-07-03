#!/bin/bash

# MIA Project Setup Script
# This script helps you set up the MIA project environment for conda

echo "🔬 MIA - Medical Image Analysis Setup"
echo "====================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if py38 environment exists
if conda env list | grep -q "py38"; then
    echo "✅ Found conda environment: py38"
    echo "📦 Activating py38 environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate py38
    
    # Check Python version in the environment
    python_version=$(python --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
    echo "✅ Python version in py38: $python_version"
else
    echo "❌ Conda environment 'py38' not found."
    echo "💡 Please create it first with:"
    echo "   conda create -n py38 python=3.8"
    echo "   conda activate py38"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/wsi
mkdir -p data/wsi_output
mkdir -p models/pretrained/NuLite
mkdir -p output/features
mkdir -p output/graph_output
mkdir -p logs

# Install Python dependencies
echo "📦 Installing Python dependencies in py38 environment..."
pip install -r requirements.txt

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA detected - GPU acceleration available"
    # Install PyTorch with CUDA support for conda
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
else
    echo "⚠️  CUDA not detected - using CPU version"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🐍 Environment: py38 (conda)"
echo "📍 Current working directory: $(pwd)"
echo ""
echo "Next steps:"
echo "1. Make sure you're in the py38 environment: conda activate py38"
echo "2. Download the NuLite pretrained model and place it in: models/pretrained/NuLite/"
echo "3. Place your WSI files in: data/wsi/"
echo "4. Prepare your label CSV file in: data/"
echo "5. Run the pipeline with: bash random_sample_nuclei_detection.sh"
echo ""
echo "For more information, see README.md"
