# Models Directory

This directory contains model definitions and pretrained weights.

## Structure

```
models/
├── README.md                    # This file
├── __init__.py                 # Package initialization
├── nulite.py                   # NuLite model definition
├── utils.py                    # Model utilities
├── encoders/                   # Encoder architectures
│   ├── __init__.py
│   └── fastvit.py             # FastViT encoder
└── pretrained/                # Pretrained model weights (not in Git)
    └── NuLite/                # NuLite pretrained models
        ├── NuLite-T-Weights.pth  # Tiny model (47MB)
        ├── NuLite-M-Weights.pth  # Medium model (131MB)
        └── NuLite-H-Weights.pth  # Heavy model (184MB)
```

## Pretrained Models

**Note**: Pretrained model files (.pth) are not included in the Git repository due to size constraints.

### NuLite Models

The NuLite models are used for nuclei detection and classification:

- **NuLite-T** (Tiny): 47MB - Fastest inference, lower accuracy
- **NuLite-M** (Medium): 131MB - Balanced speed and accuracy
- **NuLite-H** (Heavy): 184MB - Best accuracy, slower inference

### Download Instructions

1. **Download from original source**: 
   ```bash
   # Create the directory
   mkdir -p models/pretrained/NuLite/
   
   # Download the models (replace with actual download URLs)
   # wget -O models/pretrained/NuLite/NuLite-H-Weights.pth [DOWNLOAD_URL]
   # wget -O models/pretrained/NuLite/NuLite-M-Weights.pth [DOWNLOAD_URL]
   # wget -O models/pretrained/NuLite/NuLite-T-Weights.pth [DOWNLOAD_URL]
   ```

2. **Copy from existing installation**:
   ```bash
   cp /path/to/your/models/NuLite-*.pth models/pretrained/NuLite/
   ```

3. **Verify download**:
   ```bash
   ls -lh models/pretrained/NuLite/
   # Should show:
   # NuLite-H-Weights.pth (184M)
   # NuLite-M-Weights.pth (131M)  
   # NuLite-T-Weights.pth (47M)
   ```

## Model Usage

The models are used in the pipeline as follows:

1. **Random Sampling + Nuclei Detection** (`random_sample_nuclei_detection.sh`):
   ```bash
   python random_sample_nuclei_detection.py \
       --model ./models/pretrained/NuLite/NuLite-H-Weights.pth
   ```

2. **Direct Inference** (`run_inference_wsi.py`):
   ```bash
   python run_inference_wsi.py \
       --model ./models/pretrained/NuLite/NuLite-H-Weights.pth
   ```

## Model Information

### Input Requirements
- **Image size**: 1024x1024 pixels (configurable)
- **Magnification**: 20x (recommended)
- **Format**: RGB images from WSI

### Output Format
- **Detection**: Cell bounding boxes and confidence scores
- **Classification**: Cell type predictions (e.g., tumor, lymphocyte, etc.)
- **Export**: GeoJSON format with spatial coordinates

### Performance
- **NuLite-H**: Best for research and high-accuracy requirements
- **NuLite-M**: Good balance for most applications
- **NuLite-T**: Fast inference for real-time or resource-constrained scenarios

## Alternative Storage

For sharing or backing up models, consider:

1. **Git LFS** (Large File Storage):
   ```bash
   git lfs track "*.pth"
   git add .gitattributes
   ```

2. **External storage**:
   - Google Drive / Dropbox shared links
   - Institutional file servers
   - Docker images with models included

3. **Model hubs**:
   - Hugging Face Model Hub
   - TensorFlow Hub
   - PyTorch Hub
