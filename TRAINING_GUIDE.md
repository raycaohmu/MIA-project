# Training Guide for CellNet Model

This guide explains how to use the enhanced training system for the CellNet model.

## Quick Start

### 1. Prepare Your Data

Ensure you have your datasets ready in `.pt` format:
- `label0_dataset.pt` - Dataset for label 0 (e.g., non-tumor)
- `label1_dataset.pt` - Dataset for label 1 (e.g., tumor)

### 2. Simple Training

Use the quick training script:

```bash
# Create default configuration
python quick_train.py --create-config

# Edit the generated config file if needed
nano train_config_default.json

# Start training
python quick_train.py --config train_config_default.json
```

### 3. Command Line Training

You can also specify parameters directly:

```bash
python quick_train.py \
    --label0-data ./data/label0_dataset.pt \
    --label1-data ./data/label1_dataset.pt \
    --output-dir ./output/my_training \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda
```

## Enhanced Features

### Training Monitoring
- **Real-time logging**: Console and file logging
- **Training curves**: Automatic plotting of loss, accuracy, and learning rate
- **Early stopping**: Prevents overfitting
- **Checkpointing**: Saves best model and periodic checkpoints

### Advanced Options
- **Learning rate scheduling**: StepLR scheduler
- **Gradient clipping**: For training stability
- **Detailed metrics**: Classification report, AUC, confusion matrix
- **Test evaluation**: Comprehensive test set evaluation

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `label0_ds_pt` | Path to label 0 dataset | `./data/label0_dataset.pt` |
| `label1_ds_pt` | Path to label 1 dataset | `./data/label1_dataset.pt` |
| `node_feat_dim` | Node feature dimension | `512` |
| `batch_size` | Training batch size | `32` |
| `lr` | Learning rate | `0.001` |
| `momentum` | SGD momentum | `0.9` |
| `weight_decay` | Weight decay (L2 regularization) | `1e-4` |
| `epochs` | Maximum training epochs | `100` |
| `res_dir` | Results directory | `./output/training_results` |
| `device` | Training device | `cuda` |
| `patience` | Early stopping patience | `15` |
| `save_best_only` | Save only best model | `false` |
| `scheduler_step_size` | LR scheduler step size | `30` |
| `scheduler_gamma` | LR scheduler decay factor | `0.1` |

## Output Files

After training, you'll find these files in your results directory:

```
results/
├── training.log                    # Training log
├── training_config.json            # Configuration used
├── training_history.json           # Training metrics history
├── training_curves.png             # Training plots
├── best_model.pth                  # Best model checkpoint
├── latest_checkpoint.pth           # Latest checkpoint
├── test_results.json               # Test evaluation results
├── label0_split_info.json          # Dataset split information
├── label1_split_info.json          # Dataset split information
└── model_epoch_*.pth               # Epoch checkpoints (if enabled)
```

## Usage Examples

### Example 1: Basic Training
```bash
python quick_train.py \
    --label0-data ./data/normal_cells.pt \
    --label1-data ./data/tumor_cells.pt \
    --epochs 50
```

### Example 2: High-Performance Training
```bash
python quick_train.py \
    --config config_gpu.json \
    --batch-size 64 \
    --lr 0.01 \
    --epochs 200
```

### Example 3: CPU Training (Small Dataset)
```bash
python quick_train.py \
    --device cpu \
    --batch-size 8 \
    --epochs 30
```

## Resuming Training

To resume training from a checkpoint:

```python
from train import resume_training

# Resume from checkpoint with 50 additional epochs
trainer = resume_training('./output/training_results/latest_checkpoint.pth', 50)
trainer.train_process()
```

## Monitoring Training

### Real-time Monitoring
- Watch the console output for real-time metrics
- Check the log file: `tail -f ./output/training_results/training.log`

### Training Plots
Training curves are automatically generated and saved as `training_curves.png`:
- Loss curves (train/validation)
- Accuracy curves (train/validation)  
- Learning rate schedule
- Training time per epoch

### Early Stopping
Training will automatically stop if validation accuracy doesn't improve for `patience` epochs.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

2. **Dataset not found**
   - Check file paths in configuration
   - Ensure datasets are in `.pt` format

3. **Slow training**
   - Increase batch size if memory allows
   - Use GPU if available
   - Reduce logging frequency

### Performance Tips

1. **GPU Training**: Always use GPU if available
2. **Batch Size**: Use largest batch size that fits in memory
3. **Learning Rate**: Start with 0.001, adjust based on convergence
4. **Early Stopping**: Use patience=10-20 to avoid overfitting

## Advanced Usage

### Custom Training Loop
For advanced users, you can use the `Trainer` class directly:

```python
from train import Trainer

config = {
    'label0_ds_pt': './data/label0.pt',
    'label1_ds_pt': './data/label1.pt',
    'node_feat_dim': 512,
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 100,
    'res_dir': './output/custom_training',
    'device': 'cuda'
}

trainer = Trainer(**config)
trainer.train_process()
test_results = trainer.evaluate_test_set()
```

### Hyperparameter Tuning
Create multiple configuration files for different hyperparameter combinations:

```bash
# config_lr001.json - Learning rate 0.001
# config_lr01.json  - Learning rate 0.01
# config_batch16.json - Batch size 16
# config_batch64.json - Batch size 64

for config in config_*.json; do
    python quick_train.py --config $config
done
```

## Results Interpretation

### Metrics Explanation
- **Accuracy**: Overall classification accuracy
- **AUC**: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Expected Results
- Training accuracy should increase steadily
- Validation accuracy should follow training but may plateau
- AUC > 0.7 indicates good performance
- AUC > 0.8 indicates excellent performance

For questions or issues, check the training log file for detailed error messages.
