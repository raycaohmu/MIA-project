#!/usr/bin/env python3
"""
Quick training script for CellNet model.
This script provides a simple interface to start training with customizable parameters.
"""

import argparse
import json
import os
import sys
from train import Trainer, create_trainer_from_config

def create_default_config():
    """Create default configuration for training."""
    return {
        'label0_ds_pt': './data/label0_dataset.pt',
        'label1_ds_pt': './data/label1_dataset.pt', 
        'node_feat_dim': 512,
        'batch_size': 16,  # Smaller batch size for memory efficiency
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 50,  # Reduced for testing
        'res_dir': './output/training_results',
        'device': 'cuda',
        'patience': 10,
        'save_best_only': False,
        'scheduler_step_size': 20,
        'scheduler_gamma': 0.1
    }

def main():
    parser = argparse.ArgumentParser(description='Train CellNet Model')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to configuration JSON file')
    parser.add_argument('--label0-data', type=str, default=None,
                      help='Path to label 0 dataset (.pt file)')
    parser.add_argument('--label1-data', type=str, default=None,
                      help='Path to label 1 dataset (.pt file)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu)')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                      help='Node feature dimension')
    parser.add_argument('--create-config', action='store_true',
                      help='Create default configuration file and exit')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = 'train_config_default.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration created: {config_path}")
        print("Edit this file and run with --config option")
        return
    
    # Load configuration
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file {args.config} not found!")
            sys.exit(1)
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        print("Using default configuration (use --create-config to save it)")
    
    # Override config with command line arguments
    if args.label0_data:
        config['label0_ds_pt'] = args.label0_data
    if args.label1_data:
        config['label1_ds_pt'] = args.label1_data
    if args.output_dir:
        config['res_dir'] = args.output_dir
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.device:
        config['device'] = args.device
    if args.node_feat_dim:
        config['node_feat_dim'] = args.node_feat_dim
    
    # Validate required files
    if not os.path.exists(config['label0_ds_pt']):
        print(f"Error: Label 0 dataset not found: {config['label0_ds_pt']}")
        print("Please check the file path or create the dataset first.")
        sys.exit(1)
    
    if not os.path.exists(config['label1_ds_pt']):
        print(f"Error: Label 1 dataset not found: {config['label1_ds_pt']}")
        print("Please check the file path or create the dataset first.")
        sys.exit(1)
    
    # Display configuration
    print("="*60)
    print("CELLNET TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("="*60)
    
    try:
        # Initialize and start training
        trainer = Trainer(**config)
        trainer.train_process()
        
        # Evaluate test set
        test_results = trainer.evaluate_test_set()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
        print(f"AUC Score: {test_results['auc_score']:.4f}")
        print(f"Results saved to: {config['res_dir']}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
