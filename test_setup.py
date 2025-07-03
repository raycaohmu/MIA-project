#!/usr/bin/env python3
"""
Test script to validate the training setup and data loading.
This script helps debug issues before starting actual training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch_geometric.data import DataLoader

def test_imports():
    """Test if all required packages are available."""
    print("Testing imports...")
    try:
        import torch
        import torch_geometric
        import numpy
        import matplotlib
        import sklearn
        import seaborn
        print("✓ All required packages available")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_gpu():
    """Test GPU availability and memory."""
    print("\nTesting GPU...")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"✓ GPU available: {gpu_name}")
        print(f"✓ GPU memory: {total_memory:.1f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ GPU memory allocation test passed")
        except Exception as e:
            print(f"✗ GPU memory test failed: {e}")
            return False
        return True
    else:
        print("⚠ GPU not available, will use CPU")
        return True

def test_model_creation():
    """Test model creation and basic operations."""
    print("\nTesting model creation...")
    try:
        from construct_model import CellNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CellNet(
            in_channels=512,
            out_channels=2,
            device=device,
            batch=True
        )
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully with {total_params:,} parameters")
        
        return True, model, device
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_dataset_loading(label0_path, label1_path):
    """Test dataset loading and structure."""
    print(f"\nTesting dataset loading...")
    
    # Check if files exist
    if not os.path.exists(label0_path):
        print(f"✗ Label 0 dataset not found: {label0_path}")
        return False, None, None
    
    if not os.path.exists(label1_path):
        print(f"✗ Label 1 dataset not found: {label1_path}")
        return False, None, None
    
    try:
        # Load datasets
        label0_ds = torch.load(label0_path, map_location='cpu')
        label1_ds = torch.load(label1_path, map_location='cpu')
        
        print(f"✓ Label 0 dataset loaded: {len(label0_ds)} samples")
        print(f"✓ Label 1 dataset loaded: {len(label1_ds)} samples")
        
        # Check sample structure
        if len(label0_ds) > 0:
            sample = label0_ds[0]
            print(f"✓ Sample structure:")
            print(f"  - Node features: {sample.x.shape}")
            print(f"  - Edge index: {sample.edge_index.shape}")
            print(f"  - Edge attributes: {sample.edge_attr.shape}")
            print(f"  - Label: {sample.y}")
            print(f"  - Cell types: {sample.cell_type.shape}")
            if hasattr(sample, 'pid'):
                print(f"  - Patient ID: {sample.pid}")
        
        return True, label0_ds, label1_ds
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_data_loader(label0_ds, label1_ds, batch_size=4):
    """Test data loader creation and batch loading."""
    print(f"\nTesting data loader...")
    try:
        # Create small combined dataset for testing
        test_data = label0_ds[:2] + label1_ds[:2] if len(label0_ds) >= 2 and len(label1_ds) >= 2 else label0_ds[:1] + label1_ds[:1]
        
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        
        # Test loading one batch
        for batch in test_loader:
            print(f"✓ Batch loaded successfully:")
            print(f"  - Batch size: {batch.batch[-1].item() + 1}")
            print(f"  - Node features: {batch.x.shape}")
            print(f"  - Edge index: {batch.edge_index.shape}")
            print(f"  - Edge attributes: {batch.edge_attr.shape}")
            print(f"  - Labels: {batch.y.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward(model, label0_ds, label1_ds, device):
    """Test model forward pass."""
    print(f"\nTesting model forward pass...")
    try:
        # Create small test batch
        test_data = [label0_ds[0], label1_ds[0]] if len(label0_ds) > 0 and len(label1_ds) > 0 else [label0_ds[0]]
        test_loader = DataLoader(test_data, batch_size=len(test_data))
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                
                print(f"✓ Forward pass successful:")
                print(f"  - Input batch size: {batch.batch[-1].item() + 1}")
                print(f"  - Output shape: {outputs.shape}")
                print(f"  - Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                
                # Test softmax
                probs = F.softmax(outputs, dim=1)
                print(f"  - Probabilities: {probs.cpu().numpy()}")
                
                break
        
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(model, label0_ds, label1_ds, device):
    """Test one training step."""
    print(f"\nTesting training step...")
    try:
        # Create small test batch
        test_data = [label0_ds[0], label1_ds[0]] if len(label0_ds) > 0 and len(label1_ds) > 0 else [label0_ds[0]]
        test_loader = DataLoader(test_data, batch_size=len(test_data))
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y.squeeze())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"✓ Training step successful:")
            print(f"  - Loss: {loss.item():.4f}")
            print(f"  - Gradients computed successfully")
            
            break
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("="*60)
    print("CELLNET TRAINING SETUP VALIDATION")
    print("="*60)
    
    # Test configuration
    label0_path = "./data/label0_dataset.pt"
    label1_path = "./data/label1_dataset.pt"
    
    # Allow custom paths via command line
    if len(sys.argv) > 2:
        label0_path = sys.argv[1]
        label1_path = sys.argv[2]
        print(f"Using custom dataset paths:")
        print(f"  Label 0: {label0_path}")
        print(f"  Label 1: {label1_path}")
    
    success_count = 0
    total_tests = 6
    
    # Run tests
    if test_imports():
        success_count += 1
    
    if test_gpu():
        success_count += 1
    
    model_success, model, device = test_model_creation()
    if model_success:
        success_count += 1
    
    data_success, label0_ds, label1_ds = test_dataset_loading(label0_path, label1_path)
    if data_success:
        success_count += 1
        
        if test_data_loader(label0_ds, label1_ds):
            success_count += 1
        
        if model_success and test_model_forward(model, label0_ds, label1_ds, device):
            # Only test training step if forward pass works
            if test_training_step(model, label0_ds, label1_ds, device):
                success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY: {success_count}/{total_tests} tests passed")
    print("="*60)
    
    if success_count == total_tests:
        print("🎉 All tests passed! Your setup is ready for training.")
        print("\nTo start training, run:")
        print("  python quick_train.py --create-config")
        print("  python quick_train.py --config train_config_default.json")
    else:
        print("⚠ Some tests failed. Please resolve the issues before training.")
        
        if not data_success:
            print("\nTo create test datasets, you may need to run:")
            print("  python dataset_create.py")
    
    print("="*60)

if __name__ == "__main__":
    main()
