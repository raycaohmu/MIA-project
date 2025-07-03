from dataset_create import NucleiData
from construct_model import CellNet
import torch
from torch_geometric.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns


class Trainer:

    def __init__(self, label0_ds_pt: str, label1_ds_pt: str,
                 node_feat_dim: int, batch_size: int = 32, 
                 lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 1e-4,
                 epochs: int = 100, res_dir: str = "./output/models", 
                 device: str = "cuda", patience: int = 10, 
                 save_best_only: bool = True, scheduler_step_size: int = 30,
                 scheduler_gamma: float = 0.1):
        """
        Initialize Trainer with enhanced features.
        
        Args:
            label0_ds_pt: Path to label 0 dataset
            label1_ds_pt: Path to label 1 dataset
            node_feat_dim: Node feature dimension
            batch_size: Batch size for training
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay for regularization
            epochs: Maximum number of epochs
            res_dir: Result directory for saving models and logs
            device: Device for training (cuda/cpu)
            patience: Early stopping patience
            save_best_only: Whether to save only the best model
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.label0_ds_pt = label0_ds_pt
        self.label1_ds_pt = label1_ds_pt
        self.batch_size = batch_size
        self.node_feat_dim = node_feat_dim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.res_dir = res_dir
        self.patience = patience
        self.save_best_only = save_best_only
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # Create result directory
        os.makedirs(self.res_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_epoch = 0
        
        # Timing
        self.start_time = None
        self.epoch_times = []

        # Set up logging with file handler
        self._setup_logging()

        # Prepare the datasets
        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()
        self.logger.info("Data preparation complete.")

        # Prepare the model
        self.model = CellNet(
            in_channels=self.node_feat_dim,
            out_channels=2,
            device=self.device,
            batch=True
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Save training configuration
        self._save_config()

    def _setup_logging(self):
        """Setup logging with both console and file handlers."""
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set up logger
        self.logger = logging.getLogger(f"Trainer_{int(time.time())}")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.res_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def _save_config(self):
        """Save training configuration to JSON file."""
        config = {
            'label0_ds_pt': self.label0_ds_pt,
            'label1_ds_pt': self.label1_ds_pt,
            'node_feat_dim': self.node_feat_dim,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'patience': self.patience,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_gamma': self.scheduler_gamma,
            'device': str(self.device),
            'save_best_only': self.save_best_only
        }
        
        config_path = os.path.join(self.res_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Training configuration saved to {config_path}")

    def _prepare_data(self):
        """Prepare train, validation, and test data loaders."""
        # Load the datasets
        self.logger.info("Loading datasets...")
        
        if not os.path.exists(self.label0_ds_pt):
            raise FileNotFoundError(f"Label 0 dataset not found: {self.label0_ds_pt}")
        if not os.path.exists(self.label1_ds_pt):
            raise FileNotFoundError(f"Label 1 dataset not found: {self.label1_ds_pt}")
            
        label0_ds = torch.load(self.label0_ds_pt, map_location='cpu')
        label1_ds = torch.load(self.label1_ds_pt, map_location='cpu')
        
        self.logger.info(f"Label 0 dataset size: {len(label0_ds)}")
        self.logger.info(f"Label 1 dataset size: {len(label1_ds)}")
        
        # Check data consistency
        self._validate_datasets(label0_ds, label1_ds)
        
        # Split datasets
        self.logger.info("Splitting label 0 dataset...")
        label0_train_ds, label0_val_ds, label0_test_ds = self._split_dataset(label0_ds, "label0")
        
        self.logger.info("Splitting label 1 dataset...")
        label1_train_ds, label1_val_ds, label1_test_ds = self._split_dataset(label1_ds, "label1")

        # Combine datasets
        train_list = label0_train_ds + label1_train_ds
        val_list = label0_val_ds + label1_val_ds
        test_list = label0_test_ds + label1_test_ds
        
        # Create data loaders
        train_loader = DataLoader(train_list, batch_size=self.batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
        val_loader = DataLoader(val_list, batch_size=self.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
        test_loader = DataLoader(test_list, batch_size=self.batch_size, shuffle=False,
                               num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
        
        self.logger.info(f"Data loaders created - Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
        
        return train_loader, val_loader, test_loader
    
    def _validate_datasets(self, label0_ds: List, label1_ds: List):
        """Validate dataset consistency."""
        if len(label0_ds) == 0 or len(label1_ds) == 0:
            raise ValueError("One or both datasets are empty")
            
        # Check node feature dimensions
        sample0 = label0_ds[0]
        sample1 = label1_ds[0]
        
        if sample0.x.shape[1] != self.node_feat_dim:
            raise ValueError(f"Label 0 dataset node features ({sample0.x.shape[1]}) don't match expected ({self.node_feat_dim})")
        if sample1.x.shape[1] != self.node_feat_dim:
            raise ValueError(f"Label 1 dataset node features ({sample1.x.shape[1]}) don't match expected ({self.node_feat_dim})")
            
        self.logger.info("Dataset validation passed")

    def _split_dataset(self, dataset: List, label: str, train_ratio: float = 0.7, val_ratio: float = 0.1, random_seed: int = 2195719):
        """
        Split dataset by patient ID to avoid data leakage.
        
        Args:
            dataset: List of graph data
            label: Label name for logging
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            random_seed: Random seed for reproducibility
        """
        # Extract unique patient IDs
        pids = np.array(list(set([graph.pid.numpy()[0, 0] for graph in dataset])))
        n_pids = len(pids)
        
        # Calculate split sizes
        n_train = int(n_pids * train_ratio)
        n_val = int(n_pids * val_ratio)
        n_test = n_pids - n_train - n_val
        
        self.logger.info(f"Splitting {label} dataset: {n_train} train patients, {n_val} val patients, {n_test} test patients")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Split patient IDs
        pids_train = np.random.choice(pids, n_train, replace=False)
        pids_rest = np.array([pid for pid in pids if pid not in pids_train])
        
        np.random.seed(random_seed)
        pids_val = np.random.choice(pids_rest, n_val, replace=False)
        pids_test = np.array([pid for pid in pids_rest if pid not in pids_val])

        # Split datasets by patient ID
        train_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_train]
        val_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_val]
        test_ds = [graph for graph in dataset if graph.pid.numpy()[0, 0] in pids_test]
        
        self.logger.info(f"{label} split completed - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        # Save patient ID splits for reproducibility
        split_info = {
            'train_pids': pids_train.tolist(),
            'val_pids': pids_val.tolist(),
            'test_pids': pids_test.tolist(),
            'train_size': len(train_ds),
            'val_size': len(val_ds),
            'test_size': len(test_ds)
        }
        
        split_path = os.path.join(self.res_dir, f'{label}_split_info.json')
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        return train_ds, val_ds, test_ds
    
    def train_process(self):
        """Enhanced training process with early stopping, scheduling, and comprehensive logging."""
        self.logger.info("Starting enhanced training process...")
        self.start_time = time.time()
        
        # Initialize optimizer and scheduler
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.scheduler_step_size, 
            gamma=self.scheduler_gamma
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  - Optimizer: SGD (lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})")
        self.logger.info(f"  - Scheduler: StepLR (step_size={self.scheduler_step_size}, gamma={self.scheduler_gamma})")
        self.logger.info(f"  - Early stopping patience: {self.patience}")
        self.logger.info(f"  - Device: {self.device}")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(optimizer, criterion, epoch)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(criterion, epoch)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} Summary: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                f"LR={current_lr:.6f}, Time={epoch_time:.2f}s"
            )
            
            # Save model checkpoint
            self._save_checkpoint(epoch, val_loss, val_acc, optimizer, scheduler)
            
            # Early stopping check
            if self._check_early_stopping(val_loss, val_acc, epoch):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
                
            # Save training plots periodically
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._plot_training_curves()
        
        # Final evaluation and summary
        self._finish_training()
    
    def _train_epoch(self, optimizer: optim.Optimizer, criterion: torch.nn.Module, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch)
            loss = criterion(outputs, batch.y.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                pred_probs = F.softmax(outputs, dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
                true_labels = batch.y.squeeze()
                acc = (pred_labels == true_labels).float().mean().item()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            # Log batch progress periodically
            if (i + 1) % max(1, len(self.train_loader) // 5) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}, Batch {i + 1}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}, Acc={acc:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, criterion: torch.nn.Module, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                loss = criterion(outputs, batch.y.squeeze())
                
                # Calculate accuracy
                pred_probs = F.softmax(outputs, dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
                true_labels = batch.y.squeeze()
                acc = (pred_labels == true_labels).float().mean().item()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
                
                # Store predictions for detailed analysis
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
                all_probs.extend(pred_probs.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        # Calculate additional metrics
        if epoch % 10 == 0:  # Detailed metrics every 10 epochs
            self._log_detailed_metrics(all_labels, all_preds, all_probs, epoch)
        
        return avg_loss, avg_acc
    
    def _log_detailed_metrics(self, true_labels: List, pred_labels: List, pred_probs: List, epoch: int):
        """Log detailed validation metrics."""
        try:
            # Classification report
            report = classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'], output_dict=True)
            
            # AUC score
            prob_class1 = np.array(pred_probs)[:, 1]
            auc_score = roc_auc_score(true_labels, prob_class1)
            
            self.logger.info(f"Epoch {epoch + 1} Detailed Metrics:")
            self.logger.info(f"  - AUC Score: {auc_score:.4f}")
            self.logger.info(f"  - Class 0 - Precision: {report['Class 0']['precision']:.4f}, Recall: {report['Class 0']['recall']:.4f}, F1: {report['Class 0']['f1-score']:.4f}")
            self.logger.info(f"  - Class 1 - Precision: {report['Class 1']['precision']:.4f}, Recall: {report['Class 1']['recall']:.4f}, F1: {report['Class 1']['f1-score']:.4f}")
            
        except Exception as e:
            self.logger.warning(f"Could not calculate detailed metrics: {e}")
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, 
                        optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': {
                'node_feat_dim': self.node_feat_dim,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay
            }
        }
        
        # Always save latest checkpoint
        latest_path = os.path.join(self.res_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
            best_path = os.path.join(self.res_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save periodic checkpoints
        if not self.save_best_only:
            epoch_path = os.path.join(self.res_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)
    
    def _check_early_stopping(self, val_loss: float, val_acc: float, epoch: int) -> bool:
        """Check early stopping condition."""
        if val_acc > self.best_val_acc:
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                return True
        return False
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Loss curves
            ax1.plot(epochs, self.train_losses, label='Train Loss', color='blue')
            ax1.plot(epochs, self.val_losses, label='Val Loss', color='red')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy curves
            ax2.plot(epochs, self.train_accs, label='Train Acc', color='blue')
            ax2.plot(epochs, self.val_accs, label='Val Acc', color='red')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            # Learning rate curve
            ax3.plot(epochs, self.learning_rates, label='Learning Rate', color='green')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
            
            # Training time per epoch
            if self.epoch_times:
                ax4.plot(epochs[:len(self.epoch_times)], self.epoch_times, label='Epoch Time', color='orange')
                ax4.set_title('Training Time per Epoch')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Time (seconds)')
                ax4.legend()
                ax4.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.res_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create training plots: {e}")
    
    def _finish_training(self):
        """Finish training and provide summary."""
        total_time = time.time() - self.start_time
        
        # Create final plots
        self._plot_training_curves()
        
        # Save final training history
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_training_time': total_time
        }
        
        history_path = os.path.join(self.res_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Training summary
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch + 1})")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Final training accuracy: {self.train_accs[-1]:.4f}")
        self.logger.info(f"Final validation accuracy: {self.val_accs[-1]:.4f}")
        self.logger.info(f"Results saved to: {self.res_dir}")
        self.logger.info("=" * 60)
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate model on test set."""
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.res_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for evaluation")
        else:
            self.logger.warning("Best model not found, using current model state")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                loss = criterion(outputs, batch.y.squeeze())
                
                pred_probs = F.softmax(outputs, dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
                true_labels = batch.y.squeeze()
                
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
                all_probs.extend(pred_probs.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate metrics
        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        test_loss = total_loss / num_batches
        
        # Detailed metrics
        report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], output_dict=True)
        prob_class1 = np.array(all_probs)[:, 1]
        auc_score = roc_auc_score(all_labels, prob_class1)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        test_results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        # Save test results
        results_path = os.path.join(self.res_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                  for k, v in test_results.items() if k not in ['probabilities']}
            json.dump(serializable_results, f, indent=2)
        
        # Log results
        self.logger.info("=" * 50)
        self.logger.info("TEST SET EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Test Accuracy: {test_acc:.4f}")
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"AUC Score: {auc_score:.4f}")
        self.logger.info(f"Class 0 - Precision: {report['Class 0']['precision']:.4f}, Recall: {report['Class 0']['recall']:.4f}, F1: {report['Class 0']['f1-score']:.4f}")
        self.logger.info(f"Class 1 - Precision: {report['Class 1']['precision']:.4f}, Recall: {report['Class 1']['recall']:.4f}, F1: {report['Class 1']['f1-score']:.4f}")
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"{cm}")
        self.logger.info("=" * 50)
        
        return test_results


def main():
    """
    Main function to demonstrate trainer usage.
    Modify the paths and parameters according to your setup.
    """
    # Configuration
    config = {
        'label0_ds_pt': './data/label0_dataset.pt',  # Path to label 0 dataset
        'label1_ds_pt': './data/label1_dataset.pt',  # Path to label 1 dataset
        'node_feat_dim': 512,  # Node feature dimension (adjust based on your features)
        'batch_size': 32,
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 100,
        'res_dir': './output/training_results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 15,  # Early stopping patience
        'save_best_only': False,  # Save checkpoints for each epoch
        'scheduler_step_size': 30,  # LR scheduler step size
        'scheduler_gamma': 0.1,  # LR scheduler gamma
    }
    
    print("="*60)
    print("ENHANCED CELLNET TRAINER")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = Trainer(**config)
        
        # Start training
        trainer.train_process()
        
        # Evaluate on test set
        test_results = trainer.evaluate_test_set()
        
        print("\nTraining completed successfully!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
        print(f"Results saved to: {config['res_dir']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the dataset files exist and paths are correct.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()


def create_trainer_from_config(config_path: str) -> Trainer:
    """
    Create trainer from JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Trainer instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return Trainer(**config)


def resume_training(checkpoint_path: str, additional_epochs: int = 50) -> Trainer:
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        additional_epochs: Number of additional epochs to train
        
    Returns:
        Trainer instance
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create trainer
    trainer = Trainer(**config)
    
    # Load checkpoint
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.train_losses = checkpoint['train_losses']
    trainer.train_accs = checkpoint['train_accs']
    trainer.val_losses = checkpoint['val_losses']
    trainer.val_accs = checkpoint['val_accs']
    trainer.learning_rates = checkpoint['learning_rates']
    trainer.best_val_acc = checkpoint['best_val_acc']
    trainer.best_epoch = checkpoint['best_epoch']
    
    # Update epochs
    trainer.epochs = len(trainer.train_losses) + additional_epochs
    
    print(f"Resuming training from epoch {checkpoint['epoch']}")
    print(f"Previous best validation accuracy: {trainer.best_val_acc:.4f}")
    
    return trainer


if __name__ == "__main__":
    main()