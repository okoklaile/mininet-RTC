"""
Training script for BC-GCC
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import os
import sys

from config import Config
from model import GCCBC_LSTM, CombinedLoss
from dataset import create_dataloaders, normalize_features, denormalize_target
from torch.utils.data import TensorDataset


class Trainer:
    """Trainer for BC-GCC model"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Set device
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.SEED)
            # Enable cuDNN benchmark for faster training (RTX 3090 optimization)
            torch.backends.cudnn.benchmark = True
        
        # Create directories
        Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
        Path(config.LOG_DIR).mkdir(exist_ok=True)
        
        # Create model
        self.model = GCCBC_LSTM(config).to(self.device)
        print(f"\nModel created with {self.model.count_parameters():,} parameters")
        
        # Create dataloaders (use preprocessed data if available)
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders(config)
        
        # Loss function
        self.criterion = CombinedLoss().to(self.device)
        
        # Optimizer
        if config.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
        elif config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        # Learning rate scheduler
        if config.SCHEDULER == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.SCHEDULER_PATIENCE,
                factor=config.SCHEDULER_FACTOR,
                verbose=True,
            )
        elif config.SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
            )
        else:
            self.scheduler = None
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Mixed precision training (for RTX 3090)
        self.use_amp = config.USE_AMP and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using mixed precision training (AMP) for faster training on RTX 3090")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
    
    def _create_dataloaders(self, config):
        """
        Create dataloaders, using preprocessed data if available
        
        Returns:
            train_loader, val_loader, test_loader
        """
        processed_dir = Path(config.DATA_DIR) / 'processed'
        train_path = processed_dir / 'train_tensors.pt'
        val_path = processed_dir / 'val_tensors.pt'
        test_path = processed_dir / 'test_tensors.pt'
        
        # Check if preprocessed data exists
        if train_path.exists() and val_path.exists() and test_path.exists():
            print("\n" + "="*80)
            print("Found preprocessed data! Loading for fast training...")
            print("="*80)
            
            # Load preprocessed tensors
            print(f"Loading train data from {train_path}...")
            train_data = torch.load(train_path)
            print(f"Loading val data from {val_path}...")
            val_data = torch.load(val_path)
            print(f"Loading test data from {test_path}...")
            test_data = torch.load(test_path)
            
            # Create TensorDatasets
            train_dataset = TensorDataset(
                train_data['features'],
                train_data['targets'],
                train_data['weights']
            )
            val_dataset = TensorDataset(
                val_data['features'],
                val_data['targets'],
                val_data['weights']
            )
            test_dataset = TensorDataset(
                test_data['features'],
                test_data['targets'],
                test_data['weights']
            )
            
            print(f"\nLoaded preprocessed data:")
            print(f"  Train: {len(train_dataset):,} samples")
            print(f"  Val:   {len(val_dataset):,} samples")
            print(f"  Test:  {len(test_dataset):,} samples")
            
            # Create DataLoaders (no workers needed for tensor data!)
            from torch.utils.data import DataLoader
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,  # No workers needed - data already in memory!
                pin_memory=True if config.DEVICE == 'cuda' else False,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True if config.DEVICE == 'cuda' else False,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True if config.DEVICE == 'cuda' else False,
            )
            
            print("\n⚡ Using FAST mode: preprocessed tensor loading")
            print("   Expected speed: 500-800 it/s")
            print("="*80)
            
            return train_loader, val_loader, test_loader
        
        else:
            # Preprocessed data not found, use original method
            print("\n" + "="*80)
            print("Preprocessed data not found. Using original data loading...")
            print("="*80)
            print("\nTo enable fast training:")
            print("  1. Run: python3 prepare_data.py")
            print("  2. Wait for preprocessing (5-10 minutes)")
            print("  3. Run training again for 10-15x speedup")
            print("="*80 + "\n")
            
            return create_dataloaders(config)
    
    def _normalize_targets(self, targets):
        """Normalize targets (bandwidth) to [0, 1] range"""
        min_val = self.config.NORM_STATS['bandwidth_prediction']['min']
        max_val = self.config.NORM_STATS['bandwidth_prediction']['max']
        return (targets - min_val) / (max_val - min_val)
    
    def _denormalize_targets(self, targets_normalized):
        """Denormalize targets back to bps"""
        min_val = self.config.NORM_STATS['bandwidth_prediction']['min']
        max_val = self.config.NORM_STATS['bandwidth_prediction']['max']
        return targets_normalized * (max_val - min_val) + min_val
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (features, targets, weights) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            weights = weights.to(self.device)
            
            # Normalize features and targets to [0, 1] range
            features = normalize_features(features, self.config)
            targets_normalized = self._normalize_targets(targets)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions, _ = self.model(features)
                    loss = self.criterion(predictions, targets_normalized, weights)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training (no AMP)
                predictions, _ = self.model(features)
                loss = self.criterion(predictions, targets_normalized, weights)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update stats
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_samples:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # Log to tensorboard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        total_mape = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets, weights in tqdm(self.val_loader, desc='Validating'):
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                weights = weights.to(self.device)
                
                # Normalize features and targets
                features = normalize_features(features, self.config)
                targets_normalized = self._normalize_targets(targets)
                
                # Forward pass
                predictions, _ = self.model(features)
                
                # Compute loss (on normalized scale)
                loss = self.criterion(predictions, targets_normalized, weights)
                
                # Denormalize for metric computation (in bps)
                predictions_bps = self._denormalize_targets(predictions)
                targets_bps = targets  # Already in bps
                
                # Compute metrics (in bps)
                mae = torch.abs(predictions_bps - targets_bps).mean()
                mape = (torch.abs(predictions_bps - targets_bps) / (targets_bps + 1.0)).mean()
                
                # Update stats
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                total_mape += mape.item() * batch_size
                total_samples += batch_size
                
                # Store for analysis (in bps)
                all_predictions.extend(predictions_bps.cpu().numpy())
                all_targets.extend(targets_bps.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        avg_mape = total_mape / total_samples
        
        # Compute R2 score
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - targets_np.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return avg_loss, avg_mae, avg_mape, r2
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config.CHECKPOINT_DIR) / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.CHECKPOINT_DIR) / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model (val_loss={self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_loss, val_mae, val_mape, val_r2 = self.validate()
                
                # Log to tensorboard
                self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/mae', val_mae, epoch)
                self.writer.add_scalar('val/mape', val_mape, epoch)
                self.writer.add_scalar('val/r2', val_r2, epoch)
                
                # Print stats
                print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val MAE: {val_mae:.2f} bps")
                print(f"  Val MAPE: {val_mape*100:.2f}%")
                print(f"  Val R²: {val_r2:.4f}")
                
                # Learning rate scheduler
                if self.scheduler is not None:
                    if self.config.SCHEDULER == 'plateau':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Check if best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Save latest checkpoint
                self.save_checkpoint(is_best=False)
                
                # Early stopping
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*80)
        
        # Close writer
        self.writer.close()
    
    def test(self):
        """Test on test set"""
        # Load best model
        best_path = Path(self.config.CHECKPOINT_DIR) / 'best.pt'
        if best_path.exists():
            self.load_checkpoint(best_path)
        
        print("\n" + "="*80)
        print("Testing on test set...")
        print("="*80)
        
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets, weights in tqdm(self.test_loader, desc='Testing'):
                features = features.to(self.device)
                targets = targets.to(self.device)
                weights = weights.to(self.device)
                
                # Normalize features and targets
                features = normalize_features(features, self.config)
                targets_normalized = self._normalize_targets(targets)
                
                # Forward pass
                predictions, _ = self.model(features)
                
                # Compute loss (on normalized scale)
                loss = self.criterion(predictions, targets_normalized, weights)
                
                # Denormalize for metrics (in bps)
                predictions_bps = self._denormalize_targets(predictions)
                targets_bps = targets  # Already in bps
                mae = torch.abs(predictions_bps - targets_bps).mean()
                
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                total_samples += batch_size
                
                # Store for analysis (in bps)
                all_predictions.extend(predictions_bps.cpu().numpy())
                all_targets.extend(targets_bps.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - targets_np.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nTest Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  MAE: {avg_mae:.2f} bps ({avg_mae/1e6:.2f} Mbps)")
        print(f"  R²: {r2:.4f}")


def main():
    # Load config
    config = Config()
    config.print_config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()
    
    # Test
    trainer.test()


if __name__ == '__main__':
    main()
