"""
Main Training Script for ATN Deer Age Recognition
Implements two-phase training: Triplet Loss → Augmented Triplet Loss
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import numpy as np
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extractor import create_feature_extractor
from models.atn_loss import TripletLoss, AugmentedTripletLoss, CombinedLoss
from utils.data_loader import create_data_loaders
from training import config


class ATNTrainer:
    """Trainer for Augmented Triplet Network"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        """
        Args:
            model: Feature extractor model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.triplet_loss = TripletLoss(
            margin=config.TRIPLET_MARGIN,
            distance_metric=config.DISTANCE_METRIC
        )
        
        self.atn_loss = AugmentedTripletLoss(
            alpha=config.ATN_ALPHA,
            beta=config.ATN_BETA,
            distance_metric=config.DISTANCE_METRIC
        )
        
        # Optimizer
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.phase = 'triplet'  # 'triplet' or 'atn'
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'phase': []
        }
    
    def setup_optimizer(self, learning_rate, weight_decay=1e-4):
        """Setup optimizer and scheduler with differential learning rates"""
        # Define parameter groups
        # The backbone (feature extractor) usually needs a lower LR than the new head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Determine backbone LR factor from config if available (default 0.1)
        lr_factor = getattr(config, 'BACKBONE_LR_FACTOR', 0.1)
        
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * lr_factor},
            {'params': head_params, 'lr': learning_rate}
        ]
        
        if config.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=weight_decay
            )
        elif config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=weight_decay
            )
        elif config.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=config.MOMENTUM,
                weight_decay=weight_decay
            )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE,
                min_lr=config.LR_MIN
            )
        elif config.LR_SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=config.LR_FACTOR
            )
        elif config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=config.LR_MIN
            )
    
    def train_epoch_triplet(self):
        """Train one epoch with triplet loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Triplet]")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch
            if isinstance(batch_data[0], tuple):
                # Triplet format: (anchor, positive, negative), labels
                (anchor_img, positive_img, negative_img), labels = batch_data
                anchor_img = anchor_img.to(self.device)
                positive_img = positive_img.to(self.device)
                negative_img = negative_img.to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor_img)
                positive_emb = self.model(positive_img)
                negative_emb = self.model(negative_img)
                
                # Compute loss
                loss, stats = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            else:
                # Single image format - create triplets on the fly
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Simple triplet mining: for each anchor, find positive and negative
                batch_size = embeddings.size(0)
                anchor_emb = embeddings
                
                # Find positive (same label)
                positive_emb = []
                negative_emb = []
                
                for i in range(batch_size):
                    anchor_label = labels[i]
                    
                    # Find positive (same label, different index)
                    pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=self.device) != i)
                    if pos_mask.sum() > 0:
                        pos_idx = torch.where(pos_mask)[0][0]
                        positive_emb.append(embeddings[pos_idx])
                    else:
                        positive_emb.append(embeddings[i])  # Use same if no other positive
                    
                    # Find negative (different label)
                    neg_mask = labels != anchor_label
                    if neg_mask.sum() > 0:
                        neg_idx = torch.where(neg_mask)[0][0]
                        negative_emb.append(embeddings[neg_idx])
                    else:
                        negative_emb.append(embeddings[(i+1) % batch_size])  # Fallback
                
                positive_emb = torch.stack(positive_emb)
                negative_emb = torch.stack(negative_emb)
                
                # Compute loss
                loss, stats = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch_atn(self):
        """Train one epoch with ATN loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [ATN]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            
            # Compute ATN loss
            loss, stats = self.atn_loss(embeddings, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'close_pairs': stats.get('num_close_pairs', 0)
            })
            
            # Log to tensorboard
            if batch_idx % config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/ClosePairs', stats.get('num_close_pairs', 0), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                # Compute loss based on current phase
                if self.phase == 'triplet':
                    # Simple validation with triplet loss
                    batch_size = embeddings.size(0)
                    if batch_size < 2:
                        continue
                    
                    # Create simple triplets for validation
                    anchor_emb = embeddings[:-1]
                    positive_emb = embeddings[1:]
                    negative_emb = torch.roll(embeddings, 1, dims=0)[:-1]
                    
                    loss, _ = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                else:
                    loss, _ = self.atn_loss(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'phase': self.phase,
            'history': self.history
        }
        
        filepath = config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load model checkpoint"""
        filepath = config.CHECKPOINT_DIR / filename
        
        if not filepath.exists():
            print(f"Checkpoint not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.phase = checkpoint.get('phase', 'triplet')
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'phase': []})
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.current_epoch}, phase: {self.phase}")
        return True
    
    def train(self, num_epochs, phase='triplet', checkpoint_suffix=''):
        """
        Train model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            phase: 'triplet' or 'atn'
            checkpoint_suffix: Suffix to add to checkpoint filenames
        """
        self.phase = phase
        print(f"\n{'='*60}")
        print(f"Starting training - Phase: {phase.upper()} {checkpoint_suffix}")
        print(f"{'='*60}\n")
        
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train one epoch
            if phase == 'triplet':
                train_loss = self.train_epoch_triplet()
            else:
                train_loss = self.train_epoch_atn()
            
            # Validate
            val_loss = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['phase'].append(phase)
            
            # Log to tensorboard
            self.writer.add_scalar(f'Loss/Train{checkpoint_suffix}', train_loss, self.current_epoch)
            self.writer.add_scalar(f'Loss/Val{checkpoint_suffix}', val_loss, self.current_epoch)
            
            # Print epoch summary
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # Learning rate scheduler
            if config.LR_SCHEDULER == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_name = f'best_model{checkpoint_suffix}.pth'
                self.save_checkpoint(best_name)
                print(f"  [+] New best model saved as {best_name}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Periodic checkpoint
            if (self.current_epoch % config.SAVE_FREQUENCY == 0):
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}{checkpoint_suffix}.pth')
            
            # Early stopping
            if config.EARLY_STOPPING and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed - Phase: {phase.upper()}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")


def main():
    """Main training function"""
    print("="*80)
    print("ATN DEER AGE RECOGNITION - TRAINING")
    print("="*80)
    
    # Print configuration
    config.print_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase2_only', action='store_true', help='Skip Phase 1 and start Phase 2')
    parser.add_argument('--kfold', action='store_true', help='Train 5 models using K-Fold cross-validation')
    args = parser.parse_args()

    num_folds = 5 if args.kfold else 1
    from utils.kfold_loader import create_kfold_loaders

    for fold in range(num_folds):
        if args.kfold:
            print(f"\n\n" + "#"*80)
            print(f" TRAINING FOLD {fold+1}/{num_folds} ".center(80, '#'))
            print("#"*80 + "\n")
            
            loaders = create_kfold_loaders(
                data_dir=config.RAW_DATA_DIR,
                fold_idx=fold,
                num_folds=num_folds,
                batch_size=config.PHASE1_BATCH_SIZE,
                random_seed=config.RANDOM_SEED
            )
            train_loader = loaders['train']
            val_loader = loaders['val']
        else:
            # Standard single split
            data_loaders = create_data_loaders(
                data_dir=config.RAW_DATA_DIR,
                batch_size=config.PHASE1_BATCH_SIZE,
                random_seed=config.RANDOM_SEED
            )
            train_loader = data_loaders['train']
            val_loader = data_loaders['val']
        
        # Create model and trainer for this fold
        model = create_feature_extractor(
            embedding_dim=config.EMBEDDING_DIM,
            backbone=config.BACKBONE,
            pretrained=config.PRETRAINED,
            device=config.DEVICE
        )
        
        trainer = ATNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.DEVICE
        )
        
        # Suffix for fold checkpoints
        suffix = f"_fold_{fold+1}" if args.kfold else ""
        
        # Phase 1: Standard Triplet Loss
        if not args.phase2_only:
            print("\n" + "="*80)
            print(f"PHASE 1: STANDARD TRIPLET LOSS {suffix}")
            print("="*80)
            
            trainer.setup_optimizer(
                learning_rate=config.PHASE1_LEARNING_RATE,
                weight_decay=config.PHASE1_WEIGHT_DECAY
            )
            
            try:
                trainer.train(
                    num_epochs=config.PHASE1_EPOCHS,
                    phase='triplet',
                    checkpoint_suffix=suffix
                )
            except KeyboardInterrupt:
                print(f"Phase 1 (Fold {fold+1}) interrupted. Saving checkpoint...")
                trainer.save_checkpoint(f'checkpoint_interrupted{suffix}.pth')
                return
        else:
            print(f"\nSkipping Phase 1 for {suffix} as requested. Loading best model...")
        
        # Phase 2: Augmented Triplet Loss
        print("\n" + "="*80)
        print(f"PHASE 2: AUGMENTED TRIPLET LOSS {suffix}")
        print("="*80)
        
        # Phase 2: ATN
        trainer.load_checkpoint(f'best_model{suffix}.pth')
        trainer.setup_optimizer(config.PHASE2_LEARNING_RATE, config.PHASE2_WEIGHT_DECAY)
        trainer.train(num_epochs=config.PHASE2_EPOCHS, phase='atn', checkpoint_suffix=suffix)
        
        # Save final fold model
        trainer.save_checkpoint(f'final_model{suffix}.pth')
        
        # Save training history
        history_path = config.RESULTS_DIR / f'training_history{suffix}.json'
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)
        
        print("\n" + "="*80)
        print(f"FOLD {fold+1} COMPLETED!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print("="*80)
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED!")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print("="*80)
    print(f"Models saved in: {config.CHECKPOINT_DIR}")
    print(f"Logs saved in: {config.LOG_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
