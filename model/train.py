#!/usr/bin/env python3
"""
FONTe AI - Training Script

Train the SVG Path Transformer model.
Supports both local training and Google Colab.

Usage:
    # Local training
    python train.py --epochs 10 --batch-size 32
    
    # Colab training (uses GPU)
    python train.py --epochs 50 --batch-size 64 --device cuda

Author: FONTe AI Project
"""

import os
import sys
import json
import argparse
import logging
import time
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.fonte_model import FonteModel, ModelConfig, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class FonteDataset(Dataset):
    """PyTorch Dataset for FONTe training data"""
    
    def __init__(self, data_path: Path, max_length: int = 512):
        self.max_length = max_length
        
        # Load binary dataset for efficiency
        if data_path.suffix == '.bin':
            self.token_ids, self.lengths, self.max_length, self.vocab_size = \
                self._load_binary(data_path)
        else:
            # Load JSON
            self.token_ids, self.lengths = self._load_json(data_path)
    
    def _load_binary(self, path: Path) -> Tuple[List, List, int, int]:
        """Load binary format dataset"""
        with open(path, 'rb') as f:
            num_sequences, max_length, vocab_size = struct.unpack('III', f.read(12))
            
            token_ids = []
            lengths = []
            
            for _ in range(num_sequences):
                length = struct.unpack('H', f.read(2))[0]
                tokens = list(struct.unpack(f'{max_length}H', f.read(max_length * 2)))
                
                lengths.append(length)
                token_ids.append(tokens)
        
        logger.info(f"Loaded {len(token_ids)} sequences from {path}")
        return token_ids, lengths, max_length, vocab_size
    
    def _load_json(self, path: Path) -> Tuple[List, List]:
        """Load JSON format dataset"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        token_ids = []
        lengths = []
        
        for seq in data['sequences']:
            ids = seq['token_ids']
            # Pad to max_length
            padded = ids + [0] * (self.max_length - len(ids))
            padded = padded[:self.max_length]
            
            token_ids.append(padded)
            lengths.append(min(len(ids), self.max_length))
        
        logger.info(f"Loaded {len(token_ids)} sequences from {path}")
        return token_ids, lengths
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.token_ids[idx], dtype=torch.long),
            'length': self.lengths[idx],
        }


# ============================================================================
# TRAINING
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    train_path: Path = Path('./TOKENIZED/train.bin')
    val_path: Path = Path('./TOKENIZED/val.bin')
    
    # Model
    model_size: str = 'medium'  # small, medium, large
    vocab_size: int = 1105
    max_seq_length: int = 512
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Device
    device: str = 'cpu'
    
    # Saving
    save_dir: Path = Path('./checkpoints')
    save_every: int = 5
    
    # Logging
    log_every: int = 50


def train_epoch(
    model: FonteModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainingConfig,
    epoch: int,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(config.device)
        
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if batch_idx % config.log_every == 0:
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}"
            )
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: FonteModel,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> float:
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(config.device)
        
        outputs = model(input_ids, labels=input_ids)
        total_loss += outputs['loss'].item()
        num_batches += 1
    
    return total_loss / num_batches


def train(config: TrainingConfig):
    """Main training loop"""
    logger.info("=" * 60)
    logger.info("FONTe AI - Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Model size: {config.model_size}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = FonteDataset(config.train_path, config.max_seq_length)
    val_dataset = FonteDataset(config.val_path, config.max_seq_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for compatibility
        pin_memory=config.device == 'cuda',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config.model_size, config.vocab_size)
    model = model.to(config.device)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Create save directory
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = config.save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model_size': config.model_size,
            'vocab_size': config.vocab_size,
            'max_seq_length': config.max_seq_length,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        }, f, indent=2)
    
    # Training loop
    best_val_loss = float('inf')
    training_history = []
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, config)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': epoch_time,
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = config.save_dir / 'best_model.pt'
            model.save(best_path)
            logger.info(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % config.save_every == 0:
            ckpt_path = config.save_dir / f'checkpoint_epoch_{epoch}.pt'
            model.save(ckpt_path)
            logger.info(f"💾 Saved checkpoint: {ckpt_path}")
    
    total_time = time.time() - start_time
    
    # Save final model
    final_path = config.save_dir / 'final_model.pt'
    model.save(final_path)
    
    # Save training history
    history_path = config.save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final model: {final_path}")
    print(f"Best model: {config.save_dir / 'best_model.pt'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train FONTe AI model')
    
    # Data
    parser.add_argument('--train-path', type=Path, default=Path('./TOKENIZED/train.bin'))
    parser.add_argument('--val-path', type=Path, default=Path('./TOKENIZED/val.bin'))
    
    # Model
    parser.add_argument('--model-size', type=str, default='medium',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--vocab-size', type=int, default=1105)
    parser.add_argument('--max-seq-length', type=int, default=512)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])
    
    # Saving
    parser.add_argument('--save-dir', type=Path, default=Path('./checkpoints'))
    parser.add_argument('--save-every', type=int, default=5)
    
    # Logging
    parser.add_argument('--log-every', type=int, default=50)
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create config
    config = TrainingConfig(
        train_path=args.train_path.resolve(),
        val_path=args.val_path.resolve(),
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        device=args.device,
        save_dir=args.save_dir.resolve(),
        save_every=args.save_every,
        log_every=args.log_every,
    )
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
