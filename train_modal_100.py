#!/usr/bin/env python3
"""
FONTe AI - Modal Training Script (100-Font Sample)

OPTIMIZED FOR B200:
- Large batch size (2048) with gradient accumulation if needed
- BF16 mixed precision training
- Flash Attention via scaled_dot_product_attention
- Preload data to GPU
- No torch.compile (uses too much memory for CUDA graphs)

Run with: modal run train_modal_100.py -d
"""

import modal
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG - OPTIMIZED FOR B200
# ═══════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 2048     # Large batch, but not full dataset
EPOCHS = 200
LR = 5e-4             # Higher LR for large batch
DEVICE = 'cuda'
SAVE_EVERY = 50       # Save less frequently

# ═══════════════════════════════════════════════════════════════════════════
# MODAL APP
# ═══════════════════════════════════════════════════════════════════════════

app = modal.App("fonte-ai-100")
vol = modal.Volume.from_name("fonte-data-100")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")

@app.function(image=image, gpu="B200", timeout=7200, volumes={"/data": vol})
def train():
    """Train on 100-font sample - FULLY OPTIMIZED FOR B200"""
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import json
    import struct
    import math
    import time

    print("=" * 70)
    print("FONTe AI - B200 OPTIMIZED Training")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE (with Flash Attention)
    # ═══════════════════════════════════════════════════════════════════════

    @dataclass
    class ModelConfig:
        vocab_size: int = 1106
        max_seq_length: int = 512
        d_model: int = 256
        n_heads: int = 4
        n_layers: int = 6
        d_ff: int = 1024
        dropout: float = 0.1
        pad_token_id: int = 0

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class MultiHeadAttention(nn.Module):
        """Using PyTorch's scaled_dot_product_attention for Flash Attention"""
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.wq = nn.Linear(d_model, d_model)
            self.wk = nn.Linear(d_model, d_model)
            self.wv = nn.Linear(d_model, d_model)
            self.wo = nn.Linear(d_model, d_model)
            self.dropout = dropout
            
        def forward(self, x, mask=None):
            B, L, D = x.shape
            q = self.wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            k = self.wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            v = self.wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            
            # Use Flash Attention via scaled_dot_product_attention
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,  # Use is_causal instead for efficiency
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            out = out.transpose(1, 2).reshape(B, L, -1)
            return self.wo(out)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
            self.n1 = nn.LayerNorm(d_model)
            self.n2 = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            x = x + self.drop(self.attn(self.n1(x)))
            return x + self.drop(self.ff(self.n2(x)))

    class FonteModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
            self.pos = PositionalEncoding(cfg.d_model, cfg.max_seq_length, cfg.dropout)
            self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.emb.weight
        
        def forward(self, input_ids, labels=None):
            x = self.pos(self.emb(input_ids))
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.norm(x))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.cfg.vocab_size), labels[:, 1:].reshape(-1), ignore_index=self.cfg.pad_token_id)
            return {"loss": loss, "logits": logits}

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD DATA TO GPU
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_binary_to_gpu(bin_path, device):
        """Load entire binary dataset directly to GPU memory"""
        with open(bin_path, 'rb') as f:
            header = f.read(12)
            count, max_len, vocab_size = struct.unpack('III', header)
            data = f.read()
        
        # Parse all sequences at once
        bytes_per_seq = 2 + max_len * 2
        all_tokens = []
        for i in range(count):
            offset = i * bytes_per_seq + 2  # skip 2-byte length prefix
            tokens = struct.unpack(f'{max_len}H', data[offset:offset + max_len * 2])
            all_tokens.append(tokens)
        
        # Convert to tensor and move to GPU
        tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
        print(f"  Loaded {count:,} sequences to GPU ({tensor.numel() * 2 / 1e6:.1f} MB)")
        return tensor, count, max_len, vocab_size

    # Load everything to GPU
    print("\n📥 Loading data directly to GPU...")
    train_data, train_count, max_len, vocab_size = load_binary_to_gpu("/data/train.bin", DEVICE)
    val_data, val_count, _, _ = load_binary_to_gpu("/data/val.bin", DEVICE)
    
    print(f"\n📊 Dataset: {train_count:,} train / {val_count:,} val")
    print(f"   Sequence length: {max_len}, Vocab size: {vocab_size}")
    print(f"   Batch size: {BATCH_SIZE} ({(train_count + BATCH_SIZE - 1) // BATCH_SIZE} batches/epoch)")
    
    # GPU memory check
    print(f"\n🎮 GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used / {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB total")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODEL SETUP WITH BF16
    # ═══════════════════════════════════════════════════════════════════════
    
    cfg = ModelConfig()
    model = FonteModel(cfg).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: {param_count/1e6:.1f}M params")
    print(f"   Using Flash Attention (scaled_dot_product_attention)")
    
    # Optimizer
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=True)
    n_batches = (train_count + BATCH_SIZE - 1) // BATCH_SIZE
    sched = CosineAnnealingLR(opt, T_max=EPOCHS * n_batches)
    
    print(f"   LR: {LR}, Epochs: {EPOCHS}")
    print(f"   Using: BF16 mixed precision + fused AdamW")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP - MINI-BATCH WITH DATA ON GPU
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70 + "\n")
    
    best_loss = float('inf')
    history = []
    start_time = time.time()
    epoch = 0
    
    # Create random indices for shuffling (on GPU)
    train_indices = torch.arange(train_count, device=DEVICE)
    
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            
            # Shuffle training data
            perm = torch.randperm(train_count, device=DEVICE)
            shuffled_train = train_data[perm]
            
            # ─── TRAIN (mini-batches) ───
            model.train()
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, train_count, BATCH_SIZE):
                batch = shuffled_train[i:i+BATCH_SIZE]
                
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(batch, batch)
                    loss = out['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            train_loss_val = total_loss / n_batches
            
            # ─── VALIDATE (single batch since val is small) ───
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                val_out = model(val_data, val_data)
                val_loss = val_out['loss'].item()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            history.append({'epoch': epoch, 'train_loss': train_loss_val, 'val_loss': val_loss})
            
            # Track best
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            # Print every 10 epochs or if improved
            if epoch % 10 == 0 or epoch <= 5 or is_best:
                improved = " ⭐ BEST" if is_best else ""
                print(f"E{epoch:3d}/{EPOCHS} | train: {train_loss_val:.4f} | val: {val_loss:.4f} | {epoch_time*1000:.0f}ms{improved}")
            
            # Save checkpoints
            if epoch % SAVE_EVERY == 0:
                torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, f"/data/epoch_{epoch}.pt")
                torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, "/data/best_model.pt")
                with open("/data/history.json", "w") as f:
                    json.dump(history, f)
                vol.commit()
                print(f"   💾 Checkpoint saved (epoch {epoch})")
    
    except Exception as e:
        print(f"\n❌ ERROR at epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()
        # Save emergency checkpoint
        try:
            torch.save({
                'config': cfg.__dict__, 
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'error': str(e)
            }, "/data/emergency_checkpoint.pt")
            with open("/data/history.json", "w") as f:
                json.dump(history, f)
            vol.commit()
            print("Emergency checkpoint saved!")
        except Exception as e2:
            print(f"Failed to save emergency checkpoint: {e2}")
        raise
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SAVE
    # ═══════════════════════════════════════════════════════════════════════
    
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, "/data/final_model.pt")
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, "/data/best_model.pt")
    
    with open("/data/history.json", "w") as f:
        json.dump(history, f)
    
    total_time = time.time() - start_time
    vol.commit()
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Best val_loss: {best_loss:.4f}")
    print(f"   Throughput: {EPOCHS/total_time:.1f} epochs/sec")
    print("=" * 70)
    
    return {"best_val_loss": best_loss, "total_time_sec": total_time, "epochs_per_sec": EPOCHS/total_time}

@app.local_entrypoint()
def main():
    print("🚀 Starting FONTe AI training on Modal B200...")
    print("   Using: BF16 + Flash Attention + Large Batch")
    result = train.remote()
    print(f"\n✅ Training complete: {result}")
