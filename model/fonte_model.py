#!/usr/bin/env python3
"""
FONTe AI - SVG Path Transformer Model

A transformer-based model that generates SVG path commands.
Input: Style token + Character token
Output: Sequence of path tokens (M, L, C, Q, Z + coordinates)

Architecture:
- Embedding layer for all tokens (commands, coords, style, char)
- Transformer decoder (autoregressive generation)
- Language model head for next-token prediction

Designed for:
- CPU training (small, efficient)
- Google Colab T4 GPU (fits in 16GB)
- Fast inference

Author: FONTe AI Project
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 1105          # From vocabulary.json
    max_seq_length: int = 512       # Maximum sequence length
    d_model: int = 256              # Embedding dimension
    n_heads: int = 4                # Attention heads
    n_layers: int = 6               # Transformer layers
    d_ff: int = 1024                # FFN hidden dimension
    dropout: float = 0.1            # Dropout rate
    
    # Special token IDs (from vocabulary)
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ModelConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def small(cls) -> 'ModelConfig':
        """Small model for testing (~3M params)"""
        return cls(
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
        )
    
    @classmethod
    def medium(cls) -> 'ModelConfig':
        """Medium model for training (~12M params)"""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=6,
            d_ff=1024,
        )
    
    @classmethod
    def large(cls) -> 'ModelConfig':
        """Large model for quality (~50M params)"""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        # FFN with residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class FonteModel(nn.Module):
    """
    FONTe AI - SVG Path Transformer
    
    Generates SVG path tokens autoregressively.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model,
            padding_idx=config.pad_token_id
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            config.max_seq_length,
            config.dropout
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target token IDs for loss computation
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        result = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten and compute cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        style_id: int,
        char_id: int,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate SVG path tokens for a given style and character.
        
        Args:
            style_id: Style token ID
            char_id: Character token ID
            max_length: Maximum generation length
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token IDs (1, seq_len)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Start with [SOS, STYLE, CHAR]
        tokens = torch.tensor(
            [[self.config.sos_token_id, style_id, char_id]],
            device=device
        )
        
        for _ in range(max_length - 3):
            # Get logits for last position
            outputs = self.forward(tokens)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if EOS generated
            if next_token.item() == self.config.eos_token_id:
                break
        
        return tokens
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'config': self.config.__dict__,
            'state_dict': self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'FonteModel':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model


def create_model(size: str = 'medium', vocab_size: int = 1105) -> FonteModel:
    """Create a model with the specified size"""
    if size == 'small':
        config = ModelConfig.small()
    elif size == 'medium':
        config = ModelConfig.medium()
    elif size == 'large':
        config = ModelConfig.large()
    else:
        raise ValueError(f"Unknown size: {size}")
    
    config.vocab_size = vocab_size
    return FonteModel(config)


# ============================================================================
# TESTING
# ============================================================================

def test_model():
    """Test model forward pass and generation"""
    print("Testing FONTe Model...")
    
    # Create small model for testing
    config = ModelConfig.small()
    model = FonteModel(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    style_id = 28  # <STYLE:serif>
    char_id = 37   # <CHAR:A>
    
    generated = model.generate(style_id, char_id, max_length=50, temperature=0.8)
    print(f"Generated sequence: {generated.shape}")
    print(f"Tokens: {generated[0].tolist()[:20]}...")
    
    print("✅ Model test passed!")


if __name__ == '__main__':
    test_model()
