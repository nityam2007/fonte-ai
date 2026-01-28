#!/usr/bin/env python3
"""
FONTe AI - Font Generation Script

Usage:
    python scripts/generate_font.py --model TRAINED/epoch_1.pt --char A
    python scripts/generate_font.py --model TRAINED/epoch_50.pt --style serif --chars ABC
    python scripts/generate_font.py --model TRAINED/best_model.pt --all-chars --output generated/
"""

import argparse
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import shared constants
from constants import (
    VOCAB_SIZE, MAX_SEQ_LENGTH, CANVAS_SIZE,
    PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID,
    NEG_TOKEN_ID, COMMAND_START, COMMAND_END,
    STYLE_START, STYLE_END,
    CHAR_START, CHAR_END, CHARS,
    COORD_START, COORD_END, COORD_MAX,
    is_command_token, is_style_token, is_char_token, is_coord_token,
    coord_to_token_id, token_id_to_coord, char_to_token_id, token_id_to_char,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model Architecture (matching Modal notebook)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE  # 1106: includes <NEG> token
    max_seq_length: int = MAX_SEQ_LENGTH  # 512
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    pad_token_id: int = PAD_TOKEN_ID  # 0
    sos_token_id: int = SOS_TOKEN_ID  # 1
    eos_token_id: int = EOS_TOKEN_ID  # 2


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
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.n1(x), mask))
        return x + self.drop(self.ff(self.n2(x)))


class FonteModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        # Note: mask is optional - models trained with Flash Attention don't have it
        # self.register_buffer('mask', torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)))

    def forward(self, input_ids, labels=None):
        x = self.pos(self.emb(input_ids))
        # Create causal mask on the fly if needed
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.norm(x))
        return {'logits': logits}

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FonteModel':
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        # Load state dict with strict=False to handle missing 'mask' buffer
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    @torch.no_grad()
    def generate(self, style_id: int, char_id: int, max_length: int = 512, 
                 temperature: float = 1.0, top_k: int = 50,
                 repetition_penalty: float = 1.2) -> List[int]:
        """Generate tokens for a single glyph.
        
        Args:
            repetition_penalty: Penalty for repeating recent tokens. >1.0 discourages repetition.
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([[self.config.sos_token_id, style_id, char_id]], device=device)
        
        for _ in range(max_length - 3):
            outputs = self.forward(tokens)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply repetition penalty to recently generated tokens
            if repetition_penalty != 1.0:
                # Get last 20 tokens (or all if less)
                recent_tokens = tokens[0, -20:].tolist()
                for token_id in set(recent_tokens):
                    # Don't penalize special tokens (0-3) or commands (4-23)
                    if token_id >= NEG_TOKEN_ID:  # Only penalize coord/style/char tokens
                        logits[0, token_id] /= repetition_penalty
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
        
        return tokens[0].tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Token Mappings (from shared constants module)
# ═══════════════════════════════════════════════════════════════════════════

# Vocabulary Layout (from constants.py):
# 0-3:                    Special tokens (PAD, SOS, EOS, UNK)
# COMMAND_START-COMMAND_END (4-24): Commands including <NEG>
# STYLE_START-STYLE_END (25-29):    Style tokens
# CHAR_START-CHAR_END (30-105):     Character tokens
# COORD_START-COORD_END (106-1105): Coordinate tokens (0-999)

STYLE_IDS = {
    'serif': STYLE_START,      # 25
    'sans-serif': STYLE_START + 1,  # 26
    'monospace': STYLE_START + 2,   # 27
    'handwriting': STYLE_START + 3, # 28
    'display': STYLE_START + 4,     # 29
}

# Character mappings (uses shared CHARS constant)
CHAR_IDS = {char: CHAR_START + i for i, char in enumerate(CHARS)}

# Reverse mappings
ID_TO_STYLE = {v: k for k, v in STYLE_IDS.items()}
ID_TO_CHAR = {v: k for k, v in CHAR_IDS.items()}

# Command tokens (for decoding)
COMMANDS = ['M', 'm', 'L', 'l', 'H', 'h', 'V', 'v', 'C', 'c', 'S', 's', 'Q', 'q', 'T', 't', 'A', 'a', 'Z', 'z']
COMMAND_IDS = {cmd: COMMAND_START + i for i, cmd in enumerate(COMMANDS)}
ID_TO_COMMAND = {v: k for k, v in COMMAND_IDS.items()}

# Coordinate tokens use COORD_START and COORD_END from constants


# ═══════════════════════════════════════════════════════════════════════════
# Token to SVG Conversion
# ═══════════════════════════════════════════════════════════════════════════

def tokens_to_svg_path(tokens: List[int], canvas_size: int = CANVAS_SIZE) -> str:
    """Convert token sequence to SVG path data.
    
    Uses constants for token ID ranges to avoid hardcoded values.
    Also maintains backward compatibility with old models that used UNK (ID 3).
    """
    path_parts = []
    i = 0
    next_is_negative = False  # Track if next coord should be negative
    
    # Skip SOS, style, char tokens at the start
    while i < len(tokens) and (tokens[i] <= UNK_TOKEN_ID or (STYLE_START <= tokens[i] <= CHAR_END)):
        i += 1
    
    while i < len(tokens):
        token = tokens[i]
        
        # EOS or PAD - stop
        if token == PAD_TOKEN_ID or token == EOS_TOKEN_ID:
            break
        
        # <NEG> token - mark next coordinate as negative
        if token == NEG_TOKEN_ID:
            next_is_negative = True
            i += 1
            continue
        
        # UNK - LEGACY: treat as negative sign for old models
        # Old models trained with bug may still output UNK for negatives
        if token == UNK_TOKEN_ID:
            next_is_negative = True
            i += 1
            continue
        
        # SOS - skip
        if token == SOS_TOKEN_ID:
            i += 1
            continue
        
        # Command token (4-23, excluding NEG at 24)
        if COMMAND_START <= token < NEG_TOKEN_ID:
            cmd = ID_TO_COMMAND.get(token, '')
            path_parts.append(cmd)
            next_is_negative = False  # Reset after command
            i += 1
            continue
        
        # Style/char tokens that appear mid-sequence - skip
        if STYLE_START <= token <= CHAR_END:
            i += 1
            continue
        
        # Coordinate token
        if is_coord_token(token):
            coord = token_id_to_coord(token)
            # Scale from 0-999 to 0-canvas_size
            scaled = coord * canvas_size / 1000
            # Apply negative sign if <NEG> or UNK preceded this coordinate
            if next_is_negative:
                scaled = -scaled
                next_is_negative = False
            path_parts.append(f"{scaled:.1f}")
            i += 1
            continue
        
        i += 1
    
    return ' '.join(path_parts)


def create_svg(path_data: str, canvas_size: int = 128, char: str = '?', style: str = 'unknown') -> str:
    """Create complete SVG from path data."""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {canvas_size} {canvas_size}" xmlns="http://www.w3.org/2000/svg">
  <!-- Generated by FONTe AI -->
  <!-- Style: {style}, Character: {char} -->
  <path d="{path_data}" fill="#000"/>
</svg>'''


# ═══════════════════════════════════════════════════════════════════════════
# Main Generation Functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_glyph(model: FonteModel, style: str, char: str, 
                   temperature: float = 0.8, top_k: int = 50, max_length: int = 512,
                   repetition_penalty: float = 1.2) -> dict:
    """Generate a single glyph."""
    if style not in STYLE_IDS:
        raise ValueError(f"Unknown style: {style}. Choose from: {list(STYLE_IDS.keys())}")
    if char not in CHAR_IDS:
        raise ValueError(f"Unknown character: {char}. Supported: {CHARS}")
    
    style_id = STYLE_IDS[style]
    char_id = CHAR_IDS[char]
    
    tokens = model.generate(style_id, char_id, max_length=max_length, 
                           temperature=temperature, top_k=top_k,
                           repetition_penalty=repetition_penalty)
    path_data = tokens_to_svg_path(tokens)
    svg = create_svg(path_data, char=char, style=style)
    
    return {
        'char': char,
        'style': style,
        'tokens': tokens,
        'token_count': len(tokens),
        'path_data': path_data,
        'svg': svg,
    }


def list_models(model_dir: str) -> List[Path]:
    """List all .pt/.pth model files in directory."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    
    models = list(model_dir.glob('*.pt')) + list(model_dir.glob('*.pth'))
    return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)


def main():
    parser = argparse.ArgumentParser(description='FONTe AI - Generate fonts from trained models')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model checkpoint (.pt/.pth)')
    parser.add_argument('--style', '-s', type=str, default='sans-serif',
                        choices=list(STYLE_IDS.keys()),
                        help='Font style (default: sans-serif)')
    parser.add_argument('--char', '-c', type=str, default=None,
                        help='Single character to generate')
    parser.add_argument('--chars', type=str, default=None,
                        help='Multiple characters to generate (e.g., "ABC")')
    parser.add_argument('--all-chars', action='store_true',
                        help='Generate all supported characters')
    parser.add_argument('--output', '-o', type=str, default='generated',
                        help='Output directory (default: generated/)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Generation temperature (default: 0.8)')
    parser.add_argument('--top-k', '-k', type=int, default=50,
                        help='Top-k sampling (default: 50)')
    parser.add_argument('--max-length', '-l', type=int, default=512,
                        help='Maximum tokens per glyph (default: 512)')
    parser.add_argument('--repetition-penalty', '-r', type=float, default=1.2,
                        help='Repetition penalty for coords (default: 1.2, >1 discourages repeats)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models in TRAINED/ directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda (default: auto)')
    
    args = parser.parse_args()
    
    # List models mode
    if args.list_models:
        print("\n📁 Available models in TRAINED/:")
        models = list_models('TRAINED')
        if not models:
            print("  No models found. Train a model first!")
        else:
            for m in models:
                size_mb = m.stat().st_size / (1024 * 1024)
                print(f"  • {m.name} ({size_mb:.1f} MB)")
        return
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load model
    print(f"\n🔧 Loading model: {args.model}")
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    model = FonteModel.load(args.model, device=device).to(device)
    print(f"✅ Model loaded on {device}")
    
    # Determine characters to generate
    if args.all_chars:
        chars = CHARS
    elif args.chars:
        chars = args.chars
    elif args.char:
        chars = args.char
    else:
        chars = 'A'  # Default
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate glyphs
    print(f"\n🎨 Generating {len(chars)} glyph(s) with style '{args.style}'...")
    print(f"   Temperature: {args.temperature}, Top-k: {args.top_k}, Max-length: {args.max_length}")
    print(f"   Repetition penalty: {args.repetition_penalty}")
    
    results = []
    for char in chars:
        try:
            result = generate_glyph(
                model, args.style, char,
                temperature=args.temperature,
                top_k=args.top_k,
                max_length=args.max_length,
                repetition_penalty=args.repetition_penalty
            )
            results.append(result)
            
            # Save SVG
            svg_path = output_dir / f"{args.style}_{char}.svg"
            with open(svg_path, 'w') as f:
                f.write(result['svg'])
            
            print(f"   ✅ '{char}' → {result['token_count']} tokens → {svg_path}")
            
        except Exception as e:
            print(f"   ❌ '{char}' failed: {e}")
    
    # Summary
    print(f"\n📊 Generated {len(results)}/{len(chars)} glyphs")
    print(f"📁 Output: {output_dir}/")
    
    # Create preview HTML
    if results:
        html_path = output_dir / 'preview.html'
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>FONTe AI - Generated Glyphs</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, 150px); gap: 10px; }}
        .glyph {{ background: white; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .glyph img {{ width: 128px; height: 128px; }}
        .glyph .label {{ margin-top: 5px; font-weight: bold; }}
        .glyph .info {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>🎨 FONTe AI - Generated Glyphs</h1>
    <p>Style: <strong>{args.style}</strong> | Model: <code>{args.model}</code></p>
    <div class="grid">
'''
        for r in results:
            svg_file = f"{args.style}_{r['char']}.svg"
            html_content += f'''        <div class="glyph">
            <img src="{svg_file}" alt="{r['char']}">
            <div class="label">{r['char']}</div>
            <div class="info">{r['token_count']} tokens</div>
        </div>
'''
        html_content += '''    </div>
</body>
</html>'''
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"🌐 Preview: {html_path}")


if __name__ == '__main__':
    main()
