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


# ═══════════════════════════════════════════════════════════════════════════
# Model Architecture (same as training)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 1105
    max_seq_length: int = 512
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class FonteModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer('causal_mask', torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)))

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return {'logits': logits}

    @torch.no_grad()
    def generate(self, style_id: int, char_id: int, max_length: int = 256, 
                 temperature: float = 1.0, top_k: int = 50) -> List[int]:
        """Generate tokens for a single glyph."""
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor([[self.config.sos_token_id, style_id, char_id]], device=device)
        
        for _ in range(max_length - 3):
            outputs = self.forward(tokens)
            logits = outputs['logits'][:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
        
        return tokens[0].tolist()

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FonteModel':
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model


# ═══════════════════════════════════════════════════════════════════════════
# Token Mappings
# ═══════════════════════════════════════════════════════════════════════════

STYLE_IDS = {
    'serif': 28,
    'sans-serif': 29,
    'monospace': 30,
    'handwriting': 31,
    'display': 32,
}

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHAR_IDS = {char: 33 + i for i, char in enumerate(CHARS)}

# Reverse mappings
ID_TO_STYLE = {v: k for k, v in STYLE_IDS.items()}
ID_TO_CHAR = {v: k for k, v in CHAR_IDS.items()}

# Command tokens (for decoding)
COMMANDS = ['M', 'm', 'L', 'l', 'H', 'h', 'V', 'v', 'C', 'c', 'S', 's', 'Q', 'q', 'T', 't', 'A', 'a', 'Z', 'z']
COMMAND_IDS = {cmd: 4 + i for i, cmd in enumerate(COMMANDS)}
ID_TO_COMMAND = {v: k for k, v in COMMAND_IDS.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Token to SVG Conversion
# ═══════════════════════════════════════════════════════════════════════════

def tokens_to_svg_path(tokens: List[int], canvas_size: int = 128) -> str:
    """Convert token sequence to SVG path data."""
    path_parts = []
    i = 0
    
    # Skip SOS, style, char tokens
    while i < len(tokens) and tokens[i] < 100:
        i += 1
    
    while i < len(tokens):
        token = tokens[i]
        
        # EOS or PAD
        if token <= 3:
            break
        
        # Command token (4-23)
        if 4 <= token <= 23:
            cmd = ID_TO_COMMAND.get(token, '')
            path_parts.append(cmd)
            i += 1
            continue
        
        # Coordinate token (105-1104) -> 0-999
        if 105 <= token <= 1104:
            coord = token - 105
            # Scale from 0-999 to 0-canvas_size
            scaled = coord * canvas_size / 1000
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
                   temperature: float = 0.8, top_k: int = 50) -> dict:
    """Generate a single glyph."""
    if style not in STYLE_IDS:
        raise ValueError(f"Unknown style: {style}. Choose from: {list(STYLE_IDS.keys())}")
    if char not in CHAR_IDS:
        raise ValueError(f"Unknown character: {char}. Supported: {CHARS}")
    
    style_id = STYLE_IDS[style]
    char_id = CHAR_IDS[char]
    
    tokens = model.generate(style_id, char_id, temperature=temperature, top_k=top_k)
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
    print(f"   Temperature: {args.temperature}, Top-k: {args.top_k}")
    
    results = []
    for char in chars:
        try:
            result = generate_glyph(
                model, args.style, char,
                temperature=args.temperature,
                top_k=args.top_k
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
