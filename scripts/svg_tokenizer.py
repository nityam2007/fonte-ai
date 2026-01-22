#!/usr/bin/env python3
"""
SVG Path Tokenizer for FONTe AI

This script converts SVG path commands into tokens for transformer training.
Treats SVG paths as a "language" with:
- Command tokens: M, L, C, Q, Z, H, V (move, line, curve, etc.)
- Coordinate tokens: Quantized to 0-999 range

Example:
    Path: "M 100 200 L 150 300 Z"
    Tokens: [M, 100, 200, L, 150, 300, Z]

Usage:
    python svg_tokenizer.py --analyze    # Analyze dataset, build vocabulary
    python svg_tokenizer.py --tokenize   # Tokenize all SVGs
    python svg_tokenizer.py --test       # Test tokenization

Author: FONTe AI Project
"""

import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# VOCABULARY DEFINITION
# ============================================================================

# Special tokens
PAD_TOKEN = "<PAD>"      # Padding
SOS_TOKEN = "<SOS>"      # Start of sequence
EOS_TOKEN = "<EOS>"      # End of sequence
UNK_TOKEN = "<UNK>"      # Unknown token

# SVG Path command tokens
# Uppercase = absolute, lowercase = relative
PATH_COMMANDS = [
    'M', 'm',  # MoveTo
    'L', 'l',  # LineTo
    'H', 'h',  # Horizontal LineTo
    'V', 'v',  # Vertical LineTo
    'C', 'c',  # Cubic Bezier
    'S', 's',  # Smooth Cubic Bezier
    'Q', 'q',  # Quadratic Bezier
    'T', 't',  # Smooth Quadratic Bezier
    'A', 'a',  # Arc
    'Z', 'z',  # ClosePath
]

# Coordinate quantization
COORD_MIN = 0
COORD_MAX = 999
COORD_RANGE = COORD_MAX - COORD_MIN + 1

# Style tokens (5 styles)
STYLE_TOKENS = [
    "<STYLE:serif>",
    "<STYLE:sans-serif>",
    "<STYLE:monospace>",
    "<STYLE:handwriting>",
    "<STYLE:display>",
]

# Character tokens (72 characters we extract)
CHAR_SET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!@#$%&*()-+=[]"
)


@dataclass
class Vocabulary:
    """Token vocabulary for SVG paths"""
    
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.token_to_id:
            self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build the complete vocabulary"""
        tokens = []
        
        # Special tokens (0-3)
        tokens.extend([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
        
        # Path commands (4-27)
        tokens.extend(PATH_COMMANDS)
        
        # Style tokens (28-32)
        tokens.extend(STYLE_TOKENS)
        
        # Character tokens (33-104) - as conditioning
        for char in CHAR_SET:
            tokens.append(f"<CHAR:{char}>")
        
        # Coordinate tokens (105-1104) - quantized 0-999
        for i in range(COORD_RANGE):
            tokens.append(f"<COORD:{i}>")
        
        # Build mappings
        for idx, token in enumerate(tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def encode(self, token: str) -> int:
        """Convert token to ID"""
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])
    
    def decode(self, token_id: int) -> str:
        """Convert ID to token"""
        return self.id_to_token.get(token_id, UNK_TOKEN)
    
    def encode_coord(self, value: float, scale: float = 1.0) -> int:
        """Quantize and encode a coordinate value"""
        # Scale to 0-999 range
        quantized = int(value * scale)
        quantized = max(COORD_MIN, min(COORD_MAX, quantized))
        return self.encode(f"<COORD:{quantized}>")
    
    def decode_coord(self, token_id: int) -> Optional[float]:
        """Decode a coordinate token to value"""
        token = self.decode(token_id)
        if token.startswith("<COORD:"):
            return float(token[7:-1])
        return None
    
    def save(self, path: Path):
        """Save vocabulary to JSON"""
        data = {
            "token_to_id": self.token_to_id,
            "vocab_size": len(self),
            "special_tokens": [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
            "num_commands": len(PATH_COMMANDS),
            "num_styles": len(STYLE_TOKENS),
            "num_chars": len(CHAR_SET),
            "coord_range": COORD_RANGE,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved vocabulary ({len(self)} tokens) to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        """Load vocabulary from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        vocab = cls()
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(k): v for k, v in 
                            {v: k for k, v in data["token_to_id"].items()}.items()}
        return vocab


# ============================================================================
# PATH TOKENIZATION
# ============================================================================

def parse_svg_path(svg_content: str) -> Optional[str]:
    """Extract path data from SVG content"""
    match = re.search(r'd="([^"]+)"', svg_content)
    if match:
        return match.group(1)
    return None


def tokenize_path(path_data: str) -> List[str]:
    """
    Tokenize SVG path data into command and coordinate tokens.
    
    Example:
        Input:  "M 100 200 L 150 300 Z"
        Output: ["M", "<COORD:100>", "<COORD:200>", "L", "<COORD:150>", "<COORD:300>", "Z"]
    """
    tokens = []
    
    # Split path into commands and numbers
    # This regex captures: commands (letters) and numbers (including negative/decimal)
    pattern = r'([MmLlHhVvCcSsQqTtAaZz])|(-?[\d.]+)'
    
    for match in re.finditer(pattern, path_data):
        command = match.group(1)
        number = match.group(2)
        
        if command:
            tokens.append(command)
        elif number:
            # Quantize coordinate to 0-999
            try:
                value = float(number)
                # Clamp to valid range
                quantized = int(max(COORD_MIN, min(COORD_MAX, abs(value))))
                # Add sign token for negative values
                if value < 0:
                    tokens.append("<NEG>")
                tokens.append(f"<COORD:{quantized}>")
            except ValueError:
                continue
    
    return tokens


def normalize_path_for_tokenization(path_data: str, canvas_size: int = 128) -> List[str]:
    """
    Normalize path coordinates to 0-999 range and tokenize.
    This creates a scale-invariant representation.
    """
    tokens = []
    scale = 999.0 / canvas_size  # Scale 128 -> 999
    
    pattern = r'([MmLlHhVvCcSsQqTtAaZz])|(-?[\d.]+)'
    
    for match in re.finditer(pattern, path_data):
        command = match.group(1)
        number = match.group(2)
        
        if command:
            tokens.append(command)
        elif number:
            try:
                value = float(number)
                # Scale to 0-999 range
                scaled = value * scale
                quantized = int(max(COORD_MIN, min(COORD_MAX, abs(scaled))))
                if value < 0:
                    tokens.append("<NEG>")
                tokens.append(f"<COORD:{quantized}>")
            except ValueError:
                continue
    
    return tokens


def detokenize_path(tokens: List[str], canvas_size: int = 128) -> str:
    """
    Convert tokens back to SVG path data.
    
    Example:
        Input:  ["M", "<COORD:100>", "<COORD:200>", "L", "<COORD:150>", "<COORD:300>", "Z"]
        Output: "M 100 200 L 150 300 Z"
    """
    parts = []
    scale = canvas_size / 999.0  # Scale 999 -> 128
    is_negative = False
    
    for token in tokens:
        if token in PATH_COMMANDS:
            parts.append(token)
            is_negative = False
        elif token == "<NEG>":
            is_negative = True
        elif token.startswith("<COORD:"):
            value = float(token[7:-1]) * scale
            if is_negative:
                value = -value
            parts.append(f"{value:.1f}")
            is_negative = False
    
    return " ".join(parts)


# ============================================================================
# SEQUENCE CREATION
# ============================================================================

@dataclass
class TokenizedGlyph:
    """A tokenized glyph with metadata"""
    font_name: str
    character: str
    unicode: str
    style: str
    path_tokens: List[str]
    token_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "font_name": self.font_name,
            "character": self.character,
            "unicode": self.unicode,
            "style": self.style,
            "path_tokens": self.path_tokens,
            "token_ids": self.token_ids,
            "seq_length": len(self.token_ids),
        }


def create_training_sequence(
    glyph: TokenizedGlyph,
    vocab: Vocabulary,
    max_length: int = 512
) -> Dict:
    """
    Create a training sequence with conditioning tokens.
    
    Format: [SOS] [STYLE] [CHAR] [PATH_TOKENS...] [EOS] [PAD...]
    
    The model learns: given style + char, predict the path tokens.
    """
    sequence = []
    
    # Start token
    sequence.append(vocab.encode(SOS_TOKEN))
    
    # Style conditioning
    style_token = f"<STYLE:{glyph.style}>"
    sequence.append(vocab.encode(style_token))
    
    # Character conditioning
    char_token = f"<CHAR:{glyph.character}>"
    sequence.append(vocab.encode(char_token))
    
    # Path tokens
    for token in glyph.path_tokens:
        if token in PATH_COMMANDS:
            sequence.append(vocab.encode(token))
        elif token.startswith("<COORD:") or token == "<NEG>":
            sequence.append(vocab.encode(token))
    
    # End token
    sequence.append(vocab.encode(EOS_TOKEN))
    
    # Truncate if too long
    if len(sequence) > max_length:
        sequence = sequence[:max_length-1] + [vocab.encode(EOS_TOKEN)]
    
    # Pad to max_length
    pad_id = vocab.encode(PAD_TOKEN)
    while len(sequence) < max_length:
        sequence.append(pad_id)
    
    return {
        "input_ids": sequence,
        "length": min(len(glyph.path_tokens) + 3, max_length),  # +3 for SOS, STYLE, CHAR
        "style": glyph.style,
        "character": glyph.character,
        "font_name": glyph.font_name,
    }


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

def analyze_dataset(dataset_dir: Path, limit: int = None) -> Dict:
    """
    Analyze the dataset to understand path complexity and token distribution.
    """
    logger.info(f"Analyzing dataset: {dataset_dir}")
    
    stats = {
        "total_glyphs": 0,
        "total_tokens": 0,
        "command_counts": Counter(),
        "seq_lengths": [],
        "coord_distribution": Counter(),
        "fonts_by_style": Counter(),
        "glyphs_by_char": Counter(),
        "max_seq_length": 0,
        "min_seq_length": float('inf'),
    }
    
    style_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    for style_dir in style_dirs:
        style = style_dir.name
        font_dirs = list(style_dir.iterdir())
        
        if limit:
            font_dirs = font_dirs[:limit]
        
        for font_dir in font_dirs:
            if not font_dir.is_dir():
                continue
            
            stats["fonts_by_style"][style] += 1
            
            svg_files = list(font_dir.glob("uni*.svg"))
            for svg_file in svg_files:
                try:
                    with open(svg_file, 'r') as f:
                        content = f.read()
                    
                    path_data = parse_svg_path(content)
                    if not path_data:
                        continue
                    
                    tokens = tokenize_path(path_data)
                    
                    # Count commands
                    for token in tokens:
                        if token in PATH_COMMANDS:
                            stats["command_counts"][token] += 1
                        elif token.startswith("<COORD:"):
                            coord = int(token[7:-1])
                            bucket = coord // 100  # 0-9 buckets
                            stats["coord_distribution"][bucket] += 1
                    
                    # Track sequence lengths
                    seq_len = len(tokens)
                    stats["seq_lengths"].append(seq_len)
                    stats["total_tokens"] += seq_len
                    stats["max_seq_length"] = max(stats["max_seq_length"], seq_len)
                    stats["min_seq_length"] = min(stats["min_seq_length"], seq_len)
                    stats["total_glyphs"] += 1
                    
                    # Track character distribution
                    unicode_name = svg_file.stem  # e.g., "uni0041"
                    stats["glyphs_by_char"][unicode_name] += 1
                    
                except Exception as e:
                    logger.debug(f"Error analyzing {svg_file}: {e}")
    
    # Calculate statistics
    if stats["seq_lengths"]:
        stats["avg_seq_length"] = sum(stats["seq_lengths"]) / len(stats["seq_lengths"])
        stats["median_seq_length"] = sorted(stats["seq_lengths"])[len(stats["seq_lengths"]) // 2]
    
    return stats


def print_analysis(stats: Dict):
    """Print analysis results"""
    print("\n" + "=" * 60)
    print("FONTe AI - Dataset Analysis")
    print("=" * 60)
    
    print(f"\n📊 Overview:")
    print(f"  Total glyphs:     {stats['total_glyphs']:,}")
    print(f"  Total tokens:     {stats['total_tokens']:,}")
    print(f"  Avg tokens/glyph: {stats.get('avg_seq_length', 0):.1f}")
    print(f"  Min seq length:   {stats['min_seq_length']}")
    print(f"  Max seq length:   {stats['max_seq_length']}")
    print(f"  Median seq len:   {stats.get('median_seq_length', 0)}")
    
    print(f"\n📝 Command Distribution:")
    for cmd, count in stats["command_counts"].most_common(10):
        pct = count / sum(stats["command_counts"].values()) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cmd:3} {count:8,} ({pct:5.1f}%) {bar}")
    
    print(f"\n🎨 Fonts by Style:")
    for style, count in stats["fonts_by_style"].most_common():
        print(f"  {style:15} {count:5}")
    
    print(f"\n📈 Coordinate Distribution (by 100s):")
    for bucket in range(10):
        count = stats["coord_distribution"].get(bucket, 0)
        total = sum(stats["coord_distribution"].values())
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket*100:3}-{bucket*100+99:3}: {count:8,} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SVG Path Tokenizer for FONTe AI'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=Path,
        default=Path('./DATASET_NORMALIZED'),
        help='Path to normalized dataset'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./TOKENIZED'),
        help='Output directory for tokenized data'
    )
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze dataset and print statistics'
    )
    parser.add_argument(
        '--build-vocab',
        action='store_true',
        help='Build and save vocabulary'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test tokenization with examples'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit fonts per style (for testing)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    
    if args.analyze:
        # Analyze dataset
        stats = analyze_dataset(dataset_dir, limit=args.limit)
        print_analysis(stats)
        
        # Save stats
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = output_dir / "analysis.json"
        
        # Convert counters to dicts for JSON
        stats_json = {
            k: dict(v) if isinstance(v, Counter) else v 
            for k, v in stats.items()
            if k != "seq_lengths"  # Don't save full list
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"\nSaved analysis to {stats_file}")
    
    elif args.build_vocab:
        # Build and save vocabulary
        vocab = Vocabulary()
        output_dir.mkdir(parents=True, exist_ok=True)
        vocab.save(output_dir / "vocabulary.json")
        
        print(f"\n📚 Vocabulary Built:")
        print(f"  Total tokens:    {len(vocab)}")
        print(f"  Special tokens:  4 (PAD, SOS, EOS, UNK)")
        print(f"  Commands:        {len(PATH_COMMANDS)}")
        print(f"  Styles:          {len(STYLE_TOKENS)}")
        print(f"  Characters:      {len(CHAR_SET)}")
        print(f"  Coordinates:     {COORD_RANGE} (0-999)")
    
    elif args.test:
        # Test tokenization
        print("\n🧪 Testing Tokenization\n")
        
        vocab = Vocabulary()
        
        # Test path
        test_path = "M 100 200 L 150 300 C 200 400 250 350 300 300 Z"
        print(f"Original path: {test_path}")
        
        tokens = tokenize_path(test_path)
        print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
        
        # Encode to IDs
        ids = [vocab.encode(t) for t in tokens]
        print(f"Token IDs: {ids[:20]}...")
        
        # Decode back
        decoded_tokens = [vocab.decode(i) for i in ids]
        print(f"Decoded: {decoded_tokens[:20]}...")
        
        # Back to path
        reconstructed = detokenize_path(tokens)
        print(f"Reconstructed: {reconstructed}")
        
        # Test with real SVG
        print("\n📄 Testing with real SVG:")
        svg_files = list(dataset_dir.glob("*/*/uni0041.svg"))[:1]
        if svg_files:
            with open(svg_files[0], 'r') as f:
                content = f.read()
            path = parse_svg_path(content)
            if path:
                print(f"Path data: {path[:100]}...")
                tokens = tokenize_path(path)
                print(f"Token count: {len(tokens)}")
                print(f"Commands: {[t for t in tokens if t in PATH_COMMANDS]}")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
