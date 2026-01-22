#!/usr/bin/env python3
"""
FONTe AI - Dataset Pipeline

Creates training-ready data from tokenized SVGs:
1. Tokenize all glyphs
2. Filter by sequence length
3. Create PyTorch-compatible dataset
4. Save in efficient format for Colab

Usage:
    python create_dataset.py --turbo                    # Full dataset
    python create_dataset.py --limit 100 --max-len 256  # Test run

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from svg_tokenizer import (
    Vocabulary, tokenize_path, parse_svg_path,
    PATH_COMMANDS, CHAR_SET, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Unicode to character mapping
UNICODE_TO_CHAR = {}
for char in CHAR_SET:
    unicode_hex = f"uni{ord(char):04X}"
    UNICODE_TO_CHAR[unicode_hex] = char


@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""
    max_seq_length: int = 512      # Max tokens per sequence
    min_seq_length: int = 10       # Min tokens (filter empty glyphs)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


def process_font(args: Tuple[Path, str, Vocabulary, DatasetConfig]) -> List[Dict]:
    """
    Process a single font directory and return tokenized sequences.
    Runs in parallel worker process.
    """
    font_dir, style, vocab, config = args
    sequences = []
    
    try:
        svg_files = list(font_dir.glob("uni*.svg"))
        
        for svg_file in svg_files:
            unicode_name = svg_file.stem  # e.g., "uni0041"
            
            # Get character
            char = UNICODE_TO_CHAR.get(unicode_name)
            if char is None:
                continue
            
            try:
                with open(svg_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                path_data = parse_svg_path(content)
                if not path_data:
                    continue
                
                # Tokenize path
                path_tokens = tokenize_path(path_data)
                
                # Skip if too short or too long
                if len(path_tokens) < config.min_seq_length:
                    continue
                if len(path_tokens) > config.max_seq_length - 4:  # Reserve for SOS, STYLE, CHAR, EOS
                    continue
                
                # Create token sequence
                # Format: [SOS] [STYLE] [CHAR] [PATH...] [EOS]
                token_ids = []
                
                # SOS
                token_ids.append(vocab.encode(SOS_TOKEN))
                
                # Style conditioning
                style_token = f"<STYLE:{style}>"
                token_ids.append(vocab.encode(style_token))
                
                # Character conditioning
                char_token = f"<CHAR:{char}>"
                token_ids.append(vocab.encode(char_token))
                
                # Path tokens
                for token in path_tokens:
                    token_ids.append(vocab.encode(token))
                
                # EOS
                token_ids.append(vocab.encode(EOS_TOKEN))
                
                sequences.append({
                    'font_name': font_dir.name,
                    'character': char,
                    'unicode': unicode_name,
                    'style': style,
                    'token_ids': token_ids,
                    'length': len(token_ids),
                })
                
            except Exception as e:
                continue
    
    except Exception as e:
        pass
    
    return sequences


def create_dataset(
    dataset_dir: Path,
    output_dir: Path,
    vocab: Vocabulary,
    config: DatasetConfig,
    workers: int = 6,
    limit: int = None,
    turbo: bool = False
):
    """
    Create the full tokenized dataset.
    """
    logger.info(f"Creating dataset from {dataset_dir}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Workers: {workers}")
    
    # Collect all font directories with their styles
    font_tasks = []
    for style_dir in dataset_dir.iterdir():
        if not style_dir.is_dir():
            continue
        style = style_dir.name
        
        font_dirs = list(style_dir.iterdir())
        if limit:
            font_dirs = font_dirs[:limit]
        
        for font_dir in font_dirs:
            if font_dir.is_dir():
                font_tasks.append((font_dir, style, vocab, config))
    
    logger.info(f"Processing {len(font_tasks)} fonts...")
    
    # Process in parallel
    all_sequences = []
    start_time = time.time()
    completed = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_font, task): task for task in font_tasks}
        
        for future in as_completed(futures):
            try:
                sequences = future.result()
                all_sequences.extend(sequences)
                completed += 1
                
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(font_tasks) - completed) / rate if rate > 0 else 0
                    print(f"\r  {completed}/{len(font_tasks)} fonts | {rate:.1f} fonts/sec | ETA: {eta:.0f}s", end="")
            
            except Exception as e:
                completed += 1
    
    print()
    elapsed = time.time() - start_time
    
    logger.info(f"Processed {len(font_tasks)} fonts in {elapsed:.1f}s")
    logger.info(f"Total sequences: {len(all_sequences)}")
    
    # Filter and analyze
    lengths = [s['length'] for s in all_sequences]
    logger.info(f"Sequence length stats:")
    logger.info(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
    
    # Count by style
    style_counts = {}
    for s in all_sequences:
        style_counts[s['style']] = style_counts.get(s['style'], 0) + 1
    logger.info(f"By style: {style_counts}")
    
    # Shuffle and split
    import random
    random.seed(config.seed)
    random.shuffle(all_sequences)
    
    n = len(all_sequences)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)
    
    splits = {
        'train': all_sequences[:train_end],
        'val': all_sequences[train_end:val_end],
        'test': all_sequences[val_end:],
    }
    
    logger.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Save datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, sequences in splits.items():
        # Save as JSON (readable, for debugging)
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'split': split_name,
                'count': len(sequences),
                'max_length': config.max_seq_length,
                'vocab_size': len(vocab),
                'sequences': sequences,
            }, f)
        
        logger.info(f"Saved {split_name} to {json_path}")
        
        # Also save as binary (efficient, for training)
        bin_path = output_dir / f"{split_name}.bin"
        save_binary_dataset(sequences, bin_path, config.max_seq_length, vocab)
        logger.info(f"Saved binary {split_name} to {bin_path}")
    
    # Save dataset config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'max_seq_length': config.max_seq_length,
            'min_seq_length': config.min_seq_length,
            'vocab_size': len(vocab),
            'total_sequences': len(all_sequences),
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
            'style_counts': style_counts,
            'processing_time': elapsed,
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FONTe AI - Dataset Created")
    print("=" * 60)
    print(f"Total sequences:  {len(all_sequences):,}")
    print(f"Max seq length:   {config.max_seq_length}")
    print(f"Vocab size:       {len(vocab)}")
    print(f"Processing time:  {elapsed:.1f}s")
    print("-" * 60)
    print("Splits:")
    print(f"  Train: {len(splits['train']):,} ({len(splits['train'])/n*100:.1f}%)")
    print(f"  Val:   {len(splits['val']):,} ({len(splits['val'])/n*100:.1f}%)")
    print(f"  Test:  {len(splits['test']):,} ({len(splits['test'])/n*100:.1f}%)")
    print("-" * 60)
    print("Style distribution:")
    for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
        print(f"  {style:15} {count:6,}")
    print("-" * 60)
    print(f"Output: {output_dir}")
    print("=" * 60)


def save_binary_dataset(sequences: List[Dict], path: Path, max_length: int, vocab: Vocabulary):
    """
    Save dataset in binary format for efficient loading.
    
    Format:
        Header: [num_sequences (4 bytes), max_length (4 bytes), vocab_size (4 bytes)]
        Per sequence: [length (2 bytes), token_ids (max_length * 2 bytes)]
    """
    pad_id = vocab.encode(PAD_TOKEN)
    
    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('III', len(sequences), max_length, len(vocab)))
        
        # Sequences
        for seq in sequences:
            token_ids = seq['token_ids']
            length = len(token_ids)
            
            # Pad to max_length
            padded = token_ids + [pad_id] * (max_length - length)
            padded = padded[:max_length]
            
            # Write length and tokens
            f.write(struct.pack('H', length))
            f.write(struct.pack(f'{max_length}H', *padded))


def load_binary_dataset(path: Path) -> Tuple[List[List[int]], List[int], int, int]:
    """
    Load binary dataset.
    
    Returns: (token_ids_list, lengths, max_length, vocab_size)
    """
    with open(path, 'rb') as f:
        # Header
        num_sequences, max_length, vocab_size = struct.unpack('III', f.read(12))
        
        token_ids_list = []
        lengths = []
        
        for _ in range(num_sequences):
            length = struct.unpack('H', f.read(2))[0]
            tokens = list(struct.unpack(f'{max_length}H', f.read(max_length * 2)))
            
            lengths.append(length)
            token_ids_list.append(tokens)
    
    return token_ids_list, lengths, max_length, vocab_size


def main():
    parser = argparse.ArgumentParser(
        description='Create training dataset for FONTe AI'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=Path,
        default=Path('./DATASET_NORMALIZED'),
        help='Input normalized dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./TOKENIZED'),
        help='Output directory for tokenized data'
    )
    parser.add_argument(
        '--max-len', '-m',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--turbo', '-t',
        action='store_true',
        help='Turbo mode: 80%% CPU'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit fonts per style (for testing)'
    )
    
    args = parser.parse_args()
    
    # Calculate workers
    if args.workers is None:
        total_cpus = multiprocessing.cpu_count()
        args.workers = max(1, int(total_cpus * 0.8)) if args.turbo else 4
    
    if args.turbo:
        logger.info("🚀 TURBO MODE ENABLED")
    
    logger.info(f"Workers: {args.workers}")
    
    # Load or build vocabulary
    vocab_path = args.output / "vocabulary.json"
    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
        logger.info(f"Loaded vocabulary ({len(vocab)} tokens)")
    else:
        vocab = Vocabulary()
        args.output.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        logger.info(f"Created vocabulary ({len(vocab)} tokens)")
    
    # Create config
    config = DatasetConfig(
        max_seq_length=args.max_len,
    )
    
    # Create dataset
    create_dataset(
        dataset_dir=args.dataset.resolve(),
        output_dir=args.output.resolve(),
        vocab=vocab,
        config=config,
        workers=args.workers,
        limit=args.limit,
        turbo=args.turbo,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
