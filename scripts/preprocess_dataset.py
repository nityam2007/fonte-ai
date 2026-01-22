#!/usr/bin/env python3
"""
SVG Dataset Preprocessor for FONTe AI

This script performs:
1. SVG Normalization - Standardize all glyphs to 128x128 canvas
2. Style Classification - Label fonts as serif/sans-serif/display/handwriting/monospace
3. Dataset Organization - Sort by style, create train/val/test splits

Input: Raw SVG glyphs from DATASET/
Output: Normalized dataset in DATASET_NORMALIZED/

Usage:
    python preprocess_dataset.py [--input ./DATASET] [--output ./DATASET_NORMALIZED]
    python preprocess_dataset.py --turbo  # 80% CPU, maximum speed

Author: FONTe AI Project
"""

import os
import sys
import re
import json
import argparse
import logging
import time
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass, field
import threading
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canvas size for normalized output
CANVAS_SIZE = 128  # 128x128 for efficient CPU training

# Style classification keywords
STYLE_KEYWORDS = {
    'serif': [
        'serif', 'roman', 'times', 'georgia', 'garamond', 'bodoni', 'didot',
        'caslon', 'baskerville', 'century', 'palatino', 'book', 'classical',
        'traditional', 'old', 'antiqua', 'mincho', 'ming', 'song'
    ],
    'sans-serif': [
        'sans', 'gothic', 'grotesk', 'grotesque', 'helvetica', 'arial',
        'futura', 'avenir', 'proxima', 'inter', 'roboto', 'open', 'source',
        'noto', 'ubuntu', 'lato', 'montserrat', 'poppins', 'nunito', 'work',
        'barlow', 'manrope', 'outfit', 'figtree', 'geist', 'albert'
    ],
    'monospace': [
        'mono', 'code', 'courier', 'consolas', 'terminal', 'typewriter',
        'fixed', 'inconsolata', 'fira code', 'jetbrains', 'source code',
        'roboto mono', 'ibm plex mono', 'cascadia', 'hack', 'menlo'
    ],
    'handwriting': [
        'script', 'hand', 'brush', 'cursive', 'signature', 'calligraphy',
        'handwritten', 'handwriting', 'dancing', 'pacifico', 'satisfy',
        'great vibes', 'allura', 'alex brush', 'sacramento', 'cookie',
        'kaushan', 'lobster', 'caveat', 'indie', 'patrick', 'marker'
    ],
    'display': [
        'display', 'decorative', 'fancy', 'artistic', 'poster', 'headline',
        'title', 'black', 'ultra', 'compressed', 'condensed', 'extended',
        'stencil', 'outline', 'shadow', 'retro', 'vintage', 'pixel',
        'graffiti', 'comic', 'cartoon', 'fun', 'kids', 'playfair'
    ]
}

# Priority order for classification (first match wins)
STYLE_PRIORITY = ['monospace', 'handwriting', 'display', 'serif', 'sans-serif']


@dataclass
class ProcessingStats:
    """Thread-safe processing statistics"""
    total: int = 0
    completed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, success: bool):
        with self._lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
    
    def get_progress(self) -> Tuple[int, int, float, str]:
        with self._lock:
            elapsed = time.time() - self.start_time
            if self.completed > 0:
                rate = self.completed / elapsed
                remaining = (self.total - self.completed) / rate if rate > 0 else 0
                eta = format_time(remaining)
            else:
                rate = 0
                eta = "calculating..."
            return self.completed, self.total, rate, eta


def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def classify_font_style(font_name: str) -> str:
    """
    Classify a font into a style category based on its name.
    Uses keyword matching with priority ordering.
    """
    name_lower = font_name.lower().replace('-', ' ').replace('_', ' ')
    
    # Check each style in priority order
    for style in STYLE_PRIORITY:
        keywords = STYLE_KEYWORDS[style]
        for keyword in keywords:
            if keyword in name_lower:
                return style
    
    # Default to sans-serif (most common in Google Fonts)
    return 'sans-serif'


def parse_svg_viewbox(svg_content: str) -> Tuple[float, float, float, float]:
    """Extract viewBox from SVG content"""
    try:
        # Parse viewBox attribute
        match = re.search(r'viewBox="([^"]+)"', svg_content)
        if match:
            parts = match.group(1).split()
            if len(parts) == 4:
                return tuple(float(p) for p in parts)
    except:
        pass
    return (0, 0, 1000, 1000)  # Default


def parse_svg_path(svg_content: str) -> Optional[str]:
    """Extract path data from SVG content"""
    try:
        match = re.search(r'd="([^"]+)"', svg_content)
        if match:
            return match.group(1)
    except:
        pass
    return None


def get_path_bounds(path_data: str) -> Tuple[float, float, float, float]:
    """
    Calculate approximate bounding box of SVG path.
    Returns (min_x, min_y, max_x, max_y)
    """
    # Extract all coordinate values from path
    # This handles M, L, C, Q, Z commands
    coords = []
    
    # Find all numbers (including negative and decimals)
    numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', path_data)
    
    if len(numbers) < 2:
        return (0, 0, 100, 100)
    
    # Parse as x,y pairs
    x_coords = []
    y_coords = []
    
    for i in range(0, len(numbers) - 1, 2):
        try:
            x_coords.append(float(numbers[i]))
            y_coords.append(float(numbers[i + 1]))
        except (ValueError, IndexError):
            continue
    
    if not x_coords or not y_coords:
        return (0, 0, 100, 100)
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def normalize_svg(svg_content: str, target_size: int = 128) -> str:
    """
    Normalize SVG to target_size x target_size canvas with centered glyph.
    Preserves aspect ratio and adds padding.
    """
    try:
        # Parse original viewBox
        orig_viewbox = parse_svg_viewbox(svg_content)
        orig_x, orig_y, orig_w, orig_h = orig_viewbox
        
        # Calculate scaling to fit in target canvas with padding
        padding_ratio = 0.1  # 10% padding on each side
        usable_size = target_size * (1 - 2 * padding_ratio)
        
        # Scale to fit
        scale = min(usable_size / orig_w, usable_size / orig_h) if orig_w > 0 and orig_h > 0 else 1
        
        # Calculate new dimensions
        new_w = orig_w * scale
        new_h = orig_h * scale
        
        # Calculate offset to center
        offset_x = (target_size - new_w) / 2
        offset_y = (target_size - new_h) / 2
        
        # Extract path data
        path_match = re.search(r'<path[^>]*d="([^"]+)"[^>]*/>', svg_content)
        if not path_match:
            path_match = re.search(r'<path[^>]*d="([^"]+)"[^>]*>', svg_content)
        
        if not path_match:
            return svg_content  # Return original if no path found
        
        path_data = path_match.group(1)
        
        # Create normalized SVG
        # Apply transform: translate to center, scale, and handle coordinate flip
        normalized_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {target_size} {target_size}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate({offset_x:.2f},{offset_y + new_h:.2f}) scale({scale:.6f},-{scale:.6f})">
    <path d="{path_data}" fill="#000"/>
  </g>
</svg>'''
        
        return normalized_svg
        
    except Exception as e:
        logger.debug(f"Normalization failed: {e}")
        return svg_content


def process_font_directory(args: Tuple[Path, Path, str, int]) -> Dict:
    """
    Process all SVGs in a font directory.
    Normalizes and copies to output with style classification.
    """
    input_dir, output_base, style, target_size = args
    
    result = {
        'font_name': input_dir.name,
        'style': style,
        'success': False,
        'glyphs_processed': 0,
        'errors': []
    }
    
    try:
        # Create output directory: output_base/style/font_name/
        output_dir = output_base / style / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each SVG file
        svg_files = list(input_dir.glob('uni*.svg'))
        
        for svg_file in svg_files:
            try:
                # Read original SVG
                with open(svg_file, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                # Normalize
                normalized = normalize_svg(svg_content, target_size)
                
                # Write normalized SVG
                output_path = output_dir / svg_file.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(normalized)
                
                result['glyphs_processed'] += 1
                
            except Exception as e:
                result['errors'].append(f"{svg_file.name}: {e}")
        
        # Copy metadata if exists
        metadata_file = input_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add style classification
            metadata['style'] = style
            metadata['normalized_size'] = target_size
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        result['success'] = True
        result['output_dir'] = str(output_dir)
        
    except Exception as e:
        result['errors'].append(str(e))
    
    return result


def print_progress_bar(current: int, total: int, rate: float, eta: str, width: int = 40):
    """Print a progress bar to console"""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r[{bar}] {current}/{total} ({percent*100:.1f}%) | {rate:.1f} fonts/sec | ETA: {eta}')
    sys.stdout.flush()


def get_optimal_workers(cpu_percent: int = 80) -> int:
    """Calculate optimal number of workers based on CPU percentage"""
    total_cpus = multiprocessing.cpu_count()
    workers = max(1, int(total_cpus * cpu_percent / 100))
    return workers


def create_dataset_splits(
    output_dir: Path, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Create train/val/test splits and save split information.
    Splits are stratified by style.
    """
    random.seed(seed)
    
    splits = {'train': [], 'val': [], 'test': []}
    style_counts = {}
    
    # Collect all fonts by style
    fonts_by_style = {}
    for style_dir in output_dir.iterdir():
        if style_dir.is_dir() and style_dir.name in STYLE_KEYWORDS:
            style = style_dir.name
            fonts_by_style[style] = list(style_dir.iterdir())
            style_counts[style] = len(fonts_by_style[style])
    
    # Create stratified splits
    for style, font_dirs in fonts_by_style.items():
        random.shuffle(font_dirs)
        
        n = len(font_dirs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        for i, font_dir in enumerate(font_dirs):
            font_info = {
                'name': font_dir.name,
                'style': style,
                'path': str(font_dir.relative_to(output_dir))
            }
            
            if i < train_end:
                splits['train'].append(font_info)
            elif i < val_end:
                splits['val'].append(font_info)
            else:
                splits['test'].append(font_info)
    
    # Save split files
    for split_name, fonts in splits.items():
        split_file = output_dir / f'{split_name}.json'
        with open(split_file, 'w') as f:
            json.dump({
                'split': split_name,
                'count': len(fonts),
                'fonts': fonts
            }, f, indent=2)
    
    return {
        'train_count': len(splits['train']),
        'val_count': len(splits['val']),
        'test_count': len(splits['test']),
        'style_distribution': style_counts
    }


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess and normalize SVG dataset for FONTe AI training'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('./DATASET'),
        help='Input directory with raw SVG glyphs (default: ./DATASET)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./DATASET_NORMALIZED'),
        help='Output directory for normalized data (default: ./DATASET_NORMALIZED)'
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=128,
        help='Target canvas size (default: 128)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto)'
    )
    parser.add_argument(
        '--cpu-percent', '-c',
        type=int,
        default=80,
        help='Percentage of CPU cores to use (default: 80%%)'
    )
    parser.add_argument(
        '--turbo', '-t',
        action='store_true',
        help='Enable turbo mode: 80%% CPU, minimal logging'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of fonts to process (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Turbo mode
    if args.turbo:
        args.cpu_percent = 80
        args.verbose = False
        logger.info("🚀 TURBO MODE ENABLED")
    
    # Calculate workers
    if args.workers is None:
        args.workers = get_optimal_workers(args.cpu_percent)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    
    total_cpus = multiprocessing.cpu_count()
    logger.info(f"CPU Configuration: {args.workers}/{total_cpus} cores ({args.cpu_percent}%)")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target canvas size: {args.size}x{args.size}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all font directories (those with SVG files)
    logger.info("Scanning for font directories...")
    font_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and list(item.glob('uni*.svg')):
            font_dirs.append(item)
    
    if not font_dirs:
        logger.error(f"No font directories found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(font_dirs)} font directories")
    
    # Apply limit if specified
    if args.limit:
        font_dirs = font_dirs[:args.limit]
        logger.info(f"Limited to {len(font_dirs)} fonts (--limit {args.limit})")
    
    # Classify all fonts by style
    logger.info("Classifying fonts by style...")
    style_counts = {style: 0 for style in STYLE_KEYWORDS}
    classified_fonts = []
    
    for font_dir in font_dirs:
        style = classify_font_style(font_dir.name)
        style_counts[style] += 1
        classified_fonts.append((font_dir, output_dir, style, args.size))
    
    # Print style distribution
    print("\n📊 Style Classification:")
    for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(font_dirs) * 100
            bar = '█' * int(pct / 2)
            print(f"  {style:15} {count:5} ({pct:5.1f}%) {bar}")
    print()
    
    # Initialize statistics
    stats = ProcessingStats()
    stats.total = len(font_dirs)
    stats.start_time = time.time()
    
    results = {
        'total': len(font_dirs),
        'successful': 0,
        'failed': 0,
        'total_glyphs': 0,
        'by_style': {style: 0 for style in STYLE_KEYWORDS}
    }
    
    logger.info(f"Processing {len(font_dirs)} fonts with {args.workers} parallel workers...")
    print()
    
    # Process with parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_font_directory, arg): arg[0]
            for arg in classified_fonts
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            font_dir = futures[future]
            try:
                result = future.result()
                
                if result['success']:
                    results['successful'] += 1
                    results['total_glyphs'] += result['glyphs_processed']
                    results['by_style'][result['style']] += 1
                else:
                    results['failed'] += 1
                    if args.verbose:
                        logger.warning(f"✗ {font_dir.name}: {result['errors']}")
                
                stats.update(result['success'])
                
            except Exception as e:
                results['failed'] += 1
                stats.update(False)
                if args.verbose:
                    logger.error(f"✗ {font_dir.name}: {e}")
            
            # Update progress
            completed, total, rate, eta = stats.get_progress()
            print_progress_bar(completed, total, rate, eta)
    
    print()
    
    # Create dataset splits
    logger.info("Creating train/val/test splits...")
    split_info = create_dataset_splits(output_dir)
    
    # Calculate timing
    elapsed = time.time() - stats.start_time
    
    # Save global metadata
    global_metadata = {
        'canvas_size': args.size,
        'total_fonts': results['successful'],
        'total_glyphs': results['total_glyphs'],
        'style_distribution': results['by_style'],
        'splits': split_info,
        'processing_time_seconds': elapsed,
        'source_directory': str(input_dir)
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(global_metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FONTe AI - Dataset Preprocessing Complete")
    print("="*60)
    print(f"Total fonts processed:  {results['successful']}")
    print(f"Failed:                 {results['failed']}")
    print(f"Total glyphs:           {results['total_glyphs']}")
    print(f"Canvas size:            {args.size}x{args.size}")
    print(f"Processing time:        {format_time(elapsed)}")
    print(f"Speed:                  {results['total']/elapsed:.1f} fonts/sec")
    print("-"*60)
    print("📊 Style Distribution:")
    for style, count in sorted(results['by_style'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {style:15} {count:5} fonts")
    print("-"*60)
    print(f"📂 Dataset Splits:")
    print(f"  Train: {split_info['train_count']} fonts")
    print(f"  Val:   {split_info['val_count']} fonts")
    print(f"  Test:  {split_info['test_count']} fonts")
    print("-"*60)
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
