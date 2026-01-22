#!/usr/bin/env python3
"""
TTF/OTF to SVG Glyph Extractor for FONTe AI - TURBO VERSION

This script extracts individual glyph shapes from font files and saves them as SVG files
for use in training the AI font generation model.

OPTIMIZATIONS:
- Uses 80% CPU (6 cores) by default for maximum throughput
- Batch processing with optimized I/O
- Memory-efficient processing
- Real-time progress bar with ETA
- Chunked parallel execution

Usage:
    python ttf_to_svg.py [--fonts-dir ./FONTS] [--output-dir ./DATASET] [--preview]
    python ttf_to_svg.py --turbo  # Use 80% CPU automatically

Author: FONTe AI Project
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from dataclasses import dataclass, field
import threading

try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
except ImportError:
    print("ERROR: fonttools is required. Install with: pip install fonttools")
    sys.exit(1)

# Suppress fontTools debug logging for speed
logging.getLogger('fontTools').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Character set for training
CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#()[]"

# SVG template - minimal for speed
SVG_TEMPLATE = '<?xml version="1.0" encoding="UTF-8"?>\n<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><g transform="translate(0,{baseline}) scale(1,-1)"><path d="{path_data}" fill="#000"/></g></svg>'

# Preview HTML template
HTML_PREVIEW_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>Font Preview: {font_name}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 10px; }}
        .glyph {{ background: white; padding: 10px; text-align: center; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .glyph img {{ width: 60px; height: 60px; }}
        .glyph .char {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .stats {{ background: #333; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Font Preview: {font_name}</h1>
    <div class="stats">
        <strong>Total Glyphs:</strong> {glyph_count} | 
        <strong>Font File:</strong> {font_file}
    </div>
    <div class="grid">
        {glyph_items}
    </div>
</body>
</html>'''


@dataclass
class ProcessingStats:
    """Thread-safe processing statistics"""
    total: int = 0
    completed: int = 0
    successful: int = 0
    failed: int = 0
    total_glyphs: int = 0
    start_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, success: bool, glyphs: int):
        with self._lock:
            self.completed += 1
            if success:
                self.successful += 1
                self.total_glyphs += glyphs
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


def get_unicode_name(char: str) -> str:
    """Convert character to Unicode naming convention (e.g., 'A' -> 'uni0041')"""
    return f"uni{ord(char):04X}"


def extract_glyph_fast(font: TTFont, glyph_set, cmap, char: str) -> Optional[Tuple[str, int]]:
    """
    Fast glyph extraction - minimized overhead version.
    
    Returns:
        Tuple of (path_data, width) or None if glyph not found
    """
    try:
        glyph_name = cmap.get(ord(char))
        if glyph_name is None or glyph_name not in glyph_set:
            return None
        
        pen = SVGPathPen(glyph_set)
        glyph = glyph_set[glyph_name]
        glyph.draw(pen)
        
        path_data = pen.getCommands()
        if not path_data or path_data.strip() == "":
            return None
        
        width = glyph.width if hasattr(glyph, 'width') else 1000
        return (path_data, width)
        
    except Exception:
        return None


def process_font_file_fast(args: Tuple[Path, Path, bool]) -> Dict:
    """
    Optimized font processing - single font file.
    Takes tuple for easier multiprocessing.
    """
    font_path, output_dir, generate_preview = args
    
    result = {
        'font_path': str(font_path),
        'font_name': font_path.stem,
        'success': False,
        'glyphs_extracted': 0,
        'glyphs_failed': 0,
        'errors': []
    }
    
    try:
        # Load font with minimal tables
        font = TTFont(font_path, lazy=True)
        
        # Get required data once
        cmap = font.getBestCmap()
        if cmap is None:
            result['errors'].append("No character map found")
            return result
        
        glyph_set = font.getGlyphSet()
        
        # Get font metrics once
        try:
            units_per_em = font['head'].unitsPerEm
            ascender = font['hhea'].ascent if 'hhea' in font else int(units_per_em * 0.8)
            descender = font['hhea'].descent if 'hhea' in font else int(-units_per_em * 0.2)
        except:
            units_per_em = 1000
            ascender = 800
            descender = -200
        
        height = ascender - descender
        baseline = ascender
        
        # Create output directory
        font_id = f"{font_path.parent.name}_{font_path.stem}".replace(" ", "_").replace("[", "_").replace("]", "_")
        font_output_dir = output_dir / font_id
        font_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract all glyphs - batch write for speed
        extracted_glyphs = []
        svg_files_to_write = []
        
        for char in CHAR_SET:
            glyph_result = extract_glyph_fast(font, glyph_set, cmap, char)
            
            if glyph_result:
                path_data, width = glyph_result
                unicode_name = get_unicode_name(char)
                
                # Create SVG content
                svg_content = SVG_TEMPLATE.format(
                    width=max(width, 100),
                    height=height,
                    baseline=baseline,
                    path_data=path_data
                )
                
                svg_files_to_write.append((unicode_name, svg_content))
                extracted_glyphs.append({
                    'char': char,
                    'unicode_name': unicode_name
                })
                result['glyphs_extracted'] += 1
            else:
                result['glyphs_failed'] += 1
        
        # Batch write all SVG files
        for unicode_name, svg_content in svg_files_to_write:
            svg_path = font_output_dir / f"{unicode_name}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
        
        # Save minimal metadata
        metadata = {
            'font_name': result['font_name'],
            'glyphs_count': len(extracted_glyphs),
            'font_metrics': {'height': height, 'baseline': baseline}
        }
        
        with open(font_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Generate HTML preview if requested
        if generate_preview and extracted_glyphs:
            glyph_items = "".join([
                f'<div class="glyph"><img src="{g["unicode_name"]}.svg" alt="{g["char"]}"><div class="char">{g["char"]}</div></div>'
                for g in extracted_glyphs
            ])
            
            preview_html = HTML_PREVIEW_TEMPLATE.format(
                font_name=result['font_name'],
                glyph_count=len(extracted_glyphs),
                font_file=font_path.name,
                glyph_items=glyph_items
            )
            
            with open(font_output_dir / 'preview.html', 'w') as f:
                f.write(preview_html)
        
        font.close()
        result['success'] = True
        result['output_dir'] = str(font_output_dir)
        
    except Exception as e:
        result['errors'].append(str(e))
    
    return result


def find_font_files(fonts_dir: Path) -> List[Path]:
    """Find all TTF and OTF font files in the fonts directory"""
    font_files = []
    
    # Search in common Google Fonts subdirectories
    search_dirs = ['ofl', 'apache', 'ufl']
    
    for subdir in search_dirs:
        subdir_path = fonts_dir / 'fonts-main' / subdir
        if subdir_path.exists():
            for ext in ['*.ttf', '*.otf', '*.TTF', '*.OTF']:
                font_files.extend(subdir_path.rglob(ext))
    
    # Also search directly in fonts_dir if no fonts-main structure
    if not font_files:
        for ext in ['*.ttf', '*.otf', '*.TTF', '*.OTF']:
            font_files.extend(fonts_dir.rglob(ext))
    
    return sorted(set(font_files))


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


def main():
    parser = argparse.ArgumentParser(
        description='Extract SVG glyphs from TTF/OTF fonts for AI training (TURBO VERSION)'
    )
    parser.add_argument(
        '--fonts-dir', '-f',
        type=Path,
        default=Path('./FONTS'),
        help='Directory containing font files (default: ./FONTS)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./DATASET'),
        help='Output directory for SVG files (default: ./DATASET)'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Generate HTML preview for each font'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto based on --cpu-percent)'
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
        help='Enable turbo mode: 80%% CPU, no preview, minimal logging'
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
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Batch size for progress updates (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Turbo mode settings
    if args.turbo:
        args.cpu_percent = 80
        args.verbose = False
        logger.info("🚀 TURBO MODE ENABLED - Using 80%% CPU")
    
    # Calculate workers
    if args.workers is None:
        args.workers = get_optimal_workers(args.cpu_percent)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths
    fonts_dir = args.fonts_dir.resolve()
    output_dir = args.output_dir.resolve()
    
    total_cpus = multiprocessing.cpu_count()
    logger.info(f"CPU Configuration: {args.workers}/{total_cpus} cores ({args.cpu_percent}%)")
    logger.info(f"Fonts directory: {fonts_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all font files
    logger.info("Scanning for font files...")
    font_files = find_font_files(fonts_dir)
    
    if not font_files:
        logger.error(f"No font files found in {fonts_dir}")
        logger.info("Expected structure: FONTS/fonts-main/ofl/*/font.ttf")
        sys.exit(1)
    
    logger.info(f"Found {len(font_files)} font files")
    
    # Apply limit if specified
    if args.limit:
        font_files = font_files[:args.limit]
        logger.info(f"Limited to {len(font_files)} fonts (--limit {args.limit})")
    
    # Prepare processing arguments
    process_args = [(font_path, output_dir, args.preview) for font_path in font_files]
    
    # Initialize statistics
    stats = ProcessingStats()
    stats.total = len(font_files)
    stats.start_time = time.time()
    
    results = {
        'total': len(font_files),
        'successful': 0,
        'failed': 0,
        'total_glyphs': 0,
        'fonts': []
    }
    
    logger.info(f"Processing {len(font_files)} fonts with {args.workers} parallel workers...")
    print()  # New line for progress bar
    
    # Process with optimized parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_font_file_fast, arg): arg[0] 
            for arg in process_args
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            font_path = futures[future]
            try:
                result = future.result()
                
                if result['success']:
                    results['successful'] += 1
                    results['total_glyphs'] += result['glyphs_extracted']
                else:
                    results['failed'] += 1
                    if args.verbose:
                        logger.warning(f"✗ {font_path.name}: {result['errors']}")
                
                results['fonts'].append(result)
                stats.update(result['success'], result.get('glyphs_extracted', 0))
                
            except Exception as e:
                results['failed'] += 1
                stats.update(False, 0)
                if args.verbose:
                    logger.error(f"✗ {font_path.name}: {e}")
            
            # Update progress bar
            completed, total, rate, eta = stats.get_progress()
            print_progress_bar(completed, total, rate, eta)
    
    # Final newline after progress bar
    print()
    
    # Calculate final timing
    elapsed = time.time() - stats.start_time
    
    # Save global metadata
    global_metadata = {
        'character_set': CHAR_SET,
        'unicode_mapping': {char: get_unicode_name(char) for char in CHAR_SET},
        'total_fonts': results['successful'],
        'total_glyphs': results['total_glyphs'],
        'processing_time_seconds': elapsed,
        'workers_used': args.workers,
        'fonts': [r['font_name'] for r in results['fonts'] if r['success']]
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FONTe AI - Font to SVG Conversion Complete (TURBO)")
    print("="*60)
    print(f"Total fonts found:      {results['total']}")
    print(f"Successfully processed: {results['successful']}")
    print(f"Failed:                 {results['failed']}")
    print(f"Total glyphs extracted: {results['total_glyphs']}")
    print(f"Processing time:        {format_time(elapsed)}")
    print(f"Average speed:          {results['total']/elapsed:.1f} fonts/sec")
    print(f"Workers used:           {args.workers}/{total_cpus} cores")
    print(f"Output directory:       {output_dir}")
    print("="*60)
    
    if args.preview:
        print("\nHTML previews generated in each font directory.")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
