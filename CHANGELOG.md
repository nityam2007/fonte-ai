# Changelog

All notable changes to the FONTe AI project are documented in this file.

> ⚠️ **APPEND-ONLY**: This file follows strict append-only rules. See [RULES.md](RULES.md) for details.

---

## [2026-01-22] - Initial Project Setup

### Added
- **README.md**: Project documentation with overview, structure, and getting started guide
- **requirements.txt**: Python dependencies (fonttools, pillow, cairosvg, svgwrite)
- **scripts/ttf_to_svg.py**: Font to SVG glyph extraction script
  - Supports TTF/OTF font formats
  - Extracts 72 characters (A-Z, a-z, 0-9, punctuation)
  - Generates Unicode-named SVG files (e.g., `uni0041.svg` for 'A')
  - Creates HTML preview for each font
  - Parallel processing with configurable workers
  - Metadata JSON files for each font

### Tested
- Successfully processed 5 test fonts from Google Fonts archive
- Extracted 360 glyphs total (72 per font)
- Verified SVG output format with proper viewBox and transforms
- HTML preview generation working correctly

### Infrastructure
- Project structure established:
  ```
  FONTe AI/
  ├── FONTS/          # Source fonts (Google Fonts)
  ├── DATASET/        # Output SVG glyphs
  ├── scripts/        # Utility scripts
  └── aidata/         # AI planning docs
  ```

---

## [2026-01-22] - Documentation Rules

### Added
- **CHANGELOG.md**: This append-only changelog file
- **RULES.md**: Project rules and conventions documentation

---

## [2026-01-22] - Performance Optimization (TURBO MODE)

### Changed
- **scripts/ttf_to_svg.py**: Complete rewrite for maximum performance
  - Now uses 80% CPU (6 cores) by default for parallel processing
  - Added `--turbo` flag for one-click maximum performance
  - Added `--cpu-percent` flag to control CPU usage (default: 80%)
  - Suppressed fontTools debug logging for cleaner output
  - Optimized font loading with `lazy=True` flag
  - Batch file writing for reduced I/O overhead
  - Minimal SVG template for faster generation
  - Real-time progress bar with ETA display
  - Thread-safe statistics tracking

### Performance Results
- **Speed**: 34.7 fonts/second (50 fonts in ~1 second)
- **Workers**: 6/8 cores (80% CPU utilization)
- **Estimated full dataset**: ~3824 fonts in ~2 minutes

### New Command Line Options
```
--turbo, -t       Enable turbo mode (80% CPU, minimal logging)
--cpu-percent, -c Control CPU usage percentage (default: 80)
--batch-size, -b  Batch size for progress updates
```

---

## [2026-01-22] - Full Dataset Extraction Complete

### Milestone Achieved 🎉
Successfully processed the entire Google Fonts repository!

### Final Metrics
| Metric | Value |
|--------|-------|
| Total Fonts | 3,824 |
| Success Rate | 100% |
| Glyphs Extracted | 270,252 |
| Processing Time | 2.1 minutes |
| Speed | 30.4 fonts/sec |
| Workers | 6/8 cores |

---

## [2026-01-22] - Documentation Overhaul

### Added
- **RESEARCH.md**: New append-only research journal file
  - Detailed methodology documentation
  - Extraction pipeline specifications
  - Performance optimization journey
  - Challenges and solutions
  - Dataset quality observations
  - Next steps planning

### Changed
- **README.md**: Complete rewrite in research-paper style
  - Added status badges (fonts processed, glyphs extracted)
  - Added documentation links table
  - Added current results summary
  - Added complete CLI options table
  - Added roadmap section
  - Cleaner, more professional format

- **RULES.md**: Added new file deletion policy (Rule #5)
  - NO `rm` command usage allowed
  - Always edit files, never delete
  - Bulk deletions must be requested from user
  - AI should be constructive, not destructive

---
## [2026-01-22] - Dataset Preprocessing Pipeline

### Added
- **scripts/preprocess_dataset.py**: New preprocessing script
  - SVG normalization to 128x128 canvas
  - Automatic style classification (5 categories)
  - Train/val/test split generation (80/10/10)
  - Parallel processing with 80% CPU
  - Progress bar with ETA

### Style Classification System
| Category | Description |
|----------|-------------|
| serif | Traditional fonts with serifs |
| sans-serif | Modern fonts without serifs |
| monospace | Fixed-width coding fonts |
| handwriting | Script and cursive fonts |
| display | Decorative headline fonts |

### New Output Structure
```
DATASET_NORMALIZED/
├── serif/
├── sans-serif/
├── monospace/
├── handwriting/
├── display/
├── train.json
├── val.json
└── test.json
```

### Performance (50 font test)
- Speed: 44.9 fonts/sec
- Canvas: 128x128
- Workers: 6/8 cores

### Design Decision: SVG-to-SVG Model
- Keeping vector format throughout pipeline
- No rasterization needed
- CPU-friendly for training
- Designer-first output (editable vectors)

### Updated RESEARCH.md
- Added sections 1.8 through 1.13
- Documented normalization algorithm
- Documented classification system
- Recorded preprocessing metrics
- Added SVG-to-SVG rationale

---

## [2026-01-22] - Phase 1.5 Complete (Full Dataset Preprocessing)

### Executed
- Full preprocessing run on complete dataset
- Command: `python scripts/preprocess_dataset.py --turbo`

### Results
```
============================================================
FONTe AI - Dataset Preprocessing Complete
============================================================
Total fonts processed:  3813
Failed:                 0
Total glyphs:           270252
Canvas size:            128x128
Processing time:        1.4m
Speed:                  44.0 fonts/sec
------------------------------------------------------------
📊 Style Distribution:
  sans-serif       2424 fonts (63.6%)
  serif             761 fonts (20.0%)
  display           315 fonts (8.3%)
  monospace         240 fonts (6.3%)
  handwriting        73 fonts (1.9%)
------------------------------------------------------------
📂 Dataset Splits:
  Train: 3049 fonts (80%)
  Val:   380 fonts (10%)
  Test:  384 fonts (10%)
============================================================
```

### Output Structure Created
```
DATASET_NORMALIZED/
├── sans-serif/     # 2424 fonts
├── serif/          # 761 fonts
├── display/        # 315 fonts
├── monospace/      # 240 fonts
├── handwriting/    # 73 fonts
├── train.json      # 3049 fonts
├── val.json        # 380 fonts
├── test.json       # 384 fonts
├── styles.json     # Style mapping
└── metadata.json   # Global metadata
```

### Milestone
✅ **Phase 1.5 COMPLETE** - Dataset ready for AI training

---

## [2026-01-22] - GitHub Repository & Data Policy

### Added
- **GitHub Repository**: [nityam2007/fonte-ai](https://github.com/nityam2007/fonte-ai) (Private)
- **LICENSE**: Proprietary license for code and models
- **.gitignore**: Exclude large generated files

### Data Policy Established
- `FONTS/` - NOT uploaded (clone from Google Fonts: ~2GB)
- `DATASET/` - NOT uploaded (regenerate with `ttf_to_svg.py`)
- `DATASET_NORMALIZED/` - NOT uploaded (regenerate with `preprocess_dataset.py`)

### Why Not Upload Datasets?
1. **Size**: ~3GB total would bloat repository
2. **Regenerable**: 2 commands, ~3 minutes to recreate
3. **Licensing**: Font files stay under original licenses (OFL/Apache)
4. **Reproducibility**: Scripts guarantee identical output

### Data Source
- Google Fonts repository: https://github.com/google/fonts
- ~3,800 fonts under OFL, Apache 2.0, UFL licenses
- We use fonts for training, model output is unique

### Regeneration Commands
```bash
# Clone fonts (one-time)
git clone --depth 1 https://github.com/google/fonts.git FONTS/fonts-main

# Extract SVGs (2.1 min)
python scripts/ttf_to_svg.py --turbo

# Preprocess (1.4 min)
python scripts/preprocess_dataset.py --turbo
```

---