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

## [2026-01-22] - Phase 2A: Tokenization & Model Architecture

### Added
- **scripts/svg_tokenizer.py**: SVG path tokenization system
  - 1,105 token vocabulary
  - Path commands: M, L, C, Q, H, V, Z (absolute & relative)
  - Coordinates: Quantized 0-999
  - Style tokens: 5 categories
  - Character tokens: 72 glyphs

- **scripts/create_dataset.py**: Dataset pipeline for training
  - Creates tokenized sequences from normalized SVGs
  - Binary format for efficient loading
  - Filters sequences by length (10-512 tokens)

- **model/fonte_model.py**: SVG Path Transformer
  - Transformer decoder architecture
  - 3 sizes: small (~1M), medium (~12M), large (~50M params)
  - Autoregressive generation
  - Top-k and top-p sampling

- **model/train.py**: Training script
  - Supports CPU and CUDA
  - Cosine annealing LR scheduler
  - Gradient clipping
  - Checkpoint saving

- **notebooks/FONTe_AI_Training.ipynb**: Google Colab notebook
  - Ready for free T4 GPU training
  - Upload data, train, download model
  - ~6-10 hours for 50 epochs

### Tokenization Results
```
Total sequences:  248,227
Max seq length:   512
Vocab size:       1,105
Processing time:  49.9s
------------------------------------------------------------
Splits:
  Train: 198,581 (80.0%)
  Val:   24,822 (10.0%)
  Test:  24,824 (10.0%)
------------------------------------------------------------
Style distribution:
  sans-serif      155,744 (62.7%)
  serif           50,621 (20.4%)
  display         21,033 (8.5%)
  monospace       16,520 (6.7%)
  handwriting      4,309 (1.7%)
```

### Model Architecture
- **Input**: [SOS] [STYLE] [CHAR] → sequence of path tokens
- **Output**: Next token prediction
- **Generation**: Autoregressive with temperature/top-k sampling

### Files Created
```
model/
├── fonte_model.py    # Model architecture
└── train.py          # Training script
scripts/
├── svg_tokenizer.py  # Path tokenization
└── create_dataset.py # Dataset pipeline
notebooks/
└── FONTe_AI_Training.ipynb  # Colab notebook
TOKENIZED/
├── vocabulary.json   # Token mappings
├── train.bin         # Training data (198K sequences)
├── val.bin           # Validation data (24K sequences)
├── test.bin          # Test data (24K sequences)
└── config.json       # Dataset config
```

### Milestone
✅ **Phase 2A COMPLETE** - Ready to train!

---

## [2026-01-22] - Git LFS + Colab Workflow

### Added
- **.gitattributes**: Git LFS tracking configuration
  - `*.bin` files tracked via LFS (training data)
  - `TOKENIZED/*.json` tracked via LFS

### Changed
- **notebooks/FONTe_AI_Training.ipynb**: Updated for seamless Colab workflow
  - Now clones repo with `git lfs pull` instead of file upload
  - Simplified cell structure (7 sections)
  - Auto-saves checkpoints to Google Drive
- **.gitignore**: Removed TOKENIZED exclusions (now tracked via LFS)

### Uploaded to GitHub (via LFS)
```
TOKENIZED/
├── train.bin      (379 MB)
├── val.bin        (47 MB)
├── test.bin       (47 MB)
├── vocabulary.json
├── config.json
├── analysis.json
├── train.json
├── val.json
└── test.json
Total: 442 MB via Git LFS
```

### Colab Workflow (Now)
```bash
# In Colab - just run cells in order:
!apt-get install git-lfs -qq
!git lfs install
!git clone https://github.com/nityam2007/fonte-ai.git
%cd fonte-ai
!git lfs pull
# → Training data ready!
```

---

## [2026-01-22] - Phase 2B: Training Started! 🚀

### Status
**TRAINING IN PROGRESS** on Google Colab T4 GPU

### Training Metrics (Live)
| Metric | Value |
|--------|-------|
| Platform | Google Colab (T4 GPU) |
| Batches | 3,103 per epoch |
| Batch Size | 64 |
| Speed | ~1.79 it/s |
| ETA per Epoch | ~28 minutes |
| Initial Loss | 5.58 |

### Training Configuration
```python
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MODEL = "medium" (~12M params)
OPTIMIZER = AdamW (weight_decay=0.01)
SCHEDULER = CosineAnnealingLR
```

### Milestones
- ✅ Phase 1: Dataset Extraction (270K glyphs)
- ✅ Phase 1.5: Preprocessing (3,813 fonts)
- ✅ Phase 2A: Tokenization (248K sequences)
- ✅ Phase 2A: Model Architecture
- 🔄 **Phase 2B: Training (IN PROGRESS)**
- ⏳ Phase 3: Evaluation
- ⏳ Phase 4: Font Generation

---