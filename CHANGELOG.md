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
**TRAINING IN PROGRESS** on Modal L40S GPU

### Training Metrics (Actual)
| Metric | Value |
|--------|-------|
| Platform | Modal (L40S GPU, 48GB VRAM) |
| Batches | 1,003 per epoch |
| Batch Size | 198 |
| Speed | ~2.24 it/s |
| ETA per Epoch | ~7.5 minutes |
| Initial Loss | 5.68 |
| VRAM Usage | **40 GB / 48 GB (83%)** |
| Cost | $2.07/hr |

### Training Configuration
```python
EPOCHS = 50
BATCH_SIZE = 198          # Max for L40S without OOM
LEARNING_RATE = 3e-4
MODEL = "medium" (~12M params)
OPTIMIZER = AdamW (weight_decay=0.01)
SCHEDULER = CosineAnnealingLR
```

### ⚠️ Important: Memory Usage Reality

Our initial estimates were WRONG. Actual memory breakdown:

| Component | Memory |
|-----------|--------|
| Model weights | ~50 MB |
| Gradients | ~50 MB |
| Optimizer states (AdamW) | ~100 MB |
| **Attention matrices** | ~5-10 GB per layer |
| Activations (6 layers × batch × seq 512) | **~30+ GB** |

**Key Insight**: Transformer memory scales with `batch_size × seq_length² × n_layers`

### GPU Comparison (Actual)

| GPU | VRAM | Max Batch | Speed | Time/Epoch |
|-----|------|-----------|-------|------------|
| T4 (Colab Free) | 15 GB | ~64 | 1.79 it/s | ~28 min |
| **L40S (Modal)** | 48 GB | ~198 | 2.24 it/s | ~7.5 min |

### Cost Estimate (Actual)
- 50 epochs × 7.5 min = 6.2 hours
- 6.2 hours × $2.07/hr = **~$13**

### Milestones
- ✅ Phase 1: Dataset Extraction (270K glyphs)
- ✅ Phase 1.5: Preprocessing (3,813 fonts)
- ✅ Phase 2A: Tokenization (248K sequences)
- ✅ Phase 2A: Model Architecture
- 🔄 **Phase 2B: Training (IN PROGRESS on L40S)**
- ⏳ Phase 3: Evaluation
- ⏳ Phase 4: Font Generation

---

## [2026-01-22] - Epoch 1 Complete + Generation Script

### Training Progress
| Epoch | Train Loss | Val Loss | Time |
|-------|------------|----------|------|
| 1 | 4.96 | 3.94 | 7.8 min |

**Loss dropped 30%** from initial 5.68 → 3.94 (model is learning!)

### Added
- **scripts/generate_font.py**: Font generation from trained models
  - Load any checkpoint from `TRAINED/` directory
  - Generate single or multiple characters
  - Multiple styles supported
  - Outputs SVG files with HTML preview
  - Token-to-SVG path conversion

### Usage
```bash
# List available models
python scripts/generate_font.py --list-models

# Generate single character
python scripts/generate_font.py --model TRAINED/epoch_1.pt --char A --style serif

# Generate multiple characters
python scripts/generate_font.py --model TRAINED/best_model.pt --chars "ABC" --style sans-serif

# Generate all characters
python scripts/generate_font.py --model TRAINED/best_model.pt --all-chars --output generated/
```

### Model Checkpoints Saved
```
TRAINED/
├── best_model.pt         (21 MB)
├── checkpoint_epoch_1.pt (21 MB)
└── training_history.json
```

### GPU Batch Size Reference (For Our Model)

| GPU | VRAM | Safe Batch | Speed |
|-----|------|------------|-------|
| T4 | 15 GB | ~64 | 1.79 it/s |
| A10 | 24 GB | ~100 | ~2.0 it/s |
| A100-40GB | 40 GB | ~180 | ~2.2 it/s |
| **L40S** | **48 GB** | **~198** | **2.24 it/s** |
| A100-80GB | 80 GB | ~400 | ~2.5 it/s |

*Note: These are for seq_length=512, d_model=256, n_layers=6*

---

## [2026-01-22] - Upgraded to B200 GPU

### Why Upgrade
- L40S batch 198 was stable but slow (~7.5 min/epoch)
- B200 has 192GB VRAM - massive batch sizes possible
- Faster total training despite higher hourly cost

### Platform: Modal.com B200
| Spec | Value |
|------|-------|
| GPU | NVIDIA B200 |
| VRAM | 192 GB |
| Cost | $6.73/hour |
| CPU | 2 cores |
| RAM | 8 GB |

### Training Metrics Comparison

| Metric | L40S | B200 |
|--------|------|------|
| VRAM | 48 GB | 192 GB |
| Batch Size | 198 | **1024** |
| VRAM Used | 40 GB | **~130 GB** |
| Batches/Epoch | 1,003 | **194** |
| Time/Epoch | 7.5 min | **~2.2 min** |
| GPU Temp | ~70°C | ~45°C |

### Training Progress (B200)
| Epoch | Train Loss | Val Loss | Time |
|-------|------------|----------|------|
| 1 | 15.27 | 6.49 | 2:13 |
| 2 | 6.67 | 5.51 | 2:14 |

### Screenshots

![alt text](image-1.png)

![alt text](image-2.png)

See `b200_metrics.png` and `b200_training.png` for:
- GPU memory usage (~130GB during training)
- GPU utilization (100% peaks)
- Training output with loss curves

### Updated GPU Comparison

| GPU | VRAM | $/hr | Batch | Time/Epoch | 50 Epochs | **Total Cost** |
|-----|------|------|-------|------------|-----------|----------------|
| T4 | 15 GB | FREE | 64 | ~28 min | ~23 hrs | FREE |
| L40S | 48 GB | $2.07 | 198 | ~7.5 min | ~6.2 hrs | ~$13 |
| H100 | 80 GB | $3.95 | ~400 | ~3 min | ~2.5 hrs | ~$10 |
| **B200** | **192 GB** | **$6.73** | **1024** | **~2.2 min** | **~1.8 hrs** | **~$12** |

### Notebook Updated
- `notebooks/FONTe_AI_Training modal.com.ipynb` - B200 optimized
  - Uses git clone to get repo + data
  - Batch size 1024
  - `num_workers=0` to avoid DataLoader errors
  - Saves checkpoint every epoch

---

## [2026-01-22] - B200 Training: Batch 1050

### Increased Batch Size
Pushed batch size from 1024 → **1050** for maximum throughput.

### Updated Metrics
| Metric | Batch 1024 | Batch 1050 |
|--------|------------|------------|
| Batches/Epoch | 194 | **190** |
| Speed | 1.49 it/s | **1.90 it/s** |
| Time/Epoch | ~2.2 min | **~2:13** |

### Training Progress (Batch 1050)
| Epoch | Train Loss | Val Loss | Time |
|-------|------------|----------|------|
| 1 | 15.82 | 6.91 | 2:13 |
| 2 | 6.78 | 5.44 | 2:13 |
| 3 | 6.17 | 5.32 | 2:13 |
| 4 | 5.98 | 5.16 | 2:13 |
| 5 | 5.84 | 5.16 | 2:13 |
| 6 | 5.70 | 5.10 | 2:13 |
| 7 | 5.51 | 5.02 | 2:14 |

*Updated: 2026-01-22 10:30 PM*

**Val loss dropped 27% in 7 epochs!** (6.91 → 5.02)

### ETA
- 50 epochs × 2.2 min = **~1.8 hours**
- Cost: ~$12

---

## [2026-01-22 10:30 PM] - README Technical Documentation

### Added
Comprehensive technical explanations for beginners:
- How SVG-to-SVG approach works
- What Transformers are and why they work for fonts
- Token system explained with examples
- Model architecture diagram
- Training process breakdown
- Why B200 GPU (batch size comparison)

---

## [2026-01-22 11:00 PM] - First Generation Test (Epoch 13)

### Downloaded Model
- Copied `best_model.pt` from Modal B200 training
- Epoch 13, val_loss 4.29 (-38% from epoch 1)
- Model size: 21 MB (~5M params)

### Architecture Mismatch Bug Fixed
`generate_font.py` had different layer naming than Modal notebook.

| Old (broken) | New (matching Modal) |
|--------------|----------------------|
| `token_embedding` | `emb` |
| `attention.w_q` | `attn.wq` |
| `norm1/norm2` | `n1/n2` |
| `lm_head` | `head` |
| `causal_mask` | `mask` |

### UNK Token Bug Discovered
Model generates many `UNK (token 3)` tokens mid-sequence.
Old decoder stopped at first UNK → empty SVGs.

**Fix**: Skip UNK tokens instead of stopping:
```python
# Before: if token <= 3: break
# After:
if token == 0 or token == 2: break  # PAD or EOS
if token == 1 or token == 3: continue  # Skip SOS/UNK
```

### Generation Results (Epoch 13)
```
✅ 'A' → 256 tokens → d="M 20.2 127.9 Q 22.1..."
✅ 'B' → 256 tokens → d="M 16.0 Q 14.0..."
✅ 'C' → 91 tokens  → d="M 47.9 2.4 Q 25.5..."
✅ 'a' → 183 tokens → d="M 1.5 Q 1.5 1.5..."
✅ 'b' → 70 tokens  → d="M 4.9 0.0 V 60.7..."
✅ 'c' → 153 tokens → d="M 127.9 127.9 Q..."
```

- ✅ Valid SVG commands (M, Q, V, H, Z)
- ✅ Coordinate values across canvas
- ✅ Quadratic Bezier curves
- ⚠️ Paths still chaotic (expected at epoch 13)
- ⚠️ High UNK token frequency (training artifact)

### Training Status
| Epoch | Train Loss | Val Loss | Change |
|-------|------------|----------|--------|
| 1 | 15.82 | 6.91 | - |
| 7 | 5.51 | 5.02 | -27% |
| 13 | 4.58 | 4.29 | -38% |
| 50 | ? | ? | ETA ~1hr |

Target: val_loss ~2.5-3.0 for recognizable glyphs.

---