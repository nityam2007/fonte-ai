# FONTe AI - AI Font Generator

> ✅ **TRAINED** - First successful training complete! val_loss: 5.25

> 📜 **Open Source** - Dual Licensed under GPLv3 and Apache 2.0
>
> You may choose either license. See [LICENSE](LICENSE) for details.

[github.com/nityam2007/fonte-ai](https://github.com/nityam2007/fonte-ai)

An AI-powered font generation system that learns from existing fonts and generates new, unique typefaces.

[![Status](https://img.shields.io/badge/Status-Trained-brightgreen)](https://github.com/nityam2007/fonte-ai)
[![val_loss](https://img.shields.io/badge/val__loss-5.25-blue)](https://github.com/nityam2007/fonte-ai)
[![Vocab](https://img.shields.io/badge/Vocabulary-1106%20tokens-blue)](https://github.com/nityam2007/fonte-ai)
[![Fonts](https://img.shields.io/badge/Fonts-3813-purple)](https://github.com/nityam2007/fonte-ai)
[![Sequences](https://img.shields.io/badge/Sequences-248K-orange)](https://github.com/nityam2007/fonte-ai)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen)](https://github.com/nityam2007/fonte-ai)
[![License](https://img.shields.io/badge/License-GPLv3%20%2B%20Apache%202.0-blue)](LICENSE)

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Quick start guide |
| [RESEARCH.md](RESEARCH.md) | Research journal with detailed findings (append-only) |
| [CHANGELOG.md](CHANGELOG.md) | Project change history (append-only) |
| [RULES.md](RULES.md) | Project rules and conventions |

---

## 🎯 Project Goal

Build a generative AI model that can:
1. Learn typographic patterns from thousands of existing fonts
2. Generate complete, usable font files (TTF/OTF)
3. Support style conditioning (serif, sans-serif, monospace, etc.)

---

## 🧠 How It Works (For Beginners)

### The Big Picture

Imagine teaching a computer to draw letters. That's essentially what FONTe AI does:

```
Step 1: Show the AI thousands of fonts (3,800+ fonts!)
Step 2: AI learns patterns (serifs look like this, sans-serif like that)
Step 3: AI generates NEW letters it has never seen before
Step 4: Convert to usable font files (.ttf)
```

### Why SVG Instead of Images?

| Approach | How it Works | Problem |
|----------|--------------|---------|
| **Images (Pixels)** | Treat letters as 128x128 pixel grids | Blurry when scaled, loses quality |
| **SVG (Vectors)** ✅ | Store letters as mathematical curves | Perfect at ANY size, professional quality |

**We chose SVG because:**
- Fonts are vector graphics (curves, not pixels)
- SVGs scale infinitely without losing quality
- Professional designers need editable vectors
- Smaller file sizes than images

### What is a Transformer?

You've heard of ChatGPT? It uses a "Transformer" architecture. We use the same technology, but for fonts!

```
ChatGPT: "The cat sat on the ___" → predicts "mat"
FONTe:   "M 10 20 L 50 ___" → predicts "80" (next coordinate)
```

**Why Transformers work for fonts:**
1. **Sequence understanding** - Letters are sequences of drawing commands
2. **Long-range patterns** - The top of 'A' relates to the bottom
3. **Style consistency** - Learns that serif fonts have little feet

### The Token System (How AI Reads Fonts)

Computers can't read drawings directly. We convert SVG paths into "tokens" (numbers):

```
Original SVG:  "M 10 20 L 50 80 Z"  (Move to 10,20, Line to 50,80, Close)
     ↓
Tokenized:     [1, 25, 30, 4, 116, 126, 6, 156, 186, 22, 2]
               [SOS, style, char, M, 10, 20, L, 50, 80, Z, EOS]
```

**Token categories:**
| Token Range | What It Represents | Example |
|-------------|-------------------|---------|
| 0-3 | Special markers | PAD, SOS, EOS, UNK |
| 4-24 | Drawing commands | M, L, C, Z, `<NEG>` (negative sign) |
| 25-29 | Font styles | serif, sans-serif, monospace |
| 30-105 | Characters | A-Z, a-z, 0-9, punctuation |
| 106-1105 | Coordinates | 0-999 (quantized positions) |

### Model Architecture (The Brain)

```
┌─────────────────────────────────────────────┐
│            FONTe Transformer                │
├─────────────────────────────────────────────┤
│  Input: [SOS, style, char, M, 10, 20, ...]  │
│              ↓                              │
│  ┌─────────────────────────────────┐        │
│  │  Token Embedding (256 dims)     │        │
│  │  + Positional Encoding          │        │
│  └─────────────────────────────────┘        │
│              ↓                              │
│  ┌─────────────────────────────────┐        │
│  │  Transformer Block ×6           │        │
│  │  ├─ Multi-Head Attention (4)    │        │
│  │  ├─ Feed Forward (1024)         │        │
│  │  └─ Layer Normalization         │        │
│  └─────────────────────────────────┘        │
│              ↓                              │
│  Output: Predict next token                 │
└─────────────────────────────────────────────┘
```

**Key numbers:**
| Component | Value | Why? |
|-----------|-------|------|
| d_model | 256 | Embedding dimension (how "detailed" each token is) |
| n_heads | 4 | Attention heads (different ways to look at relationships) |
| n_layers | 6 | Depth (more layers = more complex patterns) |
| d_ff | 1024 | Feed-forward width (processing power) |
| vocab_size | **1,106** | Total unique tokens |
| max_seq_len | 512 | Maximum path length |
| **Total params** | **~5M** | Relatively small, trains fast |

### Training Process

```
For each font glyph:
  1. Convert SVG path to tokens
  2. Feed tokens to model
  3. Model predicts next token
  4. Compare prediction to actual
  5. Adjust model weights (backpropagation)
  6. Repeat 248,000 × 50 epochs = 12.4 million times!
```

**Loss function:** Cross-entropy (measures how wrong predictions are)
- Random guess: ~7.0 (ln(1105))
- After training: ~1.5 (much better!)

### Generation (Making New Fonts)

Once trained, the model generates fonts autoregressively:

```python
# Start with style and character
tokens = [SOS, SANS_SERIF, 'A']

# Generate one token at a time
while not done:
    next_token = model.predict(tokens)  # What comes next?
    tokens.append(next_token)
    if next_token == EOS:
        break

# Convert back to SVG
svg_path = tokens_to_svg(tokens)
```

### Why B200 GPU?

| GPU | VRAM | Batch Size | Speed | Cost/50 epochs |
|-----|------|------------|-------|----------------|
| Your laptop | 4-8 GB | 8-16 | Days | FREE (slow) |
| Colab T4 | 15 GB | 64 | 23 hrs | FREE |
| **B200** | **192 GB** | **1050** | **1.8 hrs** | **~$12** |

**Why batch size matters:**
- Bigger batch = more examples processed in parallel
- Bigger batch = more GPU memory needed
- B200's 192GB VRAM allows batch 1050 (vs 64 on T4)
- Result: **12x faster training!**

---

## 📊 Data Source & Storage

### Why Datasets Are NOT in This Repository

| Directory | Size | Why Not Uploaded |
|-----------|------|------------------|
| `FONTS/` | ~2 GB | Google Fonts repo (clone separately) |
| `DATASET/` | ~500 MB | Generated from FONTS/ (regenerable) |
| `DATASET_NORMALIZED/` | ~600 MB | Generated from DATASET/ (regenerable) |

**Reasons:**
1. **Size** — Total ~3 GB would bloat repo and slow clones
2. **Regenerable** — All data can be recreated with 2 commands (~3 min)
3. **Licensing** — Font files remain under original licenses (OFL/Apache)
4. **Reproducibility** — Scripts ensure identical output every time

### Data Source: Google Fonts

We use the official [Google Fonts repository](https://github.com/google/fonts):

```bash
# Clone Google Fonts (one-time, ~2GB)
git clone --depth 1 https://github.com/google/fonts.git FONTS/fonts-main
```

| License | Font Count | Examples |
|---------|------------|----------|
| OFL (Open Font License) | ~3,500 | Roboto, Open Sans, Lato |
| Apache 2.0 | ~200 | Roboto Slab, Cousine |
| UFL (Ubuntu Font License) | ~50 | Ubuntu, Ubuntu Mono |

### Regenerate Dataset (3 minutes)

```bash
# Step 1: Extract SVGs from TTF fonts (2.1 min)
python scripts/ttf_to_svg.py --turbo

# Step 2: Normalize and classify (1.4 min)
python scripts/preprocess_dataset.py --turbo
```

---

## �📊 Current Results

### Phase 1: Dataset Extraction ✅ COMPLETE

| Metric | Value |
|--------|-------|
| Total Fonts Processed | 3,824 |
| Success Rate | 100% |
| Total Glyphs Extracted | 270,252 |
| Processing Time | 2.1 minutes |
| Speed | 30.4 fonts/sec |

### Phase 1.5: Preprocessing ✅ COMPLETE

| Metric | Value |
|--------|-------|
| Fonts Processed | 3,813 |
| Total Glyphs | 270,252 |
| Canvas Size | 128×128 |
| Processing Time | 1.4 minutes |
| Speed | 44.0 fonts/sec |

**Style Distribution:**
| Style | Count | % |
|-------|-------|---|
| sans-serif | 2,424 | 63.6% |
| serif | 761 | 20.0% |
| display | 315 | 8.3% |
| monospace | 240 | 6.3% |
| handwriting | 73 | 1.9% |

**Dataset Splits:**
| Split | Fonts |
|-------|-------|
| Train | 3,049 |
| Val | 380 |
| Test | 384 |

### Phase 2A: Tokenization & Model ✅ COMPLETE

| Metric | Value |
|--------|-------|
| Total Sequences | 248,227 |
| Vocabulary Size | **1,106 tokens** |
| Max Sequence Length | 512 |
| Model Parameters | ~5M (medium) |

**Tokenized Dataset (Git LFS):**
| Split | Sequences | Size |
|-------|-----------|------|
| Train | 198,581 | 379 MB |
| Val | 24,822 | 47 MB |
| Test | 24,824 | 47 MB |

### Phase 2B: Training ✅ READY

**All bugs fixed! Ready for next training run.**

| Metric | Value |
|--------|-------|
| Vocabulary | 1,106 tokens |
| UNK tokens in data | **0 (clean)** |
| Verification tests | **12/12 passing** |
| Dataset sequences | 248,227 |
| Binary format | ✅ Verified |

---

## ✅ Pre-Training Verification (2026-01-27)

Run verification before training to ensure everything is correct:

```bash
python scripts/verify_bugs.py
```

### Verification Tests (All 12 Passing)

| Test | What It Checks |
|------|----------------|
| 1 | `<NEG>` token in PATH_COMMANDS |
| 2 | `<NEG>` token in vocabulary (ID 24) |
| 3 | Tokenizer uses `<NEG>` token |
| 4 | No UNK tokens in negative coordinate test |
| 5 | Vocabulary size consistency (1,106) |
| 6 | Model vocab_size matches vocabulary |
| 7 | generate_font.py ID mappings correct |
| 8 | No UNK tokens in training data (0%) |
| 9 | Coordinate range is 0-999 |
| 10 | All notebooks have vocab_size=1106 |
| 11 | constants.py consistency |
| 12 | Binary dataset format correct |

### Past Bugs Fixed

| Bug | Impact | Status |
|-----|--------|--------|
| `<NEG>` missing from vocabulary | Training data contaminated with UNK tokens | ✅ Fixed |
| Binary format mismatch | Notebook read garbage data | ✅ Fixed |
| vocab_size = 1105 (should be 1106) | Model architecture mismatch | ✅ Fixed |

---

## 📁 Project Structure

```
FONTe AI/
├── README.md                 # Quick start guide
├── RESEARCH.md               # Research journal (append-only)
├── CHANGELOG.md              # Change history (append-only)
├── RULES.md                  # Project conventions
├── requirements.txt          # Python dependencies
├── FONTS/                    # Source fonts (NOT in repo)
├── DATASET/                  # Raw SVG glyphs (NOT in repo)
├── DATASET_NORMALIZED/       # Preprocessed SVGs (NOT in repo)
├── TOKENIZED/                # Training data (Git LFS) ✅
│   ├── train.bin             # 198K sequences
│   ├── val.bin               # 24K sequences
│   ├── test.bin              # 24K sequences
│   ├── vocabulary.json       # 1,106 token vocabulary
│   └── config.json           # Dataset config
├── model/
│   ├── fonte_model.py        # Transformer architecture
│   └── train.py              # Training script
├── scripts/
│   ├── constants.py          # Centralized token IDs
│   ├── svg_tokenizer.py      # SVG path tokenization
│   ├── ttf_to_svg.py         # Font extraction
│   ├── preprocess_dataset.py # Normalization
│   ├── create_dataset.py     # Dataset pipeline
│   ├── generate_font.py      # Font generation
│   ├── verify_bugs.py        # Verification suite (11 tests)
│   ├── analyze_codebase.py   # Code analysis
│   └── deep_bug_hunt.py      # Additional validation
├── notebooks/
│   └── FONTe_AI_Training.ipynb  # Colab training
└── generated/                # Generated fonts output
```

---

## 🚀 Quick Start

### Option A: Train in Google Colab (Recommended)
```bash
# Just open the notebook in Colab!
# It will clone the repo and pull training data via Git LFS
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nityam2007/fonte-ai/blob/master/notebooks/FONTe_AI_Training.ipynb)

### Option B: Local Development

#### 1. Clone Repository
```bash
git lfs install
git clone https://github.com/nityam2007/fonte-ai.git
cd fonte-ai
git lfs pull  # Download training data (442 MB)
```

#### 2. Install Dependencies
```bash
pip install fonttools torch  # Only 2 required dependencies!
```

#### 3. Train Model
```bash
python model/train.py --epochs 50 --batch-size 64
```

### Option C: Regenerate Dataset from Scratch

```bash
# Clone Google Fonts (2 GB)
git clone --depth 1 https://github.com/google/fonts.git FONTS/fonts-main

# Extract SVGs (2 min)
python scripts/ttf_to_svg.py --turbo

# Preprocess (1.5 min)
python scripts/preprocess_dataset.py --turbo

# Tokenize (1 min)
python scripts/create_dataset.py --workers 6
```

---

## 🛠️ Script Usage

### Extraction Script
```bash
# Turbo mode - 80% CPU, maximum speed
python scripts/ttf_to_svg.py --turbo

# With HTML previews
python scripts/ttf_to_svg.py --turbo --preview

# Custom CPU usage
python scripts/ttf_to_svg.py --cpu-percent 90

# Test run (limited fonts)
python scripts/ttf_to_svg.py --limit 100 --verbose
```

### Preprocessing Script
```bash
# Full preprocessing with turbo mode
python scripts/preprocess_dataset.py --turbo

# Test run
python scripts/preprocess_dataset.py --limit 50

# Custom canvas size
python scripts/preprocess_dataset.py --canvas-size 256 --turbo
```

### All Options:
| Flag | Description | Default |
|------|-------------|---------|
| `--turbo`, `-t` | Maximum speed mode | Off |
| `--cpu-percent`, `-c` | CPU cores to use (%) | 80 |
| `--workers`, `-w` | Explicit worker count | Auto |
| `--preview`, `-p` | Generate HTML previews | Off |
| `--limit`, `-l` | Limit fonts for testing | None |
| `--verbose`, `-v` | Detailed logging | Off |
| `--canvas-size` | Target canvas (preprocessing) | 128 |

---

## 🗺️ Roadmap

- [x] **Phase 1**: Dataset Extraction (3,824 fonts → 270K SVGs)
- [x] **Phase 1.5**: Preprocessing (normalize, classify, split)
- [x] **Phase 2A**: Tokenization (248K sequences, 1,106 vocab)
- [x] **Phase 2A**: Model Architecture (Transformer, ~5M params)
- [x] **Phase 2A**: Bug fixes & verification (12 tests passing)
- [ ] **Phase 2B**: Training (Ready to start!)
- [ ] **Phase 3**: Evaluation & Generation Quality
- [ ] **Phase 4**: Font Export (SVG → TTF)

---

## 🎯 Training on Modal.com (Recommended)

For fast training, use Modal.com with B200 GPU:

### 1. Upload Data to Modal Volume

```bash
# Install Modal CLI
pip install modal
modal setup

# Create volume and upload data
modal volume create fonte-data
modal volume put fonte-data TOKENIZED/train.bin /train.bin
modal volume put fonte-data TOKENIZED/val.bin /val.bin
modal volume put fonte-data TOKENIZED/train.json /train.json
modal volume put fonte-data TOKENIZED/val.json /val.json
```

### 2. Run Training

```bash
# Run the notebook on Modal
modal run notebooks/FONTe_AI_Training\ modal.com.ipynb
```

Or use the Python script directly:
```python
# Inside Modal notebook
python -c "from notebooks import train; train()"
```

### 3. Download Trained Models

```bash
modal volume get fonte-data /best_model.pt ./TRAINED/
modal volume get fonte-data /epoch_50.pt ./TRAINED/
```

### Expected Results

| Epoch | Val Loss | Time |
|-------|----------|------|
| 1 | ~6.9 | 2 min |
| 10 | ~4.7 | 20 min |
| 25 | ~3.5 | 50 min |
| 50 | ~3.0 | 1.8 hrs |

**Cost estimate:** ~$12 for 50 epochs on B200

---

## 📝 Unicode Naming

| Character | Unicode | Filename |
|-----------|---------|----------|
| A | U+0041 | uni0041.svg |
| a | U+0061 | uni0061.svg |
| 0 | U+0030 | uni0030.svg |
| . | U+002E | uni002E.svg |

---

## 📄 License

This project uses fonts from Google Fonts under OFL, Apache 2.0, and UFL licenses.

---

## 🔧 Troubleshooting

### Common Issues

**1. "vocab_size mismatch" error**
```bash
# Verify all files have vocab_size=1106
python scripts/verify_bugs.py
```

**2. UNK tokens appearing in output**
```bash
# Check training data is clean
python -c "import json; d=json.load(open('TOKENIZED/train.json')); print(sum(s['token_ids'].count(3) for s in d['sequences']))"
# Should print: 0
```

**3. Model loading fails**
- Ensure model was trained with vocab_size=1106
- Check that checkpoint has matching architecture

**4. Binary dataset read errors**
```bash
# Verify binary format
python scripts/verify_bugs.py  # Test 12 checks this
```

### Re-tokenize Dataset (if needed)

If you encounter data issues, regenerate from scratch:

```bash
# Regenerate vocabulary and dataset
python scripts/create_dataset.py --turbo

# Verify
python scripts/verify_bugs.py
```

---

## 📈 Training History

### First Training Run (2026-01-22)
- **Issue:** `<NEG>` token missing from vocabulary
- **Impact:** Training data had UNK contamination
- **Result:** Model learned noise, glyphs unrecognizable
- **Cost:** ~$44 wasted

### Second Training Run (2026-01-28) ✅ SUCCESS
- **Platform:** Modal.com with B200 GPU (192GB VRAM)
- **Dataset:** 5,081 train / 635 val sequences
- **Epochs:** 200 in 2.3 minutes
- **Best val_loss:** 5.25
- **Optimizations:** Flash Attention, BF16, fused AdamW
- **Result:** Model generates SVG tokens, needs more training for quality

| Metric | Value |
|--------|-------|
| Training Time | 2.3 min |
| Epochs | 200 |
| val_loss | 5.25 |
| Throughput | 635ms/epoch |

### Current State (2026-01-28)
- ✅ All bugs fixed
- ✅ 12/12 verification tests passing
- ✅ First successful training complete
- ✅ Model generating SVG tokens
- 🎯 Next: Train on full dataset (248K sequences)

---

> 📚 **For detailed research notes, see [RESEARCH.md](RESEARCH.md)**
> 
> 📋 **For change history, see [CHANGELOG.md](CHANGELOG.md)**