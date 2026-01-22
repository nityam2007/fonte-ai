# FONTe AI - AI Font Generator

> ## 🚨 **KNOWN BUG - AGENT OVERSIGHT** 🚨
> 
> **The `<NEG>` token was NEVER added to vocabulary!**
> 
> - The tokenizer uses `<NEG>` for negative coordinates
> - But `_build_vocabulary()` never includes it → encoded as `<UNK>` (token 3)
> - **Result:** Training data corrupted with UNK tokens where negatives should be
> - **Cost:** $40 USD and 47 epochs before discovery
> - **Agent claimed "verification complete"** without running end-to-end test
> 
> **Workaround Applied:** `generate_font.py` now interprets UNK as negative sign
> 
> See [RESEARCH.md](RESEARCH.md#2b9-critical-bug-discovery---agent-oversight-) for details

---

> ## ❌ **PHASE 2B RESULT: FAILED** ❌
>
> **Generated glyphs are abstract blobs, NOT recognizable letters!**
>
> | Problem | Cause |
> |---------|-------|
> | Abstract shapes instead of letters | Too many fonts (3,813) |
> | UNK token contamination | Missing `<NEG>` in vocabulary |
> | Model can't find patterns | Extreme font variety |
>
> **Next Iteration:** Train on **100 fonts** instead of 3,813
>
> See [RESEARCH.md](RESEARCH.md#2b12-visual-evaluation---glyphs-not-recognizable-) for analysis

---

> 📜 **Open Source** - Dual Licensed under GPLv3 and Apache 2.0
>
> You may choose either license. See [LICENSE](LICENSE) for details.

[github.com/nityam2007/fonte-ai](https://github.com/nityam2007/fonte-ai)

An AI-powered font generation system that learns from existing fonts and generates new, unique typefaces.

[![Status](https://img.shields.io/badge/Status-Phase%202B%20Training-yellow)](https://github.com/nityam2007/fonte-ai)
[![Fonts](https://img.shields.io/badge/Fonts-3813-blue)](https://github.com/nityam2007/fonte-ai)
[![Glyphs](https://img.shields.io/badge/Glyphs-270252-purple)](https://github.com/nityam2007/fonte-ai)
[![Sequences](https://img.shields.io/badge/Sequences-248K-orange)](https://github.com/nityam2007/fonte-ai)
[![Training](https://img.shields.io/badge/Training-B200%20GPU%20🚀-brightgreen)](https://github.com/nityam2007/fonte-ai)
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
Tokenized:     [1, 25, 29, 4, 115, 125, 6, 155, 185, 22, 2]
               [SOS, style, char, M, 10, 20, L, 50, 80, Z, EOS]
```

**Token categories:**
| Token Range | What It Represents | Example |
|-------------|-------------------|---------|
| 0-3 | Special markers | Start, End, Padding |
| 4-23 | Drawing commands | M (move), L (line), C (curve) |
| 24-28 | Font styles | serif, sans-serif, mono |
| 29-104 | Characters | A, B, C, a, b, c, 0-9 |
| 105-1104 | Coordinates | 0-999 (quantized positions) |

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
| vocab_size | 1,105 | Total unique tokens |
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
| Vocabulary Size | 1,105 tokens |
| Max Sequence Length | 512 |
| Model Parameters | ~5M (medium) |

**Tokenized Dataset (Git LFS):**
| Split | Sequences | Size |
|-------|-----------|------|
| Train | 198,581 | 379 MB |
| Val | 24,822 | 47 MB |
| Test | 24,824 | 47 MB |

### Phase 2B: Training 🔄 IN PROGRESS

**Currently training on NVIDIA B200 (192GB VRAM)!**

| Metric | Value |
|--------|-------|
| GPU | NVIDIA B200 |
| VRAM | 192 GB |
| Batch Size | 1024 |
| VRAM Used | ~130 GB |
| Time/Epoch | ~2.2 min |
| Cost | $6.73/hour |

**Training Progress:**
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 15.27 | 6.49 |
| 2 | 6.67 | 5.51 |

![B200 GPU Metrics](b200_metrics.png)

![B200 Training Output](b200_training.png)

---

## 📁 Project Structure

```
FONTe AI/
├── README.md                 # Quick start guide
├── RESEARCH.md               # Research journal (append-only)
├── CHANGELOG.md              # Change history (append-only)
├── RULES.md                  # Project conventions
├── requirements.txt          # Python dependencies
├── .gitattributes            # Git LFS configuration
├── FONTS/                    # Source fonts (NOT in repo)
├── DATASET/                  # Raw SVG glyphs (NOT in repo)
├── DATASET_NORMALIZED/       # Preprocessed SVGs (NOT in repo)
├── TOKENIZED/                # Training data (Git LFS) ✅
│   ├── train.bin             # 198K sequences (379 MB)
│   ├── val.bin               # 24K sequences (47 MB)
│   ├── test.bin              # 24K sequences (47 MB)
│   ├── vocabulary.json       # 1,105 token vocabulary
│   └── config.json           # Dataset config
├── model/
│   ├── fonte_model.py        # Transformer architecture
│   └── train.py              # Training script
├── scripts/
│   ├── ttf_to_svg.py         # Font extraction
│   ├── preprocess_dataset.py # Normalization
│   ├── svg_tokenizer.py      # SVG path tokenization
│   └── create_dataset.py     # Dataset pipeline
├── notebooks/
│   └── FONTe_AI_Training.ipynb  # Colab training
└── aidata/
    ├── planv1.md             # Original roadmap
    └── planv1.5.md           # SVG-to-SVG architecture
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
pip install fonttools torch
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
- [x] **Phase 2A**: Tokenization (248K sequences, 1,105 vocab)
- [x] **Phase 2A**: Model Architecture (Transformer, ~12M params)
- [x] **Phase 2A**: Training Pipeline (Colab + Git LFS)
- [🔄] **Phase 2B**: Training (IN PROGRESS - B200 GPU, ~2.2min/epoch)
- [ ] **Phase 3**: Evaluation & Generation Quality
- [ ] **Phase 4**: Font Export (SVG → TTF)

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

> 📚 **For detailed research notes, see [RESEARCH.md](RESEARCH.md)**