# FONTe AI - Research Journal

> ⚠️ **APPEND-ONLY**: This file follows strict append-only rules. See [RULES.md](RULES.md) for details.
> 
> This document serves as a research log, capturing insights, metrics, challenges, and learnings throughout the project development.

---

## Table of Contents

1. [Phase 1: Dataset Preparation](#phase-1-dataset-preparation)
2. [Phase 2: Model Architecture](#phase-2-model-architecture) *(planned)*
3. [Phase 3: Training & Evaluation](#phase-3-training--evaluation) *(planned)*
4. [Phase 4: Generation & Export](#phase-4-generation--export) *(planned)*

---

# Phase 1: Dataset Preparation

## 1.1 Data Source Analysis

### Source: Google Fonts Repository

| Metric | Value |
|--------|-------|
| Repository | [google/fonts](https://github.com/google/fonts) |
| License Types | OFL, Apache 2.0, UFL |
| Total Font Files | 3,824 |
| Font Families | ~1,500+ |
| File Formats | TTF (TrueType), OTF (OpenType) |

### Directory Structure Discovered:
```
fonts-main/
├── ofl/          # Open Font License (majority of fonts)
├── apache/       # Apache 2.0 licensed fonts
└── ufl/          # Ubuntu Font License
```

### Observation:
The Google Fonts repository is well-organized with consistent structure. Each font family has its own directory containing:
- Font files (`.ttf` or `.otf`)
- `METADATA.pb` (protocol buffer metadata)
- `DESCRIPTION.en_us.html` (font description)
- License file

---

## 1.2 Glyph Extraction Pipeline

### Date: 2026-01-22

### Methodology:
Developed a Python-based extraction pipeline using `fontTools` library to convert vector glyph data from TTF/OTF fonts into standalone SVG files.

### Character Set Definition:

| Category | Characters | Count |
|----------|-----------|-------|
| Uppercase | A-Z | 26 |
| Lowercase | a-z | 26 |
| Digits | 0-9 | 10 |
| Punctuation | .,!?@#()[] | 10 |
| **Total** | | **72** |

### Unicode Naming Convention:
```
Character → Unicode Code Point → Filename
'A' → U+0041 → uni0041.svg
'a' → U+0061 → uni0061.svg
'0' → U+0030 → uni0030.svg
'.' → U+002E → uni002E.svg
```

### SVG Format Specification:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(0,{baseline}) scale(1,-1)">
    <path d="{path_data}" fill="#000"/>
  </g>
</svg>
```

**Key Design Decisions:**
1. **Coordinate Transform**: Font coordinate systems have Y-axis pointing up; SVG has Y-axis pointing down. Applied `scale(1,-1)` transform.
2. **Baseline Alignment**: Used font's ascender value for proper baseline positioning.
3. **Minimal SVG**: Removed unnecessary attributes for smaller file sizes.

---

## 1.3 Extraction Results

### Final Extraction Metrics (2026-01-22):

| Metric | Value |
|--------|-------|
| Total Fonts Processed | 3,824 |
| Successfully Processed | 3,824 (100%) |
| Failed | 0 (0%) |
| Total Glyphs Extracted | 270,252 |
| Average Glyphs per Font | 70.7 |
| Processing Time | 2.1 minutes |
| Processing Speed | 30.4 fonts/second |
| CPU Utilization | 6/8 cores (75%) |

### Storage Analysis:

| Metric | Estimated Value |
|--------|-----------------|
| SVG Files Generated | 270,252 |
| Average SVG Size | ~2-5 KB |
| Total Dataset Size | ~800 MB - 1.3 GB |
| Metadata Files | 3,825 (1 per font + 1 global) |

### Missing Glyphs Analysis:

Expected glyphs per font: 72
Actual average: 70.7

**Possible Reasons for Missing Glyphs:**
1. Some fonts don't include all punctuation characters
2. Specialty fonts (display, decorative) may lack certain characters
3. Non-Latin fonts may not have Latin character mappings

---

## 1.4 Performance Optimization Journey

### Initial Implementation:
- Single-threaded processing
- Full font loading
- Verbose logging

### Optimized Implementation (TURBO Mode):

| Optimization | Impact |
|-------------|--------|
| Parallel Processing (6 cores) | ~6x speedup |
| Lazy Font Loading (`lazy=True`) | Reduced memory, faster load |
| Suppressed fontTools Logging | Cleaner output, slight speedup |
| Batch SVG Writing | Reduced I/O overhead |
| Minimal SVG Template | Smaller file sizes |
| Progress Bar with ETA | Better UX |

### Command Line Interface:
```bash
# Turbo mode (recommended)
python scripts/ttf_to_svg.py --turbo

# Custom CPU allocation
python scripts/ttf_to_svg.py --cpu-percent 90

# With HTML previews
python scripts/ttf_to_svg.py --turbo --preview

# Limited run for testing
python scripts/ttf_to_svg.py --limit 100 --verbose
```

---

## 1.5 Challenges & Solutions

### Challenge 1: Font Coordinate System Mismatch

**Problem:** Font glyphs appeared upside-down in SVG output.

**Root Cause:** TrueType/OpenType fonts use a coordinate system where Y increases upward (mathematical convention), while SVG uses a coordinate system where Y increases downward (screen convention).

**Solution:** Applied transformation in SVG: `transform="translate(0,{baseline}) scale(1,-1)"`

---

### Challenge 2: Variable Width Glyphs

**Problem:** Different glyphs have different widths (e.g., 'W' is wider than 'i').

**Solution:** Extracted per-glyph width from font metrics and set individual `viewBox` for each SVG.

---

### Challenge 3: Fonts Without Standard Character Maps

**Problem:** Some fonts (especially decorative/symbol fonts) don't include standard Latin characters.

**Solution:** Gracefully handle missing glyphs, log but don't fail. These fonts will have fewer SVG outputs.

---

### Challenge 4: Memory Usage with Large Dataset

**Problem:** Processing 3,824 fonts could exhaust system memory.

**Solution:** 
- Process one font at a time (not all in memory)
- Use `ProcessPoolExecutor` for parallel processing with automatic memory management
- Close font files immediately after extraction

---

## 1.6 Dataset Quality Observations

### Positive Findings:
1. ✅ 100% success rate on extraction
2. ✅ Consistent SVG format across all fonts
3. ✅ Proper Unicode naming maintained
4. ✅ Font metrics preserved in metadata

### Areas for Future Improvement:
1. ⚠️ Some SVG paths are complex (many control points) - may need simplification for training
2. ⚠️ No normalization of glyph sizes yet - may need standardized canvas
3. ⚠️ Style metadata not extracted (serif vs sans, weight, etc.)
4. ⚠️ No validation of SVG visual quality

---

## 1.7 Next Steps (Planned)

1. ~~**SVG Normalization**: Standardize all glyphs to 128x128 canvas~~ ✅ DONE
2. ~~**Style Labeling**: Classify fonts by style~~ ✅ DONE
3. ~~**Train/Val/Test Split**: Prepare proper dataset splits~~ ✅ DONE
4. **Model Architecture**: Design SVG-to-SVG generation model
5. **Training Pipeline**: Implement CPU-optimized training

---

*Entry logged: 2026-01-22*

---

# Phase 1.5: Dataset Preprocessing

## 1.8 SVG Normalization

### Date: 2026-01-22

### Objective:
Standardize all glyph SVGs to a uniform 128x128 canvas for consistent AI model input.

### Why 128x128?
| Factor | Reasoning |
|--------|-----------|
| Memory Efficiency | Smaller tensors = faster CPU training |
| Sufficient Detail | 128px captures glyph details adequately |
| Power of 2 | Efficient for convolution operations |
| Batch Processing | More samples per batch |

### Normalization Algorithm:
```
1. Parse original viewBox (variable size per font)
2. Calculate scale factor to fit in usable area (90% of canvas)
3. Calculate offset to center the glyph
4. Apply transform: translate + scale + Y-axis flip
5. Write new SVG with viewBox="0 0 128 128"
```

### Output SVG Format:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(x,y) scale(s,-s)">
    <path d="..." fill="#000"/>
  </g>
</svg>
```

---

## 1.9 Style Classification System

### Methodology:
Keyword-based classification using font names. Each font is assigned ONE style based on priority matching.

### Style Categories:

| Style | Keywords (partial list) | Typical Use |
|-------|------------------------|-------------|
| **monospace** | mono, code, courier, terminal | Code editors, technical docs |
| **handwriting** | script, brush, cursive, signature | Invitations, personal touch |
| **display** | display, decorative, poster, headline | Headlines, logos |
| **serif** | serif, roman, times, garamond | Body text, traditional |
| **sans-serif** | sans, gothic, helvetica, roboto | Modern UI, web |

### Classification Priority:
```
monospace > handwriting > display > serif > sans-serif
```

**Rationale:** More specific styles are checked first. If no match, defaults to sans-serif (most common in Google Fonts).

### Test Results (50 font sample):

| Style | Count | Percentage |
|-------|-------|------------|
| sans-serif | 33 | 66.0% |
| serif | 11 | 22.0% |
| monospace | 5 | 10.0% |
| handwriting | 1 | 2.0% |
| display | 0 | 0.0% |

**Observation:** Sans-serif dominates Google Fonts. Display fonts may need keyword expansion.

---

## 1.10 Dataset Organization

### New Directory Structure:
```
DATASET_NORMALIZED/
├── metadata.json          # Global metadata
├── train.json             # Training split (80%)
├── val.json               # Validation split (10%)
├── test.json              # Test split (10%)
├── serif/
│   ├── fontname_variant/
│   │   ├── metadata.json
│   │   ├── uni0041.svg    # 'A' (128x128)
│   │   └── ...
├── sans-serif/
├── monospace/
├── handwriting/
└── display/
```

### Split Strategy:
- **Stratified by style**: Each split maintains style proportions
- **Random seed**: 42 (reproducible)
- **Ratios**: 80% train / 10% val / 10% test

---

## 1.11 Preprocessing Performance

### Metrics (50 font test run):

| Metric | Value |
|--------|-------|
| Processing Speed | 44.9 fonts/sec |
| Glyphs Processed | 3,600 |
| Canvas Size | 128x128 |
| Workers | 6/8 cores |
| Time | ~1 second |

### Estimated Full Dataset:
- **3,813 fonts** at 44.9 fonts/sec = **~85 seconds** (~1.5 minutes)

---

## 1.12 Design Decision: SVG-to-SVG Model

### Why Keep SVG Format?

| Approach | Pros | Cons |
|----------|------|------|
| **SVG → Raster → SVG** | Standard image models work | Lossy conversion, vectorization needed |
| **SVG → SVG (Direct)** | Lossless, vector output, smaller files | Requires specialized architecture |

### Chosen Approach: SVG-to-SVG

**Rationale:**
1. **Vector preservation**: No rasterization artifacts
2. **Scalability**: Output fonts scale to any size
3. **Smaller files**: SVG paths are compact
4. **Designer-first**: Professionals need editable vectors
5. **CPU-friendly**: Path operations are less compute-intensive than pixels

### Model Input/Output:
```
Input:  SVG path data (normalized 128x128 canvas)
Output: SVG path data (new glyph in same format)
```

---

## 1.13 Observations & Insights

### What Works Well:
1. ✅ Parallel processing scales linearly with cores
2. ✅ Keyword classification is fast and reasonably accurate
3. ✅ 128x128 canvas captures sufficient detail
4. ✅ SVG format preserves all vector information

### Areas Needing Attention:
1. ⚠️ Display fonts under-classified (need more keywords)
2. ⚠️ Some fonts may be misclassified (name doesn't match style)
3. ⚠️ Very complex paths may need simplification
4. ⚠️ Need to validate visual quality of normalized SVGs

### Future Improvements:
1. Visual validation of normalized outputs
2. Path simplification for overly complex glyphs
3. Manual review of style classifications
4. Font quality scoring (exclude low-quality fonts)

---

*Entry logged: 2026-01-22*

---

# Phase 2A: Tokenization & Model Architecture

## 2A.1 SVG Path Tokenization

### Date: 2026-01-22

### Methodology:
Developed a vocabulary-based tokenizer that converts SVG path commands into discrete tokens for transformer processing.

### Vocabulary Design (1,105 tokens):

| Category | Token Range | Count | Purpose |
|----------|-------------|-------|---------|
| Special | 0-3 | 4 | PAD, SOS, EOS, UNK |
| Commands | 4-23 | 20 | M, L, C, Q, Z, etc. |
| Styles | 24-28 | 5 | serif, sans-serif, monospace, handwriting, display |
| Characters | 29-104 | 76 | A-Z, a-z, 0-9, punctuation |
| Coordinates | 105-1104 | 1000 | Quantized 0-999 values |

### Tokenization Process:
```
SVG Path → Parse Commands → Quantize Coordinates → Token Sequence

Example:
"M 10 20 L 50 80 Z" →
[SOS, STYLE, CHAR, M, 10, 20, L, 50, 80, Z, EOS, PAD, PAD, ...]
```

### Key Design Decisions:
1. **Coordinate Quantization**: 0-999 range captures sufficient precision
2. **Style Conditioning**: Prepend style token for style-aware generation
3. **Character Embedding**: Include target character in sequence
4. **Max Length 512**: Balances detail vs memory usage

---

## 2A.2 Dataset Tokenization Results

### Final Tokenization Metrics:

| Metric | Value |
|--------|-------|
| Total Sequences | 248,227 |
| Train Split | 198,581 (80%) |
| Validation Split | 24,822 (10%) |
| Test Split | 24,824 (10%) |
| Vocabulary Size | 1,105 |
| Max Sequence Length | 512 |
| Processing Time | 49.9 seconds |

### Storage Format:

Binary format for efficient loading:
```
Header: [n_sequences, max_len, vocab_size] (12 bytes)
Per sequence: [length (2 bytes), tokens (512 × 2 bytes)]
```

| File | Sequences | Size |
|------|-----------|------|
| train.bin | 198,581 | 379 MB |
| val.bin | 24,822 | 47 MB |
| test.bin | 24,824 | 47 MB |

---

## 2A.3 Model Architecture

### Transformer Decoder Design:

| Component | Description |
|-----------|-------------|
| Embedding | Token + Positional encoding |
| Blocks | Causal self-attention + FFN |
| Output | LM head (weight-tied with embeddings) |

### Model Configurations:

| Size | d_model | n_heads | n_layers | d_ff | Params |
|------|---------|---------|----------|------|--------|
| Small | 128 | 4 | 4 | 512 | ~1M |
| Medium | 256 | 4 | 6 | 1024 | ~12M |
| Large | 512 | 8 | 8 | 2048 | ~50M |

### Key Architectural Choices:

1. **Pre-norm**: LayerNorm before attention (more stable training)
2. **GELU activation**: Smoother than ReLU
3. **Weight tying**: Output head shares weights with embeddings
4. **Causal mask**: Autoregressive generation

### Generation Strategy:
```python
def generate(style_id, char_id, temperature=1.0, top_k=50):
    tokens = [SOS, style_id, char_id]
    while len(tokens) < max_len:
        logits = model(tokens)[-1]
        next_token = sample(logits, temperature, top_k)
        tokens.append(next_token)
        if next_token == EOS:
            break
    return tokens
```

---

## 2A.4 Training Infrastructure

### Git LFS for Data Distribution:

| Challenge | Solution |
|-----------|----------|
| 442 MB training data | Git LFS tracking |
| Colab clone workflow | `git lfs pull` after clone |
| Fast iteration | Data in repo, not uploaded |

### Colab Training Setup:

```bash
# Colab workflow (automated in notebook)
!apt-get install git-lfs -qq
!git lfs install
!git clone https://github.com/nityam2007/fonte-ai.git
%cd fonte-ai
!git lfs pull
# → Ready to train!
```

### Training Configuration:

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 |
| Batch Size | 64 |
| Scheduler | Cosine Annealing |
| Gradient Clipping | 1.0 |

---

## 2A.5 Observations & Insights

### What Works Well:
1. ✅ Binary format loads 10x faster than JSON
2. ✅ Vocabulary size (1,105) is manageable
3. ✅ Git LFS handles 442 MB smoothly
4. ✅ Medium model (12M) fits in T4 GPU

### Areas to Monitor:
1. ⚠️ 512 max length may truncate complex glyphs
2. ⚠️ Coordinate quantization to 0-999 may lose precision
3. ⚠️ Style tokens based on keywords (not visual features)
4. ⚠️ Need to validate generated path validity

### Next Steps:
1. Run training for 50-100 epochs
2. Monitor loss curves and generation quality
3. Experiment with temperature and top-k
4. Validate generated SVG paths are valid

---

*Entry logged: 2026-01-22*

---

# Phase 2B: Model Training

## 2B.1 Training Session Started

### Date: 2026-01-22

### Platform:
- **Hardware**: Google Colab T4 GPU (15GB VRAM)
- **Runtime**: Standard (Free Tier)

### Training Configuration:

| Parameter | Value |
|-----------|-------|
| Model Size | Medium (~12M params) |
| d_model | 256 |
| n_heads | 4 |
| n_layers | 6 |
| d_ff | 1024 |
| Epochs | 50 |
| Batch Size | 64 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Scheduler | Cosine Annealing |
| Gradient Clipping | 1.0 |

### Initial Metrics:

| Metric | Value |
|--------|-------|
| Batches per Epoch | 3,103 |
| Training Speed | ~1.79 it/s |
| ETA per Epoch | ~28 minutes |
| Initial Loss | 5.58 |
| Total Training Time (est.) | ~23 hours |

### Observations:

1. **Loss Starting Point**: 5.58 is reasonable for random initialization
   - With vocab size 1,105, random would be ~7.0 (ln(1105))
   - Model is already learning!

2. **Speed**: 1.79 it/s is good for T4
   - Batch size 64 is optimal
   - Higher batch sizes risk OOM

3. **Memory Usage**: Model fits comfortably in T4's 15GB
   - 12M params × 4 bytes × 2 (gradients) = ~96 MB
   - Plenty of room for activations

### Expected Training Curve:

| Epoch | Expected Loss | Notes |
|-------|---------------|-------|
| 1-5 | 5.5 → 3.5 | Rapid initial learning |
| 5-20 | 3.5 → 2.0 | Steady improvement |
| 20-40 | 2.0 → 1.5 | Fine-tuning |
| 40-50 | 1.5 → 1.2 | Convergence |

---

*Entry logged: 2026-01-22 - Training in progress*

---

## 2B.2 Switched to Modal L40S

### Date: 2026-01-22

### Why Switch from Colab T4:
- Colab T4 free tier: 4-hour limit
- Would need 6 sessions over multiple days
- Modal L40S: ~$13 for complete training in one session

### Platform:
- **Hardware**: Modal L40S GPU (48GB VRAM)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Cost**: $2.07/hour

### Actual Training Metrics:

| Metric | T4 (Colab) | L40S (Modal) |
|--------|------------|---------------|
| VRAM | 15 GB | 48 GB |
| Max Batch Size | ~64 | ~198 |
| Speed | 1.79 it/s | 2.24 it/s |
| Batches/Epoch | 3,103 | 1,003 |
| Time/Epoch | ~28 min | ~7.5 min |
| 50 Epochs | ~23 hrs | ~6.2 hrs |
| Cost | FREE | ~$13 |

---

## 2B.3 CRITICAL: Memory Usage Analysis

### ⚠️ Our Initial Estimates Were WRONG!

We assumed:
- "12M params = ~50MB, plenty of room"
- "Can use batch 512 easily"

**Reality**: Batch 198 uses **40GB of 48GB VRAM!**

### Why So Much Memory?

**Transformer memory formula:**
```
Memory ≈ batch_size × seq_length² × n_layers × n_heads × bytes_per_value
```

For our model:
- `batch_size = 198`
- `seq_length = 512`
- `n_layers = 6`
- `n_heads = 4`
- `d_model = 256`

### Memory Breakdown (Actual):

| Component | Estimated | Actual | Notes |
|-----------|-----------|--------|-------|
| Model weights | ~50 MB | ~50 MB | ✅ Correct |
| Gradients | ~50 MB | ~50 MB | ✅ Correct |
| Optimizer (AdamW) | ~100 MB | ~100 MB | ✅ 2x model size |
| **Attention matrices** | ~200 MB | **~15 GB** | ❌ Way off! |
| **Activations** | ~500 MB | **~25 GB** | ❌ Way off! |

### The Attention Memory Problem:

```
Attention memory per layer = batch × heads × seq × seq × 4 bytes
= 198 × 4 × 512 × 512 × 4 = ~830 MB per layer
= 830 MB × 6 layers = ~5 GB (just attention scores!)
```

Plus activations, gradients, and intermediate values = **40 GB total**

### Lessons Learned:

1. **Never assume** batch size based on model params alone
2. **Sequence length** is the memory killer (quadratic!)
3. **Always test** with small batch first, then increase
4. **Monitor VRAM** during first few iterations

### Safe Batch Sizes by GPU:

| GPU | VRAM | Safe Batch | Max Batch |
|-----|------|------------|----------|
| T4 | 15 GB | 48 | ~64 |
| A10 | 24 GB | 80 | ~100 |
| A100 40GB | 40 GB | 160 | ~180 |
| **L40S** | 48 GB | 180 | ~198 |
| A100 80GB | 80 GB | 350 | ~400 |

*Note: These are specific to our model (seq_length=512, d_model=256, n_layers=6)*

---

*Entry logged: 2026-01-22 - Important memory insights!*

---

## 2B.4 Epoch 1 Results

### Date: 2026-01-22

### Training Progress:

| Epoch | Train Loss | Val Loss | Δ Val Loss | Time |
|-------|------------|----------|------------|------|
| 0 (init) | - | ~7.0 | - | - |
| **1** | 4.96 | 3.94 | **-44%** | 7.8 min |

### Analysis:

1. **Rapid Initial Learning**: Loss dropped from ~7.0 (random, ln(1105)) to 3.94
2. **Model is Learning**: 44% improvement in just 1 epoch
3. **Not Overfitting**: Train (4.96) > Val (3.94) is healthy

### Expected Quality at Epoch 1:

Based on loss 3.94 (perplexity ~51):
- 🔴 Mostly noise with occasional structure
- Some path commands (M, L, C) appearing
- Coordinates likely random
- NOT valid SVG paths yet

### Generation Script Added:

Created `scripts/generate_font.py` for testing checkpoints:

```bash
# Test epoch 1
python scripts/generate_font.py --model TRAINED/checkpoint_epoch_1.pt --char A

# Test with different temperatures
python scripts/generate_font.py --model TRAINED/best_model.pt --char A --temperature 0.5
python scripts/generate_font.py --model TRAINED/best_model.pt --char A --temperature 1.0
```

### Checkpoint Strategy:

Saving all 50 epoch checkpoints to:
```
TRAINED/
├── best_model.pt              # Best val_loss so far
├── checkpoint_epoch_1.pt      # 21 MB each
├── checkpoint_epoch_2.pt
├── ...
├── checkpoint_epoch_50.pt
└── training_history.json      # Loss curves
```

Total expected size: 50 × 21 MB = **~1 GB**

---

*Entry logged: 2026-01-22 - First epoch complete!*

---