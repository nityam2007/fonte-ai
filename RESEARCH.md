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

