# FONTe AI - AI Font Generator

> **Private Repository** | [github.com/nityam2007/fonte-ai](https://github.com/nityam2007/fonte-ai)

An AI-powered font generation system that learns from existing fonts and generates new, unique typefaces.

[![Status](https://img.shields.io/badge/Status-Phase%201.5%20Complete-green)](https://github.com/nityam2007/fonte-ai)
[![Fonts](https://img.shields.io/badge/Fonts-3813-blue)](https://github.com/nityam2007/fonte-ai)
[![Glyphs](https://img.shields.io/badge/Glyphs-270252-purple)](https://github.com/nityam2007/fonte-ai)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

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

## � Data Source & Storage

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

---

## 📁 Project Structure

```
FONTe AI/
├── README.md                 # Quick start guide
├── RESEARCH.md               # Research journal (append-only)
├── CHANGELOG.md              # Change history (append-only)
├── RULES.md                  # Project conventions
├── requirements.txt          # Python dependencies
├── FONTS/                    # Source fonts (Google Fonts)
│   └── fonts-main/
│       ├── ofl/              # Open Font License
│       ├── apache/           # Apache 2.0
│       └── ufl/              # Ubuntu Font License
├── DATASET/                  # Raw extracted SVG glyphs (270K files)
│   ├── {font_name}/
│   │   ├── metadata.json
│   │   └── *.svg
│   └── metadata.json         # Global metadata
├── DATASET_NORMALIZED/       # Preprocessed for training
│   ├── {style}/              # serif, sans-serif, etc.
│   │   └── {font_name}/
│   │       ├── metadata.json
│   │       └── *.svg         # 128×128 normalized
│   ├── train.json            # Training split
│   ├── val.json              # Validation split
│   ├── test.json             # Test split
│   └── styles.json           # Style classification
├── scripts/
│   ├── ttf_to_svg.py         # Font extraction
│   └── preprocess_dataset.py # Normalization & classification
└── aidata/
    └── planv1.md             # AI model roadmap
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fonttools
```

### 2. Extract Glyphs (TURBO Mode)
```bash
python scripts/ttf_to_svg.py --turbo
```

### 3. Preprocess Dataset
```bash
python scripts/preprocess_dataset.py --turbo
```

### 4. View Results
Open any `DATASET/{font_name}/preview.html` in browser.

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
- [ ] **Phase 2**: Model Architecture (SVG-to-SVG, CPU-optimized)
- [ ] **Phase 3**: Training & Evaluation
- [ ] **Phase 4**: Font Generation & Export

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