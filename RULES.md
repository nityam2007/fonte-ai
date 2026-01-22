# Project Rules & Conventions

## 📜 CHANGELOG Rules

### Rule #1: APPEND-ONLY Policy

The `CHANGELOG.md` file is **strictly append-only**:

| ✅ Allowed | ❌ Not Allowed |
|-----------|----------------|
| Add new entries at the bottom | Delete any existing text |
| Add new sections | Modify previous entries |
| Fix typos in latest entry only | Remove old entries |
| Append clarifications | Edit historical records |

### Rule #2: Update Frequency

The changelog **MUST** be updated after every code iteration:

- Every new feature added
- Every bug fix
- Every file created or modified
- Every test performed
- Every configuration change

### Rule #3: Entry Format

Each changelog entry should follow this format:

```markdown
## [YYYY-MM-DD] - Brief Title

### Added
- Description of new features/files

### Changed
- Description of modifications

### Fixed
- Description of bug fixes

### Tested
- Description of tests performed

### Removed
- Description of deprecated items (documentation only, not deletion)
```

### Rule #4: Immutability Guarantee

Once a changelog entry is written:
1. It becomes a permanent historical record
2. Corrections must be added as new entries
3. The append-only nature ensures full audit trail
4. No force-pushes or history rewrites allowed

---

## 🗂️ File Naming Conventions

### SVG Glyph Files
- Format: `uni{XXXX}.svg` where XXXX is 4-digit hex Unicode
- Example: `uni0041.svg` for 'A', `uni0061.svg` for 'a'

### Font Directories
- Format: `{source}_{fontname}/`
- Example: `aclonica_Aclonica-Regular/`

---

## 📁 Directory Structure Rules

```
FONTe AI/
├── CHANGELOG.md      # Append-only change log
├── RULES.md          # This file (project rules)
├── README.md         # Project overview
├── requirements.txt  # Python dependencies
├── FONTS/            # Source fonts (read-only after setup)
├── DATASET/          # Generated SVG glyphs
├── scripts/          # Utility scripts
├── aidata/           # AI model plans and docs
└── models/           # Trained models (future)
```

---

## 🔒 Data Integrity Rules

1. **Source fonts** in `FONTS/` should not be modified
2. **Generated data** in `DATASET/` can be regenerated from source
3. **Metadata files** must be kept in sync with SVG files
4. **Changelog** is the source of truth for project history

---

## 🚫 File Deletion Policy

### Rule #5: NO `rm` Command Usage

**NEVER use the `rm` command to delete files.** This is strictly prohibited.

| ✅ Allowed | ❌ Not Allowed |
|-----------|----------------|
| Edit existing files | `rm filename` |
| Clear file contents | `rm -rf directory/` |
| Comment out code | Bulk file deletion via terminal |
| Deprecate via renaming | Any destructive commands |

### What To Do Instead:

1. **Single file removal needed?** → Edit the file, leave empty or add deprecation notice
2. **Bulk file removal needed?** → **ASK THE USER** to handle it manually
3. **Directory cleanup needed?** → Inform user with exact paths, let them decide
4. **Generated data cleanup?** → User runs cleanup commands themselves

### Why This Rule Exists:

- Prevents accidental data loss
- Maintains audit trail
- User retains full control over destructive operations
- AI assistance should be constructive, not destructive

### Exception:

The **ONLY** exception is when the user **explicitly requests** deletion with clear confirmation, and even then, the user should execute the command themselves.

---

*Last updated: 2026-01-22*
