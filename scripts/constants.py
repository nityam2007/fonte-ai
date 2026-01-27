#!/usr/bin/env python3
"""
Shared Constants for FONTe AI

This module defines all token IDs and vocabulary constants in one place.
All other scripts should import from here to avoid hardcoded values.

Token Layout (1106 tokens total):
    0-3:     Special tokens (PAD, SOS, EOS, UNK)
    4-24:    SVG Path commands (21 tokens including <NEG>)
    25-29:   Style tokens (5 tokens)
    30-105:  Character tokens (76 tokens)
    106-1105: Coordinate tokens (1000 tokens, 0-999)
"""

# ============================================================================
# VOCABULARY SIZE
# ============================================================================
VOCAB_SIZE = 1106

# ============================================================================
# SPECIAL TOKEN IDS
# ============================================================================
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# ============================================================================
# SVG COMMAND TOKEN IDS
# ============================================================================
COMMAND_START = 4
COMMAND_END = 24  # Last command is <NEG>
COMMAND_COUNT = COMMAND_END - COMMAND_START + 1  # 21

# Special command for negative sign
NEG_TOKEN_ID = 24

# ============================================================================
# STYLE TOKEN IDS
# ============================================================================
STYLE_START = 25
STYLE_END = 29
STYLE_COUNT = STYLE_END - STYLE_START + 1  # 5

# ============================================================================
# CHARACTER TOKEN IDS
# ============================================================================
CHAR_START = 30
CHAR_END = 105
CHAR_COUNT = CHAR_END - CHAR_START + 1  # 76

# Character set (must match svg_tokenizer.py CHAR_SET)
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%&*()-+=[]"
assert len(CHARS) == CHAR_COUNT, f"CHARS length {len(CHARS)} != CHAR_COUNT {CHAR_COUNT}"

# ============================================================================
# COORDINATE TOKEN IDS
# ============================================================================
COORD_START = 106
COORD_END = 1105
COORD_COUNT = COORD_END - COORD_START + 1  # 1000 (0-999)
COORD_MAX = 999

# ============================================================================
# MODEL DEFAULTS
# ============================================================================
MAX_SEQ_LENGTH = 512
CANVAS_SIZE = 128

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_command_token(token_id: int) -> bool:
    """Check if token ID is a command token"""
    return COMMAND_START <= token_id <= COMMAND_END

def is_style_token(token_id: int) -> bool:
    """Check if token ID is a style token"""
    return STYLE_START <= token_id <= STYLE_END

def is_char_token(token_id: int) -> bool:
    """Check if token ID is a character token"""
    return CHAR_START <= token_id <= CHAR_END

def is_coord_token(token_id: int) -> bool:
    """Check if token ID is a coordinate token"""
    return COORD_START <= token_id <= COORD_END

def coord_to_token_id(coord: int) -> int:
    """Convert coordinate value (0-999) to token ID"""
    return COORD_START + max(0, min(COORD_MAX, coord))

def token_id_to_coord(token_id: int) -> int:
    """Convert token ID to coordinate value (0-999)"""
    return token_id - COORD_START

def char_to_token_id(char: str) -> int:
    """Convert character to token ID"""
    if char in CHARS:
        return CHAR_START + CHARS.index(char)
    return UNK_TOKEN_ID

def token_id_to_char(token_id: int) -> str:
    """Convert token ID to character"""
    idx = token_id - CHAR_START
    if 0 <= idx < len(CHARS):
        return CHARS[idx]
    return "?"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_constants():
    """Validate that all constants are consistent"""
    errors = []
    
    # Check total vocab size
    expected_vocab = (4 + COMMAND_COUNT + STYLE_COUNT + CHAR_COUNT + COORD_COUNT)
    if expected_vocab != VOCAB_SIZE:
        errors.append(f"VOCAB_SIZE mismatch: expected {expected_vocab}, got {VOCAB_SIZE}")
    
    # Check ranges don't overlap
    ranges = [
        ("special", 0, 3),
        ("commands", COMMAND_START, COMMAND_END),
        ("styles", STYLE_START, STYLE_END),
        ("chars", CHAR_START, CHAR_END),
        ("coords", COORD_START, COORD_END),
    ]
    
    for i, (name1, start1, end1) in enumerate(ranges):
        for j, (name2, start2, end2) in enumerate(ranges):
            if i < j:
                # Check for overlap
                if start1 <= end2 and start2 <= end1:
                    errors.append(f"Overlap between {name1} and {name2}")
    
    # Check contiguous
    prev_end = -1
    for name, start, end in ranges:
        if start != prev_end + 1:
            errors.append(f"Gap before {name}: expected start {prev_end + 1}, got {start}")
        prev_end = end
    
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return False
    return True


if __name__ == "__main__":
    print("FONTe AI Constants Validation")
    print("=" * 50)
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"Special:    0-3     ({4} tokens)")
    print(f"Commands:   {COMMAND_START}-{COMMAND_END}   ({COMMAND_COUNT} tokens)")
    print(f"Styles:     {STYLE_START}-{STYLE_END}   ({STYLE_COUNT} tokens)")
    print(f"Chars:      {CHAR_START}-{CHAR_END}  ({CHAR_COUNT} tokens)")
    print(f"Coords:     {COORD_START}-{COORD_END} ({COORD_COUNT} tokens)")
    print()
    
    if validate_constants():
        print("✓ All constants validated successfully")
    else:
        print("✗ Validation failed!")
