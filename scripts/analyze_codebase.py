#!/usr/bin/env python3
"""
Codebase Analysis Tool for FONTe AI
Analyzes code for potential issues, optimizations, and dependencies
"""

import sys
import re
from pathlib import Path

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from svg_tokenizer import Vocabulary, PATH_COMMANDS, STYLE_TOKENS, CHAR_SET, COORD_RANGE


def analyze_token_layout():
    """Verify token ID layout is correct"""
    print("\n" + "="*70)
    print("TOKEN LAYOUT ANALYSIS")
    print("="*70)
    
    vocab = Vocabulary()
    
    # Calculate expected ranges
    special_end = 3
    commands_start = 4
    commands_end = commands_start + len(PATH_COMMANDS) - 1
    styles_start = commands_end + 1
    styles_end = styles_start + len(STYLE_TOKENS) - 1
    chars_start = styles_end + 1
    chars_end = chars_start + len(CHAR_SET) - 1
    coords_start = chars_end + 1
    coords_end = coords_start + COORD_RANGE - 1
    
    print(f"\n  Expected Layout:")
    print(f"    Special:  0-{special_end}      ({special_end + 1} tokens)")
    print(f"    Commands: {commands_start}-{commands_end}     ({len(PATH_COMMANDS)} tokens)")
    print(f"    Styles:   {styles_start}-{styles_end}     ({len(STYLE_TOKENS)} tokens)")
    print(f"    Chars:    {chars_start}-{chars_end}   ({len(CHAR_SET)} tokens)")
    print(f"    Coords:   {coords_start}-{coords_end} ({COORD_RANGE} tokens)")
    print(f"    TOTAL:    {coords_end + 1} tokens")
    
    # Verify actual tokens
    issues = []
    
    # Check <NEG> token
    neg_id = vocab.token_to_id.get('<NEG>')
    if neg_id != commands_end:
        issues.append(f"<NEG> token ID mismatch: expected {commands_end}, got {neg_id}")
    
    # Check first style (STYLE_TOKENS already include the full token name)
    first_style = STYLE_TOKENS[0]  # Already formatted as <STYLE:serif>
    first_style_id = vocab.token_to_id.get(first_style)
    if first_style_id != styles_start:
        issues.append(f"First style ID mismatch: expected {styles_start}, got {first_style_id}")
    
    # Check first char
    first_char_id = vocab.token_to_id.get(f'<CHAR:{CHAR_SET[0]}>')
    if first_char_id != chars_start:
        issues.append(f"First char ID mismatch: expected {chars_start}, got {first_char_id}")
    
    # Check coord 0
    coord_0_id = vocab.token_to_id.get('<COORD:0>')
    if coord_0_id != coords_start:
        issues.append(f"COORD:0 ID mismatch: expected {coords_start}, got {coord_0_id}")
    
    # Check coord 999
    coord_999_id = vocab.token_to_id.get('<COORD:999>')
    if coord_999_id != coords_end:
        issues.append(f"COORD:999 ID mismatch: expected {coords_end}, got {coord_999_id}")
    
    if issues:
        print(f"\n  ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"     - {issue}")
        return False
    else:
        print(f"\n  ✓ Token layout verified correct")
        return True


def analyze_dependencies():
    """Analyze 3rd party dependencies"""
    print("\n" + "="*70)
    print("DEPENDENCY ANALYSIS")
    print("="*70)
    
    files = [
        SCRIPT_DIR / "svg_tokenizer.py",
        SCRIPT_DIR / "create_dataset.py",
        SCRIPT_DIR / "generate_font.py",
    ]
    
    stdlib_modules = {
        're', 'json', 'struct', 'logging', 'argparse', 'time', 'random',
        'dataclasses', 'typing', 'pathlib', 'os', 'sys', 'math', 'collections',
        'concurrent', 'functools', 'itertools', 'io', 'gzip', 'hashlib',
        '__future__', 'abc', 'copy', 'unittest', 'threading', 'multiprocessing'
    }
    
    required_3rd_party = {
        'torch': 'Deep learning framework (required)',
        'numpy': 'Numerical computing (used by torch)',
        'fonttools': 'Font parsing library (required)',
        'svgpathtools': 'SVG path manipulation (required)',
    }
    
    unnecessary_potential = []
    
    for filepath in files:
        if not filepath.exists():
            print(f"  ⚠ File not found: {filepath}")
            continue
            
        content = filepath.read_text()
        print(f"\n  {filepath.name}:")
        
        # Find all imports
        import_lines = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
        
        for imp in import_lines:
            module = imp.split('.')[0]
            if module in stdlib_modules:
                print(f"    ✓ {module} (stdlib)")
            elif module in required_3rd_party:
                print(f"    📦 {module} - {required_3rd_party[module]}")
            else:
                print(f"    ? {module}")


def analyze_performance():
    """Analyze potential performance improvements"""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    svg_tok = SCRIPT_DIR / "svg_tokenizer.py"
    if not svg_tok.exists():
        return
    
    content = svg_tok.read_text()
    
    improvements = []
    
    # Check for compiled regex
    if 're.finditer' in content or 're.match' in content or 're.search' in content:
        if 're.compile' not in content:
            improvements.append("Regex not pre-compiled - compile patterns once for ~2x speedup")
    
    # Check for list comprehensions vs loops
    loop_count = content.count('for ') + content.count('while ')
    comprehension_count = content.count(' for ') + content.count('[')
    
    # Check for string concatenation in loops
    if '+=' in content and 'str' in content:
        improvements.append("Possible string concatenation in loop - consider using join()")
    
    if improvements:
        print(f"\n  Potential improvements:")
        for imp in improvements:
            print(f"    ⚡ {imp}")
    else:
        print(f"\n  ✓ No obvious performance issues found")


def analyze_error_handling():
    """Check error handling robustness"""
    print("\n" + "="*70)
    print("ERROR HANDLING ANALYSIS")
    print("="*70)
    
    files = [
        SCRIPT_DIR / "svg_tokenizer.py",
        SCRIPT_DIR / "create_dataset.py",
        SCRIPT_DIR / "generate_font.py",
    ]
    
    for filepath in files:
        if not filepath.exists():
            continue
            
        content = filepath.read_text()
        
        try_count = content.count('try:')
        except_count = content.count('except')
        
        # Check for bare excepts
        bare_excepts = len(re.findall(r'except\s*:', content))
        
        print(f"\n  {filepath.name}:")
        print(f"    try/except blocks: {try_count}")
        if bare_excepts > 0:
            print(f"    ⚠ Bare except clauses: {bare_excepts} (catch specific exceptions)")
        else:
            print(f"    ✓ No bare except clauses")


def analyze_arc_command():
    """Check Arc (A/a) command handling"""
    print("\n" + "="*70)
    print("ARC COMMAND SPECIAL CASE ANALYSIS")
    print("="*70)
    
    print("""
  Arc commands (A/a) in SVG have 7 parameters:
    A rx ry x-axis-rotation large-arc-flag sweep-flag x y
    
  The flags (large-arc-flag, sweep-flag) are 0 or 1.
  These can cause issues if tokenized as coordinates (0-999 range)
  since they're binary values mixed with float coordinates.
    """)
    
    # Check if we handle arc commands
    vocab = Vocabulary()
    a_in_vocab = 'A' in PATH_COMMANDS and 'a' in PATH_COMMANDS
    
    if a_in_vocab:
        print(f"  ✓ Arc commands (A/a) are in vocabulary")
        print(f"  ⚠ Note: Flag parameters (0/1) will be tokenized as <COORD:0>/<COORD:1>")
        print(f"     This is acceptable but training should learn this pattern")
    else:
        print(f"  ✗ Arc commands NOT in vocabulary - may cause issues")


def check_hardcoded_values():
    """Check for hardcoded values that should be constants"""
    print("\n" + "="*70)
    print("HARDCODED VALUES CHECK")
    print("="*70)
    
    gen_font = SCRIPT_DIR / "generate_font.py"
    if not gen_font.exists():
        return
        
    content = gen_font.read_text()
    
    # Check for hardcoded token IDs
    checks = [
        (r'\b1105\b', '1105 (COORD_END)', 3),
        (r'\b1106\b', '1106 (vocab_size)', 3),
        (r'\b106\b', '106 (COORD_START)', 2),
        (r'\b24\b', '24 (NEG token ID)', 3),
        (r'\b25\b', '25 (STYLE_START)', 2),
        (r'\b29\b', '29 (STYLE_END)', 2),
        (r'\b30\b', '30 (CHAR_START)', 2),
        (r'\b105\b', '105 (CHAR_END)', 2),
    ]
    
    print(f"\n  generate_font.py:")
    for pattern, desc, max_expected in checks:
        count = len(re.findall(pattern, content))
        if count > max_expected:
            print(f"    ⚠ {desc}: {count} occurrences (consider using constant)")
        else:
            print(f"    ✓ {desc}: {count} occurrences")


def run_full_analysis():
    """Run complete codebase analysis"""
    print("="*70)
    print("FONTe AI - CODEBASE ANALYSIS")
    print("="*70)
    
    analyze_token_layout()
    analyze_dependencies()
    analyze_performance()
    analyze_error_handling()
    analyze_arc_command()
    check_hardcoded_values()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_full_analysis()
