#!/usr/bin/env python3
"""Deep bug hunt to find more issues in FONTe AI codebase"""

import sys
import re
import json
from pathlib import Path

print('='*70)
print('DEEP BUG HUNT - Finding More Issues')
print('='*70)

issues = []
warnings = []

# 1. Check notebooks for hardcoded constants
print('\n[1] CHECKING NOTEBOOKS FOR HARDCODED VALUES...')
notebooks = list(Path('notebooks').glob('*.ipynb'))
for nb in notebooks:
    content = nb.read_text()
    
    # Check for old vocab_size - but ignore comments mentioning "was 1105"
    if 'vocab_size' in content:
        # Look for actual assignments of 1105, not comments about it
        if re.search(r'vocab_size\s*[=:]\s*1105\b', content) and 'was 1105' not in content:
            issues.append(f'{nb.name}: Still has old vocab_size=1105')
        # Also check for int = 1105 pattern
        if re.search(r'vocab_size:\s*int\s*=\s*1105\b', content):
            issues.append(f'{nb.name}: Still has old vocab_size=1105')
    
    # Check for hardcoded token IDs
    patterns = [
        (r'COORD_START\s*=\s*105\b', 'old COORD_START=105'),
        (r'COORD_END\s*=\s*1104\b', 'old COORD_END=1104'),
    ]
    
    for pattern, desc in patterns:
        if re.search(pattern, content):
            issues.append(f'{nb.name}: {desc}')

# 2. Check if notebooks define ModelConfig correctly
print('\n[2] CHECKING NOTEBOOK MODEL CONFIGS...')
for nb in notebooks:
    content = nb.read_text()
    data = json.loads(content)
    
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # Check for COORD_START/COORD_END in notebooks
            if 'COORD_START' in source:
                match = re.search(r'COORD_START\s*=\s*(\d+)', source)
                if match and int(match.group(1)) != 106:
                    issues.append(f'{nb.name}: COORD_START={match.group(1)} (should be 106)')
            
            if 'COORD_END' in source:
                match = re.search(r'COORD_END\s*=\s*(\d+)', source)
                if match and int(match.group(1)) != 1105:
                    issues.append(f'{nb.name}: COORD_END={match.group(1)} (should be 1105)')

# 3. Check create_dataset.py for any issues
print('\n[3] CHECKING create_dataset.py...')
with open('scripts/create_dataset.py', 'r') as f:
    content = f.read()

# Check if it imports from constants (optional - it uses svg_tokenizer which is fine)
if 'from constants import' not in content and 'import constants' not in content:
    warnings.append('create_dataset.py: Could use constants module for consistency (optional)')

# 4. Check for edge cases in tokenization
print('\n[4] CHECKING TOKENIZATION EDGE CASES...')
sys.path.insert(0, 'scripts')
from svg_tokenizer import Vocabulary, tokenize_path

vocab = Vocabulary()

# Test edge cases
test_cases = [
    ('M 0 0 L 999 999 Z', 'Max coords'),
    ('M -500 -500 L 500 500 Z', 'Large negatives'),
    ('M 0.5 0.5 L 1.5 1.5 Z', 'Decimals'),
    ('M 1000 1000 Z', 'Out of range coords'),
    ('A 10 10 0 0 1 50 50', 'Arc command with flags'),
]

for path, desc in test_cases:
    tokens = tokenize_path(path)
    encoded = vocab.encode_sequence(tokens)  # Use encode_sequence for list
    
    if 3 in encoded:  # UNK
        issues.append(f'Tokenization issue with {desc}: produces UNK')

# 5. Check svg_tokenizer for SVGTokenizer class
print('\n[5] CHECKING svg_tokenizer.py...')
with open('scripts/svg_tokenizer.py', 'r') as f:
    tok_content = f.read()

if 'class SVGTokenizer' in tok_content:
    if 'PATH_TOKEN_PATTERN' not in tok_content:
        issues.append('svg_tokenizer.py: SVGTokenizer may not use compiled regex')

# Print results
print('\n' + '='*70)
print('BUG HUNT RESULTS')
print('='*70)

if issues:
    print(f'\n❌ Found {len(issues)} potential issues:\n')
    for i, issue in enumerate(issues, 1):
        print(f'  {i}. {issue}')
else:
    print('\n✅ No issues found!')

if warnings:
    print(f'\n⚠️ {len(warnings)} warnings:\n')
    for w in warnings:
        print(f'  - {w}')
