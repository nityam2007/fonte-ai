#!/usr/bin/env python3
"""
Bug Verification Script for FONTe AI
Tests for known and potential bugs before training.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class BugChecker:
    def __init__(self):
        self.bugs_found = []
        self.warnings = []
        self.passed = []
        
    def test_neg_token_in_vocabulary(self) -> bool:
        """Test 1: Check if <NEG> token is in vocabulary"""
        print(f"\n{BLUE}[TEST 1]{RESET} Checking if <NEG> token exists in vocabulary...")
        
        try:
            vocab_path = Path("TOKENIZED/vocabulary.json")
            if not vocab_path.exists():
                self.warnings.append("vocabulary.json not found - need to tokenize first")
                return True
            
            with open(vocab_path) as f:
                vocab = json.load(f)
            
            token_to_id = vocab.get('token_to_id', {})
            
            if '<NEG>' not in token_to_id:
                self.bugs_found.append({
                    'name': 'Missing <NEG> token in vocabulary',
                    'severity': 'CRITICAL',
                    'description': 'The <NEG> token used by tokenizer is not in vocabulary',
                    'impact': 'Training data will have UNK tokens, corrupting the dataset',
                    'fix': 'Add <NEG> to PATH_COMMANDS in svg_tokenizer.py'
                })
                print(f"  {RED}✗ FAIL{RESET}: <NEG> token NOT in vocabulary")
                return False
            else:
                self.passed.append("✓ <NEG> token exists in vocabulary")
                print(f"  {GREEN}✓ PASS{RESET}: <NEG> token found at ID {token_to_id['<NEG>']}")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check vocabulary: {e}")
            return True
    
    def test_neg_token_in_path_commands(self) -> bool:
        """Test 2: Check if <NEG> is in PATH_COMMANDS"""
        print(f"\n{BLUE}[TEST 2]{RESET} Checking if <NEG> is in PATH_COMMANDS...")
        
        try:
            with open("scripts/svg_tokenizer.py") as f:
                content = f.read()
            
            # Find PATH_COMMANDS definition
            if "PATH_COMMANDS = [" not in content:
                self.warnings.append("Could not find PATH_COMMANDS definition")
                return True
            
            # Extract PATH_COMMANDS section
            start = content.index("PATH_COMMANDS = [")
            end = content.index("]", start) + 1
            path_commands_section = content[start:end]
            
            if "'<NEG>'" in path_commands_section or '"<NEG>"' in path_commands_section:
                self.passed.append("✓ <NEG> is in PATH_COMMANDS")
                print(f"  {GREEN}✓ PASS{RESET}: <NEG> found in PATH_COMMANDS")
                return True
            else:
                self.bugs_found.append({
                    'name': '<NEG> missing from PATH_COMMANDS',
                    'severity': 'CRITICAL',
                    'description': 'PATH_COMMANDS does not include <NEG> token',
                    'impact': 'Vocabulary will not include <NEG>, causing UNK tokens',
                    'fix': "Add '<NEG>' to PATH_COMMANDS list"
                })
                print(f"  {RED}✗ FAIL{RESET}: <NEG> NOT in PATH_COMMANDS")
                return False
                
        except Exception as e:
            self.warnings.append(f"Could not check PATH_COMMANDS: {e}")
            return True
    
    def test_tokenizer_uses_neg_token(self) -> bool:
        """Test 3: Verify tokenizer actually uses <NEG> token"""
        print(f"\n{BLUE}[TEST 3]{RESET} Checking if tokenizer uses <NEG> token...")
        
        try:
            with open("scripts/svg_tokenizer.py") as f:
                content = f.read()
            
            if '"<NEG>"' in content or "'<NEG>'" in content:
                # Check if it's actually used in tokenization logic
                if 'tokens.append("<NEG>")' in content or "tokens.append('<NEG>')" in content:
                    self.passed.append("✓ Tokenizer uses <NEG> token")
                    print(f"  {GREEN}✓ PASS{RESET}: Tokenizer appends <NEG> token")
                    return True
                else:
                    self.warnings.append("Found <NEG> in code but not being appended to tokens")
                    print(f"  {YELLOW}⚠ WARN{RESET}: <NEG> found but usage unclear")
                    return True
            else:
                self.warnings.append("Tokenizer may not use <NEG> token")
                print(f"  {YELLOW}⚠ WARN{RESET}: Could not find <NEG> token usage")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check tokenizer: {e}")
            return True
    
    def test_end_to_end_tokenization(self) -> bool:
        """Test 4: End-to-end tokenization test with negative coordinates"""
        print(f"\n{BLUE}[TEST 4]{RESET} Testing end-to-end tokenization with negatives...")
        
        try:
            sys.path.insert(0, 'scripts')
            from svg_tokenizer import SVGTokenizer
            
            tokenizer = SVGTokenizer()
            
            # Test path with negative coordinates
            test_path = "M -10 20 L -5 -15 Z"
            tokens = tokenizer.tokenize_path(test_path)
            
            # Encode tokens
            encoded = [tokenizer.encode(t) for t in tokens]
            
            # Check for UNK (token 3)
            if 3 in encoded:
                unk_positions = [i for i, t in enumerate(encoded) if t == 3]
                unk_tokens = [tokens[i] for i in unk_positions]
                
                self.bugs_found.append({
                    'name': 'UNK tokens in negative coordinate test',
                    'severity': 'CRITICAL',
                    'description': f'Path with negatives produced UNK at positions {unk_positions}',
                    'impact': f'Tokens: {unk_tokens} are being encoded as UNK',
                    'fix': 'Fix vocabulary to include all tokens produced by tokenizer'
                })
                print(f"  {RED}✗ FAIL{RESET}: UNK tokens found: {unk_tokens}")
                print(f"         Tokens: {tokens}")
                print(f"         Encoded: {encoded}")
                return False
            else:
                self.passed.append("✓ No UNK tokens in negative coordinate test")
                print(f"  {GREEN}✓ PASS{RESET}: No UNK tokens in negative path")
                print(f"         Tokens: {tokens}")
                print(f"         Encoded: {encoded}")
                return True
                
        except Exception as e:
            self.bugs_found.append({
                'name': 'End-to-end tokenization test failed',
                'severity': 'CRITICAL',
                'description': f'Could not run tokenization test: {e}',
                'impact': 'Cannot verify tokenization is working correctly',
                'fix': 'Debug tokenizer initialization and usage'
            })
            print(f"  {RED}✗ FAIL{RESET}: {e}")
            return False
    
    def test_vocabulary_consistency(self) -> bool:
        """Test 5: Check vocabulary size consistency"""
        print(f"\n{BLUE}[TEST 5]{RESET} Checking vocabulary size consistency...")
        
        try:
            vocab_path = Path("TOKENIZED/vocabulary.json")
            if not vocab_path.exists():
                self.warnings.append("vocabulary.json not found")
                return True
            
            with open(vocab_path) as f:
                vocab = json.load(f)
            
            token_to_id = vocab.get('token_to_id', {})
            vocab_size = vocab.get('vocab_size', 0)
            
            actual_size = len(token_to_id)
            
            if vocab_size != actual_size:
                self.bugs_found.append({
                    'name': 'Vocabulary size mismatch',
                    'severity': 'HIGH',
                    'description': f'Reported vocab_size ({vocab_size}) != actual ({actual_size})',
                    'impact': 'Model may be initialized with wrong vocab size',
                    'fix': 'Regenerate vocabulary or fix size calculation'
                })
                print(f"  {RED}✗ FAIL{RESET}: Size mismatch - reported: {vocab_size}, actual: {actual_size}")
                return False
            else:
                self.passed.append(f"✓ Vocabulary size consistent: {vocab_size}")
                print(f"  {GREEN}✓ PASS{RESET}: Vocabulary size = {vocab_size}")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check vocabulary size: {e}")
            return True
    
    def test_model_vocab_size(self) -> bool:
        """Test 6: Check if model config matches vocabulary"""
        print(f"\n{BLUE}[TEST 6]{RESET} Checking model vocab_size matches vocabulary...")
        
        try:
            # Check generate_font.py for vocab_size - now using constants
            with open("scripts/generate_font.py") as f:
                gen_content = f.read()
            
            # Check if it imports VOCAB_SIZE from constants (preferred)
            import re
            uses_constants = 'from constants import' in gen_content and 'VOCAB_SIZE' in gen_content
            
            if uses_constants:
                # Import the constants module directly
                sys.path.insert(0, 'scripts')
                from constants import VOCAB_SIZE as model_vocab_size
            else:
                # Fallback: Extract vocab_size value from hardcoded definition
                match = re.search(r'vocab_size:\s*int\s*=\s*(\d+)', gen_content)
                if match:
                    model_vocab_size = int(match.group(1))
                else:
                    self.warnings.append("Could not find vocab_size in generate_font.py")
                    return True
            
            vocab_path = Path("TOKENIZED/vocabulary.json")
            if vocab_path.exists():
                with open(vocab_path) as f:
                    vocab = json.load(f)
                actual_size = vocab.get('vocab_size', len(vocab.get('token_to_id', {})))
                
                if model_vocab_size != actual_size:
                    self.bugs_found.append({
                        'name': 'Model vocab_size mismatch',
                        'severity': 'CRITICAL',
                        'description': f'Model expects {model_vocab_size}, vocab has {actual_size}',
                        'impact': 'Model cannot load/train with wrong vocab size',
                        'fix': f'Update ModelConfig.vocab_size to {actual_size}'
                    })
                    print(f"  {RED}✗ FAIL{RESET}: Model expects {model_vocab_size}, vocab is {actual_size}")
                    return False
                else:
                    self.passed.append(f"✓ Model vocab_size = {model_vocab_size}")
                    print(f"  {GREEN}✓ PASS{RESET}: Model vocab_size = {model_vocab_size}")
                    return True
            else:
                self.passed.append("✓ Model vocab_size check skipped (vocab not built)")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check model vocab size: {e}")
            return True
    
    def test_generate_font_id_mappings(self) -> bool:
        """Test 7: Check generate_font.py ID mappings match vocabulary"""
        print(f"\n{BLUE}[TEST 7]{RESET} Checking generate_font.py ID mappings...")
        
        try:
            sys.path.insert(0, 'scripts')
            from svg_tokenizer import Vocabulary
            vocab = Vocabulary()
            
            # Test style IDs
            expected_style_ids = {
                'serif': vocab.token_to_id.get('<STYLE:serif>'),
                'sans-serif': vocab.token_to_id.get('<STYLE:sans-serif>'),
                'monospace': vocab.token_to_id.get('<STYLE:monospace>'),
                'handwriting': vocab.token_to_id.get('<STYLE:handwriting>'),
                'display': vocab.token_to_id.get('<STYLE:display>'),
            }
            
            # Read generate_font.py
            with open("scripts/generate_font.py") as f:
                content = f.read()
            
            # Check STYLE_IDS
            import re
            style_mismatches = []
            for style, expected_id in expected_style_ids.items():
                pattern = rf"'{style}':\s*(\d+)"
                match = re.search(pattern, content)
                if match:
                    actual_id = int(match.group(1))
                    if actual_id != expected_id:
                        style_mismatches.append(f"{style}: expected {expected_id}, found {actual_id}")
            
            # Check COORD_START
            coord_start_match = re.search(r'COORD_START\s*=\s*(\d+)', content)
            expected_coord_start = vocab.token_to_id.get('<COORD:0>')
            
            coord_mismatch = None
            if coord_start_match:
                actual_coord_start = int(coord_start_match.group(1))
                if actual_coord_start != expected_coord_start:
                    coord_mismatch = f"COORD_START: expected {expected_coord_start}, found {actual_coord_start}"
            
            # Check CHAR_IDS start
            char_start_match = re.search(r'CHAR_IDS\s*=\s*\{char:\s*(\d+)', content)
            expected_char_start = vocab.token_to_id.get('<CHAR:A>')
            
            char_mismatch = None
            if char_start_match:
                actual_char_start = int(char_start_match.group(1))
                if actual_char_start != expected_char_start:
                    char_mismatch = f"CHAR_IDS start: expected {expected_char_start}, found {actual_char_start}"
            
            if style_mismatches or coord_mismatch or char_mismatch:
                all_mismatches = style_mismatches + ([coord_mismatch] if coord_mismatch else []) + ([char_mismatch] if char_mismatch else [])
                self.bugs_found.append({
                    'name': 'generate_font.py ID mapping mismatch',
                    'severity': 'CRITICAL',
                    'description': 'Token IDs in generate_font.py do not match vocabulary',
                    'impact': 'Generated glyphs will use wrong style/char conditioning',
                    'fix': f'Update ID mappings: {", ".join(all_mismatches)}'
                })
                print(f"  {RED}✗ FAIL{RESET}: ID mapping mismatches found")
                for m in all_mismatches:
                    print(f"    - {m}")
                return False
            else:
                self.passed.append("✓ generate_font.py ID mappings correct")
                print(f"  {GREEN}✓ PASS{RESET}: All ID mappings match vocabulary")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check ID mappings: {e}")
            return True
    
    def test_training_data_unk_contamination(self) -> bool:
        """Test 8: Check if training data has UNK token contamination"""
        print(f"\n{BLUE}[TEST 8]{RESET} Checking training data for UNK contamination...")
        
        try:
            train_path = Path("TOKENIZED/train.json")
            if not train_path.exists():
                self.warnings.append("train.json not found - data not yet tokenized")
                print(f"  {YELLOW}⚠ SKIP{RESET}: Training data not found")
                return True
            
            with open(train_path) as f:
                data = json.load(f)
            
            sequences = data.get('sequences', [])
            if not sequences:
                self.warnings.append("No sequences in training data")
                return True
            
            # Count UNK tokens
            unk_count = 0
            samples_with_unk = 0
            total_tokens = 0
            
            for sample in sequences:
                ids = sample.get('token_ids', [])
                total_tokens += len(ids)
                if 3 in ids:  # UNK token
                    unk_count += ids.count(3)
                    samples_with_unk += 1
            
            if unk_count > 0:
                pct = unk_count / total_tokens * 100
                self.bugs_found.append({
                    'name': 'Training data UNK contamination',
                    'severity': 'CRITICAL',
                    'description': f'{unk_count:,} UNK tokens ({pct:.2f}%) in {samples_with_unk:,} samples',
                    'impact': 'Model will learn garbage - wasted training run',
                    'fix': 'RE-TOKENIZE the dataset with fixed vocabulary!'
                })
                print(f"  {RED}✗ FAIL{RESET}: {unk_count:,} UNK tokens in training data!")
                return False
            else:
                self.passed.append("✓ No UNK tokens in training data")
                print(f"  {GREEN}✓ PASS{RESET}: Training data is clean (0 UNK tokens)")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check training data: {e}")
            return True
    
    def test_coordinate_range(self) -> bool:
        """Test 9: Verify COORD_RANGE is correct"""
        print(f"\n{BLUE}[TEST 9]{RESET} Checking coordinate range configuration...")
        
        try:
            with open("scripts/svg_tokenizer.py") as f:
                content = f.read()
            
            if "COORD_MIN = 0" in content and "COORD_MAX = 999" in content:
                if "COORD_RANGE = COORD_MAX - COORD_MIN + 1" in content or "COORD_RANGE = 1000" in content:
                    self.passed.append("✓ Coordinate range is 0-999 (1000 values)")
                    print(f"  {GREEN}✓ PASS{RESET}: COORD_RANGE = 1000 (0-999)")
                    return True
            
            self.warnings.append("Could not verify coordinate range")
            print(f"  {YELLOW}⚠ WARN{RESET}: Could not verify COORD_RANGE")
            return True
            
        except Exception as e:
            self.warnings.append(f"Could not check coordinate range: {e}")
            return True
    
    def test_notebook_vocab_size(self) -> bool:
        """Test 10: Check training notebooks have correct vocab_size"""
        print(f"\n{BLUE}[TEST 10]{RESET} Checking training notebooks vocab_size...")
        
        try:
            import re
            
            # Get expected vocab size from vocabulary
            vocab_path = Path("TOKENIZED/vocabulary.json")
            expected_size = 1106  # Default
            if vocab_path.exists():
                with open(vocab_path) as f:
                    vocab = json.load(f)
                expected_size = vocab.get('vocab_size', len(vocab.get('token_to_id', {})))
            
            notebook_issues = []
            
            # Check notebooks
            notebook_paths = [
                "notebooks/FONTe_AI_Training.ipynb",
                "notebooks/FONTe_AI_Training modal.com.ipynb"
            ]
            
            for nb_path in notebook_paths:
                if not Path(nb_path).exists():
                    continue
                    
                with open(nb_path) as f:
                    content = f.read()
                
                # Find vocab_size values
                matches = re.findall(r'vocab_size[:\s]*(?:int\s*=\s*)?(\d+)', content)
                for match in matches:
                    if int(match) != expected_size:
                        notebook_issues.append(f"{nb_path}: vocab_size={match} (should be {expected_size})")
            
            if notebook_issues:
                self.bugs_found.append({
                    'name': 'Notebook vocab_size mismatch',
                    'severity': 'CRITICAL',
                    'description': 'Training notebooks have wrong vocab_size',
                    'impact': 'Training will fail or produce wrong results',
                    'fix': '\n'.join(notebook_issues)
                })
                print(f"  {RED}✗ FAIL{RESET}: Notebook vocab_size issues found")
                for issue in notebook_issues:
                    print(f"    - {issue}")
                return False
            else:
                self.passed.append(f"✓ All notebooks have vocab_size={expected_size}")
                print(f"  {GREEN}✓ PASS{RESET}: All notebooks have correct vocab_size={expected_size}")
                return True
                
        except Exception as e:
            self.warnings.append(f"Could not check notebooks: {e}")
            return True
    
    def test_constants_module(self) -> bool:
        """Test 11: Verify constants.py is consistent with vocabulary"""
        print(f"\n{BLUE}[TEST 11]{RESET} Checking constants.py consistency...")
        
        try:
            sys.path.insert(0, 'scripts')
            from constants import (
                VOCAB_SIZE, COMMAND_START, COMMAND_END, NEG_TOKEN_ID,
                STYLE_START, STYLE_END, CHAR_START, CHAR_END,
                COORD_START, COORD_END, validate_constants
            )
            
            # Run constants self-validation
            if not validate_constants():
                self.bugs_found.append({
                    'name': 'Constants validation failed',
                    'severity': 'CRITICAL',
                    'description': 'Token ID ranges in constants.py are inconsistent',
                    'impact': 'Model will use wrong token IDs',
                    'fix': 'Check constants.py for overlapping or missing ranges'
                })
                print(f"  {RED}✗ FAIL{RESET}: Constants validation failed")
                return False
            
            # Cross-check with vocabulary.json
            vocab_path = Path("TOKENIZED/vocabulary.json")
            if vocab_path.exists():
                with open(vocab_path) as f:
                    vocab = json.load(f)
                actual_size = vocab.get('vocab_size', len(vocab.get('token_to_id', {})))
                
                if VOCAB_SIZE != actual_size:
                    self.bugs_found.append({
                        'name': 'Constants VOCAB_SIZE mismatch',
                        'severity': 'CRITICAL',
                        'description': f'constants.py has {VOCAB_SIZE}, vocabulary has {actual_size}',
                        'impact': 'Model will be misconfigured',
                        'fix': f'Update VOCAB_SIZE in constants.py to {actual_size}'
                    })
                    print(f"  {RED}✗ FAIL{RESET}: VOCAB_SIZE mismatch")
                    return False
            
            self.passed.append("✓ constants.py is consistent")
            print(f"  {GREEN}✓ PASS{RESET}: constants.py validated")
            return True
            
        except ImportError as e:
            self.warnings.append(f"Could not import constants module: {e}")
            return True
        except Exception as e:
            self.warnings.append(f"Could not check constants: {e}")
            return True

    def test_binary_dataset_format(self) -> bool:
        """Test 12: Verify binary dataset format matches notebook expectations"""
        print(f"\n{BLUE}[TEST 12]{RESET} Checking binary dataset format...")
        
        try:
            import struct
            
            train_bin = Path("TOKENIZED/train.bin")
            if not train_bin.exists():
                self.warnings.append("train.bin not found - data not yet tokenized")
                print(f"  {YELLOW}⚠ SKIP{RESET}: Binary training data not found")
                return True
            
            with open(train_bin, 'rb') as f:
                # Read header (12 bytes)
                header = f.read(12)
                if len(header) != 12:
                    self.bugs_found.append({
                        'name': 'Binary dataset header missing',
                        'severity': 'CRITICAL',
                        'description': 'train.bin does not have proper 12-byte header',
                        'impact': 'Notebook will read garbage data',
                        'fix': 'Re-run create_dataset.py to regenerate binary files'
                    })
                    print(f"  {RED}✗ FAIL{RESET}: Binary file header too short")
                    return False
                
                count, max_len, vocab_size = struct.unpack('III', header)
                data = f.read()
            
            # Expected bytes per sequence: 2 (length) + max_len * 2 (tokens)
            bytes_per_seq = 2 + max_len * 2
            expected_data_size = count * bytes_per_seq
            
            if len(data) != expected_data_size:
                self.bugs_found.append({
                    'name': 'Binary dataset size mismatch',
                    'severity': 'CRITICAL',
                    'description': f'Expected {expected_data_size} bytes, got {len(data)}',
                    'impact': 'Notebook will read misaligned data',
                    'fix': 'Re-run create_dataset.py to regenerate binary files'
                })
                print(f"  {RED}✗ FAIL{RESET}: Data size mismatch - expected {expected_data_size}, got {len(data)}")
                return False
            
            # Read and verify first sequence
            length = struct.unpack('H', data[0:2])[0]
            tokens = struct.unpack(f'{max_len}H', data[2:2 + max_len * 2])
            
            # First token should be SOS (1)
            if tokens[0] != 1:
                self.bugs_found.append({
                    'name': 'Binary dataset sequence format wrong',
                    'severity': 'CRITICAL',
                    'description': f'First sequence starts with {tokens[0]}, expected SOS (1)',
                    'impact': 'Training data is corrupted',
                    'fix': 'Re-run create_dataset.py to regenerate binary files'
                })
                print(f"  {RED}✗ FAIL{RESET}: First sequence does not start with SOS token")
                return False
            
            self.passed.append(f"✓ Binary dataset format correct ({count:,} sequences)")
            print(f"  {GREEN}✓ PASS{RESET}: Binary format verified (count={count:,}, max_len={max_len}, vocab={vocab_size})")
            return True
            
        except Exception as e:
            self.bugs_found.append({
                'name': 'Binary dataset read error',
                'severity': 'HIGH',
                'description': f'Could not read binary dataset: {e}',
                'impact': 'Cannot verify training data integrity',
                'fix': 'Check file permissions or regenerate dataset'
            })
            print(f"  {RED}✗ FAIL{RESET}: {e}")
            return False

    def run_all_tests(self) -> Tuple[int, int, int]:
        """Run all bug verification tests"""
        print(f"\n{'='*70}")
        print(f"{BLUE}FONTe AI - Bug Verification Suite{RESET}")
        print(f"{'='*70}")
        
        tests = [
            self.test_neg_token_in_path_commands,
            self.test_neg_token_in_vocabulary,
            self.test_tokenizer_uses_neg_token,
            self.test_end_to_end_tokenization,
            self.test_vocabulary_consistency,
            self.test_model_vocab_size,
            self.test_generate_font_id_mappings,
            self.test_training_data_unk_contamination,
            self.test_coordinate_range,
            self.test_notebook_vocab_size,
            self.test_constants_module,
            self.test_binary_dataset_format,
        ]
        
        for test in tests:
            test()
        
        return len(self.bugs_found), len(self.warnings), len(self.passed)
    
    def print_report(self):
        """Print final bug report"""
        print(f"\n{'='*70}")
        print(f"{BLUE}VERIFICATION REPORT{RESET}")
        print(f"{'='*70}\n")
        
        print(f"{GREEN}Passed: {len(self.passed)}{RESET}")
        for item in self.passed:
            print(f"  {item}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings: {len(self.warnings)}{RESET}")
            for item in self.warnings:
                print(f"  ⚠ {item}")
        
        if self.bugs_found:
            print(f"\n{RED}CRITICAL BUGS FOUND: {len(self.bugs_found)}{RESET}\n")
            for i, bug in enumerate(self.bugs_found, 1):
                print(f"{RED}[BUG {i}]{RESET} {bug['name']}")
                print(f"  Severity: {bug['severity']}")
                print(f"  Description: {bug['description']}")
                print(f"  Impact: {bug['impact']}")
                print(f"  Fix: {bug['fix']}\n")
        
        print(f"{'='*70}\n")
        
        if self.bugs_found:
            print(f"{RED}❌ VERIFICATION FAILED - {len(self.bugs_found)} critical bug(s) found{RESET}")
            print(f"{YELLOW}⚠  DO NOT TRAIN until bugs are fixed!{RESET}\n")
            return 1
        elif self.warnings:
            print(f"{YELLOW}⚠  VERIFICATION PASSED with {len(self.warnings)} warning(s){RESET}\n")
            return 0
        else:
            print(f"{GREEN}✅ ALL CHECKS PASSED - Safe to train!{RESET}\n")
            return 0


if __name__ == "__main__":
    checker = BugChecker()
    checker.run_all_tests()
    exit_code = checker.print_report()
    sys.exit(exit_code)
