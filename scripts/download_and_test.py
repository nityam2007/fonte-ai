#!/usr/bin/env python3
"""
Download models from Modal volume and test generation.
"""

import modal
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD FROM MODAL
# ═══════════════════════════════════════════════════════════════════════════

def download_models():
    """Download all model files from Modal volume"""
    print("Connecting to Modal volume...")
    vol = modal.Volume.from_name("fonte-data-100")
    
    # Create local directory
    os.makedirs("TRAINED", exist_ok=True)
    
    # List files in volume
    print("\nFiles in Modal volume:")
    print("-" * 40)
    
    files_to_download = []
    for entry in vol.listdir("/"):
        print(f"  {entry.path}")
        if entry.path.endswith(".pt") or entry.path.endswith(".json"):
            files_to_download.append(entry.path)
    
    print("-" * 40)
    
    # Download each file
    for filepath in files_to_download:
        local_path = os.path.join("TRAINED", os.path.basename(filepath))
        print(f"Downloading {filepath} -> {local_path}")
        
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(filepath):
                f.write(chunk)
    
    print(f"\n✅ Downloaded {len(files_to_download)} files to TRAINED/")
    return files_to_download

if __name__ == "__main__":
    download_models()
