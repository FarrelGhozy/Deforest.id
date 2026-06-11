#!/usr/bin/env python3
"""Runner untuk re-export data satelit via WSL"""
import ee
import sys
import subprocess
from pathlib import Path

KEY_FILE = Path("/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json")
SRC_DIR = Path("/mnt/c/Users/LABTI/Deforest.id/services/gee-export/src")
DATA_DIR = Path("/mnt/c/Users/LABTI/Deforest.id/data")

creds = ee.ServiceAccountCredentials(
    "tes1-998@form-sembako-chain.iam.gserviceaccount.com",
    str(KEY_FILE),
)
ee.Initialize(credentials=creds, project="form-sembako-chain")
print("[AUTH OK]")

for i in [1, 2, 3, 4, 5, 6, 7]:
    geojson = str(DATA_DIR / f"sample_{i}.geojson")
    prefix = f"hl_sample_{i}"
    print(f"\n{'='*60}")
    print(f"[EXPORT] Sample {i}: {prefix}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-m", "src.run", "export",
         "--geojson", geojson,
         "--prefix", prefix,
         "--wait"],
        cwd=str(SRC_DIR),
        capture_output=True, text=True, timeout=7200
    )
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr[:2000], file=sys.stderr)
    if result.returncode != 0:
        print(f"[FAIL] Sample {i}")
        sys.exit(result.returncode)
    print(f"[DONE] Sample {i}")

print("\n[ALL DONE] Semua export selesai!")
