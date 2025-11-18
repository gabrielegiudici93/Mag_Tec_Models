#!/usr/bin/env python3
"""
Generate plots for test 1 data collection.
This script generates plots for all HDF5 files in force_0to3N_step0.5N_single_test1.
"""

import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_ROOT))

from training.plot_raw_data import main as plot_main

# Test 1 directory
TEST1_DIR = CURRENT_DIR / "data" / "force_0to3N_step0.5N_single_test1"

if not TEST1_DIR.exists():
    print(f"Error: Test 1 directory not found: {TEST1_DIR}")
    sys.exit(1)

# Find all HDF5 files
h5_files = list(TEST1_DIR.glob("*.h5"))

if not h5_files:
    print(f"No HDF5 files found in {TEST1_DIR}")
    sys.exit(1)

print(f"Found {len(h5_files)} HDF5 files in test 1")
print("=" * 80)

# Generate all plots using the new structure
print("\n" + "=" * 80)
print("Generating plots with new structure...")
print("=" * 80)

original_argv = sys.argv.copy()
sys.argv = ['plot_raw_data.py', '--h5-files'] + [str(f) for f in sorted(h5_files)] + ['--output-dir', str(TEST1_DIR)]

try:
    plot_main()
    print("✓ All plots generated successfully")
except Exception as e:
    print(f"⚠️  Error generating plots: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.argv = original_argv

print("\n" + "=" * 80)
print(f"✓ All plots generated for test 1")
print(f"  Output directory: {TEST1_DIR}")
print("=" * 80)


Generate plots for test 1 data collection.
This script generates plots for all HDF5 files in force_0to3N_step0.5N_single_test1.
"""

import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_ROOT))

from training.plot_raw_data import main as plot_main

# Test 1 directory
TEST1_DIR = CURRENT_DIR / "data" / "force_0to3N_step0.5N_single_test1"

if not TEST1_DIR.exists():
    print(f"Error: Test 1 directory not found: {TEST1_DIR}")
    sys.exit(1)

# Find all HDF5 files
h5_files = list(TEST1_DIR.glob("*.h5"))

if not h5_files:
    print(f"No HDF5 files found in {TEST1_DIR}")
    sys.exit(1)

print(f"Found {len(h5_files)} HDF5 files in test 1")
print("=" * 80)

# Generate all plots using the new structure
print("\n" + "=" * 80)
print("Generating plots with new structure...")
print("=" * 80)

original_argv = sys.argv.copy()
sys.argv = ['plot_raw_data.py', '--h5-files'] + [str(f) for f in sorted(h5_files)] + ['--output-dir', str(TEST1_DIR)]

try:
    plot_main()
    print("✓ All plots generated successfully")
except Exception as e:
    print(f"⚠️  Error generating plots: {e}")
    import traceback
    traceback.print_exc()
finally:
    sys.argv = original_argv

print("\n" + "=" * 80)
print(f"✓ All plots generated for test 1")
print(f"  Output directory: {TEST1_DIR}")
print("=" * 80)







