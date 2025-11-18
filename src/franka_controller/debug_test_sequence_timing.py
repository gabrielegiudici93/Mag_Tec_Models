#!/usr/bin/env python3
"""
Debug test for sequence timing - single point, center offset, 0% stretch only.
Tests that sequence_start and sequence_end are correctly placed.
"""

import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_ROOT))

# Import config and override for debug
from franka_controller import config as config_module

# Debug directory
DEBUG_DIR = CURRENT_DIR / "data" / "debug_sequence_timing"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Override config for debug test
# Use the same position as single point test (position 32)
TARGET_POSITION_ID = 32
TARGET_POSITION_COORDS = [0.493776, 0.440500, 0.031311]  # Same coordinates as single point test

config_module.DATA_DIR = DEBUG_DIR
config_module.SELECTED_POSITIONS = [TARGET_POSITION_ID]  # Single position - same as single point test
config_module.SELECTED_OFFSETS = ['center']  # Only center offset
config_module.MAIN_GRID_POSITIONS[TARGET_POSITION_ID] = TARGET_POSITION_COORDS  # Set exact coordinates
config_module.NUMBER_OF_PRESSES = 5  # 5 presses for debugging
config_module.FORCE_CONTROLLED_PRESS = True
config_module.FORCE_MIN = 0.0
config_module.FORCE_MAX = 3.0
config_module.FORCE_STEP_SIZE = 0.5  # 0.5N steps for faster testing
config_module.FORCE_STEP_DELAY = 1.0
config_module.FORCE_TOLERANCE = 0.01

# Stretch level (0%)
config_module.CURRENT_STRETCH_VALUE = 0.0
config_module.CURRENT_STRETCH_LABEL = "stretch_000pct"

print("=" * 80)
print("DEBUG TEST - Sequence Timing (0% stretch, center only)")
print("=" * 80)
print(f"Target position ID: {TARGET_POSITION_ID}")
print(f"Target coordinates: {TARGET_POSITION_COORDS}")
print(f"Output directory: {DEBUG_DIR}")
print(f"Stretch level: 0%")
print(f"Presses: {config_module.NUMBER_OF_PRESSES}")
print(f"Force-controlled: {config_module.FORCE_CONTROLLED_PRESS}")
print(f"Force range: {config_module.FORCE_MIN} to {config_module.FORCE_MAX} N (step: {config_module.FORCE_STEP_SIZE} N)")
print("=" * 80)

try:
    # Import and run the main data collection
    from franka_controller import franka_skin_test
    
    print("\nStarting data collection...")
    franka_skin_test.main()
    
    # Find the latest HDF5 file
    h5_files = sorted(DEBUG_DIR.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h5_files:
        latest_h5 = h5_files[0]
        print(f"\n✓ Data collection complete!")
        print(f"  Latest HDF5 file: {latest_h5.name}")
        
        # Generate plots
        print("\nGenerating plots...")
        from training.plot_raw_data import main as plot_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['plot_raw_data.py', '--h5-file', str(latest_h5), '--output-dir', str(DEBUG_DIR)]
        
        try:
            plot_main()
            print(f"\n✓ Plots generated in {DEBUG_DIR}")
        finally:
            sys.argv = original_argv
    else:
        print("\n⚠️  No HDF5 files found in debug directory")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Test interrupted by user")
except Exception as e:
    print(f"\n❌ Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug test complete")
print("=" * 80)


Debug test for sequence timing - single point, center offset, 0% stretch only.
Tests that sequence_start and sequence_end are correctly placed.
"""

import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_ROOT))

# Import config and override for debug
from franka_controller import config as config_module

# Debug directory
DEBUG_DIR = CURRENT_DIR / "data" / "debug_sequence_timing"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Override config for debug test
# Use the same position as single point test (position 32)
TARGET_POSITION_ID = 32
TARGET_POSITION_COORDS = [0.493776, 0.440500, 0.031311]  # Same coordinates as single point test

config_module.DATA_DIR = DEBUG_DIR
config_module.SELECTED_POSITIONS = [TARGET_POSITION_ID]  # Single position - same as single point test
config_module.SELECTED_OFFSETS = ['center']  # Only center offset
config_module.MAIN_GRID_POSITIONS[TARGET_POSITION_ID] = TARGET_POSITION_COORDS  # Set exact coordinates
config_module.NUMBER_OF_PRESSES = 5  # 5 presses for debugging
config_module.FORCE_CONTROLLED_PRESS = True
config_module.FORCE_MIN = 0.0
config_module.FORCE_MAX = 3.0
config_module.FORCE_STEP_SIZE = 0.5  # 0.5N steps for faster testing
config_module.FORCE_STEP_DELAY = 1.0
config_module.FORCE_TOLERANCE = 0.01

# Stretch level (0%)
config_module.CURRENT_STRETCH_VALUE = 0.0
config_module.CURRENT_STRETCH_LABEL = "stretch_000pct"

print("=" * 80)
print("DEBUG TEST - Sequence Timing (0% stretch, center only)")
print("=" * 80)
print(f"Target position ID: {TARGET_POSITION_ID}")
print(f"Target coordinates: {TARGET_POSITION_COORDS}")
print(f"Output directory: {DEBUG_DIR}")
print(f"Stretch level: 0%")
print(f"Presses: {config_module.NUMBER_OF_PRESSES}")
print(f"Force-controlled: {config_module.FORCE_CONTROLLED_PRESS}")
print(f"Force range: {config_module.FORCE_MIN} to {config_module.FORCE_MAX} N (step: {config_module.FORCE_STEP_SIZE} N)")
print("=" * 80)

try:
    # Import and run the main data collection
    from franka_controller import franka_skin_test
    
    print("\nStarting data collection...")
    franka_skin_test.main()
    
    # Find the latest HDF5 file
    h5_files = sorted(DEBUG_DIR.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h5_files:
        latest_h5 = h5_files[0]
        print(f"\n✓ Data collection complete!")
        print(f"  Latest HDF5 file: {latest_h5.name}")
        
        # Generate plots
        print("\nGenerating plots...")
        from training.plot_raw_data import main as plot_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['plot_raw_data.py', '--h5-file', str(latest_h5), '--output-dir', str(DEBUG_DIR)]
        
        try:
            plot_main()
            print(f"\n✓ Plots generated in {DEBUG_DIR}")
        finally:
            sys.argv = original_argv
    else:
        print("\n⚠️  No HDF5 files found in debug directory")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Test interrupted by user")
except Exception as e:
    print(f"\n❌ Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug test complete")
print("=" * 80)









