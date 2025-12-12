#!/usr/bin/env python3
"""
Train single-point models using normalized features (15 features).

This script is a wrapper around train_single_point_models.py that automatically
uses the normalized feature method (15 features: normalized magnitude per sensor).

The normalized method:
1. For each sensor, normalizes its [x,y,z] components using its own min/max values
2. Computes the magnitude sqrt(x_norm² + y_norm² + z_norm²) from normalized components
3. Uses these 15 magnitudes (one per sensor) as features
4. Applies StandardScaler to the 15 magnitudes

Usage:
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --cleaned
"""

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import main as train_main


def main():
    parser = argparse.ArgumentParser(
        description="Train single-point models using normalized features (15 features)"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing cleaned HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned)'
    )
    parser.add_argument(
        '--cleaned',
        action='store_true',
        help='If specified, automatically append /cleaned to data-dir'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0, passed to train_single_point_models.py)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.cleaned:
        data_dir = args.data_dir / "cleaned"
    else:
        data_dir = args.data_dir
    
    # If data_dir is "cleaned", use it directly; if it contains "cleaned/data", use the parent
    if data_dir.name == "cleaned":
        # Already pointing to cleaned directory
        pass
    elif (data_dir / "data").exists():
        # Already pointing to cleaned directory with data subdirectory
        pass
    else:
        # Check if parent is cleaned
        if data_dir.parent.name == "cleaned":
            data_dir = data_dir.parent
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        print(f"   Hint: Use --cleaned flag if you want to use {args.data_dir}/cleaned")
        return 1
    
    # Check if directory contains HDF5 files (either directly or in data/ subdirectory)
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        # Try in data/ subdirectory (cleaned dataset structure)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            h5_files = list(data_subdir.glob("*.h5"))
    
    if not h5_files:
        print(f"⚠️  Warning: No HDF5 files found in {data_dir}")
        if (data_dir / "data").exists():
            print(f"   Also checked: {data_dir / 'data'}")
        print(f"   Looking for files matching pattern: *_stretch_*pct.h5 or *_cleaned.h5 or test_*_*_cleaned.h5")
        return 1
    
    print("="*80)
    print("TRAINING SINGLE-POINT MODELS - NORMALIZED FEATURES (15 features)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Feature method: normalized (15 features)")
    print(f"Found {len(h5_files)} HDF5 file(s)")
    print("="*80)
    
    # Prepare arguments for train_single_point_models.py
    # We need to modify sys.argv to pass arguments to train_main
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'train_single_point_models.py',
            '--data-dir', str(data_dir),
            '--feature-method', 'normalized',
            '--z-threshold', str(args.z_threshold)
        ]
        
        # Call the training function
        train_main()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"  Models saved to: models/")
    print(f"  Feature method: normalized (15 features)")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




Train single-point models using normalized features (15 features).

This script is a wrapper around train_single_point_models.py that automatically
uses the normalized feature method (15 features: normalized magnitude per sensor).

The normalized method:
1. For each sensor, normalizes its [x,y,z] components using its own min/max values
2. Computes the magnitude sqrt(x_norm² + y_norm² + z_norm²) from normalized components
3. Uses these 15 magnitudes (one per sensor) as features
4. Applies StandardScaler to the 15 magnitudes

Usage:
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --cleaned
"""

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import main as train_main


def main():
    parser = argparse.ArgumentParser(
        description="Train single-point models using normalized features (15 features)"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing cleaned HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned)'
    )
    parser.add_argument(
        '--cleaned',
        action='store_true',
        help='If specified, automatically append /cleaned to data-dir'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0, passed to train_single_point_models.py)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.cleaned:
        data_dir = args.data_dir / "cleaned"
    else:
        data_dir = args.data_dir
    
    # If data_dir is "cleaned", use it directly; if it contains "cleaned/data", use the parent
    if data_dir.name == "cleaned":
        # Already pointing to cleaned directory
        pass
    elif (data_dir / "data").exists():
        # Already pointing to cleaned directory with data subdirectory
        pass
    else:
        # Check if parent is cleaned
        if data_dir.parent.name == "cleaned":
            data_dir = data_dir.parent
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        print(f"   Hint: Use --cleaned flag if you want to use {args.data_dir}/cleaned")
        return 1
    
    # Check if directory contains HDF5 files (either directly or in data/ subdirectory)
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        # Try in data/ subdirectory (cleaned dataset structure)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            h5_files = list(data_subdir.glob("*.h5"))
    
    if not h5_files:
        print(f"⚠️  Warning: No HDF5 files found in {data_dir}")
        if (data_dir / "data").exists():
            print(f"   Also checked: {data_dir / 'data'}")
        print(f"   Looking for files matching pattern: *_stretch_*pct.h5 or *_cleaned.h5 or test_*_*_cleaned.h5")
        return 1
    
    print("="*80)
    print("TRAINING SINGLE-POINT MODELS - NORMALIZED FEATURES (15 features)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Feature method: normalized (15 features)")
    print(f"Found {len(h5_files)} HDF5 file(s)")
    print("="*80)
    
    # Prepare arguments for train_single_point_models.py
    # We need to modify sys.argv to pass arguments to train_main
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'train_single_point_models.py',
            '--data-dir', str(data_dir),
            '--feature-method', 'normalized',
            '--z-threshold', str(args.z_threshold)
        ]
        
        # Call the training function
        train_main()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"  Models saved to: models/")
    print(f"  Feature method: normalized (15 features)")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

Train single-point models using normalized features (15 features).

This script is a wrapper around train_single_point_models.py that automatically
uses the normalized feature method (15 features: normalized magnitude per sensor).

The normalized method:
1. For each sensor, normalizes its [x,y,z] components using its own min/max values
2. Computes the magnitude sqrt(x_norm² + y_norm² + z_norm²) from normalized components
3. Uses these 15 magnitudes (one per sensor) as features
4. Applies StandardScaler to the 15 magnitudes

Usage:
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --cleaned
"""

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import main as train_main


def main():
    parser = argparse.ArgumentParser(
        description="Train single-point models using normalized features (15 features)"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing cleaned HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned)'
    )
    parser.add_argument(
        '--cleaned',
        action='store_true',
        help='If specified, automatically append /cleaned to data-dir'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0, passed to train_single_point_models.py)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.cleaned:
        data_dir = args.data_dir / "cleaned"
    else:
        data_dir = args.data_dir
    
    # If data_dir is "cleaned", use it directly; if it contains "cleaned/data", use the parent
    if data_dir.name == "cleaned":
        # Already pointing to cleaned directory
        pass
    elif (data_dir / "data").exists():
        # Already pointing to cleaned directory with data subdirectory
        pass
    else:
        # Check if parent is cleaned
        if data_dir.parent.name == "cleaned":
            data_dir = data_dir.parent
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        print(f"   Hint: Use --cleaned flag if you want to use {args.data_dir}/cleaned")
        return 1
    
    # Check if directory contains HDF5 files (either directly or in data/ subdirectory)
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        # Try in data/ subdirectory (cleaned dataset structure)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            h5_files = list(data_subdir.glob("*.h5"))
    
    if not h5_files:
        print(f"⚠️  Warning: No HDF5 files found in {data_dir}")
        if (data_dir / "data").exists():
            print(f"   Also checked: {data_dir / 'data'}")
        print(f"   Looking for files matching pattern: *_stretch_*pct.h5 or *_cleaned.h5 or test_*_*_cleaned.h5")
        return 1
    
    print("="*80)
    print("TRAINING SINGLE-POINT MODELS - NORMALIZED FEATURES (15 features)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Feature method: normalized (15 features)")
    print(f"Found {len(h5_files)} HDF5 file(s)")
    print("="*80)
    
    # Prepare arguments for train_single_point_models.py
    # We need to modify sys.argv to pass arguments to train_main
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'train_single_point_models.py',
            '--data-dir', str(data_dir),
            '--feature-method', 'normalized',
            '--z-threshold', str(args.z_threshold)
        ]
        
        # Call the training function
        train_main()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"  Models saved to: models/")
    print(f"  Feature method: normalized (15 features)")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())




Train single-point models using normalized features (15 features).

This script is a wrapper around train_single_point_models.py that automatically
uses the normalized feature method (15 features: normalized magnitude per sensor).

The normalized method:
1. For each sensor, normalizes its [x,y,z] components using its own min/max values
2. Computes the magnitude sqrt(x_norm² + y_norm² + z_norm²) from normalized components
3. Uses these 15 magnitudes (one per sensor) as features
4. Applies StandardScaler to the 15 magnitudes

Usage:
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned
    python3 src/training/train_normalized_features.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --cleaned
"""

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import main as train_main


def main():
    parser = argparse.ArgumentParser(
        description="Train single-point models using normalized features (15 features)"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing cleaned HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned)'
    )
    parser.add_argument(
        '--cleaned',
        action='store_true',
        help='If specified, automatically append /cleaned to data-dir'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0, passed to train_single_point_models.py)'
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.cleaned:
        data_dir = args.data_dir / "cleaned"
    else:
        data_dir = args.data_dir
    
    # If data_dir is "cleaned", use it directly; if it contains "cleaned/data", use the parent
    if data_dir.name == "cleaned":
        # Already pointing to cleaned directory
        pass
    elif (data_dir / "data").exists():
        # Already pointing to cleaned directory with data subdirectory
        pass
    else:
        # Check if parent is cleaned
        if data_dir.parent.name == "cleaned":
            data_dir = data_dir.parent
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        print(f"   Hint: Use --cleaned flag if you want to use {args.data_dir}/cleaned")
        return 1
    
    # Check if directory contains HDF5 files (either directly or in data/ subdirectory)
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        # Try in data/ subdirectory (cleaned dataset structure)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            h5_files = list(data_subdir.glob("*.h5"))
    
    if not h5_files:
        print(f"⚠️  Warning: No HDF5 files found in {data_dir}")
        if (data_dir / "data").exists():
            print(f"   Also checked: {data_dir / 'data'}")
        print(f"   Looking for files matching pattern: *_stretch_*pct.h5 or *_cleaned.h5 or test_*_*_cleaned.h5")
        return 1
    
    print("="*80)
    print("TRAINING SINGLE-POINT MODELS - NORMALIZED FEATURES (15 features)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Feature method: normalized (15 features)")
    print(f"Found {len(h5_files)} HDF5 file(s)")
    print("="*80)
    
    # Prepare arguments for train_single_point_models.py
    # We need to modify sys.argv to pass arguments to train_main
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'train_single_point_models.py',
            '--data-dir', str(data_dir),
            '--feature-method', 'normalized',
            '--z-threshold', str(args.z_threshold)
        ]
        
        # Call the training function
        train_main()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"  Models saved to: models/")
    print(f"  Feature method: normalized (15 features)")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())