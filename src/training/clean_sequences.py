#!/usr/bin/env python3
"""
Clean sequences from HDF5 files by removing outliers and first sequence per offset.

This script:
1. Loads sequences from HDF5 files
2. Removes the first sequence per offset (warm-up/calibration)
3. Removes 2 outliers per offset independently (using same logic as training)
4. Saves cleaned sequences to new HDF5 files with preserved structure
5. Generates plots for the cleaned dataset in a 'plots' subfolder

The cleaned files can be used directly for training without additional cleaning steps.

Usage:
    # Clean all HDF5 files in a directory
    python3 src/training/clean_sequences.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50
    
    # Clean a single HDF5 file
    python3 src/training/clean_sequences.py --h5-file data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/force_0.0to3.0N_step0.1N_single_test50_stretch_000pct.h5
    
    # Specify custom output directory
    python3 src/training/clean_sequences.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --output-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50_cleaned
    
    # Custom outlier removal parameters
    python3 src/training/clean_sequences.py --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50 --z-threshold 2.5 --remove-per-offset 3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import h5py

# Add src to path
SRC_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_ROOT))

# Import cleaning functions from train_single_point_models
from training.train_single_point_models import (
    load_sequences_from_h5 as load_sequences_base,
    remove_outliers,
    identify_outliers
)


def load_sequences_from_h5(h5_path: Path) -> List[Dict]:
    """
    Load all press sequences from an HDF5 file, including positions if available.
    Extended version that also loads positions.
    """
    sequences = []
    
    with h5py.File(h5_path, 'r') as f:
        if 'presses' not in f:
            return sequences
        
        presses = f['presses']
        for press_key in sorted(presses.keys()):
            press = presses[press_key]
            
            # Load data
            if 'forces' not in press or 'stretchmagtec' not in press:
                continue
            
            forces = press['forces'][:]  # [samples, 6]
            stretchmagtec = press['stretchmagtec'][:]  # [samples, 15, 3]
            
            # Get positions if available
            positions = None
            if 'positions' in press:
                positions = press['positions'][:]
            
            # Get metadata
            label = press.attrs.get('label', '')
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            
            stretch_level = press.attrs.get('stretch_level', np.nan)
            stretch_label = press.attrs.get('stretch_label', 'unknown')
            if isinstance(stretch_label, bytes):
                stretch_label = stretch_label.decode('utf-8')
            
            # Extract offset from label
            offset = 'unknown'
            for offset_key in ['center', 'ne', 'nw', 'sw', 'se']:
                if offset_key in label.lower():
                    offset = offset_key
                    break
            
            # Get timestamps
            if 'timestamps' in press:
                timestamps = press['timestamps'][:]
            else:
                timestamps = np.arange(len(forces)) / 100.0
            
            # Calculate statistics
            fz = forces[:, 2]  # Fz component
            duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else len(forces) / 100.0
            
            seq_dict = {
                'press_key': press_key,
                'label': label,
                'offset': offset,
                'stretch_level': float(stretch_level),
                'stretch_label': stretch_label,
                'forces': forces,
                'stretchmagtec': stretchmagtec,
                'timestamps': timestamps,
                'fz': fz,
                'duration': duration,
                'num_samples': len(forces),
                'fz_min': float(np.min(fz)),
                'fz_max': float(np.max(fz)),
                'fz_mean': float(np.mean(fz)),
                'fz_std': float(np.std(fz)),
            }
            
            # Add positions if available
            if positions is not None:
                seq_dict['positions'] = positions
            
            # Copy all other attributes from original press
            for attr_key in press.attrs.keys():
                if attr_key not in ['label', 'stretch_level', 'stretch_label']:
                    attr_value = press.attrs[attr_key]
                    if isinstance(attr_value, bytes):
                        seq_dict[attr_key] = attr_value.decode('utf-8')
                    elif isinstance(attr_value, (int, float, np.number)):
                        seq_dict[attr_key] = float(attr_value)
                    else:
                        seq_dict[attr_key] = attr_value
            
            sequences.append(seq_dict)
    
    return sequences


def save_cleaned_sequences(sequences: List[Dict], output_path: Path, original_h5_path: Path):
    """
    Save cleaned sequences to a new HDF5 file, preserving the original structure.
    
    Args:
        sequences: List of cleaned sequence dictionaries
        output_path: Path where to save the cleaned HDF5 file
        original_h5_path: Path to original HDF5 file (to copy attributes)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read original file attributes
    with h5py.File(original_h5_path, 'r') as f_orig:
        original_attrs = dict(f_orig.attrs)
    
    # Write cleaned sequences
    with h5py.File(output_path, 'w') as f:
        # Copy original file attributes
        for key, value in original_attrs.items():
            if isinstance(value, bytes):
                f.attrs[key] = value
            elif isinstance(value, str):
                f.attrs[key] = value.encode('utf-8') if isinstance(value, str) else value
            else:
                try:
                    f.attrs[key] = value
                except (TypeError, ValueError):
                    # Skip attributes that can't be written
                    pass
        
        # Create presses group
        presses_group = f.create_group("presses")
        
        # Save each cleaned sequence
        for seq_idx, seq in enumerate(sequences):
            press_key = f"press_{seq_idx:03d}"
            press_group = presses_group.create_group(press_key)
            
            # Save data
            press_group.create_dataset("forces", data=seq['forces'])
            press_group.create_dataset("stretchmagtec", data=seq['stretchmagtec'])
            
            # Save timestamps if available
            if 'timestamps' in seq and seq['timestamps'] is not None:
                press_group.create_dataset("timestamps", data=seq['timestamps'])
            
            # Save positions if available
            if 'positions' in seq and seq['positions'] is not None:
                press_group.create_dataset("positions", data=seq['positions'])
            
            # Save metadata attributes
            if 'label' in seq:
                press_group.attrs['label'] = seq['label'].encode('utf-8') if isinstance(seq['label'], str) else seq['label']
            if 'stretch_level' in seq:
                press_group.attrs['stretch_level'] = float(seq['stretch_level'])
            if 'stretch_label' in seq:
                press_group.attrs['stretch_label'] = seq['stretch_label'].encode('utf-8') if isinstance(seq['stretch_label'], str) else seq['stretch_label']
            if 'offset' in seq:
                press_group.attrs['offset'] = seq['offset'].encode('utf-8') if isinstance(seq['offset'], str) else seq['offset']
            
            # Copy all other attributes from original sequence
            for attr_key, attr_value in seq.items():
                if attr_key not in ['press_key', 'forces', 'stretchmagtec', 'timestamps', 'positions', 
                                   'fz', 'duration', 'num_samples', 'fz_min', 'fz_max', 'fz_mean', 'fz_std',
                                   'label', 'stretch_level', 'stretch_label', 'offset']:
                    try:
                        if isinstance(attr_value, str):
                            press_group.attrs[attr_key] = attr_value.encode('utf-8')
                        elif isinstance(attr_value, (int, float, np.number)):
                            press_group.attrs[attr_key] = float(attr_value)
                        elif isinstance(attr_value, (list, np.ndarray)):
                            press_group.attrs[attr_key] = str(attr_value)
                        else:
                            press_group.attrs[attr_key] = attr_value
                    except (TypeError, ValueError):
                        # Skip attributes that can't be written
                        pass
        
        # Also save continuous data arrays for backward compatibility (concatenate all sequences)
        if sequences:
            all_forces = np.vstack([seq['forces'] for seq in sequences])
            all_stretchmagtec = np.vstack([seq['stretchmagtec'] for seq in sequences])
            
            # Concatenate timestamps
            all_timestamps = []
            current_time = 0.0
            for seq in sequences:
                if 'timestamps' in seq and seq['timestamps'] is not None:
                    seq_times = seq['timestamps'] + current_time
                    all_timestamps.extend(seq_times)
                    current_time = seq_times[-1] + 0.01  # Small gap between sequences
                else:
                    # Generate timestamps if missing
                    n_samples = len(seq['forces'])
                    seq_times = np.arange(n_samples) / 100.0 + current_time
                    all_timestamps.extend(seq_times)
                    current_time = seq_times[-1] + 0.01
            
            all_timestamps = np.array(all_timestamps)
            
            # Create labels array
            all_labels = []
            for seq in sequences:
                label = seq.get('label', '')
                n_samples = len(seq['forces'])
                all_labels.extend([label.encode('utf-8') if isinstance(label, str) else label] * n_samples)
            all_labels = np.array(all_labels, dtype='S64')
            
            f.create_dataset("forces", data=all_forces)
            f.create_dataset("stretchmagtec", data=all_stretchmagtec)
            f.create_dataset("timestamps", data=all_timestamps)
            f.create_dataset("labels", data=all_labels)
            
            # Save positions if available
            if all('positions' in seq and seq['positions'] is not None for seq in sequences):
                all_positions = np.vstack([seq['positions'] for seq in sequences])
                f.create_dataset("positions", data=all_positions)
    
    print(f"  ✓ Saved cleaned sequences to: {output_path}")
    original_count = len(load_sequences_from_h5(original_h5_path))
    print(f"    Original: {original_count} sequences")
    print(f"    Cleaned: {len(sequences)} sequences")
    print(f"    Removed: {original_count - len(sequences)} sequences")
    
    return output_path


def extract_test_info(h5_path: Path) -> tuple:
    """
    Extract test ID and stretch level from HDF5 filename.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Tuple of (test_id, stretch_level) or (None, None) if not found
    """
    import re
    filename = h5_path.name
    
    # Pattern: force_0.0to3.0N_step0.1N_single_test50_stretch_000pct.h5
    # Extract test number: test50 -> 50
    test_match = re.search(r'test(\d+)', filename)
    if test_match:
        test_num = test_match.group(1)
    else:
        return None, None
    
    # Extract stretch level: stretch_000pct -> 000
    stretch_match = re.search(r'stretch_(\d+)pct', filename)
    if stretch_match:
        stretch_level = stretch_match.group(1)
    else:
        return None, None
    
    return test_num, stretch_level


def clean_single_file(h5_path: Path, output_path: Path, z_threshold: float = 3.0, remove_per_offset: int = 2):
    """
    Clean sequences from a single HDF5 file.
    
    Args:
        h5_path: Path to input HDF5 file
        output_path: Path to output cleaned HDF5 file
        z_threshold: Z-score threshold for outlier detection
        remove_per_offset: Number of outliers to remove per offset
    """
    print(f"\n{'='*80}")
    print(f"Cleaning: {h5_path.name}")
    print(f"{'='*80}")
    
    # Load sequences
    sequences = load_sequences_from_h5(h5_path)
    print(f"Loaded {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print("  ⚠️  No sequences found, skipping...")
        return
    
    # Count sequences per offset before cleaning
    initial_offset_counts = {}
    for seq in sequences:
        offset = seq.get('offset', 'unknown')
        initial_offset_counts[offset] = initial_offset_counts.get(offset, 0) + 1
    print(f"  Initial sequences per offset: {initial_offset_counts}")
    
    # NOTE: First sequence per offset was already removed during data collection
    # Do NOT remove it again here
    print(f"\n  NOTE: First sequence per offset was already removed during data collection")
    print(f"  Skipping first sequence removal (using all {len(sequences)} sequences)")
    sequences_after_first_removal = sequences
    
    # Remove outliers (2 per offset/location, independently) from sequences
    n_locations = len(initial_offset_counts)
    print(f"\n  Removing outliers ({remove_per_offset} per location, independently)...")
    print(f"  Expected after outlier removal: {len(sequences_after_first_removal)} - ({n_locations} locations * {remove_per_offset} outliers) = {len(sequences_after_first_removal) - (n_locations * remove_per_offset)} sequences")
    cleaned_sequences, outlier_indices = remove_outliers(sequences_after_first_removal, z_threshold, remove_per_offset=remove_per_offset)
    print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
    
    # Count sequences per offset to verify
    offset_counts = {}
    for seq in cleaned_sequences:
        offset = seq.get('offset', 'unknown')
        offset_counts[offset] = offset_counts.get(offset, 0) + 1
    print(f"  Sequences per location after cleaning: {offset_counts}")
    n_locations = len(offset_counts)
    if n_locations > 0:
        expected_per_location = len(sequences_after_first_removal) // n_locations - remove_per_offset
        print(f"  Expected per location: {expected_per_location} sequences")
    
    # Save cleaned sequences
    saved_path = save_cleaned_sequences(cleaned_sequences, output_path, h5_path)
    
    return cleaned_sequences, saved_path


def main():
    parser = argparse.ArgumentParser(
        description="Clean sequences from HDF5 files by removing outliers and first sequence per offset"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Directory containing HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test50)'
    )
    parser.add_argument(
        '--h5-file',
        type=Path,
        help='Single HDF5 file to clean'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for cleaned files (default: same as input with _cleaned suffix)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    parser.add_argument(
        '--remove-per-offset',
        type=int,
        default=2,
        help='Number of outliers to remove per offset (default: 2)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_cleaned',
        help='Suffix to add to cleaned file names (default: _cleaned)'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.h5_file:
        parser.error("Either --data-dir or --h5-file must be provided")
    
    print("="*80)
    print("CLEANING SEQUENCES FROM HDF5 FILES")
    print("="*80)
    
    h5_files = []
    
    if args.h5_file:
        # Single file mode
        if not args.h5_file.exists():
            print(f"❌ Error: File not found: {args.h5_file}")
            return
        
        h5_files.append(args.h5_file)
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = args.h5_file.parent / "cleaned"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        raw_plots_dir = output_dir / "raw_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract test ID and stretch level for new naming
        test_num, stretch_level = extract_test_info(args.h5_file)
        if test_num and stretch_level:
            # New format: test_50_000_cleaned.h5
            new_filename = f"test_{test_num}_{stretch_level}_cleaned.h5"
            output_path = data_dir / new_filename
        else:
            # Fallback to original naming
            output_path = data_dir / args.h5_file.name.replace('.h5', f'{args.suffix}.h5')
        
        _, saved_path = clean_single_file(args.h5_file, output_path, args.z_threshold, args.remove_per_offset)
        
        # Generate plots for cleaned file (raw plots)
        if saved_path:
            print(f"\n{'='*80}")
            print("GENERATING RAW PLOTS FOR CLEANED DATASET")
            print(f"{'='*80}")
            
            raw_plots_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                from training.plot_raw_data import main as plot_main
                import sys as plot_sys
                
                # Save original sys.argv
                original_argv = plot_sys.argv.copy()
                
                # Generate plots for single cleaned file
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-file', str(saved_path),
                    '--output-dir', str(raw_plots_dir)
                ]
                
                print(f"  Generating raw plots from cleaned file...")
                plot_main()
                print(f"\n  ✓ Raw plots generated for cleaned dataset")
                print(f"    Location: {raw_plots_dir}")
                
                # Restore original sys.argv
                plot_sys.argv = original_argv
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate plots: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.data_dir:
        # Directory mode - find all HDF5 files
        if not args.data_dir.exists():
            print(f"❌ Error: Directory not found: {args.data_dir}")
            return
        
        # Find all HDF5 files in the directory
        h5_files = list(args.data_dir.glob("*.h5"))
        
        if not h5_files:
            print(f"⚠️  No HDF5 files found in {args.data_dir}")
            return
        
        print(f"\nFound {len(h5_files)} HDF5 file(s):")
        for h5_file in sorted(h5_files):
            print(f"  - {h5_file.name}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = args.data_dir / "cleaned"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        data_dir = output_dir / "data"
        raw_plots_dir = output_dir / "raw_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean each file
        cleaned_files = []
        for h5_file in sorted(h5_files):
            # Extract test ID and stretch level for new naming
            test_num, stretch_level = extract_test_info(h5_file)
            if test_num and stretch_level:
                # New format: test_50_000_cleaned.h5
                new_filename = f"test_{test_num}_{stretch_level}_cleaned.h5"
                output_path = data_dir / new_filename
            else:
                # Fallback to original naming
                output_path = data_dir / h5_file.name.replace('.h5', f'{args.suffix}.h5')
            
            _, saved_path = clean_single_file(h5_file, output_path, args.z_threshold, args.remove_per_offset)
            if saved_path:
                cleaned_files.append(saved_path)
        
        # Generate plots for cleaned files (raw plots)
        if cleaned_files:
            print(f"\n{'='*80}")
            print("GENERATING RAW PLOTS FOR CLEANED DATASET")
            print(f"{'='*80}")
            
            raw_plots_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                from training.plot_raw_data import main as plot_main
                import sys as plot_sys
                
                # Save original sys.argv
                original_argv = plot_sys.argv.copy()
                
                # Generate plots using all cleaned files
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-files'
                ] + [str(f) for f in cleaned_files] + [
                    '--output-dir', str(raw_plots_dir)
                ]
                
                print(f"  Generating raw plots from {len(cleaned_files)} cleaned file(s)...")
                plot_main()
                print(f"\n  ✓ Raw plots generated for cleaned dataset")
                print(f"    Location: {raw_plots_dir}")
                
                # Restore original sys.argv
                plot_sys.argv = original_argv
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate plots: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"✓ Cleaning complete!")
        print(f"  Cleaned files saved to: {data_dir}")
        if cleaned_files:
            print(f"  Raw plots saved to: {raw_plots_dir}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()