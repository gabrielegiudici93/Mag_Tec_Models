#!/usr/bin/env python3
"""Simple script to inspect HDF5 file contents."""

import h5py
from pathlib import Path
import sys


def inspect_h5_file(h5_path: Path):
    """Print all keys and datasets in an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"File: {h5_path.name}")
    print(f"{'='*80}")
    
    if not h5_path.exists():
        print(f"❌ File does not exist!")
        return
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get all dataset names (labels)
            all_datasets = []
            all_groups = []
            
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append(name)
                elif isinstance(obj, h5py.Group):
                    all_groups.append(name)
            
            f.visititems(collect_names)
            
            print(f"\nTop-level keys: {list(f.keys())}")
            
            # Show unique dataset patterns
            unique_patterns = set()
            for ds_name in all_datasets:
                if '/' in ds_name:
                    pattern = '/'.join(ds_name.split('/')[:-1]) + '/*'
                    unique_patterns.add(pattern)
                else:
                    unique_patterns.add(ds_name)
            
            print(f"\nDataset structure:")
            for pattern in sorted(unique_patterns):
                count = sum(1 for ds in all_datasets if ds == pattern or ds.startswith(pattern.replace('*', '')))
                print(f"  - {pattern} ({count} datasets)")
            
            # Show first few actual dataset names as examples
            print(f"\nExample dataset names (first 5):")
            for ds_name in sorted(set(all_datasets))[:5]:
                print(f"  - {ds_name}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    # Real data (cleaned) - test50
    real_data_dir = Path("data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data")
    
    # Sim data - test5
    sim_data_dir = Path("data/simulation/test5")
    
    print("="*80)
    print("REAL DATA (Cleaned) - Test50")
    print("="*80)
    
    if real_data_dir.exists():
        h5_files = sorted(real_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {real_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {real_data_dir}")
    
    print("\n" + "="*80)
    print("SIMULATION DATA - Test5")
    print("="*80)
    
    if sim_data_dir.exists():
        h5_files = sorted(sim_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {sim_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {sim_data_dir}")


if __name__ == "__main__":
    main()



import h5py
from pathlib import Path
import sys


def inspect_h5_file(h5_path: Path):
    """Print all keys and datasets in an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"File: {h5_path.name}")
    print(f"{'='*80}")
    
    if not h5_path.exists():
        print(f"❌ File does not exist!")
        return
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get all dataset names (labels)
            all_datasets = []
            all_groups = []
            
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append(name)
                elif isinstance(obj, h5py.Group):
                    all_groups.append(name)
            
            f.visititems(collect_names)
            
            print(f"\nTop-level keys: {list(f.keys())}")
            
            # Show unique dataset patterns
            unique_patterns = set()
            for ds_name in all_datasets:
                if '/' in ds_name:
                    pattern = '/'.join(ds_name.split('/')[:-1]) + '/*'
                    unique_patterns.add(pattern)
                else:
                    unique_patterns.add(ds_name)
            
            print(f"\nDataset structure:")
            for pattern in sorted(unique_patterns):
                count = sum(1 for ds in all_datasets if ds == pattern or ds.startswith(pattern.replace('*', '')))
                print(f"  - {pattern} ({count} datasets)")
            
            # Show first few actual dataset names as examples
            print(f"\nExample dataset names (first 5):")
            for ds_name in sorted(set(all_datasets))[:5]:
                print(f"  - {ds_name}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    # Real data (cleaned) - test50
    real_data_dir = Path("data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data")
    
    # Sim data - test5
    sim_data_dir = Path("data/simulation/test5")
    
    print("="*80)
    print("REAL DATA (Cleaned) - Test50")
    print("="*80)
    
    if real_data_dir.exists():
        h5_files = sorted(real_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {real_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {real_data_dir}")
    
    print("\n" + "="*80)
    print("SIMULATION DATA - Test5")
    print("="*80)
    
    if sim_data_dir.exists():
        h5_files = sorted(sim_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {sim_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {sim_data_dir}")


if __name__ == "__main__":
    main()



import h5py
from pathlib import Path
import sys


def inspect_h5_file(h5_path: Path):
    """Print all keys and datasets in an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"File: {h5_path.name}")
    print(f"{'='*80}")
    
    if not h5_path.exists():
        print(f"❌ File does not exist!")
        return
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get all dataset names (labels)
            all_datasets = []
            all_groups = []
            
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append(name)
                elif isinstance(obj, h5py.Group):
                    all_groups.append(name)
            
            f.visititems(collect_names)
            
            print(f"\nTop-level keys: {list(f.keys())}")
            
            # Show unique dataset patterns
            unique_patterns = set()
            for ds_name in all_datasets:
                if '/' in ds_name:
                    pattern = '/'.join(ds_name.split('/')[:-1]) + '/*'
                    unique_patterns.add(pattern)
                else:
                    unique_patterns.add(ds_name)
            
            print(f"\nDataset structure:")
            for pattern in sorted(unique_patterns):
                count = sum(1 for ds in all_datasets if ds == pattern or ds.startswith(pattern.replace('*', '')))
                print(f"  - {pattern} ({count} datasets)")
            
            # Show first few actual dataset names as examples
            print(f"\nExample dataset names (first 5):")
            for ds_name in sorted(set(all_datasets))[:5]:
                print(f"  - {ds_name}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    # Real data (cleaned) - test50
    real_data_dir = Path("data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data")
    
    # Sim data - test5
    sim_data_dir = Path("data/simulation/test5")
    
    print("="*80)
    print("REAL DATA (Cleaned) - Test50")
    print("="*80)
    
    if real_data_dir.exists():
        h5_files = sorted(real_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {real_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {real_data_dir}")
    
    print("\n" + "="*80)
    print("SIMULATION DATA - Test5")
    print("="*80)
    
    if sim_data_dir.exists():
        h5_files = sorted(sim_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {sim_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {sim_data_dir}")


if __name__ == "__main__":
    main()



import h5py
from pathlib import Path
import sys


def inspect_h5_file(h5_path: Path):
    """Print all keys and datasets in an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"File: {h5_path.name}")
    print(f"{'='*80}")
    
    if not h5_path.exists():
        print(f"❌ File does not exist!")
        return
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get all dataset names (labels)
            all_datasets = []
            all_groups = []
            
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append(name)
                elif isinstance(obj, h5py.Group):
                    all_groups.append(name)
            
            f.visititems(collect_names)
            
            print(f"\nTop-level keys: {list(f.keys())}")
            
            # Show unique dataset patterns
            unique_patterns = set()
            for ds_name in all_datasets:
                if '/' in ds_name:
                    pattern = '/'.join(ds_name.split('/')[:-1]) + '/*'
                    unique_patterns.add(pattern)
                else:
                    unique_patterns.add(ds_name)
            
            print(f"\nDataset structure:")
            for pattern in sorted(unique_patterns):
                count = sum(1 for ds in all_datasets if ds == pattern or ds.startswith(pattern.replace('*', '')))
                print(f"  - {pattern} ({count} datasets)")
            
            # Show first few actual dataset names as examples
            print(f"\nExample dataset names (first 5):")
            for ds_name in sorted(set(all_datasets))[:5]:
                print(f"  - {ds_name}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    # Real data (cleaned) - test50
    real_data_dir = Path("data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data")
    
    # Sim data - test5
    sim_data_dir = Path("data/simulation/test5")
    
    print("="*80)
    print("REAL DATA (Cleaned) - Test50")
    print("="*80)
    
    if real_data_dir.exists():
        h5_files = sorted(real_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {real_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {real_data_dir}")
    
    print("\n" + "="*80)
    print("SIMULATION DATA - Test5")
    print("="*80)
    
    if sim_data_dir.exists():
        h5_files = sorted(sim_data_dir.glob("*.h5"))
        print(f"\nFound {len(h5_files)} HDF5 files in {sim_data_dir}")
        for h5_file in h5_files:
            inspect_h5_file(h5_file)
    else:
        print(f"❌ Directory does not exist: {sim_data_dir}")


if __name__ == "__main__":
    main()