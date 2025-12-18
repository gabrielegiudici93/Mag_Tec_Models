#!/usr/bin/env python3
"""
Analyze why location classification performance degrades when adding shear forces.
"""

import sys
from pathlib import Path
import numpy as np
import h5py

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_combined_shear_normal import load_sequences_from_h5, filter_high_displacement


def analyze_sequence_cutting(sequences, stretch_label):
    """Analyze how sequences are cut by Fz and Fx/Fy limits."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING SEQUENCE CUTTING: {stretch_label}")
    print(f"{'='*80}")
    
    total_sequences = len(sequences)
    sequences_after_fz_cut = 0
    sequences_after_shear_cut = 0
    sequences_completely_removed = 0
    
    original_lengths = []
    after_fz_lengths = []
    after_shear_lengths = []
    
    location_counts_before = {}
    location_counts_after_fz = {}
    location_counts_after_shear = {}
    
    for seq in sequences:
        offset = seq.get('offset', 'unknown')
        location_counts_before[offset] = location_counts_before.get(offset, 0) + 1
        
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is None:
            continue
        
        fx_raw = forces[:, 0]
        fy_raw = forces[:, 1]
        fz_raw = np.abs(forces[:, 2])
        
        original_length = len(magnetic)
        original_lengths.append(original_length)
        
        # Apply displacement filter
        mag_mask = filter_high_displacement(magnetic, 95.0)
        fz_mask = filter_high_displacement(fz_raw, 95.0)
        combined_mask = mag_mask & fz_mask
        
        magnetic = magnetic[combined_mask]
        fx_raw = fx_raw[combined_mask]
        fy_raw = fy_raw[combined_mask]
        fz_raw = fz_raw[combined_mask]
        
        if len(magnetic) == 0:
            sequences_completely_removed += 1
            continue
        
        # Cut at Fz = 3N
        fz_cut_mask = fz_raw <= 3.0
        if np.any(~fz_cut_mask):
            first_exceed_idx = np.where(~fz_cut_mask)[0]
            if len(first_exceed_idx) > 0:
                cut_idx = first_exceed_idx[0]
                magnetic = magnetic[:cut_idx]
                fx_raw = fx_raw[:cut_idx]
                fy_raw = fy_raw[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
        
        if len(magnetic) == 0:
            sequences_completely_removed += 1
            continue
        
        sequences_after_fz_cut += 1
        after_fz_lengths.append(len(magnetic))
        location_counts_after_fz[offset] = location_counts_after_fz.get(offset, 0) + 1
        
        # Cut at Fx/Fy = ±2N
        fx_cut_mask = (fx_raw >= -2.0) & (fx_raw <= 2.0)
        fy_cut_mask = (fy_raw >= -2.0) & (fy_raw <= 2.0)
        combined_shear_cut_mask = fx_cut_mask & fy_cut_mask
        
        if np.any(~combined_shear_cut_mask):
            exceed_indices = np.where(~combined_shear_cut_mask)[0]
            if len(exceed_indices) > 0:
                cut_idx = exceed_indices[0]
                magnetic = magnetic[:cut_idx]
                fx_raw = fx_raw[:cut_idx]
                fy_raw = fy_raw[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
        
        if len(magnetic) == 0:
            sequences_completely_removed += 1
            continue
        
        sequences_after_shear_cut += 1
        after_shear_lengths.append(len(magnetic))
        location_counts_after_shear[offset] = location_counts_after_shear.get(offset, 0) + 1
    
    print(f"\nSequences:")
    print(f"  Original: {total_sequences}")
    print(f"  After Fz cut (≤3N): {sequences_after_fz_cut} ({sequences_after_fz_cut/total_sequences*100:.1f}%)")
    print(f"  After Fx/Fy cut (±2N): {sequences_after_shear_cut} ({sequences_after_shear_cut/total_sequences*100:.1f}%)")
    print(f"  Completely removed: {sequences_completely_removed} ({sequences_completely_removed/total_sequences*100:.1f}%)")
    
    print(f"\nSequence lengths:")
    if original_lengths:
        print(f"  Original: mean={np.mean(original_lengths):.1f}, median={np.median(original_lengths):.1f}, min={np.min(original_lengths)}, max={np.max(original_lengths)}")
    if after_fz_lengths:
        print(f"  After Fz cut: mean={np.mean(after_fz_lengths):.1f}, median={np.median(after_fz_lengths):.1f}, min={np.min(after_fz_lengths)}, max={np.max(after_fz_lengths)}")
    if after_shear_lengths:
        print(f"  After Fx/Fy cut: mean={np.mean(after_shear_lengths):.1f}, median={np.median(after_shear_lengths):.1f}, min={np.min(after_shear_lengths)}, max={np.max(after_shear_lengths)}")
    
    print(f"\nLocation distribution:")
    print(f"  Before cutting:")
    for loc in sorted(location_counts_before.keys()):
        print(f"    {loc}: {location_counts_before[loc]}")
    
    print(f"  After Fz cut:")
    for loc in sorted(location_counts_after_fz.keys()):
        print(f"    {loc}: {location_counts_after_fz[loc]}")
    
    print(f"  After Fx/Fy cut:")
    for loc in sorted(location_counts_after_shear.keys()):
        print(f"    {loc}: {location_counts_after_shear[loc]}")
    
    # Check if some locations are completely removed
    removed_locations = set(location_counts_before.keys()) - set(location_counts_after_shear.keys())
    if removed_locations:
        print(f"\n  ⚠️  WARNING: These locations were completely removed: {removed_locations}")
    
    # Check for severe imbalance
    if location_counts_after_shear:
        counts = list(location_counts_after_shear.values())
        max_count = max(counts)
        min_count = min(counts)
        if min_count == 0:
            print(f"\n  ⚠️  WARNING: Some locations have 0 sequences after cutting!")
        elif max_count / min_count > 5:
            print(f"\n  ⚠️  WARNING: Severe imbalance: max={max_count}, min={min_count}, ratio={max_count/min_count:.1f}x")


def main():
    # Load shear forces data
    shear_file = Path("data/Multiple_Points/shear_forces_test51/shear_forces_test46_shear_stretch_000pct.h5")
    
    print(f"Loading sequences from {shear_file.name}...")
    sequences = load_sequences_from_h5(shear_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Analyze cutting
    analyze_sequence_cutting(sequences, "000pct")
    
    # Also check normal forces for comparison
    normal_file = Path("data/Multiple_Points/2.5mm_single_test42/2.5mm_single_test41_stretch_000pct.h5")
    print(f"\n\nLoading normal forces from {normal_file.name}...")
    normal_sequences = load_sequences_from_h5(normal_file)
    print(f"Loaded {len(normal_sequences)} sequences")
    
    print(f"\n{'='*80}")
    print("COMPARISON: Normal forces don't have Fx/Fy cutting")
    print(f"{'='*80}")
    print(f"Normal forces sequences: {len(normal_sequences)}")
    print(f"Shear forces sequences (after all cuts): {len(sequences)}")
    
    # Check location distribution in normal vs shear
    normal_locations = {}
    for seq in normal_sequences:
        offset = seq.get('offset', 'unknown')
        normal_locations[offset] = normal_locations.get(offset, 0) + 1
    
    shear_locations = {}
    for seq in sequences:
        offset = seq.get('offset', 'unknown')
        shear_locations[offset] = shear_locations.get(offset, 0) + 1
    
    print(f"\nLocation distribution comparison:")
    print(f"  Normal forces:")
    for loc in sorted(normal_locations.keys()):
        print(f"    {loc}: {normal_locations[loc]}")
    print(f"  Shear forces (original, before cutting):")
    for loc in sorted(shear_locations.keys()):
        print(f"    {loc}: {shear_locations[loc]}")


if __name__ == '__main__':
    main()

