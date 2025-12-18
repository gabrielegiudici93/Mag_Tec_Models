#!/usr/bin/env python3
"""
Analyze Fz cutting - why does it remove 85% of sequences?
"""

import sys
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_combined_shear_normal import load_sequences_from_h5, filter_high_displacement


def analyze_fz_cutting(sequences):
    """Analyze why Fz cutting removes so many sequences."""
    
    total_sequences = len(sequences)
    sequences_after_displacement_filter = 0
    sequences_after_fz_cut = 0
    sequences_completely_removed = 0
    
    fz_max_values = []
    sequences_cut_by_fz = []
    
    for seq in sequences:
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is None:
            continue
        
        fx_raw = forces[:, 0]
        fy_raw = forces[:, 1]
        fz_raw = np.abs(forces[:, 2])
        
        original_length = len(magnetic)
        
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
        
        sequences_after_displacement_filter += 1
        
        # Check Fz values
        fz_max = np.max(fz_raw)
        fz_max_values.append(fz_max)
        
        # Cut at Fz = 3N
        fz_cut_mask = fz_raw <= 3.0
        if np.any(~fz_cut_mask):
            first_exceed_idx = np.where(~fz_cut_mask)[0]
            if len(first_exceed_idx) > 0:
                cut_idx = first_exceed_idx[0]
                sequences_cut_by_fz.append({
                    'offset': seq.get('offset', 'unknown'),
                    'fz_max': fz_max,
                    'original_length': original_length,
                    'length_after_displacement': len(magnetic),
                    'cut_at': cut_idx,
                    'length_after_cut': cut_idx,
                })
                magnetic = magnetic[:cut_idx]
                fx_raw = fx_raw[:cut_idx]
                fy_raw = fy_raw[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
        
        if len(magnetic) == 0:
            sequences_completely_removed += 1
            continue
        
        sequences_after_fz_cut += 1
    
    print(f"\n{'='*80}")
    print("FZ CUTTING ANALYSIS")
    print(f"{'='*80}")
    print(f"Total sequences: {total_sequences}")
    print(f"After displacement filter: {sequences_after_displacement_filter} ({sequences_after_displacement_filter/total_sequences*100:.1f}%)")
    print(f"After Fz cut (≤3N): {sequences_after_fz_cut} ({sequences_after_fz_cut/total_sequences*100:.1f}%)")
    print(f"Completely removed: {sequences_completely_removed} ({sequences_completely_removed/total_sequences*100:.1f}%)")
    
    print(f"\nFz statistics:")
    if fz_max_values:
        print(f"  Max Fz: {np.max(fz_max_values):.4f}N")
        print(f"  Min Fz: {np.min(fz_max_values):.4f}N")
        print(f"  Mean Fz max: {np.mean(fz_max_values):.4f}N")
        print(f"  Median Fz max: {np.median(fz_max_values):.4f}N")
        print(f"  Sequences with Fz > 3N: {len([v for v in fz_max_values if v > 3.0])} ({len([v for v in fz_max_values if v > 3.0])/len(fz_max_values)*100:.1f}%)")
    
    print(f"\nSequences cut by Fz: {len(sequences_cut_by_fz)}")
    if sequences_cut_by_fz:
        cut_lengths = [s['length_after_cut'] for s in sequences_cut_by_fz]
        print(f"  Mean length after cut: {np.mean(cut_lengths):.1f} samples")
        print(f"  Min length after cut: {np.min(cut_lengths)} samples")
        print(f"  Sequences cut to 0 length: {len([s for s in sequences_cut_by_fz if s['length_after_cut'] == 0])}")
        
        # Check if sequences are cut too early
        very_short_cuts = [s for s in sequences_cut_by_fz if s['length_after_cut'] < 10]
        print(f"  Sequences cut to <10 samples: {len(very_short_cuts)}")
        
        if very_short_cuts:
            print(f"    Examples:")
            for s in very_short_cuts[:5]:
                print(f"      Location {s['offset']}: Fz_max={s['fz_max']:.4f}N, cut at {s['length_after_cut']} samples (original: {s['original_length']})")


def main():
    shear_file = Path("data/Multiple_Points/shear_forces_test51/shear_forces_test46_shear_stretch_000pct.h5")
    
    print(f"Loading sequences from {shear_file.name}...")
    sequences = load_sequences_from_h5(shear_file)
    print(f"Loaded {len(sequences)} sequences")
    
    analyze_fz_cutting(sequences)


if __name__ == '__main__':
    main()

