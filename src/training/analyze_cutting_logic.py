#!/usr/bin/env python3
"""
Analyze what happens when we cut sequences at different thresholds.
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


def analyze_cutting_at_threshold(sequences, fx_threshold, fy_threshold, stretch_label):
    """Analyze how many sequences survive when cutting at given thresholds."""
    
    total_sequences = len(sequences)
    sequences_after_fz_cut = 0
    sequences_after_shear_cut = 0
    sequences_completely_removed = 0
    
    sequences_cut_by_fx = 0
    sequences_cut_by_fy = 0
    sequences_cut_by_both = 0
    
    fx_exceedances = []
    fy_exceedances = []
    
    for seq in sequences:
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is None:
            continue
        
        fx_raw = forces[:, 0]
        fy_raw = forces[:, 1]
        fz_raw = np.abs(forces[:, 2])
        
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
        
        # Check Fx/Fy exceedances
        fx_exceeds = np.any((fx_raw < -fx_threshold) | (fx_raw > fx_threshold))
        fy_exceeds = np.any((fy_raw < -fy_threshold) | (fy_raw > fy_threshold))
        
        if fx_exceeds:
            fx_exceedances.append({
                'offset': seq.get('offset', 'unknown'),
                'fx_max': np.max(np.abs(fx_raw)),
                'fx_min': np.min(fx_raw),
                'fx_max_abs': np.max(np.abs(fx_raw)),
            })
            sequences_cut_by_fx += 1
        
        if fy_exceeds:
            fy_exceedances.append({
                'offset': seq.get('offset', 'unknown'),
                'fy_max': np.max(np.abs(fy_raw)),
                'fy_min': np.min(fy_raw),
                'fy_max_abs': np.max(np.abs(fy_raw)),
            })
            sequences_cut_by_fy += 1
        
        if fx_exceeds and fy_exceeds:
            sequences_cut_by_both += 1
        
        # Cut at Fx/Fy thresholds
        fx_cut_mask = (fx_raw >= -fx_threshold) & (fx_raw <= fx_threshold)
        fy_cut_mask = (fy_raw >= -fy_threshold) & (fy_raw <= fy_threshold)
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
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS: Cutting at Fx=±{fx_threshold}N, Fy=±{fy_threshold}N")
    print(f"{'='*80}")
    print(f"Total sequences: {total_sequences}")
    print(f"After Fz cut (≤3N): {sequences_after_fz_cut} ({sequences_after_fz_cut/total_sequences*100:.1f}%)")
    print(f"After Fx/Fy cut: {sequences_after_shear_cut} ({sequences_after_shear_cut/total_sequences*100:.1f}%)")
    print(f"Completely removed: {sequences_completely_removed} ({sequences_completely_removed/total_sequences*100:.1f}%)")
    print(f"\nExceedances:")
    print(f"  Sequences with Fx exceeding ±{fx_threshold}N: {sequences_cut_by_fx} ({sequences_cut_by_fx/sequences_after_fz_cut*100:.1f}% of Fz-surviving)")
    print(f"  Sequences with Fy exceeding ±{fy_threshold}N: {sequences_cut_by_fy} ({sequences_cut_by_fy/sequences_after_fz_cut*100:.1f}% of Fz-surviving)")
    print(f"  Sequences with both exceeding: {sequences_cut_by_both}")
    
    if fx_exceedances:
        fx_max_abs_values = [e['fx_max_abs'] for e in fx_exceedances]
        print(f"\n  Fx exceedances:")
        print(f"    Max |Fx|: {max(fx_max_abs_values):.4f}N")
        print(f"    Mean |Fx| max: {np.mean(fx_max_abs_values):.4f}N")
        print(f"    Min |Fx| max: {min(fx_max_abs_values):.4f}N")
    
    if fy_exceedances:
        fy_max_abs_values = [e['fy_max_abs'] for e in fy_exceedances]
        print(f"\n  Fy exceedances:")
        print(f"    Max |Fy|: {max(fy_max_abs_values):.4f}N")
        print(f"    Mean |Fy| max: {np.mean(fy_max_abs_values):.4f}N")
        print(f"    Min |Fy| max: {min(fy_max_abs_values):.4f}N")
    
    return sequences_after_shear_cut, sequences_completely_removed


def main():
    shear_file = Path("data/Multiple_Points/shear_forces_test51/shear_forces_test46_shear_stretch_000pct.h5")
    
    print(f"Loading sequences from {shear_file.name}...")
    sequences = load_sequences_from_h5(shear_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Test different thresholds
    thresholds = [
        (2.0, 2.0),
        (2.5, 2.5),
        (3.0, 3.0),
        (4.0, 4.0),
    ]
    
    results = []
    for fx_thresh, fy_thresh in thresholds:
        survived, removed = analyze_cutting_at_threshold(sequences, fx_thresh, fy_thresh, "000pct")
        results.append((fx_thresh, survived, removed))
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Threshold':<15} {'Sequences Survived':<20} {'Sequences Removed':<20} {'Survival Rate':<15}")
    print("-" * 80)
    for fx_thresh, survived, removed in results:
        total = len(sequences)
        survival_rate = survived / total * 100
        print(f"±{fx_thresh:.1f}N{'':<10} {survived:<20} {removed:<20} {survival_rate:.1f}%")


if __name__ == '__main__':
    main()

