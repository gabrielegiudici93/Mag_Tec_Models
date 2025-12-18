#!/usr/bin/env python3
"""
Find an adequate shear force threshold that doesn't cut any sequences.
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


def analyze_shear_ranges(sequences, stretch_label):
    """Analyze Fx and Fy ranges to find adequate thresholds."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING SHEAR RANGES: {stretch_label}")
    print(f"{'='*80}")
    
    all_fx = []
    all_fy = []
    all_fz = []
    
    sequences_with_exceeded_fx = []
    sequences_with_exceeded_fy = []
    
    for seq_idx, seq in enumerate(sequences):
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
        
        fx_raw = fx_raw[combined_mask]
        fy_raw = fy_raw[combined_mask]
        fz_raw = fz_raw[combined_mask]
        
        if len(fx_raw) == 0:
            continue
        
        # Cut at Fz = 3N (as we do in training)
        fz_cut_mask = fz_raw <= 3.0
        if np.any(~fz_cut_mask):
            first_exceed_idx = np.where(~fz_cut_mask)[0]
            if len(first_exceed_idx) > 0:
                cut_idx = first_exceed_idx[0]
                fx_raw = fx_raw[:cut_idx]
                fy_raw = fy_raw[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
        
        if len(fx_raw) == 0:
            continue
        
        # Check if this sequence exceeds ±2N
        fx_max = np.max(np.abs(fx_raw))
        fy_max = np.max(np.abs(fy_raw))
        
        if fx_max > 2.0:
            sequences_with_exceeded_fx.append({
                'seq_idx': seq_idx,
                'offset': seq.get('offset', 'unknown'),
                'fx_max': fx_max,
                'fx_min': np.min(fx_raw),
                'fx_range': np.max(fx_raw) - np.min(fx_raw),
            })
        
        if fy_max > 2.0:
            sequences_with_exceeded_fy.append({
                'seq_idx': seq_idx,
                'offset': seq.get('offset', 'unknown'),
                'fy_max': fy_max,
                'fy_min': np.min(fy_raw),
                'fy_range': np.max(fy_raw) - np.min(fy_raw),
            })
        
        all_fx.extend(fx_raw)
        all_fy.extend(fy_raw)
        all_fz.extend(fz_raw)
    
    all_fx = np.array(all_fx)
    all_fy = np.array(all_fy)
    all_fz = np.array(all_fz)
    
    print(f"\nOverall statistics (after Fz cut at 3N):")
    print(f"  Fx: min={np.min(all_fx):.4f}N, max={np.max(all_fx):.4f}N, range={np.max(all_fx)-np.min(all_fx):.4f}N")
    print(f"  Fy: min={np.min(all_fy):.4f}N, max={np.max(all_fy):.4f}N, range={np.max(all_fy)-np.min(all_fy):.4f}N")
    print(f"  Fz: min={np.min(all_fz):.4f}N, max={np.max(all_fz):.4f}N")
    
    print(f"\nAbsolute values:")
    print(f"  |Fx|: max={np.max(np.abs(all_fx)):.4f}N")
    print(f"  |Fy|: max={np.max(np.abs(all_fy)):.4f}N")
    
    print(f"\nSequences exceeding ±2N:")
    print(f"  Fx: {len(sequences_with_exceeded_fx)} sequences ({len(sequences_with_exceeded_fx)/len(sequences)*100:.1f}%)")
    print(f"  Fy: {len(sequences_with_exceeded_fy)} sequences ({len(sequences_with_exceeded_fy)/len(sequences)*100:.1f}%)")
    
    if sequences_with_exceeded_fx:
        fx_max_values = [s['fx_max'] for s in sequences_with_exceeded_fx]
        print(f"\n  Fx exceedances:")
        print(f"    Max |Fx|: {max(fx_max_values):.4f}N")
        print(f"    Mean |Fx| max: {np.mean(fx_max_values):.4f}N")
        print(f"    Percentiles: 95%={np.percentile(fx_max_values, 95):.4f}N, 99%={np.percentile(fx_max_values, 99):.4f}N")
        
        # Group by location
        fx_by_location = {}
        for s in sequences_with_exceeded_fx:
            loc = s['offset']
            if loc not in fx_by_location:
                fx_by_location[loc] = []
            fx_by_location[loc].append(s['fx_max'])
        
        print(f"    By location:")
        for loc in sorted(fx_by_location.keys()):
            print(f"      {loc}: max={max(fx_by_location[loc]):.4f}N, count={len(fx_by_location[loc])}")
    
    if sequences_with_exceeded_fy:
        fy_max_values = [s['fy_max'] for s in sequences_with_exceeded_fy]
        print(f"\n  Fy exceedances:")
        print(f"    Max |Fy|: {max(fy_max_values):.4f}N")
        print(f"    Mean |Fy| max: {np.mean(fy_max_values):.4f}N")
        print(f"    Percentiles: 95%={np.percentile(fy_max_values, 95):.4f}N, 99%={np.percentile(fy_max_values, 99):.4f}N")
        
        # Group by location
        fy_by_location = {}
        for s in sequences_with_exceeded_fy:
            loc = s['offset']
            if loc not in fy_by_location:
                fy_by_location[loc] = []
            fy_by_location[loc].append(s['fy_max'])
        
        print(f"    By location:")
        for loc in sorted(fy_by_location.keys()):
            print(f"      {loc}: max={max(fy_by_location[loc]):.4f}N, count={len(fy_by_location[loc])}")
    
    # Find threshold that includes all sequences
    fx_abs_max = np.max(np.abs(all_fx))
    fy_abs_max = np.max(np.abs(all_fy))
    
    # Use the maximum of both, rounded up to next 0.5N
    recommended_threshold = np.ceil(max(fx_abs_max, fy_abs_max) * 2) / 2
    
    print(f"\n{'='*80}")
    print("RECOMMENDED THRESHOLD")
    print(f"{'='*80}")
    print(f"To include ALL sequences without cutting:")
    print(f"  Max |Fx|: {fx_abs_max:.4f}N")
    print(f"  Max |Fy|: {fy_abs_max:.4f}N")
    print(f"  Recommended threshold: ±{recommended_threshold:.1f}N")
    print(f"  (This ensures no sequences are cut)")
    
    # Also check percentiles to see if we can use a lower threshold for most sequences
    fx_abs_percentiles = np.percentile(np.abs(all_fx), [90, 95, 99, 99.9])
    fy_abs_percentiles = np.percentile(np.abs(all_fy), [90, 95, 99, 99.9])
    
    print(f"\nPercentile analysis (to see if lower threshold covers most sequences):")
    print(f"  |Fx| percentiles: 90%={fx_abs_percentiles[0]:.4f}N, 95%={fx_abs_percentiles[1]:.4f}N, 99%={fx_abs_percentiles[2]:.4f}N, 99.9%={fx_abs_percentiles[3]:.4f}N")
    print(f"  |Fy| percentiles: 90%={fy_abs_percentiles[0]:.4f}N, 95%={fy_abs_percentiles[1]:.4f}N, 99%={fy_abs_percentiles[2]:.4f}N, 99.9%={fy_abs_percentiles[3]:.4f}N")
    
    return recommended_threshold, fx_abs_max, fy_abs_max


def main():
    # Analyze all stretch levels
    stretch_files = {
        '000pct': Path("data/Multiple_Points/shear_forces_test51/shear_forces_test46_shear_stretch_000pct.h5"),
        '010pct': Path("data/Multiple_Points/shear_forces_test51/shear_forces_test50_shear_stretch_010pct.h5"),
        '020pct': Path("data/Multiple_Points/shear_forces_test51/shear_forces_test51_shear_stretch_020pct.h5"),
    }
    
    thresholds = {}
    fx_maxes = {}
    fy_maxes = {}
    
    for stretch, file_path in stretch_files.items():
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Loading {stretch} from {file_path.name}...")
        sequences = load_sequences_from_h5(file_path)
        print(f"Loaded {len(sequences)} sequences")
        
        threshold, fx_max, fy_max = analyze_shear_ranges(sequences, stretch)
        thresholds[stretch] = threshold
        fx_maxes[stretch] = fx_max
        fy_maxes[stretch] = fy_max
    
    # Overall recommendation
    print(f"\n{'='*80}")
    print("OVERALL RECOMMENDATION")
    print(f"{'='*80}")
    
    if thresholds:
        overall_max_fx = max(fx_maxes.values())
        overall_max_fy = max(fy_maxes.values())
        overall_threshold = max(thresholds.values())
        
        print(f"Across all stretch levels:")
        print(f"  Max |Fx|: {overall_max_fx:.4f}N")
        print(f"  Max |Fy|: {overall_max_fy:.4f}N")
        print(f"  Recommended threshold: ±{overall_threshold:.1f}N")
        print(f"\n  This threshold will preserve ALL sequences across all stretch levels")
        
        print(f"\nPer-stretch thresholds:")
        for stretch in sorted(thresholds.keys()):
            print(f"  {stretch}: ±{thresholds[stretch]:.1f}N (Fx max: {fx_maxes[stretch]:.4f}N, Fy max: {fy_maxes[stretch]:.4f}N)")


if __name__ == '__main__':
    main()

