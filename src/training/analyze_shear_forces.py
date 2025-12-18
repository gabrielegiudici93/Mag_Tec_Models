#!/usr/bin/env python3
"""
Analyze shear forces distribution in collected data.

This script helps understand:
1. What range of shear forces is present in the data
2. How shear forces vary across different points
3. What is the baseline/bias when pressing (no intentional shear)
4. Whether shear forces are consistent enough for regression
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_sequences_from_h5(h5_file: Path) -> List[Dict]:
    """Load sequences from HDF5 file."""
    sequences = []
    
    with h5py.File(h5_file, 'r') as f:
        # Check if it's the new format (top-level datasets + presses group)
        if 'presses' in f and 'forces' in f:
            # New format: top-level datasets + presses group
            all_forces = np.array(f['forces'])
            all_labels = f['labels']
            all_stretchmagtec = np.array(f['stretchmagtec']) if 'stretchmagtec' in f else None
            
            presses_group = f['presses']
            for press_name in sorted(presses_group.keys()):
                if press_name.startswith('press_'):
                    press_group = presses_group[press_name]
                    
                    # Get indices from attributes
                    start_idx = press_group.attrs.get('start_idx', None)
                    end_idx = press_group.attrs.get('end_idx', None)
                    label = press_group.attrs.get('label', 'unknown')
                    
                    if start_idx is not None and end_idx is not None:
                        # Extract offset from label
                        offset = 'unknown'
                        if 'pos_' in label:
                            parts = label.split('_')
                            for i, part in enumerate(parts):
                                if part == 'pos' and i + 1 < len(parts):
                                    offset = parts[i + 1]
                                    break
                        
                        # Extract forces for this press
                        forces = all_forces[start_idx:end_idx+1] if end_idx < len(all_forces) else all_forces[start_idx:]
                        stretchmagtec = all_stretchmagtec[start_idx:end_idx+1] if all_stretchmagtec is not None and end_idx < len(all_stretchmagtec) else (all_stretchmagtec[start_idx:] if all_stretchmagtec is not None else None)
                        
                        if len(forces) > 0:
                            sequences.append({
                                'label': label,
                                'offset': offset,
                                'forces': forces,
                                'stretchmagtec': stretchmagtec,
                            })
        else:
            # Old format: point groups
            for point_name in f.keys():
                if point_name.startswith('point_'):
                    point_group = f[point_name]
                    for press_name in point_group.keys():
                        if press_name.startswith('press_'):
                            press_group = point_group[press_name]
                            
                            # Get label
                            label = press_group.attrs.get('label', 'unknown')
                            
                            # Extract offset from label
                            offset = 'unknown'
                            if 'pos_' in label:
                                parts = label.split('_')
                                for i, part in enumerate(parts):
                                    if part == 'pos' and i + 1 < len(parts):
                                        offset = parts[i + 1]
                                        break
                            
                            # Load data
                            if 'shear_data' in press_group:
                                shear_group = press_group['shear_data']
                                
                                forces = np.array(shear_group['forces']) if 'forces' in shear_group else None
                                stretchmagtec = np.array(shear_group['stretchmagtec']) if 'stretchmagtec' in shear_group else None
                                
                                if forces is not None and len(forces) > 0:
                                    sequences.append({
                                        'label': label,
                                        'offset': offset,
                                        'forces': forces,
                                        'stretchmagtec': stretchmagtec,
                                    })
    
    return sequences


def analyze_shear_forces(h5_file: Path, output_dir: Path = None):
    """Analyze shear forces distribution."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING SHEAR FORCES: {h5_file.name}")
    print(f"{'='*80}")
    
    sequences = load_sequences_from_h5(h5_file)
    print(f"Loaded {len(sequences)} sequences")
    
    if len(sequences) == 0:
        print("❌ No sequences found!")
        return
    
    # Extract forces
    all_fx = []
    all_fy = []
    all_fz = []
    fx_by_point = {}
    fy_by_point = {}
    fz_by_point = {}
    
    # Also analyze forces at press start (before shear movement)
    fx_at_press_start = []
    fy_at_press_start = []
    
    for seq in sequences:
        forces = seq['forces']
        offset = seq['offset']
        
        if forces is None or len(forces) == 0:
            continue
        
        # Extract Fx, Fy, Fz
        fx = forces[:, 0]
        fy = forces[:, 1]
        fz = np.abs(forces[:, 2])  # Use absolute value for Fz
        
        all_fx.extend(fx)
        all_fy.extend(fy)
        all_fz.extend(fz)
        
        # Group by point
        if offset not in fx_by_point:
            fx_by_point[offset] = []
            fy_by_point[offset] = []
            fz_by_point[offset] = []
        
        fx_by_point[offset].extend(fx)
        fy_by_point[offset].extend(fy)
        fz_by_point[offset].extend(fz)
        
        # Get forces at press start (first 10% of samples)
        start_idx = max(1, len(fx) // 10)
        fx_at_press_start.extend(fx[:start_idx])
        fy_at_press_start.extend(fy[:start_idx])
    
    all_fx = np.array(all_fx)
    all_fy = np.array(all_fy)
    all_fz = np.array(all_fz)
    fx_at_press_start = np.array(fx_at_press_start)
    fy_at_press_start = np.array(fy_at_press_start)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Fx: min={np.min(all_fx):.4f}N, max={np.max(all_fx):.4f}N, mean={np.mean(all_fx):.4f}N, std={np.std(all_fx):.4f}N")
    print(f"Fy: min={np.min(all_fy):.4f}N, max={np.max(all_fy):.4f}N, mean={np.mean(all_fy):.4f}N, std={np.std(all_fy):.4f}N")
    print(f"Fz: min={np.min(all_fz):.4f}N, max={np.max(all_fz):.4f}N, mean={np.mean(all_fz):.4f}N, std={np.std(all_fz):.4f}N")
    
    print(f"\n{'='*80}")
    print("SHEAR FORCES AT PRESS START (baseline/bias)")
    print(f"{'='*80}")
    print(f"Fx at press start: min={np.min(fx_at_press_start):.4f}N, max={np.max(fx_at_press_start):.4f}N, mean={np.mean(fx_at_press_start):.4f}N, std={np.std(fx_at_press_start):.4f}N")
    print(f"Fy at press start: min={np.min(fy_at_press_start):.4f}N, max={np.max(fy_at_press_start):.4f}N, mean={np.mean(fy_at_press_start):.4f}N, std={np.std(fy_at_press_start):.4f}N")
    print(f"  → This represents the bias/noise when pressing without intentional shear")
    
    print(f"\n{'='*80}")
    print("STATISTICS BY POINT")
    print(f"{'='*80}")
    for offset in sorted(fx_by_point.keys()):
        fx_pt = np.array(fx_by_point[offset])
        fy_pt = np.array(fy_by_point[offset])
        fz_pt = np.array(fz_by_point[offset])
        print(f"\nPoint {offset}:")
        print(f"  Fx: min={np.min(fx_pt):.4f}N, max={np.max(fx_pt):.4f}N, mean={np.mean(fx_pt):.4f}N, std={np.std(fx_pt):.4f}N, range={np.max(fx_pt)-np.min(fx_pt):.4f}N")
        print(f"  Fy: min={np.min(fy_pt):.4f}N, max={np.max(fy_pt):.4f}N, mean={np.mean(fy_pt):.4f}N, std={np.std(fy_pt):.4f}N, range={np.max(fy_pt)-np.min(fy_pt):.4f}N")
        print(f"  Fz: min={np.min(fz_pt):.4f}N, max={np.max(fz_pt):.4f}N, mean={np.mean(fz_pt):.4f}N, std={np.std(fz_pt):.4f}N")
    
    # Generate plots
    if output_dir is None:
        output_dir = h5_file.parent / "shear_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Overall distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(all_fx, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(all_fx), color='r', linestyle='--', label=f'Mean: {np.mean(all_fx):.4f}N')
    axes[0].set_xlabel('Fx (N)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Fx Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_fy, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(all_fy), color='r', linestyle='--', label=f'Mean: {np.mean(all_fy):.4f}N')
    axes[1].set_xlabel('Fy (N)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Fy Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(all_fz, bins=50, alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(all_fz), color='r', linestyle='--', label=f'Mean: {np.mean(all_fz):.4f}N')
    axes[2].set_xlabel('Fz (N)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Fz Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"force_distributions_{h5_file.stem}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {plot_path.name}")
    
    # Plot 2: Shear forces by point
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    points = sorted(fx_by_point.keys())
    fx_means = [np.mean(np.array(fx_by_point[p])) for p in points]
    fx_stds = [np.std(np.array(fx_by_point[p])) for p in points]
    fx_mins = [np.min(np.array(fx_by_point[p])) for p in points]
    fx_maxs = [np.max(np.array(fx_by_point[p])) for p in points]
    
    fy_means = [np.mean(np.array(fy_by_point[p])) for p in points]
    fy_stds = [np.std(np.array(fy_by_point[p])) for p in points]
    fy_mins = [np.min(np.array(fy_by_point[p])) for p in points]
    fy_maxs = [np.max(np.array(fy_by_point[p])) for p in points]
    
    x_pos = np.arange(len(points))
    width = 0.35
    
    # Fx by point
    axes[0].bar(x_pos - width/2, fx_means, width, yerr=fx_stds, label='Mean ± Std', alpha=0.7, capsize=5)
    axes[0].scatter(x_pos - width/2, fx_mins, color='r', marker='_', s=100, label='Min')
    axes[0].scatter(x_pos - width/2, fx_maxs, color='r', marker='_', s=100, label='Max')
    axes[0].set_xlabel('Point', fontsize=12)
    axes[0].set_ylabel('Fx (N)', fontsize=12)
    axes[0].set_title('Fx Statistics by Point', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(points)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Fy by point
    axes[1].bar(x_pos - width/2, fy_means, width, yerr=fy_stds, label='Mean ± Std', alpha=0.7, capsize=5)
    axes[1].scatter(x_pos - width/2, fy_mins, color='r', marker='_', s=100, label='Min')
    axes[1].scatter(x_pos - width/2, fy_maxs, color='r', marker='_', s=100, label='Max')
    axes[1].set_xlabel('Point', fontsize=12)
    axes[1].set_ylabel('Fy (N)', fontsize=12)
    axes[1].set_title('Fy Statistics by Point', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(points)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plot_path = output_dir / f"shear_by_point_{h5_file.stem}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path.name}")
    
    # Plot 3: Baseline bias at press start
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(fx_at_press_start, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(fx_at_press_start), color='r', linestyle='--', label=f'Mean: {np.mean(fx_at_press_start):.4f}N')
    axes[0].axvline(np.mean(fx_at_press_start) + np.std(fx_at_press_start), color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {np.std(fx_at_press_start):.4f}N')
    axes[0].axvline(np.mean(fx_at_press_start) - np.std(fx_at_press_start), color='orange', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Fx at Press Start (N)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Fx Baseline/Bias Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(fy_at_press_start, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(fy_at_press_start), color='r', linestyle='--', label=f'Mean: {np.mean(fy_at_press_start):.4f}N')
    axes[1].axvline(np.mean(fy_at_press_start) + np.std(fy_at_press_start), color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {np.std(fy_at_press_start):.4f}N')
    axes[1].axvline(np.mean(fy_at_press_start) - np.std(fy_at_press_start), color='orange', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Fy at Press Start (N)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Fy Baseline/Bias Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"baseline_bias_{h5_file.stem}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path.name}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    fx_range = np.max(all_fx) - np.min(all_fx)
    fy_range = np.max(all_fy) - np.min(all_fy)
    fx_bias_std = np.std(fx_at_press_start)
    fy_bias_std = np.std(fy_at_press_start)
    
    print(f"\n1. SHEAR FORCE RANGE:")
    print(f"   Fx range: {fx_range:.4f}N (from {np.min(all_fx):.4f}N to {np.max(all_fx):.4f}N)")
    print(f"   Fy range: {fy_range:.4f}N (from {np.min(all_fy):.4f}N to {np.max(all_fy):.4f}N)")
    
    print(f"\n2. BASELINE BIAS:")
    print(f"   Fx bias std: {fx_bias_std:.4f}N (noise when pressing without shear)")
    print(f"   Fy bias std: {fy_bias_std:.4f}N (noise when pressing without shear)")
    
    if fx_bias_std > 0.1 or fy_bias_std > 0.1:
        print(f"\n   ⚠️  WARNING: High baseline bias detected!")
        print(f"   → Consider subtracting the mean baseline from all shear forces")
        print(f"   → Or use relative shear forces (current - baseline)")
    
    print(f"\n3. CONSISTENCY ACROSS POINTS:")
    fx_std_by_point = [np.std(np.array(fx_by_point[p])) for p in points]
    fy_std_by_point = [np.std(np.array(fy_by_point[p])) for p in points]
    fx_std_cv = np.std(fx_std_by_point) / np.mean(fx_std_by_point) if np.mean(fx_std_by_point) > 0 else 0
    fy_std_cv = np.std(fy_std_by_point) / np.mean(fy_std_by_point) if np.mean(fy_std_by_point) > 0 else 0
    
    print(f"   Fx std coefficient of variation: {fx_std_cv:.2f} ({'high' if fx_std_cv > 0.3 else 'low'} variability)")
    print(f"   Fy std coefficient of variation: {fy_std_cv:.2f} ({'high' if fy_std_cv > 0.3 else 'low'} variability)")
    
    if fx_std_cv > 0.3 or fy_std_cv > 0.3:
        print(f"\n   ⚠️  WARNING: High variability across points!")
        print(f"   → Shear forces vary significantly by contact point")
        print(f"   → Consider normalizing shear forces by point or using relative values")
    
    print(f"\n4. SUGGESTED APPROACH:")
    print(f"   a) For location classification:")
    print(f"      → Use shear forces as FEATURES (not targets)")
    print(f"      → Subtract baseline bias: Fx_corrected = Fx - mean(Fx_at_press_start)")
    print(f"      → Normalize by point if variability is high")
    print(f"   b) For force regression:")
    print(f"      → Only predict Fz (normal force) - most reliable")
    print(f"      → If predicting Fx/Fy, use relative values: Fx_rel = Fx - Fx_baseline")
    print(f"      → Or predict shear force magnitude: |F_shear| = sqrt(Fx^2 + Fy^2)")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Analyze shear forces in collected data")
    parser.add_argument('--h5-file', type=Path, required=True, help='Path to HDF5 file with shear forces data')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots (default: same as HDF5 file)')
    args = parser.parse_args()
    
    if not args.h5_file.exists():
        print(f"❌ Error: File not found: {args.h5_file}")
        return 1
    
    analyze_shear_forces(args.h5_file, args.output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

