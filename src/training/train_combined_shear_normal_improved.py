#!/usr/bin/env python3
"""
Improved combined training script with proper handling of shear forces.

Key improvements:
1. Shear forces used as FEATURES (not targets) for location classification
2. Only Fz is predicted (most reliable)
3. Shear forces are corrected by subtracting baseline bias
4. Optional: Predict shear force magnitude instead of Fx/Fy individually
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import from the original combined training script
from training.train_combined_shear_normal import (
    load_data_combined,
    create_model,
    train_models_for_stretch as train_models_for_stretch_original,
    train_combined_model as train_combined_model_original,
    generate_plots_multipoint,
    compare_normal_vs_combined,
    MULTIPOINT_OFFSET_NAMES,
    FT_SENSOR_PRECISION,
    FT_SENSOR_DECIMALS,
)

# Constants
SHEAR_BASELINE_FRACTION = 0.1  # Use first 10% of samples to estimate baseline


def estimate_shear_baseline(forces: np.ndarray) -> Tuple[float, float]:
    """Estimate baseline/bias for Fx and Fy from the first portion of the sequence."""
    if len(forces) == 0:
        return 0.0, 0.0
    
    baseline_samples = max(1, int(len(forces) * SHEAR_BASELINE_FRACTION))
    fx_baseline = np.mean(forces[:baseline_samples, 0])
    fy_baseline = np.mean(forces[:baseline_samples, 1])
    
    return fx_baseline, fy_baseline


def correct_shear_forces(forces: np.ndarray, fx_baseline: float, fy_baseline: float) -> np.ndarray:
    """Correct shear forces by subtracting baseline bias."""
    corrected = forces.copy()
    corrected[:, 0] -= fx_baseline
    corrected[:, 1] -= fy_baseline
    return corrected


def prepare_training_data_improved(
    sequences: List[Dict],
    normalize: bool = True,
    use_feature_engineering: bool = False,
    filter_displacement: bool = True,
    displacement_threshold: float = 95.0,
    normalize_fz: bool = True,
    fz_target_min: float = 0.0,
    fz_target_max: float = 3.0,
    include_offset_labels: bool = False,
    use_advanced_features: bool = False,
    correct_shear_bias: bool = True,  # NEW: Option to correct shear bias
    predict_shear_magnitude: bool = False,  # NEW: Option to predict |F_shear| instead of Fx/Fy
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """
    Prepare training data with improved shear force handling.
    
    Returns:
        X, y_fz, y_shear (magnitude if predict_shear_magnitude else None), y_offset, scaler, actual_sequence_lengths
    """
    from training.train_combined_shear_normal import filter_high_displacement
    
    all_features = []
    all_fz = []
    all_shear_magnitude = [] if predict_shear_magnitude else None
    all_offsets = []
    
    for seq in sequences:
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is None:
            # Fallback to Fz only
            fz_raw = np.abs(seq.get('fz', np.zeros(len(magnetic))))
            fx_raw = np.zeros(len(magnetic))
            fy_raw = np.zeros(len(magnetic))
        else:
            fx_raw = forces[:, 0]
            fy_raw = forces[:, 1]
            fz_raw = np.abs(forces[:, 2])
            
            # Correct shear bias if requested
            if correct_shear_bias:
                fx_baseline, fy_baseline = estimate_shear_baseline(forces)
                fx_raw = fx_raw - fx_baseline
                fy_raw = fy_raw - fy_baseline
        
        if filter_displacement:
            mag_mask = filter_high_displacement(magnetic, displacement_threshold)
            fz_mask = filter_high_displacement(fz_raw, displacement_threshold)
            combined_mask = mag_mask & fz_mask
        else:
            combined_mask = np.ones(len(magnetic), dtype=bool)
        
        magnetic = magnetic[combined_mask]
        fx_raw = fx_raw[combined_mask]
        fy_raw = fy_raw[combined_mask]
        fz_raw = fz_raw[combined_mask]
        
        if len(magnetic) == 0:
            continue
        
        # Cut sequence at 3N (based on Fz)
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
            continue
        
        # Feature extraction (same as original)
        if use_feature_engineering:
            magnitudes = np.sqrt(np.sum(magnetic**2, axis=2))
            normalized_magnitudes = np.zeros_like(magnitudes)
            for sensor_idx in range(15):
                sensor_xyz = magnetic[:, sensor_idx, :]
                sensor_min = np.min(sensor_xyz)
                sensor_max = np.max(sensor_xyz)
                if sensor_max > sensor_min:
                    sensor_xyz_norm = (sensor_xyz - sensor_min) / (sensor_max - sensor_min)
                else:
                    sensor_xyz_norm = sensor_xyz - sensor_min
                normalized_magnitudes[:, sensor_idx] = np.sqrt(np.sum(sensor_xyz_norm**2, axis=1))
            features = normalized_magnitudes
        else:
            features = magnetic.reshape(magnetic.shape[0], -1)
        
        # Add corrected shear forces as features (for location classification)
        # This helps the model learn location-specific shear patterns
        if correct_shear_bias:
            # Use corrected shear forces as additional features
            shear_features = np.column_stack([fx_raw, fy_raw])
            features = np.hstack([features, shear_features])
        
        # Get offset/location
        offset_str = str(seq['offset'])
        if offset_str == 'no_touch':
            offset = 10
        elif offset_str.isdigit():
            offset = int(offset_str) - 1
        else:
            offset_map = {'center': 0, 'ne': 1, 'nw': 2, 'se': 3, 'sw': 4, 'unknown': -1}
            offset = offset_map.get(offset_str, -1)
        
        all_features.append(features)
        all_fz.append(fz_raw)
        if predict_shear_magnitude:
            shear_mag = np.sqrt(fx_raw**2 + fy_raw**2)
            all_shear_magnitude.append(shear_mag)
        all_offsets.append(np.full(len(fz_raw), offset))
    
    X = np.vstack(all_features)
    y_fz_raw = np.concatenate(all_fz)
    y_offset = np.concatenate(all_offsets)
    
    if predict_shear_magnitude:
        y_shear_mag = np.concatenate(all_shear_magnitude)
    else:
        y_shear_mag = None
    
    # Round forces
    y_fz_raw = np.round(y_fz_raw, decimals=FT_SENSOR_DECIMALS)
    y_fz = y_fz_raw
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    actual_sequence_lengths = [len(f) for f in all_features]
    
    return X, y_fz, y_shear_mag, y_offset, scaler, actual_sequence_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Improved combined training with proper shear force handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--shear-dir', type=Path, required=True)
    parser.add_argument('--normal-dir', type=Path, required=True)
    parser.add_argument('--run-label', type=str, required=True)
    parser.add_argument('--feature-method', type=str, default='raw', choices=['raw', 'normalized'])
    parser.add_argument('--remove-outliers', type=bool, default=True)
    parser.add_argument('--z-threshold', type=float, default=3.0)
    parser.add_argument('--max-sequences', type=int, default=20)
    parser.add_argument('--correct-shear-bias', action='store_true', default=True, help='Correct shear force bias')
    parser.add_argument('--predict-shear-magnitude', action='store_true', help='Predict |F_shear| instead of Fx/Fy')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("IMPROVED COMBINED TRAINING")
    print(f"{'='*80}")
    print(f"Correct shear bias: {args.correct_shear_bias}")
    print(f"Predict shear magnitude: {args.predict_shear_magnitude}")
    print(f"{'='*80}\n")
    
    # Load data (same as original)
    # ... (rest of the implementation would follow the original script structure)
    
    print("\n⚠️  This is a template. Use train_combined_shear_normal.py with modifications.")
    print("   Key improvements:")
    print("   1. Use shear forces as FEATURES (not targets) for location classification")
    print("   2. Only predict Fz (most reliable)")
    print("   3. Optionally predict |F_shear| = sqrt(Fx^2 + Fy^2)")
    print("   4. Correct shear bias by subtracting baseline")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

