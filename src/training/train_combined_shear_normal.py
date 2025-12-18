#!/usr/bin/env python3
"""
Combined training script for shear forces + normal forces.

This script combines:
- Shear forces data (from shear_forces_test51)
- Normal forces data (from 2.5mm_single_test42)

Features:
- Force regression (Fz prediction)
- Location classification (10 positions: '1' through '10')
- Uses 20 sequences per stretch level (excluding first sequence and outliers)
- 70/30 split per stretch level group
- Focuses on RMSE metrics

Usage:
    python3 src/training/train_combined_shear_normal.py \
        --shear-dir data/Multiple_Points/shear_forces_test51 \
        --normal-dir data/Multiple_Points/2.5mm_single_test42 \
        --run-label combined_shear_normal \
        --feature-method raw
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
from sklearn.metrics import ConfusionMatrixDisplay, r2_score
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Multi-point specific configuration
MULTIPOINT_OFFSET_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'no_touch']
MULTIPOINT_N_OFFSETS = 11  # 10 locations + no_touch

# FT sensor precision constants
FT_SENSOR_PRECISION = 0.01  # N (FT-300S sensor resolution)
FT_SENSOR_DECIMALS = 2  # Decimal places for rounding


def load_sequences_from_h5(h5_path: Path) -> List[Dict]:
    """Load all press sequences from an HDF5 file (independent implementation for multi-point)."""
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
            
            # Get metadata
            label = press.attrs.get('label', '')
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            
            stretch_level = press.attrs.get('stretch_level', np.nan)
            stretch_label = press.attrs.get('stretch_label', 'unknown')
            if isinstance(stretch_label, bytes):
                stretch_label = stretch_label.decode('utf-8')
            
            # Extract offset/location from attribute first (preferred), then from label
            offset = press.attrs.get('offset', None)
            if offset is not None:
                if isinstance(offset, bytes):
                    offset = offset.decode('utf-8')
                # Ensure offset is a string
                offset = str(offset)
            else:
                # Fall back to extracting from label
                offset = 'unknown'
                import re
                
                # First check for no_touch
                if 'no_touch' in label.lower():
                    offset = 'no_touch'
                # Then check for numeric offsets (1-10) for multi-point data
                elif numeric_match := re.search(r'offset[_\s]*(\d+)|pos_\d+_(\d+)_press', label.lower()):
                    offset = numeric_match.group(1) or numeric_match.group(2)
                # Finally check for named offsets (center, ne, nw, sw, se) for single-point data
                else:
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
            
            sequences.append({
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
            })
    
    return sequences


def identify_outliers(sequences: List[Dict], z_threshold: float = 3.0) -> List[int]:
    """Identify outlier sequences using statistical methods (independent implementation)."""
    if len(sequences) < 3:
        return []
    
    durations = np.array([s['duration'] for s in sequences])
    fz_maxes = np.array([s['fz_max'] for s in sequences])
    num_samples = np.array([s['num_samples'] for s in sequences])
    
    outlier_indices = set()
    
    for feature_name, feature_values in [
        ('duration', durations),
        ('fz_max', fz_maxes),
        ('num_samples', num_samples),
    ]:
        median = np.median(feature_values)
        mad = np.median(np.abs(feature_values - median))
        
        if mad > 0:
            threshold = z_threshold if feature_name != 'duration' else 2.0
            modified_z_scores = 0.6745 * (feature_values - median) / mad
            outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
            if len(outliers) > 0 and feature_name == 'duration':
                outlier_durations = feature_values[outliers]
                print(f"    Found {len(outliers)} duration outliers: median={median:.2f}s, outliers={outlier_durations}")
            outlier_indices.update(outliers)
    
    return sorted(list(outlier_indices))


def remove_outliers(sequences: List[Dict], z_threshold: float = 3.0, remove_per_offset: int = 2) -> Tuple[List[Dict], List[int]]:
    """Remove outlier sequences and return cleaned list (independent implementation)."""
    if remove_per_offset <= 0:
        return sequences, []
    
    # Group sequences by offset/location
    sequences_by_offset = {}
    for idx, seq in enumerate(sequences):
        offset = seq.get('offset', 'unknown')
        if offset not in sequences_by_offset:
            sequences_by_offset[offset] = []
        sequences_by_offset[offset].append((idx, seq))
    
    all_outlier_indices = set()
    
    for offset, offset_sequences in sequences_by_offset.items():
        if len(offset_sequences) <= remove_per_offset:
            print(f"  Location {offset}: Only {len(offset_sequences)} sequences, skipping outlier removal")
            continue
        
        offset_seq_list = [seq for _, seq in offset_sequences]
        outlier_indices_local = identify_outliers(offset_seq_list, z_threshold)
        
        # Calculate outlier scores
        offset_seq_list_with_scores = []
        fz_values = [s.get('fz_max', 0) for s in offset_seq_list]
        duration_values = [s.get('duration', 0) for s in offset_seq_list]
        num_samples_values = [s.get('num_samples', 0) for s in offset_seq_list]
        
        fz_median = np.median(fz_values)
        fz_mad = np.median(np.abs(np.array(fz_values) - fz_median)) if len(fz_values) > 0 else 1.0
        duration_median = np.median(duration_values)
        duration_mad = np.median(np.abs(np.array(duration_values) - duration_median)) if len(duration_values) > 0 else 1.0
        num_samples_median = np.median(num_samples_values)
        num_samples_mad = np.median(np.abs(np.array(num_samples_values) - num_samples_median)) if len(num_samples_values) > 0 else 1.0
        
        for local_idx, (global_idx, seq) in enumerate(offset_sequences):
            fz = seq.get('fz_max', 0)
            duration = seq.get('duration', 0)
            num_samples = seq.get('num_samples', 0)
            
            score = 0
            if fz_mad > 0:
                score += abs((fz - fz_median) / fz_mad)
            if duration_mad > 0:
                duration_score = abs((duration - duration_median) / duration_mad)
                if duration > duration_median * 2.0:
                    duration_score *= 2.0
                score += duration_score * 1.5
            if num_samples_mad > 0:
                score += abs((num_samples - num_samples_median) / num_samples_mad)
            
            is_identified_outlier = local_idx in outlier_indices_local
            if is_identified_outlier:
                score *= 2.0
            
            offset_seq_list_with_scores.append((global_idx, score, is_identified_outlier))
        
        offset_seq_list_with_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        num_to_remove = min(remove_per_offset, len(offset_seq_list_with_scores))
        outliers_to_remove = [idx for idx, _, _ in offset_seq_list_with_scores[:num_to_remove]]
        
        all_outlier_indices.update(outliers_to_remove)
        print(f"  Location {offset}: Removing {len(outliers_to_remove)} outliers (indices: {sorted(outliers_to_remove)})")
    
    if all_outlier_indices:
        print(f"  Total outliers removed: {len(all_outlier_indices)} (indices: {sorted(all_outlier_indices)})")
        cleaned_sequences = [s for i, s in enumerate(sequences) if i not in all_outlier_indices]
    else:
        cleaned_sequences = sequences
    
    return cleaned_sequences, sorted(list(all_outlier_indices))


def balance_sequences(sequences_by_stretch: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Balance number of sequences across stretch levels by taking minimum (independent implementation)."""
    if not sequences_by_stretch:
        return sequences_by_stretch
    
    # Don't balance - keep all sequences to ensure all stretch levels are present
    return sequences_by_stretch


def filter_high_displacement(data: np.ndarray, threshold_percentile: float = 95.0) -> np.ndarray:
    """Filter out samples with high consecutive displacement (independent implementation)."""
    if len(data) < 2:
        return np.ones(len(data), dtype=bool)
    
    if data.ndim == 1:
        displacements = np.abs(np.diff(data))
    elif data.ndim == 2:
        displacements = np.linalg.norm(np.diff(data, axis=0), axis=1)
    elif data.ndim == 3:
        data_flat = data.reshape(len(data), -1)
        displacements = np.linalg.norm(np.diff(data_flat, axis=0), axis=1)
    else:
        data_flat = data.reshape(len(data), -1)
        displacements = np.linalg.norm(np.diff(data_flat, axis=0), axis=1)
    
    threshold = np.percentile(displacements, threshold_percentile)
    mask = np.ones(len(data), dtype=bool)
    mask[1:] = displacements < threshold
    
    return mask


def normalize_fz_to_range(fz: np.ndarray, target_min: float = 0.0, target_max: float = 3.0) -> np.ndarray:
    """Normalize Fz values to target range (independent implementation)."""
    fz_min = np.min(fz)
    fz_max = np.max(fz)
    
    if fz_max - fz_min < 1e-6:
        return np.full_like(fz, (target_min + target_max) / 2.0)
    
    normalized = (fz - fz_min) / (fz_max - fz_min) * (target_max - target_min) + target_min
    return normalized


def prepare_training_data(sequences: List[Dict], normalize: bool = True, use_feature_engineering: bool = False, 
                          filter_displacement: bool = True, displacement_threshold: float = 95.0,
                          normalize_fz: bool = True, fz_target_min: float = 0.0, fz_target_max: float = 3.0,
                          include_offset_labels: bool = False, use_advanced_features: bool = False,
                          cut_shear_for_regression: bool = True,
                          location_feature_method: str = 'raw',
                          remove_fz_baseline: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """
    Prepare features (magnetic), targets (Fx, Fy, Fz), and offsets from sequences.
    
    Args:
        cut_shear_for_regression: If True, cut Fx/Fy at ±2N (only affects regression targets, not features for classification)
        remove_fz_baseline: If False, do NOT remove baseline for Fz (use absolute values like old code). Default True for shear forces.
    """
    all_features = []
    all_fx = []
    all_fy = []
    all_fz = []
    all_offsets = []
    
    for seq in sequences:
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is not None:
            # Extract Fx, Fy, Fz (forces[:, 0], forces[:, 1], forces[:, 2])
            fx_raw = forces[:, 0] if forces.shape[1] >= 3 else np.zeros(len(magnetic))
            fy_raw = forces[:, 1] if forces.shape[1] >= 3 else np.zeros(len(magnetic))
            fz_raw = np.abs(forces[:, 2]) if forces.shape[1] >= 3 else np.abs(seq.get('fz', np.zeros(len(magnetic))))
        else:
            fx_raw = np.zeros(len(magnetic))
            fy_raw = np.zeros(len(magnetic))
            fz_raw = np.abs(seq.get('fz', np.zeros(len(magnetic))))
        
        # Calculate baseline BEFORE filtering (use first 10% of samples)
        # This gives us the VARIATION from initial state, not absolute values
        baseline_samples = max(1, int(len(magnetic) * 0.1))
        fx_baseline = np.mean(fx_raw[:baseline_samples]) if len(fx_raw) > 0 else 0.0
        fy_baseline = np.mean(fy_raw[:baseline_samples]) if len(fy_raw) > 0 else 0.0
        fz_baseline = np.mean(fz_raw[:baseline_samples]) if len(fz_raw) > 0 else 0.0
        
        # Apply displacement filter BEFORE removing baseline (to preserve baseline samples)
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
        
        # Now remove baseline to get relative forces (variation from initial state)
        # This is done AFTER filtering to ensure we use the same baseline for all samples
        # IMPORTANT: For Fz regression with normal forces, do NOT remove baseline (like old code)
        fx_raw = fx_raw - fx_baseline
        fy_raw = fy_raw - fy_baseline
        if remove_fz_baseline:
            fz_raw = fz_raw - fz_baseline
        # If remove_fz_baseline=False, keep absolute values (like train_multipoint.py)
        
        # CORRECT Fx inversion: FT sensor measures force FROM robot TO sensor (action-reaction)
        # When robot moves x+, sensor feels force in x- direction, so Fx is inverted
        # Extract direction from label if available
        label = seq.get('label', '')
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        
        # Check if this is a shear force sequence with direction
        if 'shear' in label.lower():
            if 'x+' in label or 'x-' in label:
                # Invert Fx for x-direction movements
                fx_raw = -fx_raw
                # Debug print for first few sequences
                if len(all_features) < 3:
                    print(f"  [DEBUG] Inverted Fx for sequence: {label[:50]}")
        
        if len(magnetic) == 0:
            continue
        
        # Cut sequence at 3N variation for Fz (keep only Fz variation <= 3N)
        # Since we removed the baseline, this is now the variation from initial state
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
        
        # Clip Fx/Fy variations to ±2N for regression targets
        # These are now variations from initial state, so ±2N is reasonable
        fx_raw = np.clip(fx_raw, -2.0, 2.0)
        fy_raw = np.clip(fy_raw, -2.0, 2.0)
        
        # Extract features based on method
        if use_feature_engineering:
            # Normalized magnitudes (existing method)
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
        elif location_feature_method == 'magnitude':
            # Use magnitude (norm) of each taxel: sqrt(x^2 + y^2 + z^2) for each of 15 sensors
            # Shape: [samples, 15] - one magnitude per taxel
            features = np.sqrt(np.sum(magnetic**2, axis=2))
        elif location_feature_method == 'magnitude_normalized':
            # Magnitude normalized per sensor
            magnitudes = np.sqrt(np.sum(magnetic**2, axis=2))  # [samples, 15]
            normalized_magnitudes = np.zeros_like(magnitudes)
            for sensor_idx in range(15):
                sensor_mag = magnitudes[:, sensor_idx]
                mag_min = np.min(sensor_mag)
                mag_max = np.max(sensor_mag)
                if mag_max > mag_min:
                    normalized_magnitudes[:, sensor_idx] = (sensor_mag - mag_min) / (mag_max - mag_min)
                else:
                    normalized_magnitudes[:, sensor_idx] = sensor_mag - mag_min
            features = normalized_magnitudes
        else:
            # Raw features (default): [samples, 45] - all xyz values flattened
            features = magnetic.reshape(magnetic.shape[0], -1)
        
        # Get offset/location (encode as integer)
        offset_str = str(seq['offset'])
        if offset_str == 'no_touch':
            # No-touch class: 10 (11 classes total: 0-10)
            offset = 10
        elif offset_str.isdigit():
            # Multi-point: '1', '2', ..., '10' -> 0, 1, ..., 9
            offset = int(offset_str) - 1
        else:
            # Single-point: named offsets
            offset_map = {'center': 0, 'ne': 1, 'nw': 2, 'se': 3, 'sw': 4, 'unknown': -1}
            offset = offset_map.get(offset_str, -1)
        
        if use_advanced_features:
            statistical_features = []
            for sensor_idx in range(15):
                sensor_xyz = magnetic[:, sensor_idx, :]
                sensor_mean = np.mean(sensor_xyz, axis=1)
                sensor_std = np.std(sensor_xyz, axis=1)
                sensor_max = np.max(sensor_xyz, axis=1)
                sensor_min = np.min(sensor_xyz, axis=1)
                statistical_features.append(np.column_stack([sensor_mean, sensor_std, sensor_max, sensor_min]))
            statistical_features = np.hstack(statistical_features)
            
            magnetic_flat = magnetic.reshape(magnetic.shape[0], -1)
            if len(magnetic_flat) > 1:
                diff_1 = np.diff(magnetic_flat, axis=0)
                derivative_1 = np.vstack([diff_1[0:1], diff_1])
            else:
                derivative_1 = np.zeros_like(magnetic_flat)
            
            if len(derivative_1) > 1:
                diff_2 = np.diff(derivative_1, axis=0)
                derivative_2 = np.vstack([diff_2[0:1], diff_2])
            else:
                derivative_2 = np.zeros_like(magnetic_flat)
            
            temporal_features = np.hstack([derivative_1, derivative_2])
            features = np.hstack([features, statistical_features, temporal_features])
        
        all_features.append(features)
        all_fx.append(fx_raw)
        all_fy.append(fy_raw)
        all_fz.append(fz_raw)
        all_offsets.append(np.full(len(fz_raw), offset))
    
    X = np.vstack(all_features)
    y_fx_raw = np.concatenate(all_fx)
    y_fy_raw = np.concatenate(all_fy)
    y_fz_raw = np.concatenate(all_fz)
    y_offset = np.concatenate(all_offsets)
    
    if include_offset_labels:
        from sklearn.preprocessing import OneHotEncoder
        valid_mask = y_offset >= 0
        # Dynamically determine categories based on unique offsets
        unique_offsets = np.unique(y_offset[valid_mask])
        if len(unique_offsets) > 0:
            offset_encoder = OneHotEncoder(sparse_output=False, categories=[unique_offsets], handle_unknown='ignore')
            offset_onehot = np.zeros((len(y_offset), len(unique_offsets)))
            offset_onehot[valid_mask] = offset_encoder.fit_transform(y_offset[valid_mask].reshape(-1, 1))
            X = np.hstack([X, offset_onehot])
            print(f"  Added offset labels as one-hot features: {offset_onehot.shape[1]} additional features")
    
    # Round forces to sensor precision
    y_fx_raw = np.round(y_fx_raw, decimals=FT_SENSOR_DECIMALS)
    y_fy_raw = np.round(y_fy_raw, decimals=FT_SENSOR_DECIMALS)
    y_fz_raw = np.round(y_fz_raw, decimals=FT_SENSOR_DECIMALS)
    
    y_fx = y_fx_raw
    y_fy = y_fy_raw
    y_fz = y_fz_raw
    
    fx_min, fx_max = np.min(y_fx_raw), np.max(y_fx_raw)
    fy_min, fy_max = np.min(y_fy_raw), np.max(y_fy_raw)
    fz_min, fz_max = np.min(y_fz_raw), np.max(y_fz_raw)
    print(f"  Fx range: [{fx_min:.3f}, {fx_max:.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"  Fy range: [{fy_min:.3f}, {fy_max:.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"  Fz range: [{fz_min:.3f}, {fz_max:.3f}] N (variation from baseline, cut at 3N variation)")
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Also return actual sequence lengths after filtering
    actual_sequence_lengths = [len(f) for f in all_features]
    
    return X, y_fx, y_fy, y_fz, y_offset, scaler, None, actual_sequence_lengths


def create_model(regressor: bool = True, use_gpu: bool = True, n_estimators: int = 200, gpu_id: int = 0):
    """Create a Random Forest model (regressor or classifier)."""
    try:
        if use_gpu:
            from cuml.ensemble import RandomForestRegressor as cuRFRegressor
            from cuml.ensemble import RandomForestClassifier as cuRFClassifier
            if regressor:
                return cuRFRegressor(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=42,
                    n_streams=1,
                )
            else:
                # cuML RandomForestClassifier doesn't support class_weight directly
                # Use sklearn's version for classifiers to support class_weight
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced',  # Handle imbalanced classes (especially no_touch)
                )
    except ImportError:
        pass
    
    # Fallback to scikit-learn
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if regressor:
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    else:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Handle imbalanced classes (especially no_touch)
        )


def train_models_for_stretch(
    sequences: List[Dict],
    stretch_label: str,
    train_ratio: float = 0.7,
    use_gpu: bool = True,
    gpu_id: int = 0,
    fz_target_min: float = 0.0,
    fz_target_max: float = 3.0,
    feature_method: str = 'raw',
    location_feature_method: str = 'raw',
) -> Dict:
    """Train force regressor and location classifier for a specific stretch level."""
    
    print(f"\n{'='*80}")
    print(f"Training models for {stretch_label}")
    print(f"{'='*80}")
    print(f"Total sequences: {len(sequences)}")
    
    # Prepare data
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        sequences, 
        normalize=True, 
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=False,  # Don't include offset labels as features
        use_advanced_features=False,
        location_feature_method=location_feature_method
    )
    print(f"Total samples: {len(X)}")
    if X.shape[1] == 15:
        if location_feature_method == 'magnitude':
            print(f"Features: {X.shape[1]} (15 magnitude features - norm of each taxel)")
        elif location_feature_method == 'magnitude_normalized':
            print(f"Features: {X.shape[1]} (15 normalized magnitude features)")
        else:
            print(f"Features: {X.shape[1]} (15 normalized per sensor)")
    elif X.shape[1] == 45:
        print(f"Features: {X.shape[1]} (45 raw features - all xyz values)")
    else:
        print(f"Features: {X.shape[1]} features")
    print(f"Fx range: [{np.min(y_fx):.3f}, {np.max(y_fx):.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"Fy range: [{np.min(y_fy):.3f}, {np.max(y_fy):.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (variation from baseline, cut at 3N)")
    
    # Count sequences per location (including no_touch)
    location_counts = {}
    for seq in sequences:
        offset = seq.get('offset', 'unknown')
        location_counts[offset] = location_counts.get(offset, 0) + 1
    print(f"Sequences per location: {location_counts}")
    unique_offsets = np.unique(y_offset)
    print(f"Location IDs in data: {unique_offsets}")
    if 10 not in unique_offsets:
        print(f"  ⚠️  WARNING: no_touch (class 10) is NOT present in the data!")
    else:
        no_touch_count = np.sum(y_offset == 10)
        print(f"  ✓ no_touch (class 10) is present: {no_touch_count} samples")
    print(f"Features normalized: mean={np.mean(X):.2f}, std={np.std(X):.2f}")
    
    # Split by sequence (70% of sequences, not samples) - RANDOM SPLIT
    sequence_lengths = actual_sequence_lengths
    n_sequences = len(sequence_lengths)
    n_train = int(n_sequences * train_ratio)
    
    # Random split
    np.random.seed(42)
    indices = np.random.permutation(n_sequences)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Map sequence indices to sample indices
    sequence_to_samples = {}
    current_idx = 0
    for seq_idx, seq_len in enumerate(sequence_lengths):
        sequence_to_samples[seq_idx] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    # Get sample indices for train/test
    train_samples = []
    test_samples = []
    for seq_idx in train_indices:
        start, end = sequence_to_samples[seq_idx]
        train_samples.extend(range(start, end))
    for seq_idx in test_indices:
        start, end = sequence_to_samples[seq_idx]
        test_samples.extend(range(start, end))
    
    X_train = X[train_samples]
    X_test = X[test_samples]
    y_fx_train = y_fx[train_samples]
    y_fx_test = y_fx[test_samples]
    y_fy_train = y_fy[train_samples]
    y_fy_test = y_fy[test_samples]
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressors for Fx, Fy, Fz
    print("\nTraining force regressors (Fx, Fy, Fz)...")
    print("  Note: Includes all sequences (contact + no_touch) to improve accuracy at zero force")
    
    from sklearn.metrics import mean_squared_error
    
    # Train Fx regressor
    fx_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
    fx_model.fit(X_train, y_fx_train)
    y_fx_pred = fx_model.predict(X_test)
    y_fx_pred_train = fx_model.predict(X_train)
    rmse_fx = np.sqrt(mean_squared_error(y_fx_test, y_fx_pred))
    rmse_fx_train = np.sqrt(mean_squared_error(y_fx_train, y_fx_pred_train))
    print(f"  Fx - Train RMSE: {rmse_fx_train:.4f} N, Test RMSE: {rmse_fx:.4f} N")
    
    # Train Fy regressor
    fy_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
    fy_model.fit(X_train, y_fy_train)
    y_fy_pred = fy_model.predict(X_test)
    y_fy_pred_train = fy_model.predict(X_train)
    rmse_fy = np.sqrt(mean_squared_error(y_fy_test, y_fy_pred))
    rmse_fy_train = np.sqrt(mean_squared_error(y_fy_train, y_fy_pred_train))
    print(f"  Fy - Train RMSE: {rmse_fy_train:.4f} N, Test RMSE: {rmse_fy:.4f} N")
    
    # Train Fz regressor
    fz_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
    fz_model.fit(X_train, y_fz_train)
    y_fz_pred = fz_model.predict(X_test)
    y_fz_pred_train = fz_model.predict(X_train)
    rmse_fz = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred))
    rmse_fz_train = np.sqrt(mean_squared_error(y_fz_train, y_fz_pred_train))
    print(f"  Fz - Train RMSE: {rmse_fz_train:.4f} N, Test RMSE: {rmse_fz:.4f} N")
    
    # Overall RMSE (average of Fx, Fy, Fz)
    rmse = (rmse_fx + rmse_fy + rmse_fz) / 3.0
    rmse_train = (rmse_fx_train + rmse_fy_train + rmse_fz_train) / 3.0
    print(f"  Overall - Train RMSE: {rmse_train:.4f} N, Test RMSE: {rmse:.4f} N")
    
    # Calculate force resolution (ΔF_min) for Fz
    unique_forces = np.unique(np.round(y_fz_test, decimals=2))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        delta_f_min = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        delta_f_min = float("nan")
    print(f"  Force resolution (ΔF_min for Fz): {delta_f_min:.6f} N")
    
    # KPM1: Force resolution <= 0.05N
    kpm1 = delta_f_min <= 0.05
    print(f"  KPM1: {'PASS' if kpm1 else 'FAIL'}")
    
    # KPM2: Overall RMSE < 0.1N (focus on RMSE only)
    kpm2 = rmse < 0.1
    print(f"  KPM2: {'PASS' if kpm2 else 'FAIL'} (Overall RMSE < 0.1N)")
    
    # Train location classifier (11 classes: '1'-'10' + 'no_touch')
    print("\nTraining location classifier (11 classes: locations 1-10 + no_touch)...")
    valid_mask = y_offset_train >= 0
    if np.sum(valid_mask) > 0 and len(np.unique(y_offset_train[valid_mask])) > 1:
        location_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
        location_model.fit(X_train[valid_mask], y_offset_train[valid_mask])
        test_valid_mask = y_offset_test >= 0
        if np.sum(test_valid_mask) > 0:
            y_location_pred = location_model.predict(X_test[test_valid_mask])
            y_location_test_valid = y_offset_test[test_valid_mask]
            
            from sklearn.metrics import accuracy_score, confusion_matrix
            location_accuracy = float(accuracy_score(y_location_test_valid, y_location_pred))
            print(f"  Accuracy: {location_accuracy:.4f}")
            print(f"  Test samples: {len(y_location_test_valid)}, Location IDs: {np.unique(y_location_test_valid)}")
            print(f"  Location mapping: 0-9 = points 1-10, 10 = no_touch")
            
            # Always use labels=np.arange(11) to ensure no_touch (class 10) is included
            cm = confusion_matrix(y_location_test_valid, y_location_pred, labels=np.arange(11))
            print(f"  Confusion matrix shape: {cm.shape}")
            print(f"  Note: Includes all 11 classes (1-10 + no_touch), even if some are missing in test set")
            
            return {
                'stretch_label': stretch_label,
                'fx_model': fx_model,
                'fy_model': fy_model,
                'fz_model': fz_model,
                'location_model': location_model,
                'scaler': scaler,
                'rmse': rmse,
                'rmse_train': rmse_train,
                'rmse_fx': rmse_fx,
                'rmse_fy': rmse_fy,
                'rmse_fz': rmse_fz,
                'rmse_fx_train': rmse_fx_train,
                'rmse_fy_train': rmse_fy_train,
                'rmse_fz_train': rmse_fz_train,
                'delta_f_min': delta_f_min,
                'kpm1': kpm1,
                'kpm2': kpm2,
                'location_accuracy': location_accuracy,
                'location_cm': cm,
                'location_test_labels': y_location_test_valid,
                'location_pred_labels': y_location_pred,
                'fx_test': y_fx_test,
                'fx_pred': y_fx_pred,
                'fy_test': y_fy_test,
                'fy_pred': y_fy_pred,
                'fz_test': y_fz_test,
                'fz_pred': y_fz_pred,
                'n_train': len(train_indices),
                'n_test': len(test_indices),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
            }
        else:
            location_model = None
            location_accuracy = 0.0
            cm = None
    else:
        location_model = None
        location_accuracy = 0.0
        cm = None
        print("  Skipped (insufficient location diversity)")
    
    return {
        'stretch_label': stretch_label,
        'fx_model': fx_model,
        'fy_model': fy_model,
        'fz_model': fz_model,
        'location_model': location_model,
        'scaler': scaler,
        'rmse': rmse,
        'rmse_train': rmse_train,
        'rmse_fx': rmse_fx,
        'rmse_fy': rmse_fy,
        'rmse_fz': rmse_fz,
        'rmse_fx_train': rmse_fx_train,
        'rmse_fy_train': rmse_fy_train,
        'rmse_fz_train': rmse_fz_train,
        'delta_f_min': delta_f_min,
        'kpm1': kpm1,
        'kpm2': kpm2,
        'location_accuracy': location_accuracy,
        'location_cm': cm,
        'location_test_labels': None,
        'location_pred_labels': None,
        'fx_test': y_fx_test,
        'fx_pred': y_fx_pred,
        'fy_test': y_fy_test,
        'fy_pred': y_fy_pred,
        'fz_test': y_fz_test,
        'fz_pred': y_fz_pred,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
    }


def train_combined_model(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    use_gpu: bool = True,
    fz_target_min: float = 0.0,
    fz_target_max: float = 3.0,
    feature_method: str = 'raw',
    location_feature_method: str = 'raw',
) -> Dict:
    """Train combined model across all stretch levels."""
    
    print(f"\n{'='*80}")
    print("Training combined model (all stretch levels)")
    print(f"{'='*80}")
    
    # Combine all sequences and ensure each has the correct stretch_label
    all_sequences = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            # Ensure stretch_label is set correctly from the dictionary key
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fx range: [{np.min(y_fx):.3f}, {np.max(y_fx):.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"Fy range: [{np.min(y_fy):.3f}, {np.max(y_fy):.3f}] N (variation from baseline, clipped at ±2N)")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (variation from baseline, cut at 3N)")
    print(f"Locations: {np.unique(y_offset)}")
    
    # Split by sequence
    sequence_lengths = actual_sequence_lengths
    n_sequences = len(sequence_lengths)
    n_train = int(n_sequences * train_ratio)
    
    np.random.seed(42)
    indices = np.random.permutation(n_sequences)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    sequence_to_samples = {}
    current_idx = 0
    for seq_idx, seq_len in enumerate(sequence_lengths):
        sequence_to_samples[seq_idx] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    train_samples = []
    test_samples = []
    for seq_idx in train_indices:
        start, end = sequence_to_samples[seq_idx]
        train_samples.extend(range(start, end))
    for seq_idx in test_indices:
        start, end = sequence_to_samples[seq_idx]
        test_samples.extend(range(start, end))
    
    X_train = X[train_samples]
    X_test = X[test_samples]
    y_fx_train = y_fx[train_samples]
    y_fx_test = y_fx[test_samples]
    y_fy_train = y_fy[train_samples]
    y_fy_test = y_fy[test_samples]
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressors for Fx, Fy, Fz
    print("\nTraining force regressors (Fx, Fy, Fz)...")
    
    from sklearn.metrics import mean_squared_error
    
    # Train Fx regressor
    fx_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=0)
    fx_model.fit(X_train, y_fx_train)
    y_fx_pred = fx_model.predict(X_test)
    y_fx_pred_train = fx_model.predict(X_train)
    rmse_fx = np.sqrt(mean_squared_error(y_fx_test, y_fx_pred))
    rmse_fx_train = np.sqrt(mean_squared_error(y_fx_train, y_fx_pred_train))
    print(f"  Fx - Train RMSE: {rmse_fx_train:.4f} N, Test RMSE: {rmse_fx:.4f} N")
    
    # Train Fy regressor
    fy_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=0)
    fy_model.fit(X_train, y_fy_train)
    y_fy_pred = fy_model.predict(X_test)
    y_fy_pred_train = fy_model.predict(X_train)
    rmse_fy = np.sqrt(mean_squared_error(y_fy_test, y_fy_pred))
    rmse_fy_train = np.sqrt(mean_squared_error(y_fy_train, y_fy_pred_train))
    print(f"  Fy - Train RMSE: {rmse_fy_train:.4f} N, Test RMSE: {rmse_fy:.4f} N")
    
    # Train Fz regressor
    fz_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=0)
    fz_model.fit(X_train, y_fz_train)
    y_fz_pred = fz_model.predict(X_test)
    y_fz_pred_train = fz_model.predict(X_train)
    rmse_fz = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred))
    rmse_fz_train = np.sqrt(mean_squared_error(y_fz_train, y_fz_pred_train))
    print(f"  Fz - Train RMSE: {rmse_fz_train:.4f} N, Test RMSE: {rmse_fz:.4f} N")
    
    # Overall RMSE (average of Fx, Fy, Fz)
    rmse = (rmse_fx + rmse_fy + rmse_fz) / 3.0
    rmse_train = (rmse_fx_train + rmse_fy_train + rmse_fz_train) / 3.0
    print(f"  Overall - Train RMSE: {rmse_train:.4f} N, Test RMSE: {rmse:.4f} N")
    
    # Calculate force resolution (ΔF_min) for Fz
    unique_forces = np.unique(np.round(y_fz_test, decimals=2))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        delta_f_min = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        delta_f_min = float("nan")
    print(f"  Force resolution (ΔF_min for Fz): {delta_f_min:.6f} N")
    
    kpm1 = delta_f_min <= 0.05
    print(f"  KPM1: {'PASS' if kpm1 else 'FAIL'}")
    
    # KPM2: Overall RMSE < 0.1N (focus on RMSE only)
    kpm2 = rmse < 0.1
    print(f"  KPM2: {'PASS' if kpm2 else 'FAIL'} (Overall RMSE < 0.1N)")
    
    # Train location classifier (11 classes: locations '1'-'10' + 'no_touch')
    # This classifier determines which of the 10 points is touched, or if no point is touched
    print("\nTraining location classifier (11 classes: locations 1-10 + no_touch)...")
    # Include all samples (including no_touch which is class 10)
    valid_mask = y_offset_train >= 0
    if np.sum(valid_mask) > 0 and len(np.unique(y_offset_train[valid_mask])) > 1:
        location_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=200, gpu_id=1)
        location_model.fit(X_train[valid_mask], y_offset_train[valid_mask])
        test_valid_mask = y_offset_test >= 0
        if np.sum(test_valid_mask) > 0:
            y_location_pred = location_model.predict(X_test[test_valid_mask])
            y_location_test_valid = y_offset_test[test_valid_mask]
            
            from sklearn.metrics import accuracy_score, confusion_matrix
            location_accuracy = float(accuracy_score(y_location_test_valid, y_location_pred))
            print(f"  Accuracy: {location_accuracy:.4f}")
            print(f"  Test samples: {len(y_location_test_valid)}, Location IDs: {np.unique(y_location_test_valid)}")
            print(f"  Location mapping: 0-9 = points 1-10, 10 = no_touch")
            
            cm = confusion_matrix(y_location_test_valid, y_location_pred, labels=np.arange(11))
            print(f"  Confusion matrix shape: {cm.shape}")
        else:
            location_model = None
            location_accuracy = 0.0
            cm = None
    else:
        location_model = None
        location_accuracy = 0.0
        cm = None
    
    # Train stretch classifier (3 classes: 000pct, 010pct, 020pct)
    # This classifier determines the stretch level based on the physical stretch level
    # - All sequences (with contact AND no_touch) are classified by their physical stretch level
    # - no_touch sequences are included and classified based on their stretch attribute (000pct/010pct/020pct)
    print("\nTraining stretch classifier (3 classes: 000pct, 010pct, 020pct)...")
    print("  Note: no_touch sequences are included and classified by their physical stretch level")
    y_stretch_train = []
    y_stretch_test = []
    
    for seq_idx in train_indices:
        seq = all_sequences[seq_idx]
        stretch_label = seq.get('stretch_label', '000pct')
        if stretch_label.startswith('stretch_'):
            stretch_label = stretch_label.replace('stretch_', '')
        if stretch_label not in ['000pct', '010pct', '020pct']:
            stretch_label = '000pct'
        seq_len = actual_sequence_lengths[seq_idx]
        stretch_num = {'000pct': 0, '010pct': 1, '020pct': 2}[stretch_label]
        y_stretch_train.extend([stretch_num] * seq_len)
    
    for seq_idx in test_indices:
        seq = all_sequences[seq_idx]
        stretch_label = seq.get('stretch_label', '000pct')
        if stretch_label.startswith('stretch_'):
            stretch_label = stretch_label.replace('stretch_', '')
        if stretch_label not in ['000pct', '010pct', '020pct']:
            stretch_label = '000pct'
        seq_len = actual_sequence_lengths[seq_idx]
        stretch_num = {'000pct': 0, '010pct': 1, '020pct': 2}[stretch_label]
        y_stretch_test.extend([stretch_num] * seq_len)
    
    y_stretch_train = np.array(y_stretch_train)
    y_stretch_test = np.array(y_stretch_test)
    
    stretch_model = None
    stretch_accuracy = 0.0
    stretch_cm = None
    stretch_labels = ['000pct', '010pct', '020pct']
    
    unique_train = np.unique(y_stretch_train)
    if len(unique_train) > 1:
        stretch_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=250, gpu_id=1)
        stretch_model.fit(X_train, y_stretch_train)
        y_stretch_pred = stretch_model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, confusion_matrix
        stretch_accuracy = float(accuracy_score(y_stretch_test, y_stretch_pred))
        print(f"  Accuracy: {stretch_accuracy:.4f}")
        print(f"  Training samples: {len(y_stretch_train)}, Test samples: {len(y_stretch_test)}")
        
        stretch_cm = confusion_matrix(y_stretch_test, y_stretch_pred, labels=np.arange(3))
        print(f"  Confusion matrix shape: {stretch_cm.shape}")
        unique_classes = np.unique(y_stretch_train)
        print(f"  Classes found: {unique_classes}")
        print(f"  Class mapping: 0=000pct, 1=010pct, 2=020pct")
        print(f"  Note: Includes both contact and no_touch sequences, classified by physical stretch level")
    else:
        print(f"  Skipped (insufficient stretch diversity: only {len(unique_train)} class(es) in training set)")
    
    return {
        'fx_model': fx_model,
        'fy_model': fy_model,
        'fz_model': fz_model,
        'location_model': location_model,
        'stretch_model': stretch_model,
        'scaler': scaler,
        'rmse': rmse,
        'rmse_train': rmse_train,
        'rmse_fx': rmse_fx,
        'rmse_fy': rmse_fy,
        'rmse_fz': rmse_fz,
        'rmse_fx_train': rmse_fx_train,
        'rmse_fy_train': rmse_fy_train,
        'rmse_fz_train': rmse_fz_train,
        'delta_f_min': delta_f_min,
        'kpm1': kpm1,
        'kpm2': kpm2,
        'location_accuracy': location_accuracy,
        'location_cm': cm,
        'stretch_accuracy': stretch_accuracy,
        'stretch_cm': stretch_cm,
        'stretch_labels': stretch_labels,
        'location_test_labels': y_location_test_valid if 'y_location_test_valid' in locals() and y_location_test_valid is not None else None,
        'location_pred_labels': y_location_pred if 'y_location_pred' in locals() and y_location_pred is not None else None,
        'fx_test': y_fx_test,
        'fx_pred': y_fx_pred,
        'fy_test': y_fy_test,
        'fy_pred': y_fy_pred,
        'fz_test': y_fz_test,
        'fz_pred': y_fz_pred,
    }


def generate_plots_multipoint(
    trained_models: Dict,
    sequences_by_stretch: Dict[str, List[Dict]],
    plots_dir: Path,
    feature_method: str,
    h5_files: Optional[Dict[str, Path]] = None,
):
    """Generate all plots for multi-point training."""
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrices for location classification
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRICES")
    print(f"{'='*80}")
    
    confusion_plots_saved = []
    
    # Per-stretch confusion matrices
    for stretch, result in trained_models.items():
        if stretch == 'combined':
            continue
        
        if result.get('location_cm') is not None:
            cm = result['location_cm']
            
            # sklearn's confusion_matrix with labels=np.arange(11) already creates 11x11
            # If it has less than 11 classes, it means some classes weren't in the test set
            # This is normal and correct - we just need to ensure we have the right shape for plotting
            
            # If matrix is smaller than 11x11, it means some classes are missing from test set
            # Create a full 11x11 matrix with zeros for missing classes
            if cm.shape[0] < 11 or cm.shape[1] < 11:
                # This should not happen if we used labels=np.arange(11), but handle it anyway
                cm_full = np.zeros((11, 11), dtype=cm.dtype)
                # Copy existing matrix (assuming it's already correctly ordered 0-10)
                max_dim = min(cm.shape[0], cm.shape[1], 11)
                cm_full[:max_dim, :max_dim] = cm[:max_dim, :max_dim]
                cm = cm_full
            elif cm.shape[0] > 11 or cm.shape[1] > 11:
                # This should never happen, but handle it
                print(f"  ⚠️  Warning: Confusion matrix has unexpected shape {cm.shape}, truncating to 11x11")
                cm = cm[:11, :11]
            
            # Normalize by row, handling empty rows (no samples in test set)
            row_sums = cm.sum(axis=1)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero for empty rows
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Always use all 11 offset names (1-10 + no_touch)
            offset_names = MULTIPOINT_OFFSET_NAMES  # All 11 classes
            
            fig, ax = plt.subplots(figsize=(10, 9))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=offset_names)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(f'Location Classification - {stretch.upper()}\nAccuracy: {result["location_accuracy"]:.2%}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Location', fontsize=12)
            ax.set_ylabel('True Location', fontsize=12)
            plt.tight_layout()
            
            plot_path = plots_dir / f"confusion_matrix_location_{stretch}_{feature_method}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_path.name}")
            confusion_plots_saved.append(plot_path)
    
    # Combined confusion matrix
    if 'combined' in trained_models:
        result = trained_models['combined']
        if result.get('location_cm') is not None:
            cm = result['location_cm']
            
            # Ensure 11x11 shape (sklearn should already create this with labels=np.arange(11))
            if cm.shape[0] < 11 or cm.shape[1] < 11:
                cm_full = np.zeros((11, 11), dtype=cm.dtype)
                max_dim = min(cm.shape[0], cm.shape[1], 11)
                cm_full[:max_dim, :max_dim] = cm[:max_dim, :max_dim]
                cm = cm_full
            elif cm.shape[0] > 11 or cm.shape[1] > 11:
                cm = cm[:11, :11]
            
            # Normalize by row, handling empty rows
            row_sums = cm.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Always use all 11 offset names (1-10 + no_touch)
            offset_names = MULTIPOINT_OFFSET_NAMES  # All 11 classes
            
            fig, ax = plt.subplots(figsize=(10, 9))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=offset_names)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(f'Location Classification - Combined (All Stretch Levels)\nAccuracy: {result["location_accuracy"]:.2%}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Location', fontsize=12)
            ax.set_ylabel('True Location', fontsize=12)
            plt.tight_layout()
            
            plot_path = plots_dir / f"confusion_matrix_location_combined_{feature_method}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_path.name}")
            confusion_plots_saved.append(plot_path)
        
        # Stretch confusion matrix (only for combined model)
        if result.get('stretch_cm') is not None:
            cm = result['stretch_cm']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Stretch labels: 3 classes (000pct, 010pct, 020pct)
            # Note: no_touch is NOT a stretch class, but no_touch sequences are included
            # and classified by their physical stretch level
            stretch_labels = ['000pct', '010pct', '020pct']
            # Only use labels that exist in the confusion matrix
            stretch_labels = stretch_labels[:cm.shape[0]]
            
            fig, ax = plt.subplots(figsize=(10, 9))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=stretch_labels)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(f'Stretch Classification - Combined\nAccuracy: {result["stretch_accuracy"]:.2%}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Stretch', fontsize=12)
            ax.set_ylabel('True Stretch', fontsize=12)
            plt.tight_layout()
            
            plot_path = plots_dir / f"confusion_matrix_stretch_combined_{feature_method}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_path.name}")
            confusion_plots_saved.append(plot_path)
    
    # 2. Force prediction plots
    print(f"\n{'='*80}")
    print("GENERATING PREDICTION PLOTS")
    print(f"{'='*80}")
    
    prediction_plots_saved = []
    
    for stretch, result in trained_models.items():
        if stretch == 'combined':
            continue
        
        # Plot Fx, Fy, Fz separately
        for force_name, force_key_test, force_key_pred, rmse_key in [
            ('Fx', 'fx_test', 'fx_pred', 'rmse_fx'),
            ('Fy', 'fy_test', 'fy_pred', 'rmse_fy'),
            ('Fz', 'fz_test', 'fz_pred', 'rmse_fz'),
        ]:
            if result.get(force_key_test) is not None and result.get(force_key_pred) is not None:
                force_test = result[force_key_test]
                force_pred = result[force_key_pred]
                
                # Scatter plot
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(force_test, force_pred, alpha=0.5, s=10)
                ax.plot([force_test.min(), force_test.max()], [force_test.min(), force_test.max()], 'r--', lw=2, label='Perfect prediction')
                ax.set_xlabel(f'True {force_name} (N)', fontsize=12)
                ax.set_ylabel(f'Predicted {force_name} (N)', fontsize=12)
                rmse_train = result.get(f'{rmse_key}_train', result.get(rmse_key, 0.0))
                rmse_test = result.get(rmse_key, 0.0)
                ax.set_title(f'{force_name} Regression - {stretch.upper()}\nTrain RMSE: {rmse_train:.4f} N, Test RMSE: {rmse_test:.4f} N', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = plots_dir / f"prediction_scatter_{force_name.lower()}_{stretch}_{feature_method}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {plot_path.name}")
                prediction_plots_saved.append(plot_path)
                
                # Residual plot
                residuals = force_test - force_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(force_test, residuals, alpha=0.5, s=10)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel(f'True {force_name} (N)', fontsize=12)
                ax.set_ylabel(f'Residual (True - Predicted) (N)', fontsize=12)
                ax.set_title(f'{force_name} Residuals - {stretch.upper()}\nTrain RMSE: {rmse_train:.4f} N, Test RMSE: {rmse_test:.4f} N', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = plots_dir / f"prediction_residuals_{force_name.lower()}_{stretch}_{feature_method}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {plot_path.name}")
                prediction_plots_saved.append(plot_path)
    
    # Combined prediction plots for Fx, Fy, Fz
    if 'combined' in trained_models:
        result = trained_models['combined']
        for force_name, force_key_test, force_key_pred, rmse_key in [
            ('Fx', 'fx_test', 'fx_pred', 'rmse_fx'),
            ('Fy', 'fy_test', 'fy_pred', 'rmse_fy'),
            ('Fz', 'fz_test', 'fz_pred', 'rmse_fz'),
        ]:
            if result.get(force_key_test) is not None and result.get(force_key_pred) is not None:
                force_test = result[force_key_test]
                force_pred = result[force_key_pred]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(force_test, force_pred, alpha=0.5, s=10)
                ax.plot([force_test.min(), force_test.max()], [force_test.min(), force_test.max()], 'r--', lw=2, label='Perfect prediction')
                ax.set_xlabel(f'True {force_name} (N)', fontsize=12)
                ax.set_ylabel(f'Predicted {force_name} (N)', fontsize=12)
                rmse_train = result.get(f'{rmse_key}_train', result.get(rmse_key, 0.0))
                rmse_test = result.get(rmse_key, 0.0)
                ax.set_title(f'{force_name} Regression - Combined\nTrain RMSE: {rmse_train:.4f} N, Test RMSE: {rmse_test:.4f} N', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = plots_dir / f"prediction_scatter_{force_name.lower()}_combined_{feature_method}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {plot_path.name}")
                prediction_plots_saved.append(plot_path)
    
    # 3. Raw data plot: Fz vs all normalized magnetic channels (if HDF5 files available)
    if h5_files:
        print(f"\n{'='*80}")
        print("GENERATING FZ VS ALL MAGNETIC CHANNELS PLOT")
        print(f"{'='*80}")
        
        try:
            # load_sequences_from_h5 is now defined locally in this file
            
            for stretch, h5_file_or_list in h5_files.items():
                # Handle both single file and list of files (regular + no_touch)
                if isinstance(h5_file_or_list, list):
                    # For the plot, use only the regular file (not no_touch)
                    h5_file = None
                    for f in h5_file_or_list:
                        if "no_touch" not in str(f):
                            h5_file = f
                            break
                    # If no regular file found, skip this stretch
                    if h5_file is None:
                        continue
                else:
                    h5_file = h5_file_or_list
                
                if h5_file and h5_file.exists():
                    stretch_output_dir = plots_dir / f"stretch_{stretch}"
                    stretch_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load sequences
                    sequences = load_sequences_from_h5(h5_file)
                    if not sequences:
                        print(f"  ⚠️  No sequences found for {stretch}")
                        continue
                    
                    # Use first sequence
                    first_seq = sequences[0]
                    
                    # Get Fz and time
                    fz = np.abs(first_seq['fz'])
                    time = first_seq['timestamps'] - first_seq['timestamps'][0]  # Start from 0
                    
                    # Get magnetic data: [samples, 15, 3]
                    magnetic = first_seq['stretchmagtec']
                    
                    # Normalize each channel independently (min-max normalization per channel)
                    magnetic_normalized = np.zeros_like(magnetic)
                    for ch in range(15):  # 15 channels
                        for comp in range(3):  # X, Y, Z components
                            ch_data = magnetic[:, ch, comp]
                            ch_min = np.min(ch_data)
                            ch_max = np.max(ch_data)
                            if ch_max > ch_min:
                                magnetic_normalized[:, ch, comp] = (ch_data - ch_min) / (ch_max - ch_min)
                            else:
                                magnetic_normalized[:, ch, comp] = ch_data - ch_min
                    
                    # Calculate magnitude for each channel (from normalized components)
                    mag_magnitudes = np.sqrt(np.sum(magnetic_normalized**2, axis=2))  # [samples, 15]
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(16, 10))
                    
                    # Plot Fz on left y-axis
                    ax.plot(time, fz, 'b-', linewidth=2.5, label='Fz FT Sensor |Fz|', alpha=0.9)
                    ax.set_xlabel('Time (s)', fontsize=14)
                    ax.set_ylabel('Fz FT Sensor |Fz| (N)', fontsize=14, color='b')
                    ax.tick_params(axis='y', labelcolor='b')
                    ax.grid(True, alpha=0.3)
                    
                    # Plot all magnetic channels (normalized magnitudes) on right y-axis
                    ax2 = ax.twinx()
                    colors = plt.cm.tab20(np.linspace(0, 1, 15))
                    for ch in range(15):
                        ax2.plot(time, mag_magnitudes[:, ch], 
                                color=colors[ch], linewidth=1.5, alpha=0.7, 
                                label=f'Ch{ch+1}' if ch < 10 else '')
                    
                    ax2.set_ylabel('Magnetic Sensor Channels (Normalized Magnitude)', fontsize=14, color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    # Combine legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, ncol=2)
                    
                    ax.set_title(f'Fz FT Sensor vs All Magnetic Channels (Normalized) - {stretch.upper()}\nFirst Sequence Only', 
                               fontsize=16, fontweight='bold')
                    
                    plt.tight_layout()
                    plot_path = stretch_output_dir / f"fz_vs_all_magnetic_channels_{stretch}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Saved: {plot_path.name}")
                    
        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate Fz vs magnetic plot: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  Total plots saved: {len(confusion_plots_saved) + len(prediction_plots_saved)}")
    print(f"  Location: {plots_dir}")


def limit_sequences_per_stretch(sequences: List[Dict], max_sequences: int = 20, random_seed: int = 42) -> List[Dict]:
    """Limit sequences to max_sequences per stretch level, randomly sampling if needed."""
    if len(sequences) <= max_sequences:
        return sequences
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(sequences))
    selected_indices = indices[:max_sequences]
    return [sequences[i] for i in selected_indices]


def limit_sequences_per_location(sequences: List[Dict], max_sequences_per_location: int = 30, random_seed: int = 42) -> List[Dict]:
    """Limit sequences to max_sequences_per_location PER LOCATION/OFFSET, randomly sampling if needed.
    
    This ensures balanced data: e.g., 30 sequences per point × 10 points = 300 sequences total.
    """
    from collections import defaultdict
    
    # Group sequences by location/offset
    sequences_by_location = defaultdict(list)
    for idx, seq in enumerate(sequences):
        location = seq.get('offset', 'unknown')
        sequences_by_location[location].append((idx, seq))
    
    # Limit each location separately
    limited_sequences = []
    np.random.seed(random_seed)
    
    for location, loc_sequences in sequences_by_location.items():
        if len(loc_sequences) > max_sequences_per_location:
            # Randomly sample to limit
            indices = np.random.permutation(len(loc_sequences))
            selected_indices = indices[:max_sequences_per_location]
            selected_sequences = [loc_sequences[i][1] for i in selected_indices]
        else:
            selected_sequences = [seq for _, seq in loc_sequences]
        
        limited_sequences.extend(selected_sequences)
    
    return limited_sequences


def limit_sequences_per_location_and_direction(sequences: List[Dict], max_sequences_per_location_per_direction: int = 20, random_seed: int = 42) -> List[Dict]:
    """Limit sequences to max_sequences_per_location_per_direction PER LOCATION AND PER DIRECTION.
    
    For shear forces: 20 sequences per point × 4 directions × 10 points = 800 sequences total.
    """
    from collections import defaultdict
    
    # Group sequences by location AND direction
    sequences_by_location_direction = defaultdict(list)
    
    for idx, seq in enumerate(sequences):
        location = seq.get('offset', 'unknown')
        label = seq.get('label', '')
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        
        # Determine specific direction (x+, x-, y+, y-)
        direction = 'unknown'
        if 'shear' in label.lower():
            if 'x+' in label:
                direction = 'x+'
            elif 'x-' in label:
                direction = 'x-'
            elif 'y+' in label:
                direction = 'y+'
            elif 'y-' in label:
                direction = 'y-'
        
        key = (location, direction)
        sequences_by_location_direction[key].append((idx, seq))
    
    # Limit each (location, direction) combination separately
    limited_sequences = []
    np.random.seed(random_seed)
    
    for (location, direction), loc_dir_sequences in sequences_by_location_direction.items():
        if len(loc_dir_sequences) > max_sequences_per_location_per_direction:
            # Randomly sample to limit
            indices = np.random.permutation(len(loc_dir_sequences))
            selected_indices = indices[:max_sequences_per_location_per_direction]
            selected_sequences = [loc_dir_sequences[i][1] for i in selected_indices]
        else:
            selected_sequences = [seq for _, seq in loc_dir_sequences]
        
        limited_sequences.extend(selected_sequences)
    
    return limited_sequences


def load_data_combined(
    shear_files: Dict[str, Path | List[Path]],
    normal_files: Dict[str, Path | List[Path]],
    remove_outliers_flag: bool = True,
    z_threshold: float = 3.0,
    max_sequences_per_stretch: int = 20,
) -> Dict[str, List[Dict]]:
    """Load and combine shear forces + normal forces data."""
    
    print(f"\n{'='*80}")
    print("LOADING COMBINED DATA (SHEAR FORCES + NORMAL FORCES)")
    print(f"{'='*80}")
    
    sequences_by_stretch = {}
    
    for stretch in ['000pct', '010pct', '020pct']:
        all_sequences = []
        
        # Load shear forces
        if stretch in shear_files:
            shear_file_or_list = shear_files[stretch]
            if isinstance(shear_file_or_list, list):
                shear_file_list = shear_file_or_list
            else:
                shear_file_list = [shear_file_or_list]
            
            for h5_file in shear_file_list:
                print(f"\nLoading SHEAR FORCES {stretch} from {h5_file.name}...")
                sequences = load_sequences_from_h5(h5_file)
                print(f"  Loaded {len(sequences)} sequences from {h5_file.name}")
                
                # Mark as shear forces
                for seq in sequences:
                    seq['stretch_label'] = stretch
                    seq['data_type'] = 'shear'
                
                all_sequences.extend(sequences)
        
        # Load normal forces
        if stretch in normal_files:
            normal_file_or_list = normal_files[stretch]
            if isinstance(normal_file_or_list, list):
                normal_file_list = normal_file_or_list
            else:
                normal_file_list = [normal_file_or_list]
            
            for h5_file in normal_file_list:
                print(f"\nLoading NORMAL FORCES {stretch} from {h5_file.name}...")
                sequences = load_sequences_from_h5(h5_file)
                print(f"  Loaded {len(sequences)} sequences from {h5_file.name}")
                
                # Mark as normal forces
                for seq in sequences:
                    seq['stretch_label'] = stretch
                    seq['data_type'] = 'normal'
                
                all_sequences.extend(sequences)
        
        sequences = all_sequences
        print(f"\n  Total sequences for {stretch}: {len(sequences)}")
        
        if sequences:
            # Count sequences per location
            initial_location_counts = {}
            for seq in sequences:
                location = seq.get('offset', 'unknown')
                initial_location_counts[location] = initial_location_counts.get(location, 0) + 1
            print(f"  Initial sequences per location: {initial_location_counts}")
            
            # Remove outliers if requested
            if remove_outliers_flag:
                print(f"  Removing outliers (2 per location, independently)...")
                cleaned_sequences, outlier_indices = remove_outliers(sequences, z_threshold=z_threshold, remove_per_offset=2)
                print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
                sequences = cleaned_sequences
            
            # Limit to max_sequences_per_stretch
            if len(sequences) > max_sequences_per_stretch:
                print(f"  Limiting to {max_sequences_per_stretch} sequences (random sampling)...")
                sequences = limit_sequences_per_stretch(sequences, max_sequences_per_stretch, random_seed=42)
                print(f"  Final sequences: {len(sequences)}")
            
            # Count final sequences per location
            final_location_counts = {}
            for seq in sequences:
                location = seq.get('offset', 'unknown')
                final_location_counts[location] = final_location_counts.get(location, 0) + 1
            print(f"  Final sequences per location: {final_location_counts}")
            
            sequences_by_stretch[stretch] = sequences
    
    return sequences_by_stretch


def compare_normal_vs_combined(
    normal_files: Dict[str, Path | List[Path]],
    combined_sequences: Dict[str, List[Dict]],
    trained_combined: Dict,
    output_dir: Path,
    feature_method: str,
    remove_outliers_flag: bool = True,
    z_threshold: float = 3.0,
    max_sequences_per_stretch: int = 20,
):
    """Compare models trained on only normal forces vs normal+shear forces."""
    
    print(f"\n{'='*80}")
    print("COMPARISON: NORMAL ONLY vs NORMAL+SHEAR")
    print(f"{'='*80}")
    
    # Load only normal forces
    normal_only_sequences = {}
    for stretch in ['000pct', '010pct', '020pct']:
        if stretch in normal_files:
            normal_file_or_list = normal_files[stretch]
            if isinstance(normal_file_or_list, list):
                normal_file_list = normal_file_or_list
            else:
                normal_file_list = [normal_file_or_list]
            
            all_sequences = []
            for h5_file in normal_file_list:
                print(f"\nLoading NORMAL ONLY {stretch} from {h5_file.name}...")
                sequences = load_sequences_from_h5(h5_file)
                print(f"  Loaded {len(sequences)} sequences from {h5_file.name}")
                
                for seq in sequences:
                    seq['stretch_label'] = stretch
                    seq['data_type'] = 'normal'
                
                all_sequences.extend(sequences)
            
            sequences = all_sequences
            if sequences:
                # Remove outliers
                if remove_outliers_flag:
                    cleaned_sequences, _ = remove_outliers(sequences, z_threshold=z_threshold, remove_per_offset=2)
                    sequences = cleaned_sequences
                
                # For normal-only comparison, use all available sequences (don't limit to max_sequences_per_stretch)
                # This allows us to use ~30 sequences instead of 20 for better training
                normal_only_sequences[stretch] = sequences
                print(f"  Final normal-only sequences for {stretch}: {len(sequences)} (using all available)")
    
    if not normal_only_sequences:
        print("  ⚠️  No normal-only sequences found, skipping comparison")
        return
    
    # Train models on normal-only data
    print(f"\n{'='*80}")
    print("TRAINING MODELS ON NORMAL-ONLY DATA")
    print(f"{'='*80}")
    
    normal_only_models = {}
    for stretch, sequences in normal_only_sequences.items():
        result = train_models_for_stretch(
            sequences,
            stretch,
            train_ratio=0.7,
            use_gpu=False,
            feature_method=feature_method,
            location_feature_method='raw',  # Use raw for normal-only comparison
        )
        normal_only_models[stretch] = result
    
    # Train combined model on normal-only
    result_normal_only_combined = train_combined_model(
        normal_only_sequences,
        train_ratio=0.7,
        use_gpu=False,
        feature_method=feature_method,
        location_feature_method='raw',
    )
    normal_only_models['combined'] = result_normal_only_combined
    
    # Also train normal-only models with magnitude features for location classification
    print(f"\n{'='*80}")
    print("TRAINING NORMAL-ONLY MODELS WITH MAGNITUDE FEATURES")
    print(f"{'='*80}")
    
    normal_only_magnitude_models = {}
    for stretch, sequences in normal_only_sequences.items():
        result = train_models_for_stretch(
            sequences,
            stretch,
            train_ratio=0.7,
            use_gpu=False,
            feature_method=feature_method,
            location_feature_method='magnitude',  # Use magnitude for location classification
        )
        normal_only_magnitude_models[stretch] = result
    
    # Train combined model on normal-only with magnitude
    result_normal_only_magnitude_combined = train_combined_model(
        normal_only_sequences,
        train_ratio=0.7,
        use_gpu=False,
        feature_method=feature_method,
        location_feature_method='magnitude',
    )
    normal_only_magnitude_models['combined'] = result_normal_only_magnitude_combined
    
    # Generate comparison plots
    plots_dir = output_dir / "plots" / "comparison"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")
    
    # Compare location classification accuracy
    comparison_data = {
        'normal_only': {},
        'normal_only_magnitude': {},
        'normal_shear': {},
    }
    
    for stretch in ['000pct', '010pct', '020pct', 'combined']:
        if stretch in normal_only_models and stretch in trained_combined:
            normal_result = normal_only_models[stretch]
            combined_result = trained_combined[stretch]
            
            comparison_data['normal_only'][stretch] = {
                'location_accuracy': normal_result.get('location_accuracy', 0.0),
                'rmse_fx': normal_result.get('rmse_fx', 0.0),
                'rmse_fy': normal_result.get('rmse_fy', 0.0),
                'rmse_fz': normal_result.get('rmse_fz', 0.0),
                'rmse': normal_result.get('rmse', 0.0),
            }
            
            comparison_data['normal_shear'][stretch] = {
                'location_accuracy': combined_result.get('location_accuracy', 0.0),
                'rmse_fx': combined_result.get('rmse_fx', 0.0),
                'rmse_fy': combined_result.get('rmse_fy', 0.0),
                'rmse_fz': combined_result.get('rmse_fz', 0.0),
                'rmse': combined_result.get('rmse', 0.0),
            }
        
        # Add normal-only magnitude results
        if stretch in normal_only_magnitude_models:
            normal_mag_result = normal_only_magnitude_models[stretch]
            comparison_data['normal_only_magnitude'][stretch] = {
                'location_accuracy': normal_mag_result.get('location_accuracy', 0.0),
                'rmse_fx': normal_mag_result.get('rmse_fx', 0.0),
                'rmse_fy': normal_mag_result.get('rmse_fy', 0.0),
                'rmse_fz': normal_mag_result.get('rmse_fz', 0.0),
                'rmse': normal_mag_result.get('rmse', 0.0),
            }
    
    # Plot comparison: Location accuracy (including normal-only magnitude)
    fig, ax = plt.subplots(figsize=(12, 6))
    stretches = ['000pct', '010pct', '020pct', 'combined']
    normal_acc = [comparison_data['normal_only'].get(s, {}).get('location_accuracy', 0.0) for s in stretches]
    normal_mag_acc = [comparison_data['normal_only_magnitude'].get(s, {}).get('location_accuracy', 0.0) for s in stretches]
    combined_acc = [comparison_data['normal_shear'].get(s, {}).get('location_accuracy', 0.0) for s in stretches]
    
    x = np.arange(len(stretches))
    width = 0.25
    ax.bar(x - width, normal_acc, width, label='Normal Only (raw)', alpha=0.8)
    ax.bar(x, normal_mag_acc, width, label='Normal Only (magnitude)', alpha=0.8)
    ax.bar(x + width, combined_acc, width, label='Normal+Shear (raw)', alpha=0.8)
    ax.set_xlabel('Stretch Level', fontsize=12)
    ax.set_ylabel('Location Classification Accuracy', fontsize=12)
    ax.set_title('Location Classification: Comparison of Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stretches)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, (n, nm, c) in enumerate(zip(normal_acc, normal_mag_acc, combined_acc)):
        ax.text(i - width, n + 0.01, f'{n:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, nm + 0.01, f'{nm:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = plots_dir / f"location_accuracy_comparison_{feature_method}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path.name}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("LOCATION CLASSIFICATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Stretch':<12} {'Normal Only (raw)':<20} {'Normal Only (magnitude)':<25} {'Normal+Shear (raw)':<20}")
    print("-" * 80)
    for stretch in stretches:
        n_acc = comparison_data['normal_only'].get(stretch, {}).get('location_accuracy', 0.0)
        nm_acc = comparison_data['normal_only_magnitude'].get(stretch, {}).get('location_accuracy', 0.0)
        c_acc = comparison_data['normal_shear'].get(stretch, {}).get('location_accuracy', 0.0)
        print(f"{stretch:<12} {n_acc:<20.4f} {nm_acc:<25.4f} {c_acc:<20.4f}")
    print(f"{'='*80}")
    
    # Plot comparison: Location accuracy (old version - keep for backward compatibility)
    fig, ax = plt.subplots(figsize=(10, 6))
    stretches = ['000pct', '010pct', '020pct', 'combined']
    normal_acc = [comparison_data['normal_only'].get(s, {}).get('location_accuracy', 0.0) for s in stretches]
    combined_acc = [comparison_data['normal_shear'].get(s, {}).get('location_accuracy', 0.0) for s in stretches]
    
    x = np.arange(len(stretches))
    width = 0.35
    ax.bar(x - width/2, normal_acc, width, label='Normal Only', alpha=0.8)
    ax.bar(x + width/2, combined_acc, width, label='Normal+Shear', alpha=0.8)
    ax.set_xlabel('Stretch Level', fontsize=12)
    ax.set_ylabel('Location Classification Accuracy', fontsize=12)
    ax.set_title('Location Classification: Normal Only vs Normal+Shear', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stretches)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, (n, c) in enumerate(zip(normal_acc, combined_acc)):
        ax.text(i - width/2, n + 0.01, f'{n:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = plots_dir / f"location_accuracy_comparison_{feature_method}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path.name}")
    
    # Plot comparison: Force RMSE (Fx, Fy, Fz)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    force_names = ['Fx', 'Fy', 'Fz']
    force_keys = ['rmse_fx', 'rmse_fy', 'rmse_fz']
    
    for idx, (force_name, force_key) in enumerate(zip(force_names, force_keys)):
        ax = axes[idx]
        normal_rmse = [comparison_data['normal_only'].get(s, {}).get(force_key, 0.0) for s in stretches]
        combined_rmse = [comparison_data['normal_shear'].get(s, {}).get(force_key, 0.0) for s in stretches]
        
        x = np.arange(len(stretches))
        width = 0.35
        ax.bar(x - width/2, normal_rmse, width, label='Normal Only', alpha=0.8)
        ax.bar(x + width/2, combined_rmse, width, label='Normal+Shear', alpha=0.8)
        ax.set_xlabel('Stretch Level', fontsize=11)
        ax.set_ylabel(f'{force_name} RMSE (N)', fontsize=11)
        ax.set_title(f'{force_name} Regression RMSE', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stretches)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (n, c) in enumerate(zip(normal_rmse, combined_rmse)):
            max_val = max(n, c)
            ax.text(i - width/2, n + max_val*0.05, f'{n:.4f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, c + max_val*0.05, f'{c:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = plots_dir / f"force_rmse_comparison_{feature_method}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path.name}")
    
    # Plot comparison: Overall RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    normal_rmse = [comparison_data['normal_only'].get(s, {}).get('rmse', 0.0) for s in stretches]
    combined_rmse = [comparison_data['normal_shear'].get(s, {}).get('rmse', 0.0) for s in stretches]
    
    x = np.arange(len(stretches))
    width = 0.35
    ax.bar(x - width/2, normal_rmse, width, label='Normal Only', alpha=0.8)
    ax.bar(x + width/2, combined_rmse, width, label='Normal+Shear', alpha=0.8)
    ax.set_xlabel('Stretch Level', fontsize=12)
    ax.set_ylabel('Overall RMSE (N)', fontsize=12)
    ax.set_title('Overall Force Regression RMSE: Normal Only vs Normal+Shear', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stretches)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (n, c) in enumerate(zip(normal_rmse, combined_rmse)):
        max_val = max(n, c) if max(n, c) > 0 else 0.01
        ax.text(i - width/2, n + max_val*0.05, f'{n:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, c + max_val*0.05, f'{c:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = plots_dir / f"overall_rmse_comparison_{feature_method}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot_path.name}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Stretch':<12} {'Metric':<25} {'Normal Only':<15} {'Normal+Shear':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for stretch in stretches:
        if stretch in comparison_data['normal_only']:
            normal = comparison_data['normal_only'][stretch]
            combined = comparison_data['normal_shear'][stretch]
            
            # Location accuracy
            loc_improve = combined['location_accuracy'] - normal['location_accuracy']
            print(f"{stretch:<12} {'Location Accuracy':<25} {normal['location_accuracy']:<15.4f} {combined['location_accuracy']:<15.4f} {loc_improve:+.4f}")
            
            # Overall RMSE
            rmse_improve = normal['rmse'] - combined['rmse']  # Positive = improvement
            print(f"{'':<12} {'Overall RMSE':<25} {normal['rmse']:<15.4f} {combined['rmse']:<15.4f} {rmse_improve:+.4f}")
            
            # Fx, Fy, Fz RMSE
            for force_name, force_key in zip(['Fx RMSE', 'Fy RMSE', 'Fz RMSE'], ['rmse_fx', 'rmse_fy', 'rmse_fz']):
                force_improve = normal[force_key] - combined[force_key]
                print(f"{'':<12} {force_name:<25} {normal[force_key]:<15.4f} {combined[force_key]:<15.4f} {force_improve:+.4f}")
            print()
    
    # Save comparison metrics
    comparison_metrics = {
        'normal_only': comparison_data['normal_only'],
        'normal_shear': comparison_data['normal_shear'],
    }
    metrics_path = plots_dir / f"comparison_metrics_{feature_method}.json"
    with open(metrics_path, 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    print(f"  ✓ Saved comparison metrics: {metrics_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined training script for shear forces + normal forces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--shear-dir',
        type=Path,
        required=True,
        help='Directory containing shear forces HDF5 files (e.g., data/Multiple_Points/shear_forces_test51)'
    )
    parser.add_argument(
        '--normal-dir',
        type=Path,
        required=True,
        help='Directory containing normal forces HDF5 files (e.g., data/Multiple_Points/2.5mm_single_test42)'
    )
    parser.add_argument(
        '--run-label',
        type=str,
        required=True,
        help='Label for this training run (e.g., combined_shear_normal)'
    )
    parser.add_argument(
        '--feature-method',
        type=str,
        default='raw',
        choices=['raw', 'normalized'],
        help='Feature extraction method: raw (45 features) or normalized (15 features)'
    )
    parser.add_argument(
        '--remove-outliers',
        type=bool,
        default=True,
        help='Remove outliers (default: True)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=20,
        help='Maximum sequences per stretch level (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Find shear forces HDF5 files
    shear_dir = Path(args.shear_dir)
    if not shear_dir.exists():
        print(f"❌ Error: Shear directory not found: {shear_dir}")
        return 1
    
    # Find normal forces HDF5 files
    normal_dir = Path(args.normal_dir)
    if not normal_dir.exists():
        print(f"❌ Error: Normal directory not found: {normal_dir}")
        return 1
    
    # Find shear forces files
    shear_files_dict = {}
    for stretch in ['000pct', '010pct', '020pct']:
        pattern = f"*shear*stretch_{stretch}.h5"
        files = list(shear_dir.glob(pattern))
        if files:
            shear_files_dict[stretch] = files[0] if len(files) == 1 else files
            print(f"Found SHEAR {stretch}: {len(files)} file(s)")
            for f in files if isinstance(files, list) else [files]:
                print(f"  - {f.name if hasattr(f, 'name') else f}")
    
    # Find normal forces files
    normal_files_dict = {}
    for stretch in ['000pct', '010pct', '020pct']:
        pattern = f"*stretch_{stretch}.h5"
        files = list(normal_dir.glob(pattern))
        # Exclude no_touch files for normal forces (they're separate)
        files = [f for f in files if 'no_touch' not in f.name]
        if files:
            normal_files_dict[stretch] = files[0] if len(files) == 1 else files
            print(f"Found NORMAL {stretch}: {len(files)} file(s)")
            for f in files if isinstance(files, list) else [files]:
                print(f"  - {f.name if hasattr(f, 'name') else f}")
    
    if not shear_files_dict and not normal_files_dict:
        print(f"❌ Error: No HDF5 files found in either directory")
        return 1
    
    # Create output directory structure (use shear_dir as base)
    output_dir = shear_dir / "cleaned_combined" / args.feature_method
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load combined data
    print(f"\n{'='*80}")
    print("LOADING COMBINED DATA")
    print(f"{'='*80}")
    sequences_by_stretch = load_data_combined(
        shear_files_dict,
        normal_files_dict,
        remove_outliers_flag=args.remove_outliers,
        z_threshold=args.z_threshold,
        max_sequences_per_stretch=args.max_sequences
    )
    
    if not sequences_by_stretch:
        print("❌ Error: No sequences loaded")
        return 1
    
    print(f"\n{'='*80}")
    print("TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Run label: {args.run_label}")
    print(f"Feature method: {args.feature_method}")
    print(f"Output directory: {output_dir}")
    print(f"Stretch levels: {list(sequences_by_stretch.keys())}")
    for stretch, seqs in sequences_by_stretch.items():
        print(f"  {stretch}: {len(seqs)} sequences")
    print(f"{'='*80}")
    
    # Train models
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"{'='*80}")
    
    trained_models = {}
    
    # Train per-stretch models with raw features (for force regression)
    for stretch, sequences in sequences_by_stretch.items():
        result = train_models_for_stretch(
            sequences,
            stretch,
            train_ratio=0.7,
            use_gpu=False,  # Set to True if GPU available
            feature_method=args.feature_method,
            location_feature_method='raw',  # Use raw for location classification
        )
        trained_models[stretch] = result
        
        # Save models
        if result.get('fx_model'):
            joblib.dump(result['fx_model'], models_dir / f"fx_regressor_{stretch}.joblib")
            print(f"  ✓ Saved: fx_regressor_{stretch}.joblib")
        if result.get('fy_model'):
            joblib.dump(result['fy_model'], models_dir / f"fy_regressor_{stretch}.joblib")
            print(f"  ✓ Saved: fy_regressor_{stretch}.joblib")
        if result.get('fz_model'):
            joblib.dump(result['fz_model'], models_dir / f"fz_regressor_{stretch}.joblib")
            print(f"  ✓ Saved: fz_regressor_{stretch}.joblib")
        if result.get('location_model'):
            joblib.dump(result['location_model'], models_dir / f"location_classifier_{stretch}.joblib")
            print(f"  ✓ Saved: location_classifier_{stretch}.joblib")
        if result.get('scaler'):
            joblib.dump(result['scaler'], models_dir / f"scaler_{stretch}.joblib")
    
    # Train combined model with raw features
    result_combined = train_combined_model(
        sequences_by_stretch,
        train_ratio=0.7,
        use_gpu=False,
        feature_method=args.feature_method,
        location_feature_method='raw',
    )
    trained_models['combined'] = result_combined
    
    # Save combined models
    if result_combined.get('fx_model'):
        joblib.dump(result_combined['fx_model'], models_dir / "fx_regressor_combined.joblib")
        print(f"  ✓ Saved: fx_regressor_combined.joblib")
    if result_combined.get('fy_model'):
        joblib.dump(result_combined['fy_model'], models_dir / "fy_regressor_combined.joblib")
        print(f"  ✓ Saved: fy_regressor_combined.joblib")
    if result_combined.get('fz_model'):
        joblib.dump(result_combined['fz_model'], models_dir / "fz_regressor_combined.joblib")
        print(f"  ✓ Saved: fz_regressor_combined.joblib")
    if result_combined.get('location_model'):
        joblib.dump(result_combined['location_model'], models_dir / "location_classifier_combined.joblib")
        print(f"  ✓ Saved: location_classifier_combined.joblib")
    if result_combined.get('scaler'):
        joblib.dump(result_combined['scaler'], models_dir / "scaler_combined.joblib")
    
    # Save stretch classifier (already trained in train_combined_model)
    if result_combined.get('stretch_model') is not None:
        joblib.dump(result_combined['stretch_model'], models_dir / "stretch_classifier_combined.joblib")
        print(f"  ✓ Saved: stretch_classifier_combined.joblib")
    else:
        print("  ⚠️ Stretch classifier not available (insufficient stretch diversity)")
    
    # Train additional models with magnitude features for location classification comparison
    print(f"\n{'='*80}")
    print("TRAINING ALTERNATIVE MODELS FOR LOCATION CLASSIFICATION")
    print(f"{'='*80}")
    
    alternative_methods = ['magnitude', 'magnitude_normalized']
    alternative_models = {}
    
    for alt_method in alternative_methods:
        print(f"\n{'='*80}")
        print(f"Training with location features: {alt_method}")
        print(f"{'='*80}")
        
        alt_trained = {}
        
        # Train per-stretch models
        for stretch, sequences in sequences_by_stretch.items():
            result = train_models_for_stretch(
                sequences,
                stretch,
                train_ratio=0.7,
                use_gpu=False,
                feature_method=args.feature_method,
                location_feature_method=alt_method,
            )
            alt_trained[stretch] = result
        
        # Train combined model
        result_combined_alt = train_combined_model(
            sequences_by_stretch,
            train_ratio=0.7,
            use_gpu=False,
            feature_method=args.feature_method,
            location_feature_method=alt_method,
        )
        alt_trained['combined'] = result_combined_alt
        
        alternative_models[alt_method] = alt_trained
        
        # Save alternative models
        alt_models_dir = models_dir / f"location_{alt_method}"
        alt_models_dir.mkdir(parents=True, exist_ok=True)
        
        for stretch, result in alt_trained.items():
            if stretch == 'combined':
                prefix = "combined"
            else:
                prefix = stretch
            
            if result.get('fx_model'):
                joblib.dump(result['fx_model'], alt_models_dir / f"fx_regressor_{prefix}.joblib")
            if result.get('fy_model'):
                joblib.dump(result['fy_model'], alt_models_dir / f"fy_regressor_{prefix}.joblib")
            if result.get('fz_model'):
                joblib.dump(result['fz_model'], alt_models_dir / f"fz_regressor_{prefix}.joblib")
            if result.get('location_model'):
                joblib.dump(result['location_model'], alt_models_dir / f"location_classifier_{prefix}.joblib")
            if result.get('scaler'):
                joblib.dump(result['scaler'], alt_models_dir / f"scaler_{prefix}.joblib")
        
        print(f"  ✓ Saved alternative models to: {alt_models_dir}")
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("LOCATION CLASSIFICATION COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'000pct':<12} {'010pct':<12} {'020pct':<12} {'Combined':<12}")
    print("-" * 80)
    
    for method_name in ['raw', 'magnitude', 'magnitude_normalized']:
        if method_name == 'raw':
            models_dict = trained_models
        else:
            models_dict = alternative_models.get(method_name, {})
        
        accuracies = []
        for stretch in ['000pct', '010pct', '020pct', 'combined']:
            if stretch in models_dict:
                acc = models_dict[stretch].get('location_accuracy', 0.0)
                accuracies.append(f"{acc:.4f}")
            else:
                accuracies.append("N/A")
        
        print(f"{method_name:<25} {accuracies[0]:<12} {accuracies[1]:<12} {accuracies[2]:<12} {accuracies[3]:<12}")
    
    print(f"{'='*80}")
    
    # Generate plots
    generate_plots_multipoint(
        trained_models,
        sequences_by_stretch,
        plots_dir,
        args.feature_method,
        None,  # No need for h5_files for plots
    )
    
    # Compare normal-only vs normal+shear
    compare_normal_vs_combined(
        normal_files_dict,
        sequences_by_stretch,
        trained_models,
        output_dir,
        args.feature_method,
        remove_outliers_flag=args.remove_outliers,
        z_threshold=args.z_threshold,
        max_sequences_per_stretch=args.max_sequences,
    )
    
    # Save metrics (focus on RMSE)
    metrics = {
        'run_label': args.run_label,
        'feature_method': args.feature_method,
        'n_stretch_levels': len(sequences_by_stretch),
        'max_sequences_per_stretch': args.max_sequences,
        'stretch_results': {},
        'combined_results': {},
    }
    
    for stretch, result in trained_models.items():
        if stretch == 'combined':
            metrics['combined_results'] = {
                'rmse': float(result['rmse']),
                'rmse_train': float(result.get('rmse_train', result['rmse'])),
                'rmse_fx': float(result.get('rmse_fx', 0.0)),
                'rmse_fy': float(result.get('rmse_fy', 0.0)),
                'rmse_fz': float(result.get('rmse_fz', 0.0)),
                'delta_f_min': float(result['delta_f_min']),
                'kpm1': bool(result['kpm1']),
                'kpm2': bool(result['kpm2']),
                'location_accuracy': float(result['location_accuracy']),
            }
        else:
            metrics['stretch_results'][stretch] = {
                'rmse': float(result['rmse']),
                'rmse_train': float(result.get('rmse_train', result['rmse'])),
                'rmse_fx': float(result.get('rmse_fx', 0.0)),
                'rmse_fy': float(result.get('rmse_fy', 0.0)),
                'rmse_fz': float(result.get('rmse_fz', 0.0)),
                'delta_f_min': float(result['delta_f_min']),
                'kpm1': bool(result['kpm1']),
                'kpm2': bool(result['kpm2']),
                'location_accuracy': float(result['location_accuracy']),
                'n_train': int(result['n_train']),
                'n_test': int(result['n_test']),
            }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved: {metrics_path}")
    
    print(f"\n{'='*80}")
    print("✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())