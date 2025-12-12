#!/usr/bin/env python3
"""
Multi-point training script for 10-position patch configuration.

This script is specifically designed for multi-point data collection where
10 positions are tested (arranged as: 2 3 6 7 10 / 1 4 5 8 9).

Features:
- Force regression (Fz prediction)
- Location classification (10 positions: '1' through '10')
- Automatic detection of 10-position configuration
- Separate from single-point training to avoid conflicts

Usage:
    python3 src/training/train_multipoint.py \
        --data-dir data/Multiple_Points/2.5mm_single_test24 \
        --run-label test24_multipoint \
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
                          include_offset_labels: bool = False, use_advanced_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """Prepare features (magnetic), targets (Fz), and offsets from sequences (independent implementation)."""
    all_features = []
    all_fz = []
    all_offsets = []
    
    for seq in sequences:
        magnetic = seq['stretchmagtec']
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        forces = seq.get('forces', None)
        if forces is not None:
            fz_raw = np.abs(forces[:, 2]) if forces.shape[1] >= 3 else np.abs(seq['fz'])
        else:
            fz_raw = np.abs(seq['fz'])
        
        if filter_displacement:
            mag_mask = filter_high_displacement(magnetic, displacement_threshold)
            fz_mask = filter_high_displacement(fz_raw, displacement_threshold)
            combined_mask = mag_mask & fz_mask
        else:
            combined_mask = np.ones(len(magnetic), dtype=bool)
        
        magnetic = magnetic[combined_mask]
        fz_raw = fz_raw[combined_mask]
        
        if len(magnetic) == 0:
            continue
        
        # Cut sequence at 3N
        fz_cut_mask = fz_raw <= 3.0
        if np.any(~fz_cut_mask):
            first_exceed_idx = np.where(~fz_cut_mask)[0]
            if len(first_exceed_idx) > 0:
                cut_idx = first_exceed_idx[0]
                magnetic = magnetic[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
        
        if len(magnetic) == 0:
            continue
        
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
        all_fz.append(fz_raw)
        all_offsets.append(np.full(len(fz_raw), offset))
    
    X = np.vstack(all_features)
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
    
    y_fz_raw = np.round(y_fz_raw, decimals=FT_SENSOR_DECIMALS)
    fz_scaler = None
    y_fz = y_fz_raw
    fz_min = np.min(y_fz_raw)
    fz_max = np.max(y_fz_raw)
    print(f"  Fz range: [{fz_min:.3f}, {fz_max:.3f}] N (not normalized, sequences cut at 3N, rounded to {FT_SENSOR_PRECISION}N precision)")
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Also return actual sequence lengths after filtering
    actual_sequence_lengths = [len(f) for f in all_features]
    
    return X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths


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
) -> Dict:
    """Train force regressor and location classifier for a specific stretch level."""
    
    print(f"\n{'='*80}")
    print(f"Training models for {stretch_label}")
    print(f"{'='*80}")
    print(f"Total sequences: {len(sequences)}")
    
    # Prepare data
    X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        sequences, 
        normalize=True, 
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=False,  # Don't include offset labels as features
        use_advanced_features=False
    )
    print(f"Total samples: {len(X)}")
    if X.shape[1] == 15:
        print(f"Features: {X.shape[1]} (15 normalized per sensor)")
    elif X.shape[1] == 45:
        print(f"Features: {X.shape[1]} (45 raw features)")
    else:
        print(f"Features: {X.shape[1]} features")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (normalized to [{fz_target_min:.1f}, {fz_target_max:.1f}])")
    
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
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressor
    # Note: Includes ALL sequences (with contact AND no_touch)
    # no_touch sequences help the model learn to predict zero force when there's no contact
    print("\nTraining force regressor...")
    print("  Note: Includes all sequences (contact + no_touch) to improve accuracy at zero force")
    force_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
    force_model.fit(X_train, y_fz_train)
    y_fz_pred = force_model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred))
    std_dev = np.std(y_fz_test - y_fz_pred)
    
    print(f"  RMSE: {rmse:.4f} N")
    print(f"  Std Dev: {std_dev:.4f} N")
    
    # Calculate force resolution (ΔF_min)
    # Round to match FT sensor precision (0.01N = 2 decimal places)
    unique_forces = np.unique(np.round(y_fz_test, decimals=2))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        delta_f_min = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        delta_f_min = float("nan")
    print(f"  Force resolution (ΔF_min): {delta_f_min:.6f} N")
    
    # KPM1: Force resolution <= 0.05N
    kpm1 = delta_f_min <= 0.05
    print(f"  KPM1: {'PASS' if kpm1 else 'FAIL'}")
    
    # KPM2: RMSE < 0.1N and STD < 0.05N
    kpm2 = rmse < 0.1 and std_dev < 0.05
    print(f"  KPM2: {'PASS' if kpm2 else 'FAIL'}")
    
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
                'force_model': force_model,
                'location_model': location_model,
                'scaler': scaler,
                'rmse': rmse,
                'std_dev': std_dev,
                'delta_f_min': delta_f_min,
                'kpm1': kpm1,
                'kpm2': kpm2,
                'location_accuracy': location_accuracy,
                'location_cm': cm,
                'location_test_labels': y_location_test_valid,
                'location_pred_labels': y_location_pred,
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
        'force_model': force_model,
        'location_model': location_model,
        'scaler': scaler,
        'rmse': rmse,
        'std_dev': std_dev,
        'delta_f_min': delta_f_min,
        'kpm1': kpm1,
        'kpm2': kpm2,
        'location_accuracy': location_accuracy,
        'location_cm': cm,
        'location_test_labels': None,
        'location_pred_labels': None,
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
    X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=False,
        use_advanced_features=False
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N")
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
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressor
    # Note: Includes ALL sequences (with contact AND no_touch)
    # no_touch sequences help the model learn to predict zero force when there's no contact
    print("\nTraining force regressor...")
    print("  Note: Includes all sequences (contact + no_touch) to improve accuracy at zero force")
    force_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=1)
    force_model.fit(X_train, y_fz_train)
    y_fz_pred = force_model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred))
    std_dev = np.std(y_fz_test - y_fz_pred)
    
    print(f"  RMSE: {rmse:.4f} N")
    print(f"  Std Dev: {std_dev:.4f} N")
    
    # Calculate force resolution (ΔF_min)
    # Round to match FT sensor precision (0.01N = 2 decimal places)
    unique_forces = np.unique(np.round(y_fz_test, decimals=2))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        delta_f_min = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        delta_f_min = float("nan")
    print(f"  Force resolution (ΔF_min): {delta_f_min:.6f} N")
    
    kpm1 = delta_f_min <= 0.05
    print(f"  KPM1: {'PASS' if kpm1 else 'FAIL'}")
    
    kpm2 = rmse < 0.1 and std_dev < 0.05
    print(f"  KPM2: {'PASS' if kpm2 else 'FAIL'}")
    
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
        'force_model': force_model,
        'location_model': location_model,
        'stretch_model': stretch_model,
        'scaler': scaler,
        'rmse': rmse,
        'std_dev': std_dev,
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
        
        if result.get('fz_test') is not None and result.get('fz_pred') is not None:
            fz_test = result['fz_test']
            fz_pred = result['fz_pred']
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(fz_test, fz_pred, alpha=0.5, s=10)
            ax.plot([fz_test.min(), fz_test.max()], [fz_test.min(), fz_test.max()], 'r--', lw=2, label='Perfect prediction')
            ax.set_xlabel('True Fz (N)', fontsize=12)
            ax.set_ylabel('Predicted Fz (N)', fontsize=12)
            ax.set_title(f'Force Regression - {stretch.upper()}\nRMSE: {result["rmse"]:.4f} N, Std Dev: {result["std_dev"]:.4f} N', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = plots_dir / f"prediction_scatter_{stretch}_{feature_method}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_path.name}")
            prediction_plots_saved.append(plot_path)
            
            # Residual plot
            residuals = fz_test - fz_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(fz_test, residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('True Fz (N)', fontsize=12)
            ax.set_ylabel('Residual (True - Predicted) (N)', fontsize=12)
            ax.set_title(f'Residuals - {stretch.upper()}\nRMSE: {result["rmse"]:.4f} N, Std Dev: {result["std_dev"]:.4f} N', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = plots_dir / f"prediction_residuals_{stretch}_{feature_method}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {plot_path.name}")
            prediction_plots_saved.append(plot_path)
    
    # Combined prediction plot
    if 'combined' in trained_models:
        result = trained_models['combined']
        if result.get('fz_test') is not None and result.get('fz_pred') is not None:
            fz_test = result['fz_test']
            fz_pred = result['fz_pred']
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(fz_test, fz_pred, alpha=0.5, s=10)
            ax.plot([fz_test.min(), fz_test.max()], [fz_test.min(), fz_test.max()], 'r--', lw=2, label='Perfect prediction')
            ax.set_xlabel('True Fz (N)', fontsize=12)
            ax.set_ylabel('Predicted Fz (N)', fontsize=12)
            ax.set_title(f'Force Regression - Combined\nRMSE: {result["rmse"]:.4f} N, Std Dev: {result["std_dev"]:.4f} N', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = plots_dir / f"prediction_scatter_combined_{feature_method}.png"
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


def load_data_multipoint(
    h5_files: Dict[str, Path | List[Path]],
    remove_outliers_flag: bool = True,
    z_threshold: float = 3.0,
) -> Dict[str, List[Dict]]:
    """Load and preprocess multi-point data (10 locations + no_touch)."""
    
    print(f"\n{'='*80}")
    print("LOADING MULTI-POINT DATA (10 LOCATIONS + NO_TOUCH)")
    print(f"{'='*80}")
    
    sequences_by_stretch = {}
    
    for stretch, h5_file_or_list in h5_files.items():
        # Handle both single file and list of files
        if isinstance(h5_file_or_list, list):
            h5_file_list = h5_file_or_list
        else:
            h5_file_list = [h5_file_or_list]
        
        all_sequences = []
        for h5_file in h5_file_list:
            file_type = "no_touch" if "no_touch" in h5_file.name else "regular"
            print(f"\nLoading {stretch} from {h5_file.name} ({file_type})...")
            sequences = load_sequences_from_h5(h5_file)
            print(f"  Loaded {len(sequences)} sequences from {h5_file.name}")
            
            # Verify that no_touch files match the stretch level
            # Note: Files may contain "000" instead of "000pct", so check for both
            if "no_touch" in h5_file.name:
                expected_stretch = stretch
                stretch_num = stretch.replace('pct', '')  # e.g., "000" from "000pct"
                if expected_stretch not in h5_file.name and stretch_num not in h5_file.name:
                    print(f"  ⚠️  WARNING: no_touch file {h5_file.name} does not match stretch level {stretch}!")
            
            # Ensure all sequences have the correct stretch_label (always override)
            for seq in sequences:
                seq['stretch_label'] = stretch  # Always set from the file's stretch level
            
            all_sequences.extend(sequences)
        
        sequences = all_sequences
        print(f"  Total sequences for {stretch}: {len(sequences)}")
        
        if sequences:
            # Count sequences per location (not offset!)
            initial_location_counts = {}
            for seq in sequences:
                # For multi-point, offset is actually the location ID ('1'-'10' or 'no_touch')
                location = seq.get('offset', 'unknown')
                initial_location_counts[location] = initial_location_counts.get(location, 0) + 1
            print(f"  Initial sequences per location: {initial_location_counts}")
            
            # Debug: check if no_touch is present
            if 'no_touch' in initial_location_counts:
                print(f"  ✓ no_touch sequences found: {initial_location_counts['no_touch']}")
            else:
                print(f"  ⚠️  WARNING: no_touch sequences NOT found in loaded data!")
                print(f"  Available locations: {list(initial_location_counts.keys())}")
            print(f"  Total initial sequences: {len(sequences)}")
            print(f"  NOTE: First sequence per location was already removed during data collection")
            
            if remove_outliers_flag:
                # Remove outliers (2 per location, independently)
                n_locations = len(initial_location_counts)
                print(f"  Removing outliers (2 per location, independently)...")
                print(f"  Expected after outlier removal: {len(sequences)} - ({n_locations} locations * 2 outliers) = {len(sequences) - (n_locations * 2)} sequences")
                cleaned_sequences, outlier_indices = remove_outliers(sequences, z_threshold=z_threshold, remove_per_offset=2)
                print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
                
                # Count sequences per location to verify
                location_counts = {}
                for seq in cleaned_sequences:
                    location = seq.get('offset', 'unknown')  # Still stored as 'offset' in sequence dict
                    location_counts[location] = location_counts.get(location, 0) + 1
                print(f"  Sequences per location after cleaning: {location_counts}")
                
                sequences_by_stretch[stretch] = cleaned_sequences
            else:
                sequences_by_stretch[stretch] = sequences
    
    return sequences_by_stretch


def main():
    parser = argparse.ArgumentParser(
        description="Multi-point training script for 10-position patch configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing HDF5 files (e.g., data/Multiple_Points/2.5mm_single_test24)'
    )
    parser.add_argument(
        '--run-label',
        type=str,
        required=True,
        help='Label for this training run (e.g., test24_multipoint)'
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
    
    args = parser.parse_args()
    
    # Find HDF5 files
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return 1
    
    # Find all HDF5 files (including no_touch files)
    h5_files_dict = {}
    for stretch in ['000pct', '010pct', '020pct']:
        pattern = f"*stretch_{stretch}.h5"
        files = list(data_dir.glob(pattern))
        
        # Also explicitly search for no_touch files (they might not match the pattern)
        no_touch_pattern = f"no_touch_*_stretch_{stretch}.h5"
        no_touch_files = list(data_dir.glob(no_touch_pattern))
        
        # Combine and remove duplicates
        all_files = list(set(files + no_touch_files))
        
        if all_files:
            # If multiple files found (e.g., regular + no_touch), use all of them
            # The load_data_multipoint function will handle multiple files per stretch
            h5_files_dict[stretch] = all_files  # Store list of files
            print(f"Found {stretch}: {len(all_files)} file(s)")
            for f in all_files:
                file_type = "no_touch" if "no_touch" in f.name else "regular"
                print(f"  - {f.name} ({file_type})")
    
    if not h5_files_dict:
        print(f"❌ Error: No HDF5 files found in {data_dir}")
        return 1
    
    # Create output directory structure
    output_dir = data_dir / "cleaned" / args.feature_method
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    data_cleaned_dir = output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if cleaned data exists
    cleaned_files = {}
    for stretch in ['000pct', '010pct', '020pct']:
        # Try multiple patterns to find all files (including no_touch)
        pattern1 = f"test_*_{stretch.replace('pct', '')}_cleaned.h5"
        pattern2 = f"*stretch_{stretch}_cleaned.h5"
        pattern3 = f"no_touch_*_stretch_{stretch}_cleaned.h5"
        
        files = list(data_cleaned_dir.glob(pattern1))
        if not files:
            files = list(data_cleaned_dir.glob(pattern2))
        # Also look for no_touch files specifically
        no_touch_files = list(data_cleaned_dir.glob(pattern3))
        if no_touch_files:
            files.extend(no_touch_files)
        
        # Remove duplicates
        files = list(set(files))
        
        if files:
            # Store all cleaned files for this stretch (including no_touch)
            cleaned_files[stretch] = files if len(files) > 1 else files[0]
            print(f"Found cleaned files for {stretch}: {len(files)} file(s)")
            for f in files if isinstance(files, list) else [files]:
                print(f"  - {f.name if hasattr(f, 'name') else f}")
    
    # Check if we need to clean: verify that all files (including no_touch) have been cleaned
    need_cleaning = False
    if not cleaned_files:
        need_cleaning = True
        print(f"Cleaned data directory is empty. Cleaning dataset...")
    else:
        # Check if all original files have corresponding cleaned files
        for stretch, h5_file_or_list in h5_files_dict.items():
            if isinstance(h5_file_or_list, list):
                h5_file_list = h5_file_or_list
            else:
                h5_file_list = [h5_file_or_list]
            
            # Count how many cleaned files we have for this stretch
            cleaned_for_stretch = cleaned_files.get(stretch, [])
            if isinstance(cleaned_for_stretch, list):
                num_cleaned = len(cleaned_for_stretch)
            else:
                num_cleaned = 1
            
            # Count how many no_touch files we have in original
            num_no_touch_original = sum(1 for f in h5_file_list if "no_touch" in f.name)
            
            # Count how many no_touch cleaned files we have
            if isinstance(cleaned_for_stretch, list):
                num_no_touch_cleaned = sum(1 for f in cleaned_for_stretch if "no_touch" in str(f))
            else:
                num_no_touch_cleaned = 1 if "no_touch" in str(cleaned_for_stretch) else 0
            
            # We should have as many cleaned files as original files
            if num_cleaned < len(h5_file_list):
                need_cleaning = True
                print(f"Missing cleaned files for {stretch}: have {num_cleaned}, need {len(h5_file_list)}")
                break
            
            # Also check specifically for no_touch files
            if num_no_touch_original > 0 and num_no_touch_cleaned == 0:
                need_cleaning = True
                print(f"Missing no_touch cleaned files for {stretch}: have {num_no_touch_cleaned}, need {num_no_touch_original}")
                break
    
    # If cleaned data doesn't exist or is incomplete, clean the dataset
    if need_cleaning:
        print(f"\n{'='*80}")
        print("CLEANING DATASET")
        print(f"{'='*80}")
        print(f"Cleaning all files (including no_touch)...")
        
        from training.clean_sequences import clean_single_file, extract_test_info
        
        for stretch, h5_file_or_list in h5_files_dict.items():
            # Handle both single file and list of files
            if isinstance(h5_file_or_list, list):
                h5_file_list = h5_file_or_list
            else:
                h5_file_list = [h5_file_or_list]
            
            # Clean all files for this stretch level
            cleaned_file_list = []
            for h5_file in h5_file_list:
                # Preserve "no_touch" in filename if present
                is_no_touch = "no_touch" in h5_file.name
                
                test_num, stretch_level = extract_test_info(h5_file)
                if test_num and stretch_level:
                    if is_no_touch:
                        new_filename = f"no_touch_test_{test_num}_{stretch_level}_cleaned.h5"
                    else:
                        new_filename = f"test_{test_num}_{stretch_level}_cleaned.h5"
                    output_path = data_cleaned_dir / new_filename
                else:
                    # Preserve original filename structure
                    output_path = data_cleaned_dir / h5_file.name.replace('.h5', '_cleaned.h5')
                
                file_type = "no_touch" if is_no_touch else "regular"
                print(f"\nCleaning {stretch} from {h5_file.name} ({file_type})...")
                _, saved_path = clean_single_file(h5_file, output_path, args.z_threshold, remove_per_offset=2)
                if saved_path:
                    cleaned_file_list.append(saved_path)
                    print(f"  ✓ Saved cleaned file: {saved_path.name}")
            
            if cleaned_file_list:
                # Store list if multiple files, single file if one
                cleaned_files[stretch] = cleaned_file_list if len(cleaned_file_list) > 1 else cleaned_file_list[0]
    else:
        print(f"\n{'='*80}")
        print("USING EXISTING CLEANED DATA")
        print(f"{'='*80}")
        for stretch, cleaned_file_or_list in cleaned_files.items():
            if isinstance(cleaned_file_or_list, list):
                for f in cleaned_file_or_list:
                    print(f"  {stretch}: {f.name}")
            else:
                print(f"  {stretch}: {cleaned_file_or_list.name}")
    
    # Use cleaned files for training
    h5_files_dict = cleaned_files
    
    print(f"\n{'='*80}")
    print("MULTI-POINT TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Run label: {args.run_label}")
    print(f"Feature method: {args.feature_method}")
    print(f"Output directory: {output_dir}")
    print(f"HDF5 files found: {len(h5_files_dict)}")
    for stretch, h5_file_or_list in h5_files_dict.items():
        if isinstance(h5_file_or_list, list):
            print(f"  {stretch}: {len(h5_file_or_list)} file(s)")
            for f in h5_file_or_list:
                print(f"    - {f.name}")
        else:
            print(f"  {stretch}: {h5_file_or_list.name}")
    print(f"{'='*80}")
    
    # Load data from cleaned files (no additional cleaning needed)
    sequences_by_stretch = load_data_multipoint(
        h5_files_dict,
        remove_outliers_flag=False,  # Already cleaned
        z_threshold=args.z_threshold,
    )
    
    # Don't balance sequences - keep all to ensure all stretch levels are present for stretch classifier
    # sequences_by_stretch = balance_sequences(sequences_by_stretch)
    
    # Train models
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"{'='*80}")
    
    trained_models = {}
    
    # Train per-stretch models
    for stretch, sequences in sequences_by_stretch.items():
        result = train_models_for_stretch(
            sequences,
            stretch,
            train_ratio=0.7,
            use_gpu=False,  # Set to True if GPU available
            feature_method=args.feature_method,
        )
        trained_models[stretch] = result
        
        # Save models
        if result['force_model']:
            joblib.dump(result['force_model'], models_dir / f"force_regressor_{stretch}.joblib")
            print(f"  ✓ Saved: force_regressor_{stretch}.joblib")
        if result['location_model']:
            joblib.dump(result['location_model'], models_dir / f"location_classifier_{stretch}.joblib")
            print(f"  ✓ Saved: location_classifier_{stretch}.joblib")
        if result['scaler']:
            joblib.dump(result['scaler'], models_dir / f"scaler_{stretch}.joblib")
    
    # Train combined model
    result_combined = train_combined_model(
        sequences_by_stretch,
        train_ratio=0.7,
        use_gpu=False,
        feature_method=args.feature_method,
    )
    trained_models['combined'] = result_combined
    
    # Save combined models
    if result_combined['force_model']:
        joblib.dump(result_combined['force_model'], models_dir / "force_regressor_combined.joblib")
        print(f"  ✓ Saved: force_regressor_combined.joblib")
    if result_combined['location_model']:
        joblib.dump(result_combined['location_model'], models_dir / "location_classifier_combined.joblib")
        print(f"  ✓ Saved: location_classifier_combined.joblib")
    if result_combined['scaler']:
        joblib.dump(result_combined['scaler'], models_dir / "scaler_combined.joblib")
    
    # Save stretch classifier (already trained in train_combined_model)
    if result_combined.get('stretch_model') is not None:
        joblib.dump(result_combined['stretch_model'], models_dir / "stretch_classifier_combined.joblib")
        print(f"  ✓ Saved: stretch_classifier_combined.joblib")
    else:
        print("  ⚠️ Stretch classifier not available (insufficient stretch diversity)")
    
    # Generate plots
    # Use cleaned files if available, otherwise original files
    h5_files_for_plots = cleaned_files if cleaned_files else h5_files_dict
    generate_plots_multipoint(
        trained_models,
        sequences_by_stretch,
        plots_dir,
        args.feature_method,
        h5_files_for_plots,
    )
    
    # Save metrics
    metrics = {
        'run_label': args.run_label,
        'feature_method': args.feature_method,
        'n_stretch_levels': len(sequences_by_stretch),
        'stretch_results': {},
        'combined_results': {},
    }
    
    for stretch, result in trained_models.items():
        if stretch == 'combined':
            metrics['combined_results'] = {
                'rmse': float(result['rmse']),
                'std_dev': float(result['std_dev']),
                'delta_f_min': float(result['delta_f_min']),
                'kpm1': bool(result['kpm1']),
                'kpm2': bool(result['kpm2']),
                'location_accuracy': float(result['location_accuracy']),
            }
        else:
            metrics['stretch_results'][stretch] = {
                'rmse': float(result['rmse']),
                'std_dev': float(result['std_dev']),
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