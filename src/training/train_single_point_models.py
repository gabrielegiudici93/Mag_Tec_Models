#!/usr/bin/env python3
"""
Train models for single-point data with automatic outlier removal and balanced datasets.

This script:
1. Loads sequences from HDF5 files for 0%, 10%, 20% stretch
2. Automatically identifies and removes outlier sequences
3. Balances the number of sequences across stretch levels
4. Trains 3 separate models (one per stretch level) and 1 combined model
5. Uses 70% of sequences for training (not 70% of samples)
6. Prints statistics for each model
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Try to import GPU-accelerated libraries
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
    from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Fallback to scikit-learn
if not CUML_AVAILABLE and not XGBOOST_AVAILABLE:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    SKLEARN_AVAILABLE = True
else:
    SKLEARN_AVAILABLE = False

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import MODELS_DIR, LOGS_DIR

# =============================================================================
# FT SENSOR PRECISION CONFIGURATION
# =============================================================================
# FT sensor precision in Newtons (standard deviation or resolution)
# This value determines:
# - Rounding precision for force values during training (decimals = -log10(FT_SENSOR_PRECISION))
# - Force resolution calculation (ΔF_min) precision
# - KPM evaluation precision
# 
# Change this value to adjust sensor precision throughout the training pipeline
# Examples:
#   FT_SENSOR_PRECISION = 0.01  # 0.01N precision (2 decimal places)
#   FT_SENSOR_PRECISION = 0.1   # 0.1N precision (1 decimal place)
#   FT_SENSOR_PRECISION = 0.05  # 0.05N precision (2 decimal places)
FT_SENSOR_PRECISION = 0.01  # FT sensor precision: 0.01N (2 decimal places)

# Calculate number of decimal places from precision
FT_SENSOR_DECIMALS = int(-np.log10(FT_SENSOR_PRECISION)) if FT_SENSOR_PRECISION > 0 else 2


def load_sequences_from_h5(h5_path: Path) -> List[Dict]:
    """Load all press sequences from an HDF5 file."""
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
            else:
                # Fall back to extracting from label
                offset = 'unknown'
                # First try to find numeric offsets (1-10) for multi-point data
                import re
                numeric_match = re.search(r'offset[_\s]*(\d+)|pos_\d+_(\d+)_press', label.lower())
                if numeric_match:
                    offset = numeric_match.group(1) or numeric_match.group(2)  # Keep as string '1', '2', ..., '10'
                else:
                    # Fall back to named offsets (center, ne, nw, sw, se) for single-point data
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
    """Identify outlier sequences using statistical methods.
    
    Uses Median Absolute Deviation (MAD) to identify outliers in:
    - Duration (with special attention to very long sequences > 20s)
    - Max Fz
    - Number of samples
    
    Returns indices of outlier sequences.
    """
    if len(sequences) < 3:
        return []
    
    # Extract features
    durations = np.array([s['duration'] for s in sequences])
    fz_maxes = np.array([s['fz_max'] for s in sequences])
    num_samples = np.array([s['num_samples'] for s in sequences])
    
    outlier_indices = set()
    
    # Check each feature with MAD
    for feature_name, feature_values in [
        ('duration', durations),
        ('fz_max', fz_maxes),
        ('num_samples', num_samples),
    ]:
        median = np.median(feature_values)
        mad = np.median(np.abs(feature_values - median))
        
        if mad > 0:
            # Use modified Z-score with MAD
            # For duration, use a more sensitive threshold (2.0) to catch sequences
            # that are significantly longer/shorter than the median, even if not > 20s
            threshold = z_threshold if feature_name != 'duration' else 2.0
            modified_z_scores = 0.6745 * (feature_values - median) / mad
            outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
            if len(outliers) > 0 and feature_name == 'duration':
                # Print info about duration outliers
                outlier_durations = feature_values[outliers]
                print(f"    Found {len(outliers)} duration outliers: median={median:.2f}s, outliers={outlier_durations}")
            outlier_indices.update(outliers)
    
    return sorted(list(outlier_indices))


def remove_outliers(sequences: List[Dict], z_threshold: float = 3.0, remove_per_offset: int = 2) -> Tuple[List[Dict], List[int]]:
    """Remove outlier sequences and return cleaned list.
    
    Args:
        sequences: List of sequence dictionaries
        z_threshold: Z-score threshold for outlier detection
        remove_per_offset: Number of outliers to remove per offset (default: 2)
    
    Returns:
        Tuple of (cleaned_sequences, outlier_indices)
    """
    if remove_per_offset <= 0:
        return sequences, []
    
    # Group sequences by offset
    sequences_by_offset = {}
    for idx, seq in enumerate(sequences):
        offset = seq.get('offset', 'unknown')
        if offset not in sequences_by_offset:
            sequences_by_offset[offset] = []
        sequences_by_offset[offset].append((idx, seq))
    
    all_outlier_indices = set()
    
    # Remove outliers independently for each offset
    for offset, offset_sequences in sequences_by_offset.items():
        if len(offset_sequences) <= remove_per_offset:
            print(f"  Offset {offset}: Only {len(offset_sequences)} sequences, skipping outlier removal")
            continue
        
        # Extract just the sequences (without indices) for outlier detection
        offset_seq_list = [seq for _, seq in offset_sequences]
        
        # Identify outliers for this offset
        outlier_indices_local = identify_outliers(offset_seq_list, z_threshold)
        
        # Calculate outlier scores for ALL sequences to find the most extreme ones
        # This ensures we can always remove remove_per_offset sequences, even if identify_outliers finds fewer
        offset_seq_list_with_scores = []
        fz_values = [s.get('max_fz', 0) for s in offset_seq_list]
        duration_values = [s.get('duration', 0) for s in offset_seq_list]
        num_samples_values = [s.get('num_samples', 0) for s in offset_seq_list]
        
        fz_median = np.median(fz_values)
        fz_mad = np.median(np.abs(np.array(fz_values) - fz_median)) if len(fz_values) > 0 else 1.0
        duration_median = np.median(duration_values)
        duration_mad = np.median(np.abs(np.array(duration_values) - duration_median)) if len(duration_values) > 0 else 1.0
        num_samples_median = np.median(num_samples_values)
        num_samples_mad = np.median(np.abs(np.array(num_samples_values) - num_samples_median)) if len(num_samples_values) > 0 else 1.0
        
        for local_idx, (global_idx, seq) in enumerate(offset_sequences):
            # Calculate a combined outlier score
            fz = seq.get('max_fz', 0)
            duration = seq.get('duration', 0)
            num_samples = seq.get('num_samples', 0)
            
            score = 0
            if fz_mad > 0:
                score += abs((fz - fz_median) / fz_mad)
            if duration_mad > 0:
                # Give duration higher weight to catch sequences significantly longer/shorter than median
                duration_score = abs((duration - duration_median) / duration_mad)
                # Extra penalty for sequences much longer than median (e.g., 2x median or more)
                if duration > duration_median * 2.0:
                    duration_score *= 2.0
                score += duration_score * 1.5  # 1.5x weight for duration
            if num_samples_mad > 0:
                score += abs((num_samples - num_samples_median) / num_samples_mad)
            
            # Prioritize sequences identified as outliers by identify_outliers
            is_identified_outlier = local_idx in outlier_indices_local
            if is_identified_outlier:
                score *= 2.0  # Double the score for identified outliers
            
            offset_seq_list_with_scores.append((global_idx, score, is_identified_outlier))
        
        # Sort by score (highest = most extreme) and take top remove_per_offset
        # First prioritize identified outliers, then by score
        offset_seq_list_with_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)  # First by outlier flag, then by score
        
        # ALWAYS remove exactly remove_per_offset sequences (or as many as available if fewer)
        num_to_remove = min(remove_per_offset, len(offset_seq_list_with_scores))
        outliers_to_remove = [idx for idx, _, _ in offset_seq_list_with_scores[:num_to_remove]]
        
        all_outlier_indices.update(outliers_to_remove)
        print(f"  Offset {offset}: Removing {len(outliers_to_remove)} outliers (indices: {sorted(outliers_to_remove)})")
    
    if all_outlier_indices:
        print(f"  Total outliers removed: {len(all_outlier_indices)} (indices: {sorted(all_outlier_indices)})")
        # Remove outliers (in reverse order to maintain indices)
        cleaned_sequences = [s for i, s in enumerate(sequences) if i not in all_outlier_indices]
    else:
        cleaned_sequences = sequences
    
    return cleaned_sequences, sorted(list(all_outlier_indices))


def balance_sequences(sequences_by_stretch: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Balance number of sequences across stretch levels by taking minimum."""
    if not sequences_by_stretch:
        return sequences_by_stretch
    
    min_count = min(len(seqs) for seqs in sequences_by_stretch.values())
    
    balanced = {}
    for stretch_label, sequences in sequences_by_stretch.items():
        if len(sequences) > min_count:
            # Randomly sample to match minimum
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(sequences), size=min_count, replace=False)
            balanced[stretch_label] = [sequences[i] for i in sorted(indices)]
            print(f"  Balanced {stretch_label}: {len(sequences)} -> {min_count} sequences")
        else:
            balanced[stretch_label] = sequences
    
    return balanced


def filter_high_displacement(data: np.ndarray, threshold_percentile: float = 95.0) -> np.ndarray:
    """Filter out samples with high consecutive displacement.
    
    Args:
        data: Array of shape [samples, ...] (can be 1D, 2D, or 3D)
        threshold_percentile: Percentile threshold for displacement (default 95%)
    
    Returns:
        Boolean mask: True for samples to keep, False for samples to remove
    """
    if len(data) < 2:
        return np.ones(len(data), dtype=bool)
    
    # Calculate displacement between consecutive samples
    # For multi-dimensional data, use Euclidean distance
    if data.ndim == 1:
        displacements = np.abs(np.diff(data))
    elif data.ndim == 2:
        # Euclidean distance for each consecutive pair
        displacements = np.linalg.norm(np.diff(data, axis=0), axis=1)
    elif data.ndim == 3:
        # Flatten last two dimensions, then calculate distance
        data_flat = data.reshape(len(data), -1)
        displacements = np.linalg.norm(np.diff(data_flat, axis=0), axis=1)
    else:
        # For higher dimensions, flatten and use Euclidean distance
        data_flat = data.reshape(len(data), -1)
        displacements = np.linalg.norm(np.diff(data_flat, axis=0), axis=1)
    
    # Calculate threshold based on percentile
    threshold = np.percentile(displacements, threshold_percentile)
    
    # Create mask: keep first sample, then keep samples where displacement < threshold
    mask = np.ones(len(data), dtype=bool)
    mask[1:] = displacements < threshold
    
    return mask


def normalize_fz_to_range(fz: np.ndarray, target_min: float = 0.0, target_max: float = 3.0) -> np.ndarray:
    """Normalize Fz values to target range [target_min, target_max].
    
    Args:
        fz: Array of Fz values
        target_min: Target minimum value (default 0.0)
        target_max: Target maximum value (default 3.0)
    
    Returns:
        Normalized Fz values in range [target_min, target_max]
    """
    fz_min = np.min(fz)
    fz_max = np.max(fz)
    
    if fz_max - fz_min < 1e-6:  # Avoid division by zero
        return np.full_like(fz, (target_min + target_max) / 2.0)
    
    # Linear normalization: map [fz_min, fz_max] -> [target_min, target_max]
    normalized = (fz - fz_min) / (fz_max - fz_min) * (target_max - target_min) + target_min
    
    return normalized


def prepare_training_data(sequences: List[Dict], normalize: bool = True, use_feature_engineering: bool = False, 
                          filter_displacement: bool = True, displacement_threshold: float = 95.0,
                          normalize_fz: bool = True, fz_target_min: float = 0.0, fz_target_max: float = 3.0,
                          include_offset_labels: bool = False, use_advanced_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """Prepare features (magnetic), targets (Fz), and offsets from sequences.
    
    Args:
        sequences: List of sequence dictionaries
        normalize: Whether to normalize features
        use_feature_engineering: Whether to add engineered features (statistics, magnitudes, etc.)
        filter_displacement: Whether to filter out samples with high consecutive displacement
        displacement_threshold: Percentile threshold for displacement filtering (default 95%)
        normalize_fz: Whether to normalize Fz to target range [fz_target_min, fz_target_max]
        fz_target_min: Target minimum for Fz normalization (default 0.0)
        fz_target_max: Target maximum for Fz normalization (default 3.0)
        include_offset_labels: Whether to add offset labels as one-hot encoded features
        use_advanced_features: Whether to add advanced features (statistical + temporal features)
    
    Returns:
        X: Features [samples, N] where N depends on feature method and advanced features
        y_fz: Fz values (normalized to [fz_target_min, fz_target_max] if normalize_fz=True)
        y_offset: Offset labels (integers: 0=center, 1=ne, 2=nw, 3=se, 4=sw, -1=unknown)
        scaler: Fitted StandardScaler (if normalize=True) or None
        fz_scaler: Tuple of (fz_min, fz_max) used for normalization or None
    """
    from sklearn.preprocessing import StandardScaler
    
    all_features = []
    all_fz = []
    all_offsets = []
    
    for seq in sequences:
        # Magnetic data: [samples, 15, 3]
        magnetic = seq['stretchmagtec']
        
        # Filter magnetic sensor: set values below 250 to 0 (noise threshold)
        magnetic = np.where(np.abs(magnetic) < 250, 0, magnetic)
        
        # FT data: [samples, 6] or [samples, 3] - we need Fz
        forces = seq.get('forces', None)
        if forces is not None:
            fz_raw = np.abs(forces[:, 2]) if forces.shape[1] >= 3 else np.abs(seq['fz'])
        else:
            fz_raw = np.abs(seq['fz'])
        
        # Apply displacement filter separately for magnetic and FT data
        if filter_displacement:
            # Filter magnetic data
            mag_mask = filter_high_displacement(magnetic, displacement_threshold)
            # Filter FT data (Fz)
            fz_mask = filter_high_displacement(fz_raw, displacement_threshold)
            # Keep samples where both sensors pass the filter
            combined_mask = mag_mask & fz_mask
        else:
            combined_mask = np.ones(len(magnetic), dtype=bool)
        
        # Apply filters
        magnetic = magnetic[combined_mask]
        fz_raw = fz_raw[combined_mask]
        
        if len(magnetic) == 0:
            print(f"    Warning: Sequence {seq.get('press_key', 'unknown')} (offset: {seq.get('offset', 'unknown')}) is empty after filtering, skipping")
            continue  # Skip empty sequences
        
        # Cut sequence at 3N: keep only samples where Fz <= 3.0N
        # Find first index where Fz exceeds 3N, then cut everything after
        fz_cut_mask = fz_raw <= 3.0
        if np.any(~fz_cut_mask):
            # Find first index where Fz exceeds 3N
            first_exceed_idx = np.where(~fz_cut_mask)[0]
            if len(first_exceed_idx) > 0:
                cut_idx = first_exceed_idx[0]
                # Cut magnetic, fz, and also timestamps if available
                magnetic = magnetic[:cut_idx]
                fz_raw = fz_raw[:cut_idx]
                # Also cut timestamps if they exist in the sequence
                if 'timestamps' in seq and len(seq['timestamps']) > cut_idx:
                    seq['timestamps'] = seq['timestamps'][:cut_idx]
        
        if len(magnetic) == 0:
            print(f"    Warning: Sequence {seq.get('press_key', 'unknown')} (offset: {seq.get('offset', 'unknown')}) is empty after filtering/cutting, skipping")
            continue  # Skip empty sequences after cutting
        
        if use_feature_engineering:
            # METHOD 1: 15 features (one per sensor), each normalized independently
            # For each sensor, normalize the [x,y,z] triple using the sensor's global max/min
            # Then use the normalized magnitude as the feature
            
            # Compute magnitude for each sensor: [samples, 15]
            magnitudes = np.sqrt(np.sum(magnetic**2, axis=2))  # [samples, 15]
            
            # For each sensor, find global min/max across all samples for normalization
            # We need to normalize each sensor's [x,y,z] values independently
            # Then compute magnitude from normalized values
            normalized_magnitudes = np.zeros_like(magnitudes)  # [samples, 15]
            
            for sensor_idx in range(15):
                # Get all [x,y,z] values for this sensor across all samples
                sensor_xyz = magnetic[:, sensor_idx, :]  # [samples, 3]
                
                # Find global min and max across all [x,y,z] values for this sensor
                sensor_min = np.min(sensor_xyz)
                sensor_max = np.max(sensor_xyz)
                
                # Normalize [x,y,z] values: (val - min) / (max - min)
                if sensor_max > sensor_min:
                    sensor_xyz_norm = (sensor_xyz - sensor_min) / (sensor_max - sensor_min)
                else:
                    # Avoid division by zero
                    sensor_xyz_norm = sensor_xyz - sensor_min
                
                # Compute magnitude from normalized [x,y,z]
                normalized_magnitudes[:, sensor_idx] = np.sqrt(np.sum(sensor_xyz_norm**2, axis=1))
            
            # Use normalized magnitudes as features: [samples, 15]
            features = normalized_magnitudes
            
            # Total: 15 features (one per sensor, normalized independently)
            # Normalization: Each sensor's [x,y,z] is normalized using min-max normalization
            #               independently (sensor 0 uses its own min/max, sensor 1 uses its own, etc.)
            #               Then magnitude is computed from normalized [x,y,z]
        else:
            # METHOD 2: 45 raw features (original approach)
            # Just flatten [samples, 15, 3] -> [samples, 45]
            # Each feature is a raw magnetic field component (Bx, By, Bz for each of 15 sensors)
            # Normalization: Applied later via StandardScaler (zero mean, unit variance)
            features = magnetic.reshape(magnetic.shape[0], -1)
        
        # Store raw Fz values (will normalize after concatenation)
        # Get offset/location (encode as integer)
        offset_str = str(seq['offset'])
        # Try to detect if it's a numeric offset (multi-point) or named offset (single-point)
        if offset_str.isdigit():
            # Multi-point: '1', '2', ..., '10' -> 0, 1, ..., 9
            offset = int(offset_str) - 1  # Convert '1'-'10' to 0-9
        else:
            # Single-point: named offsets
            offset_map = {'center': 0, 'ne': 1, 'nw': 2, 'se': 3, 'sw': 4, 'unknown': -1}
            offset = offset_map.get(offset_str, -1)
        
        # Add advanced features if requested
        if use_advanced_features:
            # Features statistiche: per ogni sensore, calcola media, std, max, min delle 3 componenti
            # Shape: [samples, 15, 3] -> [samples, 60] (15 sensori × 4 statistiche)
            statistical_features = []
            for sensor_idx in range(15):
                sensor_xyz = magnetic[:, sensor_idx, :]  # [samples, 3]
                
                # Calcola statistiche locali per ogni sample (media, std, max, min delle 3 componenti)
                sensor_mean = np.mean(sensor_xyz, axis=1)  # [samples] - media delle 3 componenti per sample
                sensor_std = np.std(sensor_xyz, axis=1)  # [samples] - std delle 3 componenti per sample
                sensor_max = np.max(sensor_xyz, axis=1)  # [samples] - max delle 3 componenti per sample
                sensor_min = np.min(sensor_xyz, axis=1)  # [samples] - min delle 3 componenti per sample
                
                # Aggiungi anche statistiche globali della sequenza (ripetute per ogni sample)
                seq_mean = np.mean(sensor_xyz)  # Scalar
                seq_std = np.std(sensor_xyz)  # Scalar
                seq_max = np.max(sensor_xyz)  # Scalar
                seq_min = np.min(sensor_xyz)  # Scalar
                
                # Usa statistiche locali (più informative) + statistiche globali = 8 features per sensore
                # Ma per mantenere il numero ragionevole, usiamo solo le statistiche locali
                # Questo dà 4 features per sensore × 15 sensori = 60 features
                statistical_features.append(np.column_stack([
                    sensor_mean,  # Media locale per sample
                    sensor_std,   # Std locale per sample
                    sensor_max,   # Max locale per sample
                    sensor_min    # Min locale per sample
                ]))
            
            # Concatena tutte le features statistiche: [samples, 60]
            statistical_features = np.hstack(statistical_features)  # [samples, 60]
            
            # Features temporali: derivate prima e seconda per ogni componente
            # Shape: [samples, 15, 3] -> [samples, 90] (45 componenti × 2 derivate)
            
            # Flatten magnetic data per calcolare derivate: [samples, 45]
            magnetic_flat = magnetic.reshape(magnetic.shape[0], -1)  # [samples, 45]
            
            # Derivata prima (velocità di cambiamento)
            if len(magnetic_flat) > 1:
                # Calcola differenza tra campioni consecutivi
                diff_1 = np.diff(magnetic_flat, axis=0)  # [samples-1, 45]
                # Per il primo sample, usa la differenza con il secondo (ripeti il primo valore)
                derivative_1 = np.vstack([diff_1[0:1], diff_1])  # [samples, 45]
            else:
                derivative_1 = np.zeros_like(magnetic_flat)
            
            # Derivata seconda (accelerazione)
            if len(derivative_1) > 1:
                # Calcola la derivata seconda come differenza della derivata prima
                diff_2 = np.diff(derivative_1, axis=0)  # [samples-1, 45]
                # Per il primo sample, usa la differenza con il secondo
                derivative_2 = np.vstack([diff_2[0:1], diff_2])  # [samples, 45]
            else:
                derivative_2 = np.zeros_like(magnetic_flat)
            
            # Concatena derivate: [samples, 90]
            temporal_features = np.hstack([derivative_1, derivative_2])  # [samples, 90]
            
            # Combina features base + statistiche + temporali
            features = np.hstack([features, statistical_features, temporal_features])
        
        all_features.append(features)
        all_fz.append(fz_raw)  # Store raw values first
        all_offsets.append(np.full(len(fz_raw), offset))
    
    # Concatenate all sequences
    X = np.vstack(all_features)
    y_fz_raw = np.concatenate(all_fz)
    y_offset = np.concatenate(all_offsets)
    
    # Add offset labels as features if requested (one-hot encoding)
    # Note: This is done AFTER advanced features are added, so offset labels are always at the end
    if include_offset_labels:
        from sklearn.preprocessing import OneHotEncoder
        # Create one-hot encoding for offset labels (5 classes: center, ne, nw, se, sw)
        # Filter out unknown offsets (-1) for encoding
        valid_mask = y_offset >= 0
        offset_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(5)], handle_unknown='ignore')
        # Fit on valid offsets only
        offset_onehot = np.zeros((len(y_offset), 5))
        if np.sum(valid_mask) > 0:
            offset_onehot[valid_mask] = offset_encoder.fit_transform(y_offset[valid_mask].reshape(-1, 1))
        # Concatenate magnetic features with offset one-hot encoding
        X = np.hstack([X, offset_onehot])
        print(f"  Added offset labels as one-hot features: {X.shape[1] - (X.shape[1] - 5)} additional features")
    
    # Print feature summary
    if use_advanced_features:
        base_features = 45 if not use_feature_engineering else 15
        if include_offset_labels:
            base_features += 5
        statistical_features = 60  # 15 sensori × 4 statistiche
        temporal_features = 90    # 45 componenti × 2 derivate
        print(f"  Feature breakdown: {base_features} base + {statistical_features} statistical + {temporal_features} temporal = {X.shape[1]} total features")
    
    # Round force values to match FT sensor precision to remove artificial precision
    # Values beyond the sensor precision are artifacts
    y_fz_raw = np.round(y_fz_raw, decimals=FT_SENSOR_DECIMALS)
    
    # Don't normalize Fz - keep original values (but sequences are cut at 3N)
    fz_scaler = None
    y_fz = y_fz_raw
    fz_min = np.min(y_fz_raw)
    fz_max = np.max(y_fz_raw)
    print(f"  Fz range: [{fz_min:.3f}, {fz_max:.3f}] N (not normalized, sequences cut at 3N, rounded to {FT_SENSOR_PRECISION}N precision)")
    
    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Also return actual sequence lengths after filtering
    actual_sequence_lengths = [len(f) for f in all_features]
    
    return X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths


def create_model(regressor: bool = True, use_gpu: bool = True, n_estimators: int = 200, gpu_id: int = 0):
    """Create a model (regressor or classifier) with GPU support if available.
    
    Args:
        regressor: If True, create regressor; if False, create classifier
        use_gpu: If True, try to use GPU acceleration
        n_estimators: Number of trees in the forest (increased for better performance)
        gpu_id: GPU device ID (0 or 1 for dual GPU setup)
    """
    # Increase n_estimators for better performance
    if regressor:
        if use_gpu and CUML_AVAILABLE:
            return cuMLRandomForestRegressor(
                n_estimators=n_estimators * 2,  # More trees for better accuracy
                max_depth=30,  # Limit depth to prevent overfitting
                random_state=42,
                n_streams=1,
            )
        elif use_gpu and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=n_estimators * 2,
                max_depth=15,  # Limit depth
                learning_rate=0.1,
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=gpu_id,
                n_jobs=1,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=n_estimators * 2,  # More trees
                max_depth=30,  # Limit depth
                min_samples_split=5,  # Prevent overfitting
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
    else:
        if use_gpu and CUML_AVAILABLE:
            return cuMLRandomForestClassifier(
                n_estimators=n_estimators * 2,
                max_depth=30,
                random_state=42,
                n_streams=1,
            )
        elif use_gpu and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=n_estimators * 2,
                max_depth=15,
                learning_rate=0.1,
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=gpu_id,
                n_jobs=1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=n_estimators * 2,  # More trees
                max_depth=30,  # Limit depth
                min_samples_split=5,  # Prevent overfitting
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
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
    use_advanced_features: bool = False,
) -> Dict:
    """Train force regressor and offset classifier for a specific stretch level."""
    print(f"\n{'='*80}")
    print(f"Training models for {stretch_label}")
    print(f"{'='*80}")
    print(f"Total sequences: {len(sequences)}")
    
    # Prepare data (with normalization, feature method from parameter, displacement filtering, NO Fz normalization - cut at 3N)
    X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        sequences, 
        normalize=True, 
        use_feature_engineering=(feature_method == 'normalized'),  # True = 15 normalized, False = 45 raw
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,  # Don't normalize, just cut at 3N
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=(feature_method == 'raw_labelled'),  # Add offset labels as features for raw_labelled
        use_advanced_features=use_advanced_features  # Add advanced features (statistical + temporal)
    )
    print(f"Total samples: {len(X)}")
    if X.shape[1] == 15:
        print(f"Features: {X.shape[1]} (15 normalized per sensor)")
    elif X.shape[1] == 45:
        print(f"Features: {X.shape[1]} (45 raw features)")
    elif X.shape[1] == 50:
        print(f"Features: {X.shape[1]} (45 raw features + 5 offset one-hot labels)")
    else:
        print(f"Features: {X.shape[1]} features")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (normalized to [{fz_target_min:.1f}, {fz_target_max:.1f}])")
    print(f"Offsets: {np.unique(y_offset)}")
    print(f"Features normalized: mean={np.mean(X):.2f}, std={np.std(X):.2f}")
    
    # Split by sequence (70% of sequences, not samples) - RANDOM SPLIT
    # Use actual sequence lengths after filtering
    sequence_lengths = actual_sequence_lengths
    
    # Random split of sequences (not sequential)
    np.random.seed(42)  # For reproducibility
    n_train_seqs = int(len(sequences) * train_ratio)
    all_indices = np.arange(len(sequences))
    np.random.shuffle(all_indices)
    train_seq_indices = sorted(all_indices[:n_train_seqs])  # Keep sorted for easier indexing
    test_seq_indices = sorted(all_indices[n_train_seqs:])
    
    # Get sample indices for train/test (reorder to match shuffled sequences)
    # First, we need to map sequence indices to sample indices
    # Build a mapping: sequence_index -> (start_sample_idx, end_sample_idx)
    sequence_to_samples = {}
    current_idx = 0
    for i, seq_len in enumerate(sequence_lengths):
        sequence_to_samples[i] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    # Verify that current_idx matches len(X)
    if current_idx != len(X):
        print(f"  Warning: Total samples mismatch! Expected {len(X)}, calculated {current_idx}")
        # Adjust to match actual X length
        if current_idx > len(X):
            # Truncate the last sequence if needed
            last_seq_idx = len(sequence_lengths) - 1
            if last_seq_idx in sequence_to_samples:
                start, end = sequence_to_samples[last_seq_idx]
                sequence_to_samples[last_seq_idx] = (start, len(X))
    
    # Now collect sample indices for train and test sequences
    train_samples = []
    test_samples = []
    for i in train_seq_indices:
        start, end = sequence_to_samples[i]
        # Ensure end doesn't exceed X length
        end = min(end, len(X))
        train_samples.extend(range(start, end))
    for i in test_seq_indices:
        start, end = sequence_to_samples[i]
        # Ensure end doesn't exceed X length
        end = min(end, len(X))
        test_samples.extend(range(start, end))
    
    # Convert to numpy arrays for indexing
    train_samples = np.array(train_samples)
    test_samples = np.array(test_samples)
    
    X_train = X[train_samples]
    X_test = X[test_samples]
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_seq_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_seq_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressor with sample weighting (more weight to low forces)
    print("\nTraining force regressor...")
    force_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
    if CUML_AVAILABLE and use_gpu:
        print(f"  Using cuML (GPU-accelerated on GPU {gpu_id})")
    elif XGBOOST_AVAILABLE and use_gpu:
        print(f"  Using XGBoost (GPU-accelerated on GPU {gpu_id})")
    else:
        print("  Using scikit-learn (CPU)")
    
    # Calculate sample weights: more weight to low forces (inverse of force)
    # This helps the model learn better at low forces where we have fewer samples
    sample_weights = 1.0 / (y_fz_train + 0.1)  # +0.1 to avoid division by zero
    sample_weights = sample_weights / np.mean(sample_weights)  # Normalize to mean=1
    
    # Check if model supports sample_weight
    try:
        force_model.fit(X_train, y_fz_train, sample_weight=sample_weights)
        print(f"  Using sample weighting (weight range: {np.min(sample_weights):.2f}-{np.max(sample_weights):.2f})")
        print(f"  Low force samples (0-1N) get {np.mean(sample_weights[y_fz_train < 1.0]):.2f}x more weight")
    except TypeError:
        # Model doesn't support sample_weight, use regular fit
        force_model.fit(X_train, y_fz_train)
        print("  Sample weighting not supported, using regular fit")
    y_fz_pred = force_model.predict(X_test)
    
    # Store predictions for plotting (before denormalization)
    y_fz_test_actual = y_fz_test.copy()
    y_fz_pred_actual = y_fz_pred.copy()
    
    rmse = float(np.sqrt(mean_squared_error(y_fz_test, y_fz_pred)))
    residuals = y_fz_test - y_fz_pred
    std_dev = float(np.std(residuals))
    
    # Calculate actual force range in data
    fz_min_actual = float(np.min(y_fz))
    fz_max_actual = float(np.max(y_fz))
    fz_train_min = float(np.min(y_fz_train))
    fz_train_max = float(np.max(y_fz_train))
    fz_test_min = float(np.min(y_fz_test))
    fz_test_max = float(np.max(y_fz_test))
    
    # Calculate force resolution (KPM1) from test data
    # Round to match FT sensor precision to get realistic estimate, not an artifact of rounding
    unique_forces = np.unique(np.round(y_fz_test, decimals=FT_SENSOR_DECIMALS))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        force_resolution = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        force_resolution = float("nan")
    
    sensitivity_target = 0.05  # KPM1 threshold
    kpm1_pass = force_resolution <= sensitivity_target if not np.isnan(force_resolution) else None
    
    # Round RMSE and STD to 2 decimal places for KPM2 check
    rmse_rounded = round(rmse, 2)
    std_dev_rounded = round(std_dev, 2)
    kpm2_pass = (rmse_rounded < 0.10) and (std_dev_rounded < 0.05)
    
    print(f"  RMSE: {rmse_rounded:.2f} N")
    print(f"  Std Dev: {std_dev_rounded:.2f} N")
    if not np.isnan(force_resolution):
        print(f"  Force resolution (ΔF_min): {force_resolution:.6f} N")
        print(f"  KPM1: {'PASS' if kpm1_pass else 'FAIL'}")
    else:
        print(f"  Force resolution (ΔF_min): N/A")
        print(f"  KPM1: N/A")
    print(f"  KPM2: {'PASS' if kpm2_pass else 'FAIL'}")
    print(f"  Actual force range in data: [{fz_min_actual:.3f}, {fz_max_actual:.3f}] N")
    print(f"  Train force range: [{fz_train_min:.3f}, {fz_train_max:.3f}] N")
    print(f"  Test force range: [{fz_test_min:.3f}, {fz_test_max:.3f}] N")
    
    # Train offset classifier
    print("\nTraining offset classifier...")
    # Filter out unknown offsets
    valid_mask = y_offset_train >= 0
    if np.sum(valid_mask) > 0 and len(np.unique(y_offset_train[valid_mask])) > 1:
        offset_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=200, gpu_id=gpu_id)
        offset_model.fit(X_train[valid_mask], y_offset_train[valid_mask])
        test_valid_mask = y_offset_test >= 0
        if np.sum(test_valid_mask) > 0:
            y_offset_pred = offset_model.predict(X_test[test_valid_mask])
            y_offset_test_valid = y_offset_test[test_valid_mask]
            offset_accuracy = float(accuracy_score(y_offset_test_valid, y_offset_pred))
            print(f"  Accuracy: {offset_accuracy:.4f}")
            print(f"  Test samples: {len(y_offset_test_valid)}, Classes: {np.unique(y_offset_test_valid)}")
            # Print confusion matrix info
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_offset_test_valid, y_offset_pred)
            print(f"  Confusion matrix shape: {cm.shape}")
            
            # Store confusion matrix for plotting
            offset_cm = cm
            offset_test_labels = y_offset_test_valid
            offset_pred_labels = y_offset_pred
        else:
            offset_accuracy = 0.0
            offset_cm = None
            offset_test_labels = None
            offset_pred_labels = None
            print("  No valid test samples for offset classification")
    else:
        offset_model = None
        offset_accuracy = 0.0
        offset_cm = None
        offset_test_labels = None
        offset_pred_labels = None
        print(f"  Skipped (insufficient offset diversity: train classes={np.unique(y_offset_train[valid_mask]) if np.sum(valid_mask) > 0 else 'none'})")
    
    return {
        'stretch_label': stretch_label,
        'force_model': force_model,
        'offset_model': offset_model,
        'scaler': scaler,  # Save scaler for feature normalization
        'fz_scaler': fz_scaler,  # Save fz_scaler for Fz denormalization
        'n_sequences': len(sequences),
        'n_train_sequences': len(train_seq_indices),
        'n_test_sequences': len(test_seq_indices),
        'n_samples': len(X),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'force_rmse': rmse,
        'force_std_dev': std_dev,
        'force_resolution_est': force_resolution,
        'kpm1_pass': kpm1_pass,
        'kpm2_pass': kpm2_pass,
        'offset_accuracy': offset_accuracy,
        'fz_min_actual': fz_min_actual,
        'fz_max_actual': fz_max_actual,
        'fz_train_min': fz_train_min,
        'fz_train_max': fz_train_max,
        'fz_test_min': fz_test_min,
        'fz_test_max': fz_test_max,
        'y_fz_test': y_fz_test_actual,  # Store for prediction plots
        'y_fz_pred': y_fz_pred_actual,  # Store for prediction plots
        'offset_confusion_matrix': offset_cm,
        'offset_test_labels': offset_test_labels,
        'offset_pred_labels': offset_pred_labels,
    }


def train_combined_model(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    use_gpu: bool = True,
    fz_target_min: float = 0.0,
    fz_target_max: float = 3.0,
    feature_method: str = 'raw',
    use_advanced_features: bool = False,
) -> Dict:
    """Train a combined model using all stretch levels."""
    print(f"\n{'='*80}")
    print(f"Training COMBINED model (all stretch levels)")
    print(f"{'='*80}")
    
    # Combine all sequences
    all_sequences = []
    all_stretches = []
    for stretch_label, sequences in sequences_by_stretch.items():
        all_sequences.extend(sequences)
        all_stretches.extend([stretch_label] * len(sequences))
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"  Per stretch: {[(k, len(v)) for k, v in sequences_by_stretch.items()]}")
    
    # Prepare data (with normalization, feature method from args, displacement filtering, NO Fz normalization - cut at 3N)
    X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),  # True = 15 normalized, False = 45 raw
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,  # Don't normalize, just cut at 3N
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max,
        include_offset_labels=(feature_method == 'raw_labelled'),  # Add offset labels as features for raw_labelled
        use_advanced_features=use_advanced_features  # Add advanced features (statistical + temporal)
    )
    
    # Encode stretch labels as integers
    stretch_map = {label: i for i, label in enumerate(sorted(set(all_stretches)))}
    y_stretch = np.array([stretch_map[label] for label in all_stretches])
    
    # Expand stretch labels based on actual filtered lengths
    y_stretch_expanded = np.concatenate([np.full(length, stretch_map[all_stretches[i]]) 
                                         for i, length in enumerate(actual_sequence_lengths)])
    
    # Ensure lengths match (should be equal to len(y_fz))
    if len(y_stretch_expanded) != len(y_fz):
        # Trim or pad to match
        if len(y_stretch_expanded) > len(y_fz):
            y_stretch_expanded = y_stretch_expanded[:len(y_fz)]
        else:
            # Pad with last stretch label (shouldn't happen, but handle it)
            y_stretch_expanded = np.concatenate([y_stretch_expanded, np.full(len(y_fz) - len(y_stretch_expanded), y_stretch_expanded[-1] if len(y_stretch_expanded) > 0 else 0)])
    
    print(f"Total samples: {len(X)}")
    if X.shape[1] == 15:
        print(f"Features: {X.shape[1]} (15 normalized per sensor)")
    elif X.shape[1] == 45:
        print(f"Features: {X.shape[1]} (45 raw features)")
    elif X.shape[1] == 50:
        print(f"Features: {X.shape[1]} (45 raw features + 5 offset one-hot labels)")
    else:
        print(f"Features: {X.shape[1]} features")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (normalized to [0, 3])")
    print(f"Stretch levels: {list(stretch_map.keys())}")
    print(f"Features normalized: mean={np.mean(X):.2f}, std={np.std(X):.2f}")
    print(f"Offset distribution: {dict(zip(*np.unique(y_offset, return_counts=True)))}")
    
    # Split by sequence (70% of sequences) - RANDOM SPLIT
    # Use actual sequence lengths after filtering
    sequence_lengths = actual_sequence_lengths
    
    # Random split of sequences (not sequential)
    np.random.seed(42)  # For reproducibility
    n_train_seqs = int(len(all_sequences) * train_ratio)
    all_indices = np.arange(len(all_sequences))
    np.random.shuffle(all_indices)
    train_seq_indices = sorted(all_indices[:n_train_seqs])  # Keep sorted for easier indexing
    test_seq_indices = sorted(all_indices[n_train_seqs:])
    
    # Get sample indices for train/test (reorder to match shuffled sequences)
    # First, we need to map sequence indices to sample indices
    # Build a mapping: sequence_index -> (start_sample_idx, end_sample_idx)
    sequence_to_samples = {}
    current_idx = 0
    for i, seq_len in enumerate(sequence_lengths):
        sequence_to_samples[i] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    # Verify that current_idx matches len(X)
    if current_idx != len(X):
        print(f"  Warning: Total samples mismatch! Expected {len(X)}, calculated {current_idx}")
        # Adjust to match actual X length
        if current_idx > len(X):
            # Truncate the last sequence if needed
            last_seq_idx = len(sequence_lengths) - 1
            if last_seq_idx in sequence_to_samples:
                start, end = sequence_to_samples[last_seq_idx]
                sequence_to_samples[last_seq_idx] = (start, len(X))
    
    # Now collect sample indices for train and test sequences
    train_samples = []
    test_samples = []
    for i in train_seq_indices:
        start, end = sequence_to_samples[i]
        # Ensure end doesn't exceed X length
        end = min(end, len(X))
        train_samples.extend(range(start, end))
    for i in test_seq_indices:
        start, end = sequence_to_samples[i]
        # Ensure end doesn't exceed X length
        end = min(end, len(X))
        test_samples.extend(range(start, end))
    
    # Convert to numpy arrays for indexing
    train_samples = np.array(train_samples)
    test_samples = np.array(test_samples)
    
    X_train = X[train_samples]
    X_test = X[test_samples]
    y_fz_train = y_fz[train_samples]
    y_fz_test = y_fz[test_samples]
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    y_stretch_train = y_stretch_expanded[train_samples]
    y_stretch_test = y_stretch_expanded[test_samples]
    
    print(f"\nTrain: {n_train_seqs} sequences, {len(X_train)} samples")
    print(f"Test: {len(all_sequences) - n_train_seqs} sequences, {len(X_test)} samples")
    
    # Train force regressor
    print("\nTraining force regressor...")
    force_model = create_model(regressor=True, use_gpu=use_gpu, n_estimators=200, gpu_id=1)
    if CUML_AVAILABLE and use_gpu:
        print("  Using cuML (GPU-accelerated on GPU 1)")
    elif XGBOOST_AVAILABLE and use_gpu:
        print("  Using XGBoost (GPU-accelerated on GPU 1)")
    else:
        print("  Using scikit-learn (CPU)")
    
    # Calculate sample weights: more weight to low forces (inverse of force)
    # This helps the model learn better at low forces where we have fewer samples
    sample_weights = 1.0 / (y_fz_train + 0.1)  # +0.1 to avoid division by zero
    sample_weights = sample_weights / np.mean(sample_weights)  # Normalize to mean=1
    
    # Check if model supports sample_weight
    try:
        force_model.fit(X_train, y_fz_train, sample_weight=sample_weights)
        print(f"  Using sample weighting (weight range: {np.min(sample_weights):.2f}-{np.max(sample_weights):.2f})")
        print(f"  Low force samples (0-1N) get {np.mean(sample_weights[y_fz_train < 1.0]):.2f}x more weight")
    except TypeError:
        # Model doesn't support sample_weight, use regular fit
        force_model.fit(X_train, y_fz_train)
        print("  Sample weighting not supported, using regular fit")
    
    y_fz_pred = force_model.predict(X_test)
    
    # Store predictions for plotting (before denormalization)
    y_fz_test_actual = y_fz_test.copy()
    y_fz_pred_actual = y_fz_pred.copy()
    
    rmse = float(np.sqrt(mean_squared_error(y_fz_test, y_fz_pred)))
    residuals = y_fz_test - y_fz_pred
    std_dev = float(np.std(residuals))
    
    # Calculate actual force range in data
    fz_min_actual = float(np.min(y_fz))
    fz_max_actual = float(np.max(y_fz))
    fz_train_min = float(np.min(y_fz_train))
    fz_train_max = float(np.max(y_fz_train))
    fz_test_min = float(np.min(y_fz_test))
    fz_test_max = float(np.max(y_fz_test))
    
    # Calculate force resolution (KPM1) from test data
    # Round to match FT sensor precision to get realistic estimate, not an artifact of rounding
    unique_forces = np.unique(np.round(y_fz_test, decimals=FT_SENSOR_DECIMALS))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        force_resolution = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
    else:
        force_resolution = float("nan")
    
    sensitivity_target = 0.05  # KPM1 threshold
    kpm1_pass = force_resolution <= sensitivity_target if not np.isnan(force_resolution) else None
    
    # Round RMSE and STD to 2 decimal places for KPM2 check
    rmse_rounded = round(rmse, 2)
    std_dev_rounded = round(std_dev, 2)
    kpm2_pass = (rmse_rounded < 0.10) and (std_dev_rounded < 0.05)
    
    print(f"  RMSE: {rmse_rounded:.2f} N")
    print(f"  Std Dev: {std_dev_rounded:.2f} N")
    if not np.isnan(force_resolution):
        print(f"  Force resolution (ΔF_min): {force_resolution:.6f} N")
        print(f"  KPM1: {'PASS' if kpm1_pass else 'FAIL'}")
    else:
        print(f"  Force resolution (ΔF_min): N/A")
        print(f"  KPM1: N/A")
    print(f"  KPM2: {'PASS' if kpm2_pass else 'FAIL'}")
    print(f"  Actual force range in data: [{fz_min_actual:.3f}, {fz_max_actual:.3f}] N")
    print(f"  Train force range: [{fz_train_min:.3f}, {fz_train_max:.3f}] N")
    print(f"  Test force range: [{fz_test_min:.3f}, {fz_test_max:.3f}] N")
    
    # Train offset classifier
    print("\nTraining offset classifier...")
    valid_mask = y_offset_train >= 0
    print(f"  Train valid samples: {np.sum(valid_mask)}/{len(y_offset_train)}")
    print(f"  Train unique offsets: {np.unique(y_offset_train[valid_mask]) if np.sum(valid_mask) > 0 else 'none'}")
    if np.sum(valid_mask) > 0 and len(np.unique(y_offset_train[valid_mask])) > 1:
        offset_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=200, gpu_id=1)
        offset_model.fit(X_train[valid_mask], y_offset_train[valid_mask])
        test_valid_mask = y_offset_test >= 0
        print(f"  Test valid samples: {np.sum(test_valid_mask)}/{len(y_offset_test)}")
        print(f"  Test unique offsets: {np.unique(y_offset_test[test_valid_mask]) if np.sum(test_valid_mask) > 0 else 'none'}")
        if np.sum(test_valid_mask) > 0:
            y_offset_pred = offset_model.predict(X_test[test_valid_mask])
            y_offset_test_valid = y_offset_test[test_valid_mask]
            offset_accuracy = float(accuracy_score(y_offset_test_valid, y_offset_pred))
            print(f"  Accuracy: {offset_accuracy:.4f}")
            print(f"  Test samples: {len(y_offset_test_valid)}, Classes: {np.unique(y_offset_test_valid)}")
            print(f"  Predicted classes: {np.unique(y_offset_pred)}")
            # Print confusion matrix info
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_offset_test_valid, y_offset_pred)
            print(f"  Confusion matrix shape: {cm.shape}")
            print(f"  Confusion matrix:\n{cm}")
            
            # Store confusion matrix for plotting
            offset_cm = cm
            offset_test_labels = y_offset_test_valid
            offset_pred_labels = y_offset_pred
        else:
            offset_accuracy = 0.0
            offset_cm = None
            offset_test_labels = None
            offset_pred_labels = None
            print("  No valid test samples for offset classification")
    else:
        offset_model = None
        offset_accuracy = 0.0
        offset_cm = None
        offset_test_labels = None
        offset_pred_labels = None
        print(f"  Skipped (insufficient offset diversity: train classes={np.unique(y_offset_train[valid_mask]) if np.sum(valid_mask) > 0 else 'none'})")
    
    # Train stretch classifier
    print("\nTraining stretch classifier...")
    if len(np.unique(y_stretch_train)) > 1:
        stretch_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=250)
        stretch_model.fit(X_train, y_stretch_train)
        y_stretch_pred = stretch_model.predict(X_test)
        stretch_accuracy = float(accuracy_score(y_stretch_test, y_stretch_pred))
        print(f"  Accuracy: {stretch_accuracy:.4f}")
        
        # Store confusion matrix for plotting
        from sklearn.metrics import confusion_matrix
        stretch_cm = confusion_matrix(y_stretch_test, y_stretch_pred)
        print(f"  Confusion matrix shape: {stretch_cm.shape}")
        print(f"  Confusion matrix:\n{stretch_cm}")
    else:
        stretch_model = None
        stretch_accuracy = 0.0
        stretch_cm = None
        print("  Skipped (insufficient stretch diversity)")
    
    return {
        'stretch_label': 'combined',
        'force_model': force_model,
        'offset_model': offset_model,
        'stretch_model': stretch_model,
        'offset_confusion_matrix': offset_cm,
        'offset_test_labels': offset_test_labels,
        'offset_pred_labels': offset_pred_labels,
        'stretch_confusion_matrix': stretch_cm,
        'stretch_test_labels': y_stretch_test,
        'stretch_pred_labels': y_stretch_pred,
        'scaler': scaler,  # Save scaler for feature normalization
        'fz_scaler': fz_scaler,  # Save fz_scaler for Fz denormalization
        'n_sequences': len(all_sequences),
        'n_train_sequences': n_train_seqs,
        'n_test_sequences': len(all_sequences) - n_train_seqs,
        'n_samples': len(X),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'force_rmse': rmse,
        'force_std_dev': std_dev,
        'force_resolution_est': force_resolution,
        'kpm1_pass': kpm1_pass,
        'kpm2_pass': kpm2_pass,
        'offset_accuracy': offset_accuracy,
        'stretch_accuracy': stretch_accuracy,
        'fz_min_actual': fz_min_actual,
        'fz_max_actual': fz_max_actual,
        'fz_train_min': fz_train_min,
        'fz_train_max': fz_train_max,
        'fz_test_min': fz_test_min,
        'fz_test_max': fz_test_max,
        'y_fz_test': y_fz_test_actual,  # Store for prediction plots
        'y_fz_pred': y_fz_pred_actual,  # Store for prediction plots
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for single-point data")
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing HDF5 files (e.g., force_0.0to3.0N_step0.1N_single_test1)'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=MODELS_DIR,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU acceleration if available (default: True)'
    )
    parser.add_argument(
        '--feature-method',
        type=str,
        choices=['raw', 'normalized', 'raw_labelled'],
        default='raw',
        help='Feature extraction method: "raw" = 45 raw features (flattened), "normalized" = 15 normalized features (one per sensor), "raw_labelled" = 50 features (45 raw + 5 offset one-hot) (default: raw)'
    )
    parser.add_argument(
        '--no-gpu',
        dest='use_gpu',
        action='store_false',
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        help='Path to save JSON metrics report (default: auto-generated)'
    )
    parser.add_argument(
        '--use-advanced-features',
        action='store_true',
        default=False,
        help='Enable advanced feature engineering: adds 60 statistical features (mean, std, max, min per sensor) + 90 temporal features (first and second derivatives) to base features. Total: base + 150 features. (default: False)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # If data_dir is inside a "cleaned" directory, save models and plots in feature_method subfolder
    if "cleaned" in str(data_dir):
        # Extract the cleaned directory path
        if data_dir.name == "cleaned":
            cleaned_dir = data_dir
        elif (data_dir.parent / "cleaned").exists():
            cleaned_dir = data_dir.parent / "cleaned"
        else:
            cleaned_dir = data_dir
        # Create subdirectories: cleaned/{feature_method}/models and cleaned/{feature_method}/plots
        feature_dir = cleaned_dir / args.feature_method
        models_dir = feature_dir / "models"
        plots_dir = feature_dir / "plots"
    else:
        # Use default models directory
        models_dir = Path(args.models_dir)
        # Create subdirectory for plots based on feature method
        plots_dir = data_dir / "plots" / args.feature_method
    
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SINGLE-POINT MODEL TRAINING")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Feature method: {args.feature_method}")
    print(f"Outlier Z-threshold: {args.z_threshold}")
    print(f"GPU acceleration: {args.use_gpu}")
    if args.use_gpu:
        if CUML_AVAILABLE:
            print(f"  → Using cuML (RAPIDS) for GPU acceleration")
        elif XGBOOST_AVAILABLE:
            print(f"  → Using XGBoost with GPU support")
        else:
            print(f"  → GPU libraries not available, falling back to CPU")
    print("="*80)
    
    # Find HDF5 files for each stretch level
    h5_files = {}
    
    # Check if we're in a cleaned directory structure (with data/ subdirectory)
    data_subdir = data_dir / "data"
    if data_subdir.exists():
        search_dir = data_subdir
    else:
        search_dir = data_dir
    
    for stretch in ['000pct', '010pct', '020pct']:
        # Try to find file with new naming: test_XX_YYY_cleaned.h5
        # Extract stretch number: 000pct -> 000
        stretch_num = stretch.replace('pct', '')
        pattern_new = f"test_*_{stretch_num}_cleaned.h5"
        files = list(search_dir.glob(pattern_new))
        
        # Also try old naming patterns
        if not files:
            pattern1 = f"*stretch_{stretch}.h5"
            pattern2 = f"*stretch_{stretch}_cleaned.h5"
            files = list(search_dir.glob(pattern1)) + list(search_dir.glob(pattern2))
        
        if not files:
            # Try in subdirectory
            stretch_dir = search_dir / f"stretch_{stretch}"
            if stretch_dir.exists():
                files = list(stretch_dir.glob("*.h5"))
        
        if files:
            h5_files[stretch] = sorted(files)[-1]  # Use latest file
            print(f"\nFound {stretch}: {h5_files[stretch]}")
        else:
            print(f"\n⚠️  No file found for {stretch}")
    
    if not h5_files:
        print("\n❌ No HDF5 files found!")
        return
    
    # Load sequences from each file
    print("\n" + "="*80)
    print("LOADING SEQUENCES")
    print("="*80)
    sequences_by_stretch = {}
    
    for stretch, h5_file in h5_files.items():
        print(f"\nLoading {stretch} from {h5_file.name}...")
        sequences = load_sequences_from_h5(h5_file)
        print(f"  Loaded {len(sequences)} sequences")
        
        if sequences:
            # Check if this file is already cleaned (has _cleaned.h5 suffix or is in cleaned/ directory)
            file_is_cleaned = "_cleaned.h5" in h5_file.name or "cleaned" in str(h5_file.parent)
            
            if file_is_cleaned:
                print(f"  ✓ File is already cleaned (detected _cleaned.h5 suffix or cleaned/ directory)")
                print(f"  ✓ Using sequences as-is (no additional cleaning)")
                # Count sequences per offset for info
                offset_counts = {}
                for seq in sequences:
                    offset = seq.get('offset', 'unknown')
                    offset_counts[offset] = offset_counts.get(offset, 0) + 1
                print(f"  Sequences per offset: {offset_counts}")
                sequences_by_stretch[stretch] = sequences
            else:
                # File is NOT cleaned - apply cleaning steps
                # Count sequences per offset before any removal
                initial_offset_counts = {}
                for seq in sequences:
                    offset = seq.get('offset', 'unknown')
                    initial_offset_counts[offset] = initial_offset_counts.get(offset, 0) + 1
                print(f"  Initial sequences per offset: {initial_offset_counts}")
                print(f"  Total initial sequences: {len(sequences)} (expected: 33*5 = 165)")
                
                # Remove first sequence per offset (warm-up/calibration sequence) BEFORE outlier removal
                print(f"  Removing first sequence per offset...")
                sequences_by_offset = {}
                for idx, seq in enumerate(sequences):
                    offset = seq.get('offset', 'unknown')
                    if offset not in sequences_by_offset:
                        sequences_by_offset[offset] = []
                    sequences_by_offset[offset].append((idx, seq))
                
                first_sequence_indices = set()
                for offset, offset_sequences in sequences_by_offset.items():
                    if len(offset_sequences) > 1:  # Only remove if there's more than one sequence
                        # Get the first sequence index (assuming sequences are in order)
                        first_idx = offset_sequences[0][0]
                        first_sequence_indices.add(first_idx)
                        print(f"    Offset {offset}: Removing first sequence (index: {first_idx})")
                
                if first_sequence_indices:
                    sequences_after_first_removal = [s for i, s in enumerate(sequences) if i not in first_sequence_indices]
                    print(f"  After removing first sequences: {len(sequences_after_first_removal)} sequences (removed {len(first_sequence_indices)} first sequences)")
                    print(f"  Expected after first removal: {len(sequences)} - {len(first_sequence_indices)} = {len(sequences) - len(first_sequence_indices)}")
                else:
                    sequences_after_first_removal = sequences
                
                # Remove outliers (2 per offset, independently) from remaining sequences
                print(f"  Removing outliers (2 per offset, independently)...")
                print(f"  Expected after outlier removal: {len(sequences_after_first_removal)} - (5 offsets * 2 outliers) = {len(sequences_after_first_removal) - 10} sequences")
                cleaned_sequences, outlier_indices = remove_outliers(sequences_after_first_removal, args.z_threshold, remove_per_offset=2)
                print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
                
                # Count sequences per offset to verify
                offset_counts = {}
                for seq in cleaned_sequences:
                    offset = seq.get('offset', 'unknown')
                    offset_counts[offset] = offset_counts.get(offset, 0) + 1
                print(f"  Sequences per offset after cleaning: {offset_counts}")
                print(f"  Expected per offset: 33 - 1 (first) - 2 (outliers) = 30")
                sequences_by_stretch[stretch] = cleaned_sequences
    
    if not sequences_by_stretch:
        print("\n❌ No sequences loaded!")
        return
    
    # Balance sequences across stretch levels
    print("\n" + "="*80)
    print("BALANCING SEQUENCES")
    print("="*80)
    sequences_by_stretch = balance_sequences(sequences_by_stretch)
    
    # Train separate models for each stretch level
    print("\n" + "="*80)
    print("TRAINING SEPARATE MODELS")
    print("="*80)
    
    # Distribute models across GPUs: 0%, 10% on GPU 0, 20% on GPU 1
    trained_models = {}
    gpu_mapping = {'000pct': 0, '010pct': 0, '020pct': 1}
    
    for stretch_label, sequences in sequences_by_stretch.items():
        # Use different GPU for each stretch level to parallelize training
        gpu_id = gpu_mapping.get(stretch_label, 0)
        result = train_models_for_stretch(sequences, stretch_label, train_ratio=0.7, use_gpu=args.use_gpu, gpu_id=gpu_id, feature_method=args.feature_method, use_advanced_features=args.use_advanced_features)
        trained_models[stretch_label] = result
        
    
    # Train combined model
    print("\n" + "="*80)
    print("TRAINING COMBINED MODEL")
    print("="*80)
    combined_result = train_combined_model(sequences_by_stretch, train_ratio=0.7, use_gpu=args.use_gpu, feature_method=args.feature_method, use_advanced_features=args.use_advanced_features)
    trained_models['combined'] = combined_result
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    for model_name, result in trained_models.items():
        if result['force_model'] is not None:
            force_path = models_dir / f"force_regressor_{model_name}.joblib"
            joblib.dump(result['force_model'], force_path)
            print(f"Saved: {force_path}")
            
            # Save scaler and fz_scaler
            if result.get('scaler') is not None:
                scaler_path = models_dir / f"scaler_{model_name}.joblib"
                joblib.dump(result['scaler'], scaler_path)
                print(f"Saved: {scaler_path}")
            if result.get('fz_scaler') is not None:
                fz_scaler_path = models_dir / f"fz_scaler_{model_name}.joblib"
                joblib.dump(result['fz_scaler'], fz_scaler_path)
                print(f"Saved: {fz_scaler_path}")
        
        if result.get('offset_model') is not None:
            offset_path = models_dir / f"offset_classifier_{model_name}.joblib"
            joblib.dump(result['offset_model'], offset_path)
            print(f"Saved: {offset_path}")
        
        if result.get('stretch_model') is not None:
            stretch_path = models_dir / f"stretch_classifier_{model_name}.joblib"
            joblib.dump(result['stretch_model'], stretch_path)
            print(f"Saved: {stretch_path}")
    
    # Generate plots from cleaned sequences and confusion matrices
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # plots_dir already created above
    
    # Plot confusion matrices
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    offset_names = ['center', 'ne', 'nw', 'se', 'sw']
    
    confusion_plots_saved = []
    for model_name, result in trained_models.items():
        if 'offset_confusion_matrix' in result and result['offset_confusion_matrix'] is not None:
            cm = result['offset_confusion_matrix']
            # Normalize by row (each row sums to 1.0) to show percentages
            # This makes it easy to see per-class accuracy: diagonal values = accuracy for that class
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=offset_names)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(f'Offset Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_offset_{model_name}_{args.feature_method}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved confusion matrix plot: {confusion_plot_path}")
            confusion_plots_saved.append(confusion_plot_path)
        
        if 'stretch_confusion_matrix' in result and result['stretch_confusion_matrix'] is not None:
            cm = result['stretch_confusion_matrix']
            # Normalize by row (each row sums to 1.0) to show percentages
            # This makes it easy to see per-class accuracy: diagonal values = accuracy for that class
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            stretch_labels = ['000pct', '010pct', '020pct']
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=stretch_labels)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(f'Stretch Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_stretch_{model_name}_{args.feature_method}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved confusion matrix plot: {confusion_plot_path}")
            confusion_plots_saved.append(confusion_plot_path)
    
    if confusion_plots_saved:
        print(f"\n  Total confusion matrix plots saved: {len(confusion_plots_saved)}")
        print(f"  Location: {plots_dir}")
    else:
        print(f"\n  ⚠️  No confusion matrices to plot")
    
    # Generate prediction plots (predicted vs actual Fz)
    print("\n" + "="*80)
    print("GENERATING PREDICTION PLOTS")
    print("="*80)
    
    prediction_plots_saved = []
    for model_name, result in trained_models.items():
        if 'y_fz_test' in result and 'y_fz_pred' in result:
            y_test = result['y_fz_test']
            y_pred = result['y_fz_pred']
            
            if len(y_test) > 0 and len(y_pred) > 0:
                # Scatter plot: Predicted vs Actual
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.scatter(y_test, y_pred, alpha=0.5, s=20)
                
                # Perfect prediction line
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
                
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Predicted Fz (N)', fontsize=12)
                ax.set_title(f'Force Prediction - {model_name}\nRMSE: {result["force_rmse"]:.4f} N', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add R² value
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
                
                plt.tight_layout()
                prediction_plot_path = plots_dir / f"prediction_scatter_{model_name}_{args.feature_method}.png"
                plt.savefig(prediction_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved prediction scatter plot: {prediction_plot_path}")
                prediction_plots_saved.append(prediction_plot_path)
                
                # Residual plot
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, residuals, alpha=0.5, s=20)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Residuals (Actual - Predicted) (N)', fontsize=12)
                ax.set_title(f'Residual Plot - {model_name}\nStd Dev: {result["force_std_dev"]:.4f} N', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                residual_plot_path = plots_dir / f"prediction_residuals_{model_name}_{args.feature_method}.png"
                plt.savefig(residual_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved residual plot: {residual_plot_path}")
                prediction_plots_saved.append(residual_plot_path)
    
    if prediction_plots_saved:
        print(f"\n  Total prediction plots saved: {len(prediction_plots_saved)}")
        print(f"  Location: {plots_dir}")
    else:
        print(f"\n  ⚠️  No prediction plots generated")
    
    # Generate plots from cleaned sequences using plot_raw_data
    print("\nGenerating plots from cleaned sequences...")
    print(f"  Plots will be saved to: {data_dir}/stretch_*_cleaned/")
    try:
        from training.plot_raw_data import main as plot_main
        import sys as plot_sys
        
        # Save original sys.argv
        original_argv = plot_sys.argv.copy()
        
        # Generate plots for each stretch level from cleaned sequences
        plot_dirs_created = []
        for stretch, sequences in sequences_by_stretch.items():
            if not sequences:
                print(f"  ⚠️  Skipping {stretch}: No sequences")
                continue
            
            h5_file = h5_files.get(stretch)
            if h5_file and h5_file.exists():
                stretch_output_dir = data_dir / f"stretch_{stretch}_cleaned"
                stretch_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Use --h5-file for individual stretch plots
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-file', str(h5_file),
                    '--output-dir', str(stretch_output_dir)
                ]
                
                try:
                    plot_main()
                    print(f"  ✓ Plots generated for {stretch} (cleaned sequences)")
                    print(f"    Location: {stretch_output_dir}")
                    plot_dirs_created.append(stretch_output_dir)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not generate plots for {stretch}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠️  Skipping {stretch}: HDF5 file not found")
        
        # Generate average plots (requires all HDF5 files together)
        print("\nGenerating average plots (across all stretch levels)...")
        if len(h5_files) >= 2:  # Need at least 2 files for average plots
            try:
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-files'
                ] + [str(h5_file) for h5_file in h5_files.values()] + [
                    '--output-dir', str(data_dir)
                ]
                plot_main()
                print(f"  ✓ Average plots generated")
                print(f"    Location: {data_dir}")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate average plots: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠️  Skipping average plots: Need at least 2 HDF5 files (found {len(h5_files)})")
        
        # Restore original sys.argv
        plot_sys.argv = original_argv
        
        if plot_dirs_created:
            print(f"\n  ✓ Total plot directories created: {len(plot_dirs_created)}")
        else:
            print(f"\n  ⚠️  No plot directories created")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Prepare JSON report (compatible with evaluate_single_point_stretch.py format)
    print("\n" + "="*80)
    print("PREPARING JSON REPORT")
    print("="*80)
    
    # Format metrics for JSON (compatible format)
    force_results_full = []
    offset_results_full = []
    
    for stretch_label in ['000pct', '010pct', '020pct']:
        if stretch_label in trained_models:
            result = trained_models[stretch_label]
            force_results_full.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'rmse': result['force_rmse'],
                'std_dev': result['force_std_dev'],
                'force_resolution_est': result.get('force_resolution_est', np.nan),
                'kpm1_pass': result.get('kpm1_pass', None),
                'kpm2_pass': result.get('kpm2_pass', None),
                'fz_min_actual': result.get('fz_min_actual', np.nan),
                'fz_max_actual': result.get('fz_max_actual', np.nan),
                'fz_train_min': result.get('fz_train_min', np.nan),
                'fz_train_max': result.get('fz_train_max', np.nan),
                'fz_test_min': result.get('fz_test_min', np.nan),
                'fz_test_max': result.get('fz_test_max', np.nan),
            })
            offset_results_full.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'accuracy': result['offset_accuracy'],
            })
        
    
    # Combined metrics
    if 'combined' in trained_models:
        combined_result = trained_models['combined']
        combined_force_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result['n_samples'],
            'sequences': combined_result.get('n_test_sequences', combined_result.get('n_sequences', 0)),
            'rmse': combined_result['force_rmse'],
            'std_dev': combined_result['force_std_dev'],
            'force_resolution_est': combined_result.get('force_resolution_est', np.nan),
            'kpm1_pass': combined_result.get('kpm1_pass', None),
            'kpm2_pass': combined_result.get('kpm2_pass', None),
            'fz_min_actual': combined_result.get('fz_min_actual', np.nan),
            'fz_max_actual': combined_result.get('fz_max_actual', np.nan),
            'fz_train_min': combined_result.get('fz_train_min', np.nan),
            'fz_train_max': combined_result.get('fz_train_max', np.nan),
            'fz_test_min': combined_result.get('fz_test_min', np.nan),
            'fz_test_max': combined_result.get('fz_test_max', np.nan),
        }
        combined_offset_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result['n_samples'],
            'sequences': combined_result.get('n_test_sequences', combined_result.get('n_sequences', 0)),
            'accuracy': combined_result['offset_accuracy'],
        }
        combined_stretch_metrics = {
            'samples': combined_result['n_samples'],
            'accuracy': combined_result.get('stretch_accuracy', 0.0),
        }
    else:
        combined_force_metrics = None
        combined_offset_metrics = None
        combined_stretch_metrics = None
    
    # Save JSON report
    # If we're in a cleaned directory, save JSON in the feature_method subfolder
    if "cleaned" in str(data_dir):
        if data_dir.name == "cleaned":
            cleaned_dir = data_dir
        elif (data_dir.parent / "cleaned").exists():
            cleaned_dir = data_dir.parent / "cleaned"
        else:
            cleaned_dir = data_dir
        feature_dir = cleaned_dir / args.feature_method
        feature_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = feature_dir
    else:
        reports_dir = Path(LOGS_DIR) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
    
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = reports_dir / f"{data_dir.name}_metrics_{args.feature_method}.json"
    
    report_payload = {
        'force_mapping_per_stretch_full': force_results_full,
        'force_mapping_combined_full': combined_force_metrics,
        'offset_classification_per_stretch_full': offset_results_full,
        'offset_classification_combined_full': combined_offset_metrics,
        'stretch_classification_combined_full': combined_stretch_metrics,
        'kpm_thresholds': {
            'accuracy_rmse_threshold': 0.10,
            'precision_std_threshold': 0.05,
            'sensitivity_target': 0.05,
        },
        'training_info': {
            'data_dir': str(data_dir),
            'outlier_z_threshold': args.z_threshold,
            'gpu_acceleration': args.use_gpu,
            'gpu_library': 'cuml' if CUML_AVAILABLE and args.use_gpu else 'xgboost' if XGBOOST_AVAILABLE and args.use_gpu else 'sklearn',
        },
    }
    
    # Custom JSON encoder to handle NaN and None
    def json_encoder(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2, default=json_encoder)
    
    print(f"Metrics report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*120)
    print("TRAINING SUMMARY")
    print("="*120)
    print(f"{'Model':<15} {'Sequences':<12} {'Train Seq':<12} {'Test Seq':<12} {'Samples':<12} {'Train Samples':<15} {'Test Samples':<15} {'RMSE [N]':<12} {'STD [N]':<12} {'ΔF_min [N]':<14} {'KPM1':<6} {'KPM2':<6} {'Offset Acc':<12}")
    print("-"*120)
    
    for model_name, result in trained_models.items():
        # Skip sensor8 models in main summary (they have their own section)
        if '_sensor8' in model_name:
            continue
        
        # Get metrics
        rmse = result.get('force_rmse', np.nan)
        std_dev = result.get('force_std_dev', np.nan)
        force_resolution = result.get('force_resolution_est', np.nan)
        kpm1_pass = result.get('kpm1_pass', None)
        kpm2_pass = result.get('kpm2_pass', None)
        offset_acc = result.get('offset_accuracy', np.nan)
        
        # Format values (round to 2 decimal places for display and KPM checks)
        rmse_rounded = round(rmse, 2) if not np.isnan(rmse) else np.nan
        std_dev_rounded = round(std_dev, 2) if not np.isnan(std_dev) else np.nan
        rmse_str = f"{rmse_rounded:.2f}" if not np.isnan(rmse_rounded) else "N/A"
        std_str = f"{std_dev_rounded:.2f}" if not np.isnan(std_dev_rounded) else "N/A"
        delta_f_str = f"{force_resolution:.4f}" if not np.isnan(force_resolution) else "N/A"
        kpm1_str = "PASS" if kpm1_pass is True else "FAIL" if kpm1_pass is False else "N/A"
        kpm2_str = "PASS" if kpm2_pass is True else "FAIL" if kpm2_pass is False else "N/A"
        offset_acc_str = f"{offset_acc:.4f}" if not np.isnan(offset_acc) else "N/A"
        
        print(f"{model_name:<15} {result['n_sequences']:<12} {result['n_train_sequences']:<12} "
              f"{result['n_test_sequences']:<12} {result['n_samples']:<12} "
              f"{result['n_train_samples']:<15} {result['n_test_samples']:<15} "
              f"{rmse_str:<12} {std_str:<12} {delta_f_str:<14} {kpm1_str:<6} {kpm2_str:<6} {offset_acc_str:<12}")
    
    print("="*100)
    print("\nForce Range Details:")
    for model_name, result in trained_models.items():
        fz_min = result.get('fz_min_actual', np.nan)
        fz_max = result.get('fz_max_actual', np.nan)
        fz_train_min = result.get('fz_train_min', np.nan)
        fz_train_max = result.get('fz_train_max', np.nan)
        fz_test_min = result.get('fz_test_min', np.nan)
        fz_test_max = result.get('fz_test_max', np.nan)
        if not np.isnan(fz_min):
            print(f"  {model_name}:")
            print(f"    Overall range: [{fz_min:.3f}, {fz_max:.3f}] N")
            print(f"    Train range: [{fz_train_min:.3f}, {fz_train_max:.3f}] N")
            print(f"    Test range: [{fz_test_min:.3f}, {fz_test_max:.3f}] N")
    print("="*100)
    print("Training complete!")
    print(f"Models saved to: {models_dir}")
    print(f"Metrics report saved to: {report_path}")


if __name__ == '__main__':
    main()
