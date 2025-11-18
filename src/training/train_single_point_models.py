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
            
            # Extract offset from label
            offset = 'unknown'
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
    - Duration
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
    
    # Check each feature
    for feature_name, feature_values in [
        ('duration', durations),
        ('fz_max', fz_maxes),
        ('num_samples', num_samples),
    ]:
        median = np.median(feature_values)
        mad = np.median(np.abs(feature_values - median))
        
        if mad > 0:
            # Use modified Z-score with MAD
            modified_z_scores = 0.6745 * (feature_values - median) / mad
            outliers = np.where(np.abs(modified_z_scores) > z_threshold)[0]
            outlier_indices.update(outliers)
    
    return sorted(list(outlier_indices))


def remove_outliers(sequences: List[Dict], z_threshold: float = 3.0) -> Tuple[List[Dict], List[int]]:
    """Remove outlier sequences and return cleaned list."""
    outlier_indices = identify_outliers(sequences, z_threshold)
    
    if outlier_indices:
        print(f"  Identified {len(outlier_indices)} outlier sequences: {outlier_indices}")
        # Remove outliers (in reverse order to maintain indices)
        cleaned_sequences = [s for i, s in enumerate(sequences) if i not in outlier_indices]
    else:
        cleaned_sequences = sequences
    
    return cleaned_sequences, outlier_indices


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
                          normalize_fz: bool = True, fz_target_min: float = 0.0, fz_target_max: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
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
    
    Returns:
        X: Features [samples, N] where N=45 (raw) if use_feature_engineering=False, or N=125 (with engineering) if True
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
            continue  # Skip empty sequences
        
        if use_feature_engineering:
            # Start with raw features: [samples, 45]
            features = magnetic.reshape(magnetic.shape[0], -1)
            
            # 1. Add magnitude for each sensor: sqrt(Bx² + By² + Bz²) [samples, 15]
            magnitudes = np.sqrt(np.sum(magnetic**2, axis=2))  # [samples, 15]
            features = np.hstack([features, magnitudes])
            
            # 2. Add statistics per sensor: mean, std, max, min [samples, 15*4 = 60]
            # Mean per sensor
            sensor_means = np.mean(magnetic, axis=2)  # [samples, 15]
            # Std per sensor
            sensor_stds = np.std(magnetic, axis=2)  # [samples, 15]
            # Max per sensor
            sensor_maxs = np.max(magnetic, axis=2)  # [samples, 15]
            # Min per sensor
            sensor_mins = np.min(magnetic, axis=2)  # [samples, 15]
            features = np.hstack([features, sensor_means, sensor_stds, sensor_maxs, sensor_mins])
            
            # 3. Add central sensor features (indices 6, 7, 8 - the 3x3 center)
            central_sensors = magnetic[:, [6, 7, 8], :]  # [samples, 3, 3]
            central_mean = np.mean(central_sensors, axis=(1, 2))  # [samples]
            central_std = np.std(central_sensors, axis=(1, 2))  # [samples]
            central_max = np.max(central_sensors, axis=(1, 2))  # [samples]
            central_min = np.min(central_sensors, axis=(1, 2))  # [samples]
            features = np.hstack([features, central_mean.reshape(-1, 1), central_std.reshape(-1, 1), 
                                 central_max.reshape(-1, 1), central_min.reshape(-1, 1)])
            
            # 4. Add differences between central and peripheral sensors
            peripheral_sensors = np.concatenate([magnetic[:, :6, :], magnetic[:, 9:, :]], axis=1)  # [samples, 12, 3]
            peripheral_mean = np.mean(peripheral_sensors, axis=(1, 2))  # [samples]
            center_periphery_diff = central_mean - peripheral_mean  # [samples]
            features = np.hstack([features, center_periphery_diff.reshape(-1, 1)])
            
            # Total: 45 (raw) + 15 (magnitudes) + 60 (stats) + 4 (central) + 1 (diff) = 125 features
        else:
            # Original: just flatten [samples, 15, 3] -> [samples, 45]
            features = magnetic.reshape(magnetic.shape[0], -1)
        
        # Store raw Fz values (will normalize after concatenation)
        # Get offset (encode as integer)
        offset_map = {'center': 0, 'ne': 1, 'nw': 2, 'se': 3, 'sw': 4, 'unknown': -1}
        offset = offset_map.get(seq['offset'], -1)
        
        all_features.append(features)
        all_fz.append(fz_raw)  # Store raw values first
        all_offsets.append(np.full(len(fz_raw), offset))
    
    # Concatenate all sequences
    X = np.vstack(all_features)
    y_fz_raw = np.concatenate(all_fz)
    y_offset = np.concatenate(all_offsets)
    
    # Normalize Fz to target range [0, 3.0] if requested (on ALL data together)
    fz_scaler = None
    if normalize_fz:
        fz_min = np.min(y_fz_raw)
        fz_max = np.max(y_fz_raw)
        y_fz = normalize_fz_to_range(y_fz_raw, fz_target_min, fz_target_max)
        fz_scaler = (fz_min, fz_max)  # Store original range for inverse transform if needed
        print(f"  Fz normalized from [{fz_min:.3f}, {fz_max:.3f}] N to [{fz_target_min:.3f}, {fz_target_max:.3f}] N")
    else:
        y_fz = y_fz_raw
    
    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y_fz, y_offset, scaler, fz_scaler


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
) -> Dict:
    """Train force regressor and offset classifier for a specific stretch level."""
    print(f"\n{'='*80}")
    print(f"Training models for {stretch_label}")
    print(f"{'='*80}")
    print(f"Total sequences: {len(sequences)}")
    
    # Prepare data (with normalization, NO feature engineering, displacement filtering, and Fz normalization)
    X, y_fz, y_offset, scaler, fz_scaler = prepare_training_data(
        sequences, 
        normalize=True, 
        use_feature_engineering=False,  # Use raw features only (45 features)
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=True,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max
    )
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]} (raw features only)")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (normalized to [{fz_target_min:.1f}, {fz_target_max:.1f}])")
    print(f"Offsets: {np.unique(y_offset)}")
    print(f"Features normalized: mean={np.mean(X):.2f}, std={np.std(X):.2f}")
    
    # Split by sequence (70% of sequences, not samples)
    # Group samples by sequence
    sequence_lengths = [len(seq['fz']) for seq in sequences]
    sequence_boundaries = np.cumsum([0] + sequence_lengths)
    
    # Split sequences
    n_train_seqs = int(len(sequences) * train_ratio)
    train_seq_indices = list(range(n_train_seqs))
    test_seq_indices = list(range(n_train_seqs, len(sequences)))
    
    # Get sample indices for train/test
    train_start = sequence_boundaries[0]
    train_end = sequence_boundaries[n_train_seqs]
    test_start = sequence_boundaries[n_train_seqs]
    test_end = sequence_boundaries[-1]
    
    X_train = X[train_start:train_end]
    X_test = X[test_start:test_end]
    y_fz_train = y_fz[train_start:train_end]
    y_fz_test = y_fz[test_start:test_end]
    y_offset_train = y_offset[train_start:train_end]
    y_offset_test = y_offset[test_start:test_end]
    
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
    
    print(f"  RMSE: {rmse:.4f} N")
    print(f"  Std Dev: {std_dev:.4f} N")
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
        else:
            offset_accuracy = 0.0
            print("  No valid test samples for offset classification")
    else:
        offset_model = None
        offset_accuracy = 0.0
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
        'offset_accuracy': offset_accuracy,
        'fz_min_actual': fz_min_actual,
        'fz_max_actual': fz_max_actual,
        'fz_train_min': fz_train_min,
        'fz_train_max': fz_train_max,
        'fz_test_min': fz_test_min,
        'fz_test_max': fz_test_max,
    }


def train_combined_model(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    use_gpu: bool = True,
    fz_target_min: float = 0.0,
    fz_target_max: float = 3.0,
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
    
    # Prepare data (with normalization, NO feature engineering, displacement filtering, and Fz normalization)
    X, y_fz, y_offset, scaler, fz_scaler = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=False,  # Use raw features only (45 features)
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=True,
        fz_target_min=fz_target_min,
        fz_target_max=fz_target_max
    )
    
    # Encode stretch labels as integers
    stretch_map = {label: i for i, label in enumerate(sorted(set(all_stretches)))}
    y_stretch = np.array([stretch_map[label] for label in all_stretches])
    
    # Recalculate stretch labels after filtering
    # We need to track which samples belong to which stretch after filtering
    # Since filtering happens inside prepare_training_data, we need to recalculate
    # by going through sequences again and tracking filtered lengths
    stretch_labels_per_seq = []
    for stretch_label, sequences in sequences_by_stretch.items():
        for seq in sequences:
            # Apply same filtering logic to get actual filtered length
            magnetic = seq['stretchmagtec']
            forces = seq.get('forces', None)
            if forces is not None:
                fz_raw = np.abs(forces[:, 2]) if forces.shape[1] >= 3 else np.abs(seq['fz'])
            else:
                fz_raw = np.abs(seq['fz'])
            
            # Apply displacement filter
            mag_mask = filter_high_displacement(magnetic, 95.0)
            fz_mask = filter_high_displacement(fz_raw, 95.0)
            combined_mask = mag_mask & fz_mask
            
            filtered_length = np.sum(combined_mask)
            stretch_labels_per_seq.append((stretch_map[stretch_label], filtered_length))
    
    # Expand stretch labels based on filtered lengths
    y_stretch_expanded = np.concatenate([np.full(length, stretch_label) for stretch_label, length in stretch_labels_per_seq])
    
    # Ensure lengths match (should be equal to len(y_fz))
    if len(y_stretch_expanded) != len(y_fz):
        # Trim or pad to match
        if len(y_stretch_expanded) > len(y_fz):
            y_stretch_expanded = y_stretch_expanded[:len(y_fz)]
        else:
            # Pad with last stretch label (shouldn't happen, but handle it)
            y_stretch_expanded = np.concatenate([y_stretch_expanded, np.full(len(y_fz) - len(y_stretch_expanded), y_stretch_expanded[-1] if len(y_stretch_expanded) > 0 else 0)])
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]} (raw features only)")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (normalized to [0, 3])")
    print(f"Stretch levels: {list(stretch_map.keys())}")
    print(f"Features normalized: mean={np.mean(X):.2f}, std={np.std(X):.2f}")
    print(f"Offset distribution: {dict(zip(*np.unique(y_offset, return_counts=True)))}")
    
    # Split by sequence (70% of sequences)
    sequence_lengths = [len(seq['fz']) for seq in all_sequences]
    sequence_boundaries = np.cumsum([0] + sequence_lengths)
    
    n_train_seqs = int(len(all_sequences) * train_ratio)
    train_start = sequence_boundaries[0]
    train_end = sequence_boundaries[n_train_seqs]
    test_start = sequence_boundaries[n_train_seqs]
    test_end = sequence_boundaries[-1]
    
    X_train = X[train_start:train_end]
    X_test = X[test_start:test_end]
    y_fz_train = y_fz[train_start:train_end]
    y_fz_test = y_fz[test_start:test_end]
    y_offset_train = y_offset[train_start:train_end]
    y_offset_test = y_offset[test_start:test_end]
    y_stretch_train = y_stretch_expanded[train_start:train_end]
    y_stretch_test = y_stretch_expanded[test_start:test_end]
    
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
    
    print(f"  RMSE: {rmse:.4f} N")
    print(f"  Std Dev: {std_dev:.4f} N")
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
        else:
            offset_accuracy = 0.0
            print("  No valid test samples for offset classification")
    else:
        offset_model = None
        offset_accuracy = 0.0
        print(f"  Skipped (insufficient offset diversity: train classes={np.unique(y_offset_train[valid_mask]) if np.sum(valid_mask) > 0 else 'none'})")
    
    # Train stretch classifier
    print("\nTraining stretch classifier...")
    if len(np.unique(y_stretch_train)) > 1:
        stretch_model = create_model(regressor=False, use_gpu=use_gpu, n_estimators=250)
        stretch_model.fit(X_train, y_stretch_train)
        y_stretch_pred = stretch_model.predict(X_test)
        stretch_accuracy = float(accuracy_score(y_stretch_test, y_stretch_pred))
        print(f"  Accuracy: {stretch_accuracy:.4f}")
    else:
        stretch_model = None
        stretch_accuracy = 0.0
        print("  Skipped (insufficient stretch diversity)")
    
    return {
        'stretch_label': 'combined',
        'force_model': force_model,
        'offset_model': offset_model,
        'stretch_model': stretch_model,
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
        'offset_accuracy': offset_accuracy,
        'stretch_accuracy': stretch_accuracy,
        'fz_min_actual': fz_min_actual,
        'fz_max_actual': fz_max_actual,
        'fz_train_min': fz_train_min,
        'fz_train_max': fz_train_max,
        'fz_test_min': fz_test_min,
        'fz_test_max': fz_test_max,
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
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SINGLE-POINT MODEL TRAINING")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
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
    for stretch in ['000pct', '010pct', '020pct']:
        # Try to find file in data_dir
        pattern = f"*stretch_{stretch}.h5"
        files = list(data_dir.glob(pattern))
        if not files:
            # Try in subdirectory
            stretch_dir = data_dir / f"stretch_{stretch}"
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
            # Remove outliers
            print(f"  Removing outliers...")
            cleaned_sequences, outlier_indices = remove_outliers(sequences, args.z_threshold)
            print(f"  After outlier removal: {len(cleaned_sequences)} sequences")
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
        result = train_models_for_stretch(sequences, stretch_label, train_ratio=0.7, use_gpu=args.use_gpu, gpu_id=gpu_id)
        trained_models[stretch_label] = result
        
    
    # Train combined model
    print("\n" + "="*80)
    print("TRAINING COMBINED MODEL")
    print("="*80)
    combined_result = train_combined_model(sequences_by_stretch, train_ratio=0.7, use_gpu=args.use_gpu)
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
                'rmse': result['force_rmse'],
                'std_dev': result['force_std_dev'],
                'force_resolution_est': np.nan,  # Not calculated in this script
                'kpm1_pass': None,
                'kpm2_pass': None,
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
                'accuracy': result['offset_accuracy'],
            })
        
    
    # Combined metrics
    if 'combined' in trained_models:
        combined_result = trained_models['combined']
        combined_force_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result['n_samples'],
            'rmse': combined_result['force_rmse'],
            'std_dev': combined_result['force_std_dev'],
            'force_resolution_est': np.nan,
            'kpm1_pass': None,
            'kpm2_pass': None,
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
    reports_dir = Path(LOGS_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = reports_dir / f"{data_dir.name}_metrics.json"
    
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
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print(f"{'Model':<15} {'Sequences':<12} {'Train Seq':<12} {'Test Seq':<12} {'Samples':<12} {'Train Samples':<15} {'Test Samples':<15} {'RMSE':<10} {'Offset Acc':<12} {'Fz Range':<15}")
    print("-"*100)
    
    for model_name, result in trained_models.items():
        # Skip sensor8 models in main summary (they have their own section)
        if '_sensor8' in model_name:
            continue
        fz_min = result.get('fz_min_actual', np.nan)
        fz_max = result.get('fz_max_actual', np.nan)
        fz_range_str = f"[{fz_min:.2f},{fz_max:.2f}]" if not np.isnan(fz_min) and not np.isnan(fz_max) else "N/A"
        print(f"{model_name:<15} {result['n_sequences']:<12} {result['n_train_sequences']:<12} "
              f"{result['n_test_sequences']:<12} {result['n_samples']:<12} "
              f"{result['n_train_samples']:<15} {result['n_test_samples']:<15} "
              f"{result['force_rmse']:<10.4f} {result['offset_accuracy']:<12.4f} {fz_range_str:<15}")
    
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

