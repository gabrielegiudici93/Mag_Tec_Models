#!/usr/bin/env python3
"""
Train separate force regressors:
- Fz regressor: using only normal forces (test42)
- Fx, Fy regressors: using only shear forces (test51)

IMPORTANT: Fx regression uses only x+ and x- sequences
           Fy regression uses only y+ and y- sequences

Usage:
    python3 src/training/train_separate_forces.py \
        --normal-dir data/Multiple_Points/2.5mm_single_test42 \
        --shear-dir data/Multiple_Points/shear_forces_test51 \
        --run-label separate_forces \
        --feature-method raw \
        --max-sequences 20
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
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import functions from train_combined_shear_normal
from training.train_combined_shear_normal import (
    load_sequences_from_h5,
    remove_outliers,
    prepare_training_data,
    create_model,
    filter_high_displacement,
)

# FT sensor precision constants
FT_SENSOR_PRECISION = 0.01  # N (FT-300S sensor resolution)


def train_fz_regressor(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    feature_method: str = 'raw',
    location_feature_method: str = 'raw',
) -> Dict:
    """Train Fz regressor using only normal forces."""
    print(f"\n{'='*80}")
    print("TRAINING FZ REGRESSOR (Normal Forces Only)")
    print(f"{'='*80}")
    
    # Combine all sequences
    all_sequences = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data - only Fz
    # IMPORTANT: Do NOT remove baseline for Fz (like old train_multipoint.py code)
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method,
        remove_fz_baseline=False  # Do NOT remove baseline for Fz (like old code)
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N (absolute values, cut at 3N)")
    
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
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train Fz regressor
    print("\nTraining Fz regressor...")
    fz_model = create_model(regressor=True, use_gpu=False, n_estimators=200, gpu_id=0)
    fz_model.fit(X_train, y_fz_train)
    y_fz_pred = fz_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred))
    r2 = r2_score(y_fz_test, y_fz_pred)
    
    print(f"  Fz - Train RMSE: {np.sqrt(mean_squared_error(y_fz_train, fz_model.predict(X_train))):.4f} N")
    print(f"  Fz - Test RMSE: {rmse:.4f} N")
    print(f"  Fz - R²: {r2:.4f}")
    
    return {
        'fz_model': fz_model,
        'scaler': scaler,
        'rmse_fz': rmse,
        'r2_fz': r2,
        'y_fz_test': y_fz_test,
        'y_fz_pred': y_fz_pred,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
    }


def train_fx_fy_regressors(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    feature_method: str = 'raw',
    location_feature_method: str = 'raw',
) -> Dict:
    """Train Fx, Fy regressors using only shear forces.
    
    IMPORTANT: For Fx regression, use only x+ and x- sequences.
    For Fy regression, use only y+ and y- sequences.
    This avoids confusion between directions.
    """
    print(f"\n{'='*80}")
    print("TRAINING FX, FY REGRESSORS (Shear Forces Only)")
    print(f"{'='*80}")
    
    # Separate sequences by direction
    fx_sequences = []  # x+ and x- sequences for Fx regression
    fy_sequences = []  # y+ and y- sequences for Fy regression
    
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
            label = seq.get('label', '')
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            
            # Check direction
            if 'shear' in label.lower():
                if 'x+' in label or 'x-' in label:
                    fx_sequences.append(seq)
                elif 'y+' in label or 'y-' in label:
                    fy_sequences.append(seq)
    
    print(f"Total sequences for Fx regression (x+ and x-): {len(fx_sequences)}")
    print(f"Total sequences for Fy regression (y+ and y-): {len(fy_sequences)}")
    
    # Prepare data for Fx regression (only x+ and x- sequences)
    X_fx, y_fx_all, y_fy_dummy, y_fz_dummy, y_offset_dummy, scaler_fx, fz_scaler_dummy, actual_sequence_lengths_fx = prepare_training_data(
        fx_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method
    )
    
    # Prepare data for Fy regression (only y+ and y- sequences)
    X_fy, y_fx_dummy, y_fy_all, y_fz_dummy, y_offset_dummy, scaler_fy, fz_scaler_dummy, actual_sequence_lengths_fy = prepare_training_data(
        fy_sequences,
        normalize=True,
        use_feature_engineering=(feature_method == 'normalized'),
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method
    )
    
    print(f"\nFx regression data: {len(X_fx)} samples")
    print(f"Fy regression data: {len(X_fy)} samples")
    print(f"Fx range: [{np.min(y_fx_all):.3f}, {np.max(y_fx_all):.3f}] N")
    print(f"Fy range: [{np.min(y_fy_all):.3f}, {np.max(y_fy_all):.3f}] N")
    
    # Split Fx sequences by sequence
    sequence_lengths_fx = actual_sequence_lengths_fx
    n_sequences_fx = len(sequence_lengths_fx)
    n_train_fx = int(n_sequences_fx * train_ratio)
    
    np.random.seed(42)
    indices_fx = np.random.permutation(n_sequences_fx)
    train_indices_fx = indices_fx[:n_train_fx]
    test_indices_fx = indices_fx[n_train_fx:]
    
    sequence_to_samples_fx = {}
    current_idx = 0
    for seq_idx, seq_len in enumerate(sequence_lengths_fx):
        sequence_to_samples_fx[seq_idx] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    train_samples_fx = []
    test_samples_fx = []
    for seq_idx in train_indices_fx:
        start, end = sequence_to_samples_fx[seq_idx]
        train_samples_fx.extend(range(start, end))
    for seq_idx in test_indices_fx:
        start, end = sequence_to_samples_fx[seq_idx]
        test_samples_fx.extend(range(start, end))
    
    X_fx_train = X_fx[train_samples_fx]
    X_fx_test = X_fx[test_samples_fx]
    y_fx_train = y_fx_all[train_samples_fx]
    y_fx_test = y_fx_all[test_samples_fx]
    
    print(f"\nFx - Train: {len(train_indices_fx)} sequences, {len(X_fx_train)} samples")
    print(f"Fx - Test: {len(test_indices_fx)} sequences, {len(X_fx_test)} samples")
    
    # Train Fx regressor (only on x+ and x- sequences)
    print("\nTraining Fx regressor (x+ and x- sequences only)...")
    fx_model = create_model(regressor=True, use_gpu=False, n_estimators=200, gpu_id=0)
    fx_model.fit(X_fx_train, y_fx_train)
    y_fx_pred = fx_model.predict(X_fx_test)
    
    rmse_fx = np.sqrt(mean_squared_error(y_fx_test, y_fx_pred))
    r2_fx = r2_score(y_fx_test, y_fx_pred)
    
    print(f"  Fx - Train RMSE: {np.sqrt(mean_squared_error(y_fx_train, fx_model.predict(X_fx_train))):.4f} N")
    print(f"  Fx - Test RMSE: {rmse_fx:.4f} N")
    print(f"  Fx - R²: {r2_fx:.4f}")
    
    # Split Fy sequences by sequence
    sequence_lengths_fy = actual_sequence_lengths_fy
    n_sequences_fy = len(sequence_lengths_fy)
    n_train_fy = int(n_sequences_fy * train_ratio)
    
    np.random.seed(42)
    indices_fy = np.random.permutation(n_sequences_fy)
    train_indices_fy = indices_fy[:n_train_fy]
    test_indices_fy = indices_fy[n_train_fy:]
    
    sequence_to_samples_fy = {}
    current_idx = 0
    for seq_idx, seq_len in enumerate(sequence_lengths_fy):
        sequence_to_samples_fy[seq_idx] = (current_idx, current_idx + seq_len)
        current_idx += seq_len
    
    train_samples_fy = []
    test_samples_fy = []
    for seq_idx in train_indices_fy:
        start, end = sequence_to_samples_fy[seq_idx]
        train_samples_fy.extend(range(start, end))
    for seq_idx in test_indices_fy:
        start, end = sequence_to_samples_fy[seq_idx]
        test_samples_fy.extend(range(start, end))
    
    X_fy_train = X_fy[train_samples_fy]
    X_fy_test = X_fy[test_samples_fy]
    y_fy_train = y_fy_all[train_samples_fy]
    y_fy_test = y_fy_all[test_samples_fy]
    
    print(f"\nFy - Train: {len(train_indices_fy)} sequences, {len(X_fy_train)} samples")
    print(f"Fy - Test: {len(test_indices_fy)} sequences, {len(X_fy_test)} samples")
    
    # Train Fy regressor (only on y+ and y- sequences)
    print("\nTraining Fy regressor (y+ and y- sequences only)...")
    fy_model = create_model(regressor=True, use_gpu=False, n_estimators=200, gpu_id=0)
    fy_model.fit(X_fy_train, y_fy_train)
    y_fy_pred = fy_model.predict(X_fy_test)
    
    rmse_fy = np.sqrt(mean_squared_error(y_fy_test, y_fy_pred))
    r2_fy = r2_score(y_fy_test, y_fy_pred)
    
    print(f"  Fy - Train RMSE: {np.sqrt(mean_squared_error(y_fy_train, fy_model.predict(X_fy_train))):.4f} N")
    print(f"  Fy - Test RMSE: {rmse_fy:.4f} N")
    print(f"  Fy - R²: {r2_fy:.4f}")
    
    return {
        'fx_model': fx_model,
        'fy_model': fy_model,
        'scaler_fx': scaler_fx,
        'scaler_fy': scaler_fy,
        'rmse_fx': rmse_fx,
        'rmse_fy': rmse_fy,
        'r2_fx': r2_fx,
        'r2_fy': r2_fy,
        'y_fx_test': y_fx_test,
        'y_fx_pred': y_fx_pred,
        'y_fy_test': y_fy_test,
        'y_fy_pred': y_fy_pred,
        'n_train_fx': len(train_indices_fx),
        'n_test_fx': len(test_indices_fx),
        'n_train_fy': len(train_indices_fy),
        'n_test_fy': len(test_indices_fy),
    }


def main():
    parser = argparse.ArgumentParser(description="Train separate force regressors")
    parser.add_argument('--normal-dir', type=Path, required=True, help='Directory with normal forces HDF5 files')
    parser.add_argument('--shear-dir', type=Path, required=True, help='Directory with shear forces HDF5 files')
    parser.add_argument('--run-label', type=str, default='separate_forces', help='Run label for output')
    parser.add_argument('--feature-method', type=str, default='raw', choices=['raw', 'normalized'], help='Feature method')
    parser.add_argument('--max-sequences', type=int, default=None, help='Max sequences per location (default: 30 for normal, 20 for shear per location per direction)')
    parser.add_argument('--remove-outliers', action='store_true', help='Remove outliers')
    parser.add_argument('--z-threshold', type=float, default=3.0, help='Z-score threshold for outliers')
    
    args = parser.parse_args()
    
    # Find normal forces files
    normal_files_dict = {}
    for stretch in ['000pct', '010pct', '020pct']:
        pattern = f"*stretch_{stretch}.h5"
        files = list(args.normal_dir.glob(pattern))
        files = [f for f in files if 'no_touch' not in f.name]
        if files:
            normal_files_dict[stretch] = files[0]
            print(f"Found NORMAL {stretch}: {files[0].name}")
    
    # Find shear forces files
    shear_files_dict = {}
    for stretch in ['000pct', '010pct', '020pct']:
        pattern = f"*shear*stretch_{stretch}.h5"
        files = list(args.shear_dir.glob(pattern))
        if files:
            shear_files_dict[stretch] = files[0]
            print(f"Found SHEAR {stretch}: {files[0].name}")
    
    # Load normal forces sequences
    print(f"\n{'='*80}")
    print("LOADING NORMAL FORCES (for Fz regression)")
    print(f"{'='*80}")
    normal_sequences = {}
    for stretch, h5_file in normal_files_dict.items():
        sequences = load_sequences_from_h5(h5_file)
        for seq in sequences:
            seq['stretch_label'] = stretch
        
        if args.remove_outliers:
            sequences, _ = remove_outliers(sequences, z_threshold=args.z_threshold, remove_per_offset=2)
        
        # Limit sequences PER LOCATION (default 30 per point)
        # 30 sequenze per punto × 10 punti = 300 sequenze totali per stretch
        max_seq_per_location = args.max_sequences if args.max_sequences else 30
        from training.train_combined_shear_normal import limit_sequences_per_location
        sequences = limit_sequences_per_location(sequences, max_seq_per_location, random_seed=42)
        
        # Count sequences per location
        location_counts = {}
        for seq in sequences:
            location = seq.get('offset', 'unknown')
            location_counts[location] = location_counts.get(location, 0) + 1
        
        normal_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences total ({max_seq_per_location} per location)")
        print(f"    Per location: {location_counts}")
    
    # Load shear forces sequences
    print(f"\n{'='*80}")
    print("LOADING SHEAR FORCES (for Fx, Fy regression)")
    print(f"{'='*80}")
    shear_sequences = {}
    for stretch, h5_file in shear_files_dict.items():
        sequences = load_sequences_from_h5(h5_file)
        for seq in sequences:
            seq['stretch_label'] = stretch
        
        if args.remove_outliers:
            sequences, _ = remove_outliers(sequences, z_threshold=args.z_threshold, remove_per_offset=2)
        
        # IMPORTANT: Limit sequences PER LOCATION AND PER DIRECTION
        # 20 sequenze per punto × 4 direzioni × 10 punti = 800 sequenze totali per stretch
        max_seq_per_location_per_direction = args.max_sequences if args.max_sequences else 20
        from training.train_combined_shear_normal import limit_sequences_per_location_and_direction
        sequences = limit_sequences_per_location_and_direction(sequences, max_seq_per_location_per_direction, random_seed=42)
        
        # Count sequences per location and direction for verification
        from collections import defaultdict
        location_direction_counts = defaultdict(int)
        for seq in sequences:
            location = seq.get('offset', 'unknown')
            label = seq.get('label', '')
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            direction = 'unknown'
            if 'shear' in label.lower():
                if 'x+' in label or 'x-' in label:
                    direction = 'x'
                elif 'y+' in label or 'y-' in label:
                    direction = 'y'
            location_direction_counts[(location, direction)] += 1
        
        shear_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences total ({max_seq_per_location_per_direction} per location per direction)")
        print(f"    Example counts: {dict(list(location_direction_counts.items())[:5])}...")
    
    # Create output directory
    output_dir = args.shear_dir / "separate_forces" / args.feature_method
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train Fz regressor (normal forces only)
    fz_result = train_fz_regressor(
        normal_sequences,
        train_ratio=0.7,
        feature_method=args.feature_method,
        location_feature_method='raw'
    )
    
    # Train Fx, Fy regressors (shear forces only)
    fx_fy_result = train_fx_fy_regressors(
        shear_sequences,
        train_ratio=0.7,
        feature_method=args.feature_method,
        location_feature_method='raw'
    )
    
    # Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")
    
    joblib.dump(fz_result['fz_model'], models_dir / "fz_regressor.joblib")
    print(f"  ✓ Saved: fz_regressor.joblib")
    
    joblib.dump(fx_fy_result['fx_model'], models_dir / "fx_regressor.joblib")
    print(f"  ✓ Saved: fx_regressor.joblib")
    
    joblib.dump(fx_fy_result['fy_model'], models_dir / "fy_regressor.joblib")
    print(f"  ✓ Saved: fy_regressor.joblib")
    
    joblib.dump(fz_result['scaler'], models_dir / "scaler_fz.joblib")
    print(f"  ✓ Saved: scaler_fz.joblib")
    
    joblib.dump(fx_fy_result['scaler_fx'], models_dir / "scaler_fx.joblib")
    print(f"  ✓ Saved: scaler_fx.joblib")
    
    joblib.dump(fx_fy_result['scaler_fy'], models_dir / "scaler_fy.joblib")
    print(f"  ✓ Saved: scaler_fy.joblib")
    
    # Save metrics
    metrics = {
        'run_label': args.run_label,
        'feature_method': args.feature_method,
        'fz_regression': {
            'rmse': float(fz_result['rmse_fz']),
            'r2': float(fz_result['r2_fz']),
            'n_train': fz_result['n_train'],
            'n_test': fz_result['n_test'],
            'dataset': 'normal_forces_only',
        },
        'fx_regression': {
            'rmse': float(fx_fy_result['rmse_fx']),
            'r2': float(fx_fy_result['r2_fx']),
            'n_train': fx_fy_result['n_train_fx'],
            'n_test': fx_fy_result['n_test_fx'],
            'dataset': 'shear_forces_only_x_directions',
            'note': 'Trained only on x+ and x- sequences',
        },
        'fy_regression': {
            'rmse': float(fx_fy_result['rmse_fy']),
            'r2': float(fx_fy_result['r2_fy']),
            'n_train': fx_fy_result['n_train_fy'],
            'n_test': fx_fy_result['n_test_fy'],
            'dataset': 'shear_forces_only_y_directions',
            'note': 'Trained only on y+ and y- sequences',
        },
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Saved metrics: {metrics_file}")
    
    print(f"\n{'='*80}")
    print("✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Metrics: {metrics_file}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
