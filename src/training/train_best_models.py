#!/usr/bin/env python3
"""
Train the best models for force regression, stretch classification, and location classification.

Uses optimal configurations:
- Fz regression: Normal forces only (30 per point), NO baseline removal
- Fx/Fy regression: Shear forces only (20 per point per direction)
- Location classification: Combined data (normal + shear), raw or magnitude features
- Stretch classification: Combined data (normal + shear)

Usage:
    python3 src/training/train_best_models.py \
        --normal-dir data/Multiple_Points/2.5mm_single_test42 \
        --shear-dir data/Multiple_Points/shear_forces_test51 \
        --run-label best_models \
        --remove-outliers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_combined_shear_normal import (
    load_sequences_from_h5,
    remove_outliers,
    prepare_training_data,
    create_model,
    limit_sequences_per_location,
    limit_sequences_per_location_and_direction,
)


def train_fz_regressor(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
) -> Dict:
    """Train Fz regressor using normal forces only (best configuration)."""
    print(f"\n{'='*80}")
    print("TRAINING FZ REGRESSOR (Normal Forces Only)")
    print(f"{'='*80}")
    
    all_sequences = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data - NO baseline removal for Fz
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method='raw',
        remove_fz_baseline=False  # NO baseline removal for Fz
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fz range: [{np.min(y_fz):.3f}, {np.max(y_fz):.3f}] N")
    
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
) -> Dict:
    """Train Fx, Fy regressors using shear forces only (best configuration)."""
    print(f"\n{'='*80}")
    print("TRAINING FX, FY REGRESSORS (Shear Forces Only)")
    print(f"{'='*80}")
    
    # Separate sequences by direction
    fx_sequences = []
    fy_sequences = []
    
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
            label = seq.get('label', '')
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            
            if 'shear' in label.lower():
                if 'x+' in label or 'x-' in label:
                    fx_sequences.append(seq)
                elif 'y+' in label or 'y-' in label:
                    fy_sequences.append(seq)
    
    print(f"Total sequences for Fx regression (x+ and x-): {len(fx_sequences)}")
    print(f"Total sequences for Fy regression (y+ and y-): {len(fy_sequences)}")
    
    # Prepare data for Fx regression
    X_fx, y_fx_all, y_fy_dummy, y_fz_dummy, y_offset_dummy, scaler_fx, fz_scaler_dummy, actual_sequence_lengths_fx = prepare_training_data(
        fx_sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method='raw'
    )
    
    # Prepare data for Fy regression
    X_fy, y_fx_dummy, y_fy_all, y_fz_dummy, y_offset_dummy, scaler_fy, fz_scaler_dummy, actual_sequence_lengths_fy = prepare_training_data(
        fy_sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method='raw'
    )
    
    print(f"\nFx regression data: {len(X_fx)} samples")
    print(f"Fy regression data: {len(X_fy)} samples")
    
    # Split Fx sequences
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
    
    # Train Fx regressor
    print("\nTraining Fx regressor...")
    fx_model = create_model(regressor=True, use_gpu=False, n_estimators=200, gpu_id=0)
    fx_model.fit(X_fx_train, y_fx_train)
    y_fx_pred = fx_model.predict(X_fx_test)
    
    rmse_fx = np.sqrt(mean_squared_error(y_fx_test, y_fx_pred))
    r2_fx = r2_score(y_fx_test, y_fx_pred)
    
    print(f"  Fx - Train RMSE: {np.sqrt(mean_squared_error(y_fx_train, fx_model.predict(X_fx_train))):.4f} N")
    print(f"  Fx - Test RMSE: {rmse_fx:.4f} N")
    print(f"  Fx - R²: {r2_fx:.4f}")
    
    # Split Fy sequences
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
    
    # Train Fy regressor
    print("\nTraining Fy regressor...")
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


def train_location_classifier(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    location_feature_method: str = 'raw',
) -> Dict:
    """Train location classifier using combined data (best configuration)."""
    print(f"\n{'='*80}")
    print(f"TRAINING LOCATION CLASSIFIER ({location_feature_method} features)")
    print(f"{'='*80}")
    
    all_sequences = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method,
        remove_fz_baseline=False
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    
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
    y_offset_train = y_offset[train_samples]
    y_offset_test = y_offset[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train location classifier
    print("\nTraining location classifier...")
    location_model = create_model(regressor=False, use_gpu=False, n_estimators=200, gpu_id=0)
    location_model.fit(X_train, y_offset_train)
    y_offset_pred = location_model.predict(X_test)
    
    accuracy = accuracy_score(y_offset_test, y_offset_pred)
    cm = confusion_matrix(y_offset_test, y_offset_pred, labels=np.arange(11))
    
    print(f"  Location - Train Accuracy: {accuracy_score(y_offset_train, location_model.predict(X_train)):.4f}")
    print(f"  Location - Test Accuracy: {accuracy:.4f}")
    
    return {
        'location_model': location_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'n_train': len(train_indices),
        'n_test': len(test_indices),
    }


def train_stretch_classifier(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    location_feature_method: str = 'raw',
) -> Dict:
    """Train stretch classifier using combined data."""
    print(f"\n{'='*80}")
    print(f"TRAINING STRETCH CLASSIFIER ({location_feature_method} features)")
    print(f"{'='*80}")
    
    all_sequences = []
    all_stretches = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
        all_stretches.extend([stretch_label_key] * len(sequences))
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data
    X, y_fx, y_fy, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        all_sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False,
        location_feature_method=location_feature_method,
        remove_fz_baseline=False
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    
    # Encode stretch labels
    stretch_map = {'000pct': 0, '010pct': 1, '020pct': 2}
    y_stretch = []
    for seq_idx, seq_len in enumerate(actual_sequence_lengths):
        stretch_label = all_stretches[seq_idx]
        stretch_num = stretch_map.get(stretch_label, 0)
        y_stretch.extend([stretch_num] * seq_len)
    y_stretch = np.array(y_stretch)
    
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
    y_stretch_train = y_stretch[train_samples]
    y_stretch_test = y_stretch[test_samples]
    
    print(f"\nTrain: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train stretch classifier
    print("\nTraining stretch classifier...")
    stretch_model = create_model(regressor=False, use_gpu=False, n_estimators=200, gpu_id=0)
    stretch_model.fit(X_train, y_stretch_train)
    y_stretch_pred = stretch_model.predict(X_test)
    
    accuracy = accuracy_score(y_stretch_test, y_stretch_pred)
    cm = confusion_matrix(y_stretch_test, y_stretch_pred, labels=np.arange(3))
    
    print(f"  Stretch - Train Accuracy: {accuracy_score(y_stretch_train, stretch_model.predict(X_train)):.4f}")
    print(f"  Stretch - Test Accuracy: {accuracy:.4f}")
    
    return {
        'stretch_model': stretch_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'n_train': len(train_indices),
        'n_test': len(test_indices),
    }


def main():
    parser = argparse.ArgumentParser(description="Train best models for all tasks")
    parser.add_argument('--normal-dir', type=Path, required=True, help='Directory with normal forces HDF5 files')
    parser.add_argument('--shear-dir', type=Path, required=True, help='Directory with shear forces HDF5 files')
    parser.add_argument('--run-label', type=str, default='best_models', help='Run label for output')
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
        
        sequences = limit_sequences_per_location(sequences, max_sequences_per_location=30, random_seed=42)
        normal_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences (30 per location)")
    
    # Load shear forces sequences
    print(f"\n{'='*80}")
    print("LOADING SHEAR FORCES (for Fx/Fy regression and classification)")
    print(f"{'='*80}")
    shear_sequences = {}
    for stretch, h5_file in shear_files_dict.items():
        sequences = load_sequences_from_h5(h5_file)
        for seq in sequences:
            seq['stretch_label'] = stretch
        
        if args.remove_outliers:
            sequences, _ = remove_outliers(sequences, z_threshold=args.z_threshold, remove_per_offset=2)
        
        sequences = limit_sequences_per_location_and_direction(sequences, max_sequences_per_location_per_direction=20, random_seed=42)
        shear_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences (20 per location per direction)")
    
    # Combine for classification tasks
    combined_sequences = {}
    for stretch in ['000pct', '010pct', '020pct']:
        combined = []
        if stretch in normal_sequences:
            combined.extend(normal_sequences[stretch])
        if stretch in shear_sequences:
            combined.extend(shear_sequences[stretch])
        combined_sequences[stretch] = combined
        print(f"  {stretch}: {len(combined)} combined sequences")
    
    # Create output directory
    output_dir = args.shear_dir / "best_models" / args.run_label
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    print(f"\n{'='*80}")
    print("TRAINING ALL MODELS")
    print(f"{'='*80}")
    
    # 1. Force regression
    fz_result = train_fz_regressor(normal_sequences, train_ratio=0.7)
    fx_fy_result = train_fx_fy_regressors(shear_sequences, train_ratio=0.7)
    
    # 2. Location classification (test both raw and magnitude)
    location_results = {}
    for method in ['raw', 'magnitude']:
        location_results[method] = train_location_classifier(combined_sequences, train_ratio=0.7, location_feature_method=method)
    
    # 3. Stretch classification (use raw features)
    stretch_result = train_stretch_classifier(combined_sequences, train_ratio=0.7, location_feature_method='raw')
    
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
    
    # Save best location classifier (choose between raw and magnitude)
    best_location_method = max(location_results.items(), key=lambda x: x[1]['accuracy'])[0]
    joblib.dump(location_results[best_location_method]['location_model'], models_dir / "location_classifier.joblib")
    joblib.dump(location_results[best_location_method]['scaler'], models_dir / "location_scaler.joblib")
    print(f"  ✓ Saved: location_classifier.joblib ({best_location_method} features)")
    
    joblib.dump(stretch_result['stretch_model'], models_dir / "stretch_classifier.joblib")
    joblib.dump(stretch_result['scaler'], models_dir / "stretch_scaler.joblib")
    print(f"  ✓ Saved: stretch_classifier.joblib")
    
    # Save scalers for force regression
    joblib.dump(fz_result['scaler'], models_dir / "scaler_fz.joblib")
    joblib.dump(fx_fy_result['scaler_fx'], models_dir / "scaler_fx.joblib")
    joblib.dump(fx_fy_result['scaler_fy'], models_dir / "scaler_fy.joblib")
    
    # Save metrics
    metrics = {
        'run_label': args.run_label,
        'force_regression': {
            'fz': {
                'rmse': float(fz_result['rmse_fz']),
                'r2': float(fz_result['r2_fz']),
                'n_train': fz_result['n_train'],
                'n_test': fz_result['n_test'],
                'dataset': 'normal_forces_only',
                'note': 'NO baseline removal',
            },
            'fx': {
                'rmse': float(fx_fy_result['rmse_fx']),
                'r2': float(fx_fy_result['r2_fx']),
                'n_train': fx_fy_result['n_train_fx'],
                'n_test': fx_fy_result['n_test_fx'],
                'dataset': 'shear_forces_only_x_directions',
            },
            'fy': {
                'rmse': float(fx_fy_result['rmse_fy']),
                'r2': float(fx_fy_result['r2_fy']),
                'n_train': fx_fy_result['n_train_fy'],
                'n_test': fx_fy_result['n_test_fy'],
                'dataset': 'shear_forces_only_y_directions',
            },
        },
        'location_classification': {
            method: {
                'accuracy': float(result['accuracy']),
                'n_train': result['n_train'],
                'n_test': result['n_test'],
            }
            for method, result in location_results.items()
        },
        'stretch_classification': {
            'accuracy': float(stretch_result['accuracy']),
            'n_train': stretch_result['n_train'],
            'n_test': stretch_result['n_test'],
            'feature_method': 'raw',
        },
        'best_location_method': best_location_method,
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Saved metrics: {metrics_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print("\n📍 FORCE REGRESSION:")
    print(f"   Fz: RMSE {metrics['force_regression']['fz']['rmse']:.4f} N, R² {metrics['force_regression']['fz']['r2']:.4f}")
    print(f"   Fx: RMSE {metrics['force_regression']['fx']['rmse']:.4f} N, R² {metrics['force_regression']['fx']['r2']:.4f}")
    print(f"   Fy: RMSE {metrics['force_regression']['fy']['rmse']:.4f} N, R² {metrics['force_regression']['fy']['r2']:.4f}")
    
    print("\n📍 LOCATION CLASSIFICATION:")
    for method, result in metrics['location_classification'].items():
        print(f"   {method:20s}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"   ✅ Best method: {best_location_method} (accuracy: {metrics['location_classification'][best_location_method]['accuracy']:.4f})")
    
    print("\n📍 STRETCH CLASSIFICATION:")
    print(f"   Accuracy: {metrics['stretch_classification']['accuracy']:.4f} ({metrics['stretch_classification']['accuracy']*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print("✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Metrics: {metrics_file}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

