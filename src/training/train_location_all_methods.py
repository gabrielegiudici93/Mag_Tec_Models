#!/usr/bin/env python3
"""
Train location classification using all available data (normal + shear forces)
with all feature methods (raw, magnitude, magnitude_normalized).

Usage:
    python3 src/training/train_location_all_methods.py \
        --normal-dir data/Multiple_Points/2.5mm_single_test42 \
        --shear-dir data/Multiple_Points/shear_forces_test51 \
        --run-label location_all_methods \
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
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Multi-point specific configuration
MULTIPOINT_OFFSET_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'no_touch']
MULTIPOINT_N_OFFSETS = 11  # 10 locations + no_touch


def train_location_classifier(
    sequences_by_stretch: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    feature_method: str = 'raw',
    location_feature_method: str = 'raw',
) -> Dict:
    """Train location classifier using all sequences."""
    print(f"\n{'='*80}")
    print(f"TRAINING LOCATION CLASSIFIER ({location_feature_method} features)")
    print(f"{'='*80}")
    
    # Combine all sequences
    all_sequences = []
    for stretch_label_key, sequences in sequences_by_stretch.items():
        for seq in sequences:
            seq['stretch_label'] = stretch_label_key
        all_sequences.extend(sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    
    # Prepare data - only for location classification (we don't need forces)
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
        remove_fz_baseline=False  # Don't remove baseline for location classification
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    
    # Count sequences per location
    location_counts = {}
    for seq in all_sequences:
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
    
    print(f"  Location - Train Accuracy: {accuracy_score(y_offset_train, location_model.predict(X_train)):.4f}")
    print(f"  Location - Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_offset_test, y_offset_pred, labels=np.arange(11))
    print(f"\n  Confusion matrix shape: {cm.shape}")
    
    return {
        'location_model': location_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'y_offset_test': y_offset_test,
        'y_offset_pred': y_offset_pred,
        'confusion_matrix': cm.tolist(),
        'n_train': len(train_indices),
        'n_test': len(test_indices),
    }


def main():
    parser = argparse.ArgumentParser(description="Train location classification with all feature methods")
    parser.add_argument('--normal-dir', type=Path, required=True, help='Directory with normal forces HDF5 files')
    parser.add_argument('--shear-dir', type=Path, required=True, help='Directory with shear forces HDF5 files')
    parser.add_argument('--run-label', type=str, default='location_all_methods', help='Run label for output')
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
    print("LOADING NORMAL FORCES")
    print(f"{'='*80}")
    normal_sequences = {}
    for stretch, h5_file in normal_files_dict.items():
        sequences = load_sequences_from_h5(h5_file)
        for seq in sequences:
            seq['stretch_label'] = stretch
        
        if args.remove_outliers:
            sequences, _ = remove_outliers(sequences, z_threshold=args.z_threshold, remove_per_offset=2)
        
        # Limit to 30 per location
        sequences = limit_sequences_per_location(sequences, max_sequences_per_location=30, random_seed=42)
        
        location_counts = {}
        for seq in sequences:
            location = seq.get('offset', 'unknown')
            location_counts[location] = location_counts.get(location, 0) + 1
        
        normal_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences total (30 per location)")
        print(f"    Per location: {dict(list(location_counts.items())[:5])}...")
    
    # Load shear forces sequences
    print(f"\n{'='*80}")
    print("LOADING SHEAR FORCES")
    print(f"{'='*80}")
    shear_sequences = {}
    for stretch, h5_file in shear_files_dict.items():
        sequences = load_sequences_from_h5(h5_file)
        for seq in sequences:
            seq['stretch_label'] = stretch
        
        if args.remove_outliers:
            sequences, _ = remove_outliers(sequences, z_threshold=args.z_threshold, remove_per_offset=2)
        
        # Limit to 20 per location per direction
        sequences = limit_sequences_per_location_and_direction(sequences, max_sequences_per_location_per_direction=20, random_seed=42)
        
        shear_sequences[stretch] = sequences
        print(f"  {stretch}: {len(sequences)} sequences total (20 per location per direction)")
    
    # Combine normal + shear sequences
    print(f"\n{'='*80}")
    print("COMBINING NORMAL + SHEAR FORCES")
    print(f"{'='*80}")
    combined_sequences = {}
    for stretch in ['000pct', '010pct', '020pct']:
        combined = []
        if stretch in normal_sequences:
            combined.extend(normal_sequences[stretch])
        if stretch in shear_sequences:
            combined.extend(shear_sequences[stretch])
        combined_sequences[stretch] = combined
        print(f"  {stretch}: {len(combined)} sequences ({len(normal_sequences.get(stretch, []))} normal + {len(shear_sequences.get(stretch, []))} shear)")
    
    # Create output directory
    output_dir = args.shear_dir / "location_classification" / args.run_label
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train with all feature methods
    feature_methods = ['raw', 'magnitude', 'magnitude_normalized']
    results = {}
    
    for location_feature_method in feature_methods:
        print(f"\n{'='*80}")
        print(f"FEATURE METHOD: {location_feature_method}")
        print(f"{'='*80}")
        
        result = train_location_classifier(
            combined_sequences,
            train_ratio=0.7,
            feature_method='raw',  # Always use raw for feature engineering
            location_feature_method=location_feature_method
        )
        
        results[location_feature_method] = result
        
        # Save model
        model_path = models_dir / f"location_classifier_{location_feature_method}.joblib"
        joblib.dump(result['location_model'], model_path)
        print(f"  ✓ Saved: {model_path.name}")
        
        scaler_path = models_dir / f"scaler_{location_feature_method}.joblib"
        joblib.dump(result['scaler'], scaler_path)
        print(f"  ✓ Saved: {scaler_path.name}")
    
    # Save metrics
    metrics = {
        'run_label': args.run_label,
        'normal_sequences_per_stretch': {k: len(v) for k, v in normal_sequences.items()},
        'shear_sequences_per_stretch': {k: len(v) for k, v in shear_sequences.items()},
        'combined_sequences_per_stretch': {k: len(v) for k, v in combined_sequences.items()},
        'results': {
            method: {
                'accuracy': float(result['accuracy']),
                'n_train': result['n_train'],
                'n_test': result['n_test'],
            }
            for method, result in results.items()
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Saved metrics: {metrics_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nLocation Classification Accuracy:")
    for method, result in results.items():
        print(f"  {method:20s}: {result['accuracy']:.4f} ({result['n_train']} train, {result['n_test']} test)")
    
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n  ✅ Best method: {best_method[0]} (accuracy: {best_method[1]['accuracy']:.4f})")
    
    print(f"\n{'='*80}")
    print("✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Metrics: {metrics_file}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

