#!/usr/bin/env python3
"""
Analyze data sufficiency by training with different amounts of data (10, 20, 30 movements).

This script loads shear forces data and trains models with progressively more data
to determine the minimum amount needed for good performance.
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List
import h5py

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_multipoint import (
    load_sequences_from_h5,
    prepare_training_data,
    create_model,
    train_models_for_stretch
)
from sklearn.metrics import mean_squared_error, accuracy_score


def group_sequences_by_point_direction(sequences: List[Dict]) -> Dict[str, List[Dict]]:
    """Group sequences by (point, direction) key."""
    grouped = {}
    for seq in sequences:
        # Extract point and direction from label
        label = seq.get('label', '')
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        
        # Extract point (e.g., "pos_1_1_press_00_shear_x+_3.0N" -> point="1")
        import re
        point_match = re.search(r'pos_(\d+)_\d+_press', label)
        direction_match = re.search(r'shear_([xy][+-])_', label)
        
        if point_match and direction_match:
            point = point_match.group(1)
            direction = direction_match.group(1)
            key = f"{point}_{direction}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(seq)
    
    return grouped


def sample_movements(grouped_sequences: Dict[str, List[Dict]], n_movements: int, random_seed: int = 42) -> List[Dict]:
    """Sample n_movements from each (point, direction) group."""
    np.random.seed(random_seed)
    sampled = []
    
    for key, seqs in grouped_sequences.items():
        if len(seqs) >= n_movements:
            # Randomly sample n_movements
            indices = np.random.choice(len(seqs), size=n_movements, replace=False)
            sampled.extend([seqs[i] for i in indices])
        else:
            # Use all available if less than n_movements
            sampled.extend(seqs)
            print(f"  Warning: {key} has only {len(seqs)} movements (requested {n_movements})")
    
    return sampled


def train_and_evaluate(sequences: List[Dict], stretch_label: str, n_movements: int) -> Dict:
    """Train models and return metrics."""
    print(f"\n{'='*70}")
    print(f"Training with {n_movements} movements per (point, direction)")
    print(f"{'='*70}")
    
    # Prepare data
    X, y_fz, y_offset, scaler, fz_scaler, actual_sequence_lengths = prepare_training_data(
        sequences,
        normalize=True,
        use_feature_engineering=False,
        filter_displacement=True,
        displacement_threshold=95.0,
        normalize_fz=False,
        fz_target_min=0.0,
        fz_target_max=3.0,
        include_offset_labels=False,
        use_advanced_features=False
    )
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    
    # Split by sequence (70% train, 30% test)
    n_sequences = len(actual_sequence_lengths)
    n_train = int(n_sequences * 0.7)
    
    np.random.seed(42)
    indices = np.random.permutation(n_sequences)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Map sequence indices to sample indices
    sequence_to_samples = {}
    current_idx = 0
    for seq_idx, seq_len in enumerate(actual_sequence_lengths):
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
    
    print(f"Train: {len(train_indices)} sequences, {len(X_train)} samples")
    print(f"Test: {len(test_indices)} sequences, {len(X_test)} samples")
    
    # Train force regressor
    print("\nTraining force regressor...")
    force_model = create_model(regressor=True, use_gpu=False, n_estimators=200, gpu_id=0)
    force_model.fit(X_train, y_fz_train)
    
    # Evaluate force regressor
    y_fz_pred_train = force_model.predict(X_train)
    y_fz_pred_test = force_model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_fz_train, y_fz_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_fz_test, y_fz_pred_test))
    std_dev_train = np.std(y_fz_train - y_fz_pred_train)
    std_dev_test = np.std(y_fz_test - y_fz_pred_test)
    
    print(f"  Train RMSE: {rmse_train:.4f} N, Std Dev: {std_dev_train:.4f} N")
    print(f"  Test RMSE:  {rmse_test:.4f} N, Std Dev: {std_dev_test:.4f} N")
    
    # Train location classifier
    print("\nTraining location classifier...")
    valid_mask = y_offset_train >= 0
    location_accuracy_train = 0.0
    location_accuracy_test = 0.0
    
    if np.sum(valid_mask) > 0 and len(np.unique(y_offset_train[valid_mask])) > 1:
        location_model = create_model(regressor=False, use_gpu=False, n_estimators=200, gpu_id=0)
        location_model.fit(X_train[valid_mask], y_offset_train[valid_mask])
        
        test_valid_mask = y_offset_test >= 0
        if np.sum(test_valid_mask) > 0:
            y_location_pred_train = location_model.predict(X_train[valid_mask])
            y_location_pred_test = location_model.predict(X_test[test_valid_mask])
            
            location_accuracy_train = float(accuracy_score(y_offset_train[valid_mask], y_location_pred_train))
            location_accuracy_test = float(accuracy_score(y_offset_test[test_valid_mask], y_location_pred_test))
            
            print(f"  Train Accuracy: {location_accuracy_train:.4f}")
            print(f"  Test Accuracy:  {location_accuracy_test:.4f}")
    
    # Check KPMs
    kpm1 = False
    unique_forces = np.unique(np.round(y_fz_test, decimals=2))
    if len(unique_forces) > 1:
        deltas = np.diff(unique_forces)
        delta_f_min = float(np.min(np.abs(deltas[np.abs(deltas) > 0])))
        kpm1 = delta_f_min <= 0.05
        print(f"  Force resolution (ΔF_min): {delta_f_min:.6f} N")
        print(f"  KPM1: {'PASS' if kpm1 else 'FAIL'}")
    
    kpm2 = rmse_test < 0.1 and std_dev_test < 0.05
    print(f"  KPM2: {'PASS' if kpm2 else 'FAIL'} (RMSE < 0.1N and STD < 0.05N)")
    
    return {
        'n_movements': n_movements,
        'n_sequences': len(sequences),
        'n_samples': len(X),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'force_rmse_train': float(rmse_train),
        'force_rmse_test': float(rmse_test),
        'force_std_train': float(std_dev_train),
        'force_std_test': float(std_dev_test),
        'location_accuracy_train': float(location_accuracy_train),
        'location_accuracy_test': float(location_accuracy_test),
        'kpm1_pass': bool(kpm1),
        'kpm2_pass': bool(kpm2),
        'overfitting_force': float(rmse_test - rmse_train),  # Positive = overfitting
        'overfitting_location': float(location_accuracy_train - location_accuracy_test),  # Positive = overfitting
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze data sufficiency with different amounts of movements")
    parser.add_argument('--data-file', type=Path, required=True, help='Path to HDF5 data file')
    parser.add_argument('--output', type=Path, default=None, help='Output JSON file for results')
    args = parser.parse_args()
    
    # Load sequences
    print(f"Loading data from {args.data_file}...")
    sequences = load_sequences_from_h5(args.data_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Group by (point, direction)
    grouped = group_sequences_by_point_direction(sequences)
    print(f"\nGrouped into {len(grouped)} (point, direction) combinations")
    
    # Count movements per group
    min_movements = min(len(seqs) for seqs in grouped.values())
    max_movements = max(len(seqs) for seqs in grouped.values())
    avg_movements = np.mean([len(seqs) for seqs in grouped.values()])
    print(f"Movements per group: min={min_movements}, max={max_movements}, avg={avg_movements:.1f}")
    
    # Test with different amounts
    results = []
    for n_movements in [10, 20, 30]:
        # Sample movements
        sampled_sequences = sample_movements(grouped, n_movements, random_seed=42)
        
        # Train and evaluate
        metrics = train_and_evaluate(sampled_sequences, '000pct', n_movements)
        results.append(metrics)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Movements':<12} {'Test RMSE':<12} {'Test Std':<12} {'Loc Acc':<12} {'KPM1':<8} {'KPM2':<8} {'Overfit (RMSE)':<15}")
    print("-" * 70)
    for r in results:
        overfit = r['force_rmse_test'] - r['force_rmse_train']
        print(f"{r['n_movements']:<12} {r['force_rmse_test']:<12.4f} {r['force_std_test']:<12.4f} "
              f"{r['location_accuracy_test']:<12.4f} {'PASS' if r['kpm1_pass'] else 'FAIL':<8} "
              f"{'PASS' if r['kpm2_pass'] else 'FAIL':<8} {overfit:<15.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Find where performance plateaus
    rmse_improvements = []
    for i in range(1, len(results)):
        rmse_improvement = results[i-1]['force_rmse_test'] - results[i]['force_rmse_test']
        rmse_improvements.append(rmse_improvement)
        print(f"RMSE improvement from {results[i-1]['n_movements']} to {results[i]['n_movements']} movements: {rmse_improvement:.4f} N")
    
    # Check if 30 movements passes KPMs
    final_result = results[-1]
    if final_result['kpm1_pass'] and final_result['kpm2_pass']:
        print(f"\n✅ With {final_result['n_movements']} movements: Both KPM1 and KPM2 PASS")
    else:
        print(f"\n⚠️  With {final_result['n_movements']} movements: Some KPMs FAIL")
        if not final_result['kpm1_pass']:
            print(f"   - KPM1 FAILED (force resolution > 0.05N)")
        if not final_result['kpm2_pass']:
            print(f"   - KPM2 FAILED (RMSE >= 0.1N or STD >= 0.05N)")
    
    # Check overfitting
    for r in results:
        overfit_force = r['force_rmse_test'] - r['force_rmse_train']
        overfit_location = r['location_accuracy_train'] - r['location_accuracy_test']
        if overfit_force > 0.05 or overfit_location > 0.1:
            print(f"\n⚠️  Overfitting detected with {r['n_movements']} movements:")
            print(f"   - Force RMSE gap: {overfit_force:.4f} N")
            print(f"   - Location accuracy gap: {overfit_location:.4f}")


if __name__ == '__main__':
    main()

