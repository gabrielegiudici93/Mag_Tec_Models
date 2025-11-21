#!/usr/bin/env python3
"""
Train force and location models directly from simulation datasets with multiple probe points.
Uses the SAME training functions as train_single_point_models.py for consistency.

Expected HDF5 structure (per stretch file):
    MagneticField   [samples, 15, 3]   raw Bx/By/Bz readings
    forcesTest      [samples, 3]       (optional) Fx/Fy/Fz ground-truth
    IdenterPosition [samples, 3]       probe pose in metres

The script:
  * infers stretch labels from filenames or attributes,
  * derives contact-location labels from the indenter XY coordinates,
  * converts continuous data into sequences (similar to real data),
  * duplicates sequences to match real data count (if needed),
  * trains per-stretch force regressors (if Fz is provided),
  * trains per-stretch location classifiers,
  * trains pooled stretch and location classifiers,
  * writes metrics and trained artefacts to data/Imported/<run_label>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import shutil

import h5py
import joblib
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import DATA_DIR  # noqa: E402

# Import training functions from train_single_point_models
from training.train_single_point_models import (  # noqa: E402
    create_model,
    filter_high_displacement,
    normalize_fz_to_range,
    prepare_training_data,
    train_models_for_stretch,
    train_combined_model,
    balance_sequences,
    remove_outliers,
)


def infer_stretch_label(path: Path, attrs: Dict[str, str]) -> str:
    if "stretch" in attrs:
        try:
            value = float(attrs["stretch"])
            return f"{int(round(value)):03d}pct"
        except (TypeError, ValueError):
            pass

    name = path.stem
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            value = int(digits[-3:])  # assume the last digits encode the stretch %
        except ValueError:
            value = int(digits)
        return f"{value:03d}pct"
    return f"{hash(name) & 0xFFFF:04x}pct"


def position_key(xy: np.ndarray, resolution_mm: float = 0.1) -> Tuple[str, Tuple[float, float]]:
    """
    DETECT CONTACT POSITION: Convert indenter XY coordinate into a discrete position label.
    
    This function is the KEY to automatic position detection:
    1. Takes X,Y coordinates in METRES (e.g., [0.0029, 0.0020])
    2. Converts to MILLIMETRES (e.g., [2.9, 2.0])
    3. ROUNDS to 0.1mm resolution (e.g., 2.9mm stays 2.9mm, 2.87mm becomes 2.9mm)
    4. Creates a label string like "x+02.9mm_y+02.0mm"
    
    Why rounding? Different samples at the "same" position will have slightly different
    coordinates due to simulation noise. Rounding groups them into discrete positions.
    
    Example:
        Input:  [0.00287, 0.00195] metres
        Step 1: [2.87, 1.95] mm
        Step 2: Round to 0.1mm → [2.9, 2.0] mm
        Output: "x+02.9mm_y+02.0mm", (2.9, 2.0)
    
    Args:
        xy: Array with [X, Y] coordinates in metres
        resolution_mm: Rounding resolution (default 0.1mm)
    
    Returns:
        Tuple of (label_string, (x_mm, y_mm))
    """
    # Convert metres to millimetres and round to 0.1mm resolution
    x_mm = round(xy[0] * 1000.0 / resolution_mm) * resolution_mm
    y_mm = round(xy[1] * 1000.0 / resolution_mm) * resolution_mm
    # Create readable label: "x+02.9mm_y+02.0mm"
    key = f"x{float(x_mm):+05.1f}mm_y{float(y_mm):+05.1f}mm"
    return key, (float(x_mm), float(y_mm))


def load_simulation_file(path: Path) -> Dict[str, np.ndarray]:
    """
    READ DATA FROM HDF5 FILE:
    
    Reads three datasets from the simulation HDF5 file:
    
    1. MagneticField [samples, 15, 3]:
       - Raw magnetic field readings from 15 sensors
       - Each sensor has 3 channels: Bx, By, Bz
       - Shape: (N samples, 15 sensors, 3 channels)
       - This becomes our FEATURES (input to ML models)
    
    2. IdenterPosition [samples, 3]:
       - Probe position in metres: [X, Y, Z]
       - Used to DETECT which contact position was pressed
       - We only use X and Y (ignore Z)
       - Example: [0.0029, 0.0020, 0.001] means probe at X=2.9mm, Y=2.0mm
    
    3. forcesTest [samples, 3] (OPTIONAL):
       - Ground-truth forces: [Fx, Fy, Fz]
       - Used as TARGET for force regression models
       - If missing, we skip force training
    
    Returns:
        Dictionary with:
        - "magnetic": Magnetic data [samples, 15, 3]
        - "forces": Force data [samples, 3] or None
        - "indenter": Position data [samples, 3] or None
        - "stretch_label": Inferred stretch level (e.g., "010pct")
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        if "MagneticField" not in f:
            raise RuntimeError(f"{path} missing 'MagneticField' dataset.")

        # Read magnetic field: [samples, 15 sensors, 3 channels]
        magnetic = f["MagneticField"][()].astype(np.float32)
        
        # Read forces (optional): [samples, 3] for Fx, Fy, Fz
        forces = f["forcesTest"][()].astype(np.float32) if "forcesTest" in f else None
        
        # Read indenter position (required for position detection): [samples, 3] for X, Y, Z
        indenter = f["IdenterPosition"][()].astype(np.float32) if "IdenterPosition" in f else None

    samples = magnetic.shape[0]
    stretch_label = infer_stretch_label(path, attrs)
    
    data = {
        "magnetic": magnetic,          # [samples, 15, 3]
        "forces": forces,             # [samples, 3] or None
        "indenter": indenter,          # [samples, 3] or None
        "stretch_label": stretch_label, # e.g., "010pct"
    }
    return data


def convert_to_sequences(
    magnetic: np.ndarray,
    forces: np.ndarray | None,
    indenter: np.ndarray | None,
    position_labels: List[str],
    stretch_label: str,
) -> List[Dict]:
    """
    Convert continuous simulation data into sequences (similar to real data structure).
    
    Groups samples by position label to create sequences.
    Each sequence represents one "press" at a specific position.
    
    Args:
        magnetic: [samples, 15, 3] magnetic field data
        forces: [samples, 3] or None, force data
        indenter: [samples, 3] or None, indenter position
        position_labels: List of position labels (one per sample)
        stretch_label: Stretch level label (e.g., "010pct")
    
    Returns:
        List of sequence dictionaries, each with:
        - 'stretchmagtec': [samples_in_seq, 15, 3]
        - 'forces': [samples_in_seq, 3] or None
        - 'fz': [samples_in_seq] (extracted from forces)
        - 'offset': position label (e.g., "center", "ne", etc.)
        - 'stretch': stretch_label
    """
    sequences = []
    
    # Group samples by position
    position_groups = defaultdict(list)
    for i, pos_label in enumerate(position_labels):
        position_groups[pos_label].append(i)
    
    # Map position labels to offset names based on RELATIVE position to center
    # Find center position (closest to origin or most common)
    center_pos = None
    center_label = None
    min_dist_to_origin = float('inf')
    
    for pos_label, indices in position_groups.items():
        # Extract coordinates from label
        import re
        match = re.match(r'x([+-]?\d+\.?\d*)mm_y([+-]?\d+\.?\d*)mm', pos_label)
        if match:
            x_mm = float(match.group(1))
            y_mm = float(match.group(2))
            dist = np.sqrt(x_mm**2 + y_mm**2)
            if dist < min_dist_to_origin:
                min_dist_to_origin = dist
                center_pos = (x_mm, y_mm)
                center_label = pos_label
    
    # If no center found, use the position with most samples
    if center_pos is None:
        center_label = max(position_groups.items(), key=lambda x: len(x[1]))[0]
        match = re.match(r'x([+-]?\d+\.?\d*)mm_y([+-]?\d+\.?\d*)mm', center_label)
        if match:
            center_pos = (float(match.group(1)), float(match.group(2)))
    
    # Map other positions relative to center
    offset_map = {center_label: "center"}
    
    for pos_label, indices in position_groups.items():
        if pos_label == center_label:
            continue
        
        # Extract coordinates from label
        match = re.match(r'x([+-]?\d+\.?\d*)mm_y([+-]?\d+\.?\d*)mm', pos_label)
        if match:
            x_mm = float(match.group(1))
            y_mm = float(match.group(2))
            
            # Calculate relative position to center
            dx = x_mm - center_pos[0]
            dy = y_mm - center_pos[1]
            
            # Map to offset based on relative position
            # Real data: ne=[-2.5, +5.0] (X=-2.5mm nord, Y=+5.0mm est), nw=[-2.5, -5.0], se=[+2.5, +5.0], sw=[+2.5, -5.0]
            # So: ne has dx<0, dy>0; nw has dx<0, dy<0; se has dx>0, dy>0; sw has dx>0, dy<0
            if dx < 0 and dy > 0:
                offset_map[pos_label] = "ne"  # Northeast (negative X, positive Y)
            elif dx < 0 and dy < 0:
                offset_map[pos_label] = "nw"  # Northwest (negative X, negative Y)
            elif dx > 0 and dy > 0:
                offset_map[pos_label] = "se"  # Southeast (positive X, positive Y)
            elif dx > 0 and dy < 0:
                offset_map[pos_label] = "sw"  # Southwest (positive X, negative Y)
            else:
                # If on axis, use closest match based on dominant direction
                if abs(dx) > abs(dy):
                    offset_map[pos_label] = "se" if dx > 0 else "ne"
                else:
                    offset_map[pos_label] = "ne" if dy > 0 else "nw"
    
    for pos_label, indices in position_groups.items():
        # Extract sequence data
        seq_magnetic = magnetic[indices]  # [samples_in_seq, 15, 3]
        
        seq_forces = None
        seq_fz = None
        if forces is not None:
            seq_forces = forces[indices]  # [samples_in_seq, 3]
            seq_fz = np.abs(seq_forces[:, 2])  # Fz component (use absolute value for training)
        
        # Downsample to match real data sequence length (~81 samples average)
        # Use linear interpolation to downsample if sequence is too long
        target_length = 81  # Average from real data
        if len(seq_magnetic) > target_length:
            # Downsample using linear interpolation
            if len(seq_magnetic) > 1:
                # Resample to target_length
                original_indices = np.linspace(0, len(seq_magnetic) - 1, len(seq_magnetic))
                target_indices = np.linspace(0, len(seq_magnetic) - 1, target_length)
                
                # Interpolate magnetic data
                seq_magnetic_downsampled = np.zeros((target_length, seq_magnetic.shape[1], seq_magnetic.shape[2]))
                for sensor in range(seq_magnetic.shape[1]):
                    for channel in range(seq_magnetic.shape[2]):
                        seq_magnetic_downsampled[:, sensor, channel] = np.interp(
                            target_indices, original_indices, seq_magnetic[:, sensor, channel]
                        )
                seq_magnetic = seq_magnetic_downsampled
                
                # Interpolate forces if available
                if seq_forces is not None:
                    seq_forces_downsampled = np.zeros((target_length, seq_forces.shape[1]))
                    for channel in range(seq_forces.shape[1]):
                        seq_forces_downsampled[:, channel] = np.interp(
                            target_indices, original_indices, seq_forces[:, channel]
                        )
                    seq_forces = seq_forces_downsampled
                    seq_fz = np.abs(seq_forces[:, 2])  # Use absolute value for training
        
        # Filter magnetic sensor: set values below 250 to 0 (noise threshold)
        seq_magnetic = np.where(np.abs(seq_magnetic) < 250, 0, seq_magnetic)
        
        # Map position to offset name
        offset = offset_map.get(pos_label, "unknown")
        
        # Calculate duration (number of samples, assuming ~100Hz sampling rate)
        duration = len(seq_magnetic) / 100.0  # Approximate duration in seconds
        
        # Calculate required fields for outlier removal
        fz_max = float(np.max(seq_fz)) if seq_fz is not None and len(seq_fz) > 0 else 0.0
        num_samples = len(seq_magnetic)
        
        sequence = {
            'stretchmagtec': seq_magnetic,
            'forces': seq_forces,
            'fz': seq_fz if seq_fz is not None else np.zeros(len(seq_magnetic)),
            'offset': offset,
            'stretch': stretch_label,
            'duration': duration,  # Required for outlier removal
            'fz_max': fz_max,      # Required for outlier removal
            'num_samples': num_samples,  # Required for outlier removal
        }
        sequences.append(sequence)
    
    return sequences


def duplicate_sequences(sequences: List[Dict], target_count: int) -> List[Dict]:
    """
    Duplicate sequences to reach target count.
    
    Args:
        sequences: List of sequence dictionaries
        target_count: Target number of sequences
    
    Returns:
        List of sequences (duplicated if needed)
    """
    import copy
    
    if len(sequences) >= target_count:
        return sequences[:target_count]
    
    # Duplicate sequences to reach target count (deep copy to avoid shared references)
    duplicated = []
    while len(duplicated) < target_count:
        # Append sequences in order until we reach target
        remaining = target_count - len(duplicated)
        to_add = min(remaining, len(sequences))
        for seq in sequences[:to_add]:
            # Deep copy the sequence dictionary
            seq_copy = copy.deepcopy(seq)
            duplicated.append(seq_copy)
    
    return duplicated[:target_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train force and location models directly from simulation HDF5 datasets."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Simulation HDF5 files (one per stretch).")
    parser.add_argument("--run-label", type=str, default=None, help="Name for the output directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination root for artefacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory.")
    parser.add_argument("--target-sequences", type=int, default=None, help="Target number of sequences per stretch (duplicate if needed).")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration if available.")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="Z-score threshold for outlier removal.")
    return parser.parse_args()


def main() -> None:
    """
    MAIN WORKFLOW:
    
    1. READ DATA: Load HDF5 files → extract MagneticField, IdenterPosition, forcesTest
    2. DETECT POSITIONS: Round X/Y coordinates to create discrete position labels
    3. CONVERT TO SEQUENCES: Group samples by position to create sequences
    4. DUPLICATE SEQUENCES: If needed, duplicate to match real data count
    5. TRAIN MODELS: Use same functions as train_single_point_models.py
       - Per-stretch force regressors (if forcesTest available)
       - Per-stretch offset classifiers
       - Pooled stretch classifier
       - Pooled offset classifier
    6. SAVE: Models and metrics JSON to data/Imported/<run_label>/
    """
    args = parse_args()
    run_label = args.run_label or f"simulation_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest_root = args.output_dir or (DATA_DIR / "Imported" / run_label)
    dest_root = dest_root.resolve()

    if dest_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination {dest_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    models_dir = dest_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Storage for sequences by stretch level
    sequences_by_stretch: Dict[str, List[Dict]] = defaultdict(list)
    position_map = {}  # Maps position labels to (x_mm, y_mm) coordinates

    # ========================================================================
    # STEP 1: READ ALL HDF5 FILES AND CONVERT TO SEQUENCES
    # ========================================================================
    for file_path in args.inputs:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        data = load_simulation_file(file_path)
        magnetic = data["magnetic"]
        forces = data["forces"]
        indenter = data["indenter"]
        stretch = data["stretch_label"]
        
        if indenter is None:
            raise RuntimeError(f"{file_path} missing 'IdenterPosition'; cannot derive contact labels.")

        # Detect contact positions from indenter coordinates
        position_labels = []
        for vec in indenter:  # vec is [X, Y, Z] in metres
            # Extract only X, Y (ignore Z) and create position label
            label, canonical_xy = position_key(vec[:2])  # vec[:2] = [X, Y]
            position_map[label] = canonical_xy  # Store unique positions
            position_labels.append(label)  # One label per sample

        # Convert to sequences (group by position)
        sequences = convert_to_sequences(magnetic, forces, indenter, position_labels, stretch)
        sequences_by_stretch[stretch].extend(sequences)

        # Copy the original file for reference
        target_copy = dest_root / file_path.name
        target_copy.write_bytes(file_path.read_bytes())

    # ========================================================================
    # STEP 2: DUPLICATE SEQUENCES IF NEEDED
    # ========================================================================
    if args.target_sequences is not None:
        print(f"\n{'='*80}")
        print("DUPLICATING SEQUENCES")
        print(f"{'='*80}")
        for stretch, sequences in sequences_by_stretch.items():
            original_count = len(sequences)
            sequences_by_stretch[stretch] = duplicate_sequences(sequences, args.target_sequences)
            print(f"{stretch}: {original_count} → {len(sequences_by_stretch[stretch])} sequences")

    # ========================================================================
    # STEP 3: REMOVE OUTLIERS AND BALANCE SEQUENCES
    # ========================================================================
    print(f"\n{'='*80}")
    print("REMOVING OUTLIERS AND BALANCING")
    print(f"{'='*80}")
    
    for stretch in list(sequences_by_stretch.keys()):
        sequences = sequences_by_stretch[stretch]
        print(f"\n{stretch}: {len(sequences)} sequences before cleaning")
        
        # Remove outliers (returns tuple: (cleaned_sequences, outlier_indices))
        cleaned_sequences, outlier_indices = remove_outliers(sequences, z_threshold=args.z_threshold)
        print(f"{stretch}: {len(cleaned_sequences)} sequences after outlier removal (removed {len(outlier_indices)} outliers)")
        
        sequences_by_stretch[stretch] = cleaned_sequences
    
    # Balance sequences across stretch levels
    sequences_by_stretch = balance_sequences(sequences_by_stretch)
    print(f"\nAfter balancing:")
    for stretch, sequences in sequences_by_stretch.items():
        print(f"  {stretch}: {len(sequences)} sequences")

    # ========================================================================
    # STEP 4: TRAIN MODELS (using same functions as real data training)
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"{'='*80}")
    
    trained_models = {}
    gpu_mapping = {'000pct': 0, '010pct': 0, '020pct': 1}
    
    # Train per-stretch models
    for stretch_label, sequences in sequences_by_stretch.items():
        if len(sequences) == 0:
            print(f"Skipping {stretch_label}: no sequences")
            continue
        
        gpu_id = gpu_mapping.get(stretch_label, 0)
        result = train_models_for_stretch(
            sequences, 
            stretch_label, 
            train_ratio=0.7, 
            use_gpu=args.use_gpu, 
            gpu_id=gpu_id,
            fz_target_min=0.0,  # Use absolute values: 0 to 3N
            fz_target_max=3.0
        )
        trained_models[stretch_label] = result
        
        # Save models
        model_name = f"{run_label}_{stretch_label}"
        if result.get('force_model') is not None:
            force_path = models_dir / f"{model_name}_force_regressor.joblib"
            joblib.dump(result['force_model'], force_path)
            print(f"Saved: {force_path}")
            
            # Save scaler and fz_scaler
            if result.get('scaler') is not None:
                scaler_path = models_dir / f"{model_name}_scaler.joblib"
                joblib.dump(result['scaler'], scaler_path)
                print(f"Saved: {scaler_path}")
            if result.get('fz_scaler') is not None:
                fz_scaler_path = models_dir / f"{model_name}_fz_scaler.joblib"
                joblib.dump(result['fz_scaler'], fz_scaler_path)
                print(f"Saved: {fz_scaler_path}")
        if result.get('offset_model') is not None:
            offset_path = models_dir / f"{model_name}_offset_classifier.joblib"
            joblib.dump(result['offset_model'], offset_path)
            print(f"Saved: {offset_path}")

    # Train combined model
    if len(sequences_by_stretch) > 1:
        print(f"\n{'='*80}")
        print("TRAINING COMBINED MODEL")
        print(f"{'='*80}")
        combined_result = train_combined_model(
            sequences_by_stretch, 
            train_ratio=0.7, 
            use_gpu=args.use_gpu,
            fz_target_min=0.0,  # Use absolute values: 0 to 3N
            fz_target_max=3.0
        )
        trained_models['combined'] = combined_result
        
        # Save combined models
        model_name = f"{run_label}_combined"
        if combined_result.get('force_model') is not None:
            force_path = models_dir / f"{model_name}_force_regressor.joblib"
            joblib.dump(combined_result['force_model'], force_path)
            print(f"Saved: {force_path}")
            
            # Save scaler and fz_scaler
            if combined_result.get('scaler') is not None:
                scaler_path = models_dir / f"{model_name}_scaler.joblib"
                joblib.dump(combined_result['scaler'], scaler_path)
                print(f"Saved: {scaler_path}")
            if combined_result.get('fz_scaler') is not None:
                fz_scaler_path = models_dir / f"{model_name}_fz_scaler.joblib"
                joblib.dump(combined_result['fz_scaler'], fz_scaler_path)
                print(f"Saved: {fz_scaler_path}")
        if combined_result.get('offset_model') is not None:
            offset_path = models_dir / f"{model_name}_offset_classifier.joblib"
            joblib.dump(combined_result['offset_model'], offset_path)
            print(f"Saved: {offset_path}")
        if combined_result.get('stretch_model') is not None:
            stretch_path = models_dir / f"{model_name}_stretch_classifier.joblib"
            joblib.dump(combined_result['stretch_model'], stretch_path)
            print(f"Saved: {stretch_path}")

    # ========================================================================
    # STEP 5: PRINT TRAINING SUMMARY (identical to physical model)
    # ========================================================================
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print(f"{'Model':<15} {'Sequences':<12} {'Train Seq':<12} {'Test Seq':<12} {'Samples':<12} {'Train Samples':<15} {'Test Samples':<15} {'RMSE':<10} {'Offset Acc':<12} {'Fz Range':<15}")
    print("-"*100)
    
    for model_name, result in trained_models.items():
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

    # ========================================================================
    # STEP 6: PREPARE METRICS JSON (compatible format)
    # ========================================================================
    print(f"\n{'='*80}")
    print("PREPARING METRICS JSON")
    print(f"{'='*80}")
    
    force_results = []
    offset_results = []
    
    for stretch_label in ['000pct', '010pct', '020pct']:
        if stretch_label in trained_models:
            result = trained_models[stretch_label]
            force_results.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'rmse': result['force_rmse'],
                'std_dev': result['force_std_dev'],
                'fz_min_actual': result.get('fz_min_actual', np.nan),
                'fz_max_actual': result.get('fz_max_actual', np.nan),
            })
            offset_results.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'accuracy': result['offset_accuracy'],
            })
    
    # Combined metrics
    combined_force_metrics = {}
    combined_offset_metrics = {}
    combined_stretch_metrics = {}
    if 'combined' in trained_models:
        combined_result = trained_models['combined']
        combined_force_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result.get('n_samples', 0),
            'sequences': combined_result.get('n_test_sequences', combined_result.get('n_sequences', 0)),
            'rmse': combined_result.get('force_rmse', np.nan),
            'std_dev': combined_result.get('force_std_dev', np.nan),
        }
        combined_offset_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result.get('n_samples', 0),
            'sequences': combined_result.get('n_test_sequences', combined_result.get('n_sequences', 0)),
            'accuracy': combined_result.get('offset_accuracy', 0.0),
        }
        combined_stretch_metrics = {
            'stretch_label': 'combined',
            'samples': combined_result.get('n_samples', 0),
            'sequences': combined_result.get('n_test_sequences', combined_result.get('n_sequences', 0)),
            'accuracy': combined_result.get('stretch_accuracy', 0.0),
        }
    
    # Custom JSON encoder
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
    
    metrics_payload = {
        'force_mapping_per_stretch_full': force_results,
        'force_mapping_combined_full': combined_force_metrics,
        'offset_classification_per_stretch_full': offset_results,
        'offset_classification_combined_full': combined_offset_metrics,
        'stretch_classification_combined_full': combined_stretch_metrics,
        'config': {
            'positions_detected': position_map,
            'target_sequences': args.target_sequences,
            'use_gpu': args.use_gpu,
            'z_threshold': args.z_threshold,
        },
    }
    
    metrics_path = dest_root / f"{run_label}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2, default=json_encoder)

    print("Training complete!")
    print(f"Models saved to: {models_dir}")
    print(f"Metrics report saved to: {metrics_path}")


if __name__ == "__main__":
    main()
