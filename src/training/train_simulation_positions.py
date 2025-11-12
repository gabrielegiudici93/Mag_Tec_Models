#!/usr/bin/env python3
"""
Train force and location models directly from simulation datasets with multiple probe points.

Expected HDF5 structure (per stretch file):
    MagneticField   [samples, 15, 3]   raw Bx/By/Bz readings
    forcesTest      [samples, 3]       (optional) Fx/Fy/Fz ground-truth
    IdenterPosition [samples, 3]       probe pose in metres

The script:
  * infers stretch labels from filenames or attributes,
  * derives contact-location labels from the indenter XY coordinates,
  * trains per-stretch force regressors (if Fz is provided),
  * trains per-stretch location classifiers,
  * trains pooled stretch and location classifiers,
  * writes metrics and trained artefacts to data/Imported/<run_label>/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import shutil

import h5py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import DATA_DIR  # noqa: E402


def infer_stretch_label(path: Path, attrs: Dict[str, str]) -> str:
    if "stretch" in attrs:
        try:
            value = float(attrs["stretch"])
            return f"stretch_{int(round(value)):03d}pct"
        except (TypeError, ValueError):
            pass

    name = path.stem
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            value = int(digits[-3:])  # assume the last digits encode the stretch %
        except ValueError:
            value = int(digits)
        return f"stretch_{value:03d}pct"
    return f"stretch_{hash(name) & 0xFFFF:04x}"


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
        - "features": Flattened magnetic data [samples, 45] (15×3 = 45 features)
        - "forces": Force data [samples, 3] or None
        - "indenter": Position data [samples, 3] or None
        - "stretch_label": Inferred stretch level (e.g., "stretch_010pct")
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
    
    # Flatten magnetic data: [samples, 15, 3] → [samples, 45]
    # This creates one feature vector per sample (45 features = 15 sensors × 3 channels)
    data = {
        "features": magnetic.reshape(samples, -1),  # Flatten to [samples, 45]
        "forces": forces,                            # [samples, 3] or None
        "indenter": indenter,                        # [samples, 3] or None
        "stretch_label": stretch_label,              # e.g., "stretch_010pct"
    }
    return data


def train_force_regressor(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    std = float(np.std(y_test - y_pred))
    return model, {"rmse": rmse, "std_dev": std, "samples": int(len(y))}


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    class_names: List[str],
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=class_names).tolist()
    metrics = {
        "label": label,
        "samples": int(len(y)),
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "classes": class_names,
    }
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train force and location models directly from simulation HDF5 datasets."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Simulation HDF5 files (one per stretch).")
    parser.add_argument("--run-label", type=str, default=None, help="Name for the output directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination root for artefacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directory.")
    parser.add_argument("--normalize", action="store_true", help="Use magnitude-normalised features.")
    return parser.parse_args()


def normalise_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def main() -> None:
    """
    MAIN WORKFLOW:
    
    1. READ DATA: Load HDF5 files → extract MagneticField, IdenterPosition, forcesTest
    2. DETECT POSITIONS: Round X/Y coordinates to create discrete position labels
    3. TRAIN MODELS:
       - Per-stretch position classifiers (one per stretch level)
       - Per-stretch force regressors (if forcesTest available)
       - Pooled stretch classifier (predicts stretch from magnetic field)
       - Pooled position classifier (works across all stretches)
    4. SAVE: Models and metrics JSON to data/Imported/<run_label>/
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

    # Storage for all data (across all stretches)
    all_features = []      # Magnetic field features [samples, 45]
    all_stretches = []     # Stretch labels (e.g., "stretch_010pct")
    all_positions = []     # Position labels (e.g., "x+02.9mm_y+02.0mm")
    all_forces = []        # Force values Fz
    position_map = {}      # Maps position labels to (x_mm, y_mm) coordinates

    # Storage per stretch level
    per_stretch_features = defaultdict(list)   # Features grouped by stretch
    per_stretch_positions = defaultdict(list)  # Position labels grouped by stretch
    per_stretch_forces = defaultdict(list)     # Forces grouped by stretch

    # ========================================================================
    # STEP 1: READ ALL HDF5 FILES AND DETECT POSITIONS
    # ========================================================================
    for file_path in args.inputs:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        data = load_simulation_file(file_path)
        features = data["features"]
        if args.normalize:
            features = normalise_features(features)

        stretch = data["stretch_label"]
        indenter = data["indenter"]
        if indenter is None:
            raise RuntimeError(f"{file_path} missing 'IdenterPosition'; cannot derive contact labels.")

        # ========================================================================
        # DETECT CONTACT POSITIONS FROM INDENTER COORDINATES
        # ========================================================================
        # For each sample, we look at where the indenter was (X, Y coordinates)
        # and round them to create discrete position labels.
        #
        # Example workflow:
        #   Sample 1: indenter = [0.0000, 0.0000] → "x+00.0mm_y+00.0mm" (center)
        #   Sample 2: indenter = [0.0029, 0.0020] → "x+02.9mm_y+02.0mm" (NE offset)
        #   Sample 3: indenter = [0.0029, -0.0020] → "x+02.9mm_y-02.0mm" (SE offset)
        #   Sample 4: indenter = [0.00287, 0.00195] → "x+02.9mm_y+02.0mm" (same as Sample 2!)
        #
        # By rounding, samples at "similar" positions get the SAME label.
        # This automatically groups samples into discrete contact positions.
        # ========================================================================
        position_labels = []
        for vec in indenter:  # vec is [X, Y, Z] in metres
            # Extract only X, Y (ignore Z) and create position label
            label, canonical_xy = position_key(vec[:2])  # vec[:2] = [X, Y]
            position_map[label] = canonical_xy  # Store unique positions
            position_labels.append(label)  # One label per sample

        forces = data["forces"]
        if forces is not None:
            fz = forces[:, 2]
        else:
            fz = None

        all_features.append(features)
        all_stretches.extend([stretch] * len(features))
        all_positions.extend(position_labels)
        if fz is not None:
            all_forces.append(fz)

        per_stretch_features[stretch].append(features)
        per_stretch_positions[stretch].extend(position_labels)
        if fz is not None:
            per_stretch_forces[stretch].append(fz)

        # Copy the original file for reference
        target_copy = dest_root / file_path.name
        target_copy.write_bytes(file_path.read_bytes())

    X_all = np.vstack(all_features)
    stretch_array = np.array(all_stretches)
    position_array = np.array(all_positions)
    force_array = np.concatenate(all_forces) if all_forces else None

    metrics_payload = {
        "config": {
            "normalized_features": bool(args.normalize),
            "positions_detected": position_map,
        },
        "per_stretch": {},
        "pooled": {},
        "notes": [],
    }

    # ========================================================================
    # TRAIN PER-STRETCH MODELS
    # ========================================================================
    # For each stretch level (0%, 10%, 20%), train separate models:
    # 1. Position classifier: Predicts which position was pressed (center, NE, SE, etc.)
    # 2. Force regressor: Predicts applied force Fz
    # ========================================================================
    for stretch, feature_blocks in per_stretch_features.items():
        X_stretch = np.vstack(feature_blocks)  # All magnetic features for this stretch
        y_pos = np.array(per_stretch_positions[stretch])  # Position labels for this stretch
        stretch_metrics = {}

        # Check how many unique positions we detected for this stretch
        unique_positions = np.unique(y_pos)
        if len(unique_positions) > 1:
            # Train position classifier: magnetic features → position label
            # Example: If we have 5 positions (center + 4 offsets), this learns to
            # distinguish between them using the magnetic field patterns
            clf, clf_metrics = train_classifier(X_stretch, y_pos, stretch, unique_positions.tolist())
            joblib.dump(clf, models_dir / f"{run_label}_{stretch}_position_classifier.joblib")
            stretch_metrics["position_classifier"] = clf_metrics
        else:
            # Only one position detected → can't train a classifier
            metrics_payload["notes"].append(
                f"Stretch {stretch}: only one contact position; skipping location classifier."
            )

        if stretch in per_stretch_forces:
            y_force = np.concatenate(per_stretch_forces[stretch])
            if len(np.unique(y_force)) > 1:
                reg, reg_metrics = train_force_regressor(X_stretch, y_force)
                joblib.dump(reg, models_dir / f"{run_label}_{stretch}_force_regressor.joblib")
                stretch_metrics["force_regressor"] = reg_metrics
            else:
                metrics_payload["notes"].append(
                    f"Stretch {stretch}: constant Fz signal; skipping force regressor."
                )

        metrics_payload["per_stretch"][stretch] = stretch_metrics

    # Pooled stretch classifier
    if len(np.unique(stretch_array)) > 1:
        stretch_clf, stretch_metrics = train_classifier(
            X_all, stretch_array, "stretch_classifier", np.unique(stretch_array).tolist()
        )
        joblib.dump(stretch_clf, models_dir / f"{run_label}_stretch_classifier.joblib")
        metrics_payload["pooled"]["stretch_classifier"] = stretch_metrics

    # Pooled position classifier (using stretch + position)
    if len(np.unique(position_array)) > 1:
        pooled_clf, pooled_metrics = train_classifier(
            X_all, position_array, "pooled_position_classifier", np.unique(position_array).tolist()
        )
        joblib.dump(pooled_clf, models_dir / f"{run_label}_position_classifier.joblib")
        metrics_payload["pooled"]["position_classifier"] = pooled_metrics
    else:
        metrics_payload["notes"].append("Only one unique contact position detected overall; skipping pooled classifier.")

    # Force regressor on pooled data
    if force_array is not None and len(np.unique(force_array)) > 1:
        pooled_reg, pooled_reg_metrics = train_force_regressor(X_all, force_array)
        joblib.dump(pooled_reg, models_dir / f"{run_label}_force_regressor.joblib")
        metrics_payload["pooled"]["force_regressor"] = pooled_reg_metrics
    else:
        metrics_payload["notes"].append("Not enough force variation in pooled data; skipping pooled force regressor.")

    # Persist metrics
    metrics_path = dest_root / f"{run_label}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(f"Training complete. Metrics saved to {metrics_path}")
    print(f"Models stored in {models_dir}")


if __name__ == "__main__":
    main()














