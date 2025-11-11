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
    Convert an indenter XY coordinate (metres) into a readable label and canonical tuple.
    The resolution parameter controls how positions are rounded (default 0.1 mm).
    """
    x_mm = round(xy[0] * 1000.0 / resolution_mm) * resolution_mm
    y_mm = round(xy[1] * 1000.0 / resolution_mm) * resolution_mm
    key = f"x{float(x_mm):+05.1f}mm_y{float(y_mm):+05.1f}mm"
    return key, (float(x_mm), float(y_mm))


def load_simulation_file(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        if "MagneticField" not in f:
            raise RuntimeError(f"{path} missing 'MagneticField' dataset.")

        magnetic = f["MagneticField"][()].astype(np.float32)
        forces = f["forcesTest"][()].astype(np.float32) if "forcesTest" in f else None
        indenter = f["IdenterPosition"][()].astype(np.float32) if "IdenterPosition" in f else None

    samples = magnetic.shape[0]
    stretch_label = infer_stretch_label(path, attrs)
    data = {
        "features": magnetic.reshape(samples, -1),
        "forces": forces,
        "indenter": indenter,
        "stretch_label": stretch_label,
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

    all_features = []
    all_stretches = []
    all_positions = []
    all_forces = []
    position_map = {}

    per_stretch_features = defaultdict(list)
    per_stretch_positions = defaultdict(list)
    per_stretch_forces = defaultdict(list)

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

        # derive discrete labels from XY coordinates
        position_labels = []
        for vec in indenter:
            label, canonical_xy = position_key(vec[:2])
            position_map[label] = canonical_xy
            position_labels.append(label)

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

    # Per-stretch training
    for stretch, feature_blocks in per_stretch_features.items():
        X_stretch = np.vstack(feature_blocks)
        y_pos = np.array(per_stretch_positions[stretch])
        stretch_metrics = {}

        unique_positions = np.unique(y_pos)
        if len(unique_positions) > 1:
            clf, clf_metrics = train_classifier(X_stretch, y_pos, stretch, unique_positions.tolist())
            joblib.dump(clf, models_dir / f"{run_label}_{stretch}_position_classifier.joblib")
            stretch_metrics["position_classifier"] = clf_metrics
        else:
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

