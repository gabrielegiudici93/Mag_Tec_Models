#!/usr/bin/env python3
"""Train baseline models directly from simulation HDF5 files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import h5py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import DATA_DIR  # noqa: E402


def infer_stretch_from_name(path: Path) -> str:
    name = path.stem
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        value = int(digits)
        return f"stretch_{value:03d}pct"
    return f"stretch_{hash(name) & 0xffff:04x}"


def load_simulation_file(path: Path, stretch_label: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "MagneticField" not in f:
            raise RuntimeError(f"{path} is missing 'MagneticField' dataset")
        magnetic = f["MagneticField"][()].astype(float)  # (samples, 15, 3)
        samples = magnetic.shape[0]
        forces = None
        if "forcesTest" in f:
            forces = f["forcesTest"][()].astype(float)

    features = magnetic.reshape(samples, -1)
    targets = forces[:, 2] if forces is not None and forces.shape[1] >= 3 else None

    return {
        "features": features,
        "fz": targets,
        "stretch_label": stretch_label,
        "samples": samples,
    }


def train_force_model(X: np.ndarray, y: np.ndarray, label: str) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    residuals = y_test - y_pred
    std = float(np.std(residuals))
    metrics = {
        "stretch_label": label,
        "samples": int(len(y)),
        "rmse": rmse,
        "std_dev": std,
    }
    return metrics, model


def train_stretch_model(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)
    metrics = {
        "samples": int(len(y)),
        "accuracy": accuracy,
        "report": report,
    }
    return metrics, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models directly from simulation HDF5 files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Simulation HDF5 files")
    parser.add_argument("--run-label", type=str, default=None, help="Name for the output run")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination folder for artefacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_label = args.run_label or f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest_root = args.output_dir or (DATA_DIR / "Imported" / run_label)
    dest_root = dest_root.resolve()

    if dest_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination {dest_root} exists. Use --overwrite to replace it.")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    stretches = []
    features_by_label: Dict[str, np.ndarray] = {}
    fz_by_label: Dict[str, np.ndarray] = {}

    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        stretch_label = infer_stretch_from_name(path)
        data = load_simulation_file(path, stretch_label)
        features_by_label[stretch_label] = data["features"]
        if data["fz"] is not None:
            fz_by_label[stretch_label] = data["fz"]
        stretches.append(stretch_label)

    metrics_payload = {
        "force_mapping_per_stretch": [],
        "stretch_classification": None,
        "notes": [],
    }

    models_dir = dest_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if fz_by_label:
        for label, X in features_by_label.items():
            if label not in fz_by_label:
                metrics_payload["notes"].append(f"No force targets for {label}; skipping force regression.")
                continue
            y = fz_by_label[label]
            if len(np.unique(y)) < 2:
                metrics_payload["notes"].append(f"Constant force signal for {label}; skipping.")
                continue
            metrics, model = train_force_model(X, y, label)
            joblib.dump(model, models_dir / f"{run_label}_{label}_force_regressor.joblib")
            metrics_payload["force_mapping_per_stretch"].append(metrics)
    else:
        metrics_payload["notes"].append("forcesTest dataset not available; force regression skipped.")

    if len(features_by_label) >= 2:
        X_all = np.vstack(list(features_by_label.values()))
        stretch_labels = []
        for label, X in features_by_label.items():
            stretch_labels.extend([label] * len(X))
        stretch_labels = np.array(stretch_labels)
        stretch_metrics, stretch_model = train_stretch_model(X_all, stretch_labels)
        joblib.dump(stretch_model, models_dir / f"{run_label}_stretch_classifier.joblib")
        metrics_payload["stretch_classification"] = stretch_metrics
    else:
        metrics_payload["notes"].append("Need at least two stretch files to train a stretch classifier.")

    metrics_path = dest_root / f"{run_label}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(f"Metrics written to {metrics_path}")
    print(f"Models stored in {models_dir}")


if __name__ == "__main__":
    main()

+import shutil
+
 if __name__ == "__main__":
     main()
