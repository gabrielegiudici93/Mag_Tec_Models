#!/usr/bin/env python3
"""
Quick script to calculate KPM1 (force resolution) on regressed/predicted force values.

This calculates the minimum distinguishable force difference in the model's predictions,
rather than the ground truth FT sensor values.

Usage:
    python3 calculate_kpm1_on_predictions.py --model path/to/model.joblib --data path/to/data.h5
    python3 calculate_kpm1_on_predictions.py --metrics-json path/to/metrics.json
"""

import argparse
import json
import numpy as np
import joblib
import h5py
from pathlib import Path


def calculate_force_resolution(forces: np.ndarray, decimals: int = 3) -> float:
    """
    Calculate the minimum force resolution (KPM1) from force values.
    
    Args:
        forces: Array of force values (ground truth or predicted)
        decimals: Number of decimal places to round to
        
    Returns:
        Minimum force resolution (delta F_min) in Newtons
    """
    if len(forces) == 0:
        return float("nan")
    
    # Round to specified decimals and get unique values
    unique_forces = np.unique(np.round(forces, decimals=decimals))
    
    if len(unique_forces) < 2:
        return float("nan")
    
    # Calculate differences between consecutive unique forces
    deltas = np.diff(unique_forces)
    
    # Find minimum non-zero difference
    non_zero_deltas = np.abs(deltas[np.abs(deltas) > 0])
    
    if len(non_zero_deltas) == 0:
        return float("nan")
    
    force_resolution = float(np.min(non_zero_deltas))
    return force_resolution


def calculate_from_model(model_path: Path, data_path: Path):
    """Calculate KPM1 from model predictions on data."""
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading data from: {data_path}")
    with h5py.File(data_path, "r") as hf:
        if "press_summaries/sensors" not in hf:
            raise ValueError(f"{data_path} does not contain press_summaries/sensors")
        
        sensors = hf["press_summaries/sensors"][:]  # (n_press, 15, 3)
        forces_gt = hf["press_summaries/forces"][:, 2]  # Fz component (ground truth)
    
    # Flatten sensor data to feature vector
    X = sensors.reshape(len(sensors), -1)  # (n_press, 45)
    
    # Get model predictions
    print("Generating predictions...")
    forces_pred = model.predict(X)
    
    # Calculate KPM1 on predictions
    resolution_pred = calculate_force_resolution(forces_pred)
    
    # Also calculate on ground truth for comparison
    resolution_gt = calculate_force_resolution(forces_gt)
    
    print("\n" + "="*60)
    print("KPM1 Force Resolution Results")
    print("="*60)
    print(f"Ground truth (FT sensor) resolution: {resolution_gt:.6f} N")
    print(f"Predicted (model) resolution:        {resolution_pred:.6f} N")
    print(f"\nTarget: ≤ 0.05 N")
    print(f"Ground truth KPM1: {'PASS' if resolution_gt <= 0.05 else 'FAIL'}")
    print(f"Predicted KPM1:    {'PASS' if resolution_pred <= 0.05 else 'FAIL'}")
    print("="*60)
    
    return {
        "ground_truth_resolution": float(resolution_gt),
        "predicted_resolution": float(resolution_pred),
        "ground_truth_kpm1_pass": resolution_gt <= 0.05,
        "predicted_kpm1_pass": resolution_pred <= 0.05,
    }


def calculate_from_metrics_json(metrics_path: Path):
    """Extract and display KPM1 info from existing metrics JSON."""
    print(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    print("\n" + "="*60)
    print("KPM1 Force Resolution from Metrics JSON")
    print("="*60)
    
    # Check for per-stretch force metrics
    if "force_mapping_per_stretch_full" in metrics:
        print("\nPer-Stretch Models (Full Sensors):")
        for result in metrics["force_mapping_per_stretch_full"]:
            stretch = result.get("stretch_label", "unknown")
            resolution = result.get("force_resolution_est", "N/A")
            kpm1_pass = result.get("kpm1_pass", None)
            print(f"  {stretch}: ΔF = {resolution:.6f} N, KPM1 = {'PASS' if kpm1_pass else 'FAIL' if kpm1_pass is not None else 'N/A'}")
    
    # Check for combined model
    if "force_mapping_combined_full" in metrics:
        result = metrics["force_mapping_combined_full"]
        resolution = result.get("force_resolution_est", "N/A")
        kpm1_pass = result.get("kpm1_pass", None)
        print(f"\nCombined Model (Full Sensors):")
        print(f"  ΔF = {resolution:.6f} N, KPM1 = {'PASS' if kpm1_pass else 'FAIL' if kpm1_pass is not None else 'N/A'}")
    
    print("\nNote: These values are calculated on ground truth FT sensor data.")
    print("To calculate on predicted values, use --model and --data options.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate KPM1 force resolution on regressed/predicted force values"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained model (joblib file)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to HDF5 data file"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Path to existing metrics JSON file (for comparison)"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places for rounding (default: 3)"
    )
    
    args = parser.parse_args()
    
    if args.metrics_json:
        calculate_from_metrics_json(args.metrics_json)
    elif args.model and args.data:
        results = calculate_from_model(args.model, args.data)
        
        # Optionally save results
        output_path = args.data.parent / "kpm1_predictions.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        parser.print_help()
        print("\nError: Either provide --model and --data, or --metrics-json")


if __name__ == "__main__":
    main()

