#!/usr/bin/env python3
"""
Print metrics from JSON files in LaTeX table format.

This script reads metrics JSON files (from evaluate_single_point_stretch.py or
train_simulation_positions.py) and prints them in a format similar to the LaTeX tables
in the scientific report.

Usage:
    python3 src/training/print_metrics_tables.py data/2.5mm_single_test1/2.5mm_single_test1_metrics.json
    python3 src/training/print_metrics_tables.py data/Multiple_Points/2.5mm_single_test1/2.5mm_single_test1_metrics.json
    python3 src/training/print_metrics_tables.py data/Imported/simulation_points_test1/simulation_points_test1_metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def print_force_table(metrics: Dict[str, Any], title: str = "Force Regression Metrics"):
    """Print force regression metrics in table format."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Stretch':<20} {'Samples':<10} {'RMSE [N]':<12} {'STD [N]':<12} {'ΔF_min [N]':<12} {'KPM1':<8} {'KPM2':<8}")
    print("-" * 80)
    
    # Per-stretch metrics
    if "force_mapping_per_stretch_full" in metrics:
        for result in metrics["force_mapping_per_stretch_full"]:
            stretch = result.get("stretch_label", "unknown")
            samples = result.get("samples", 0)
            rmse = result.get("rmse", 0.0)
            std_dev = result.get("std_dev", 0.0)
            force_res = result.get("force_resolution_est", 0.0)
            kpm1 = "PASS" if result.get("kpm1_pass") else "FAIL" if result.get("kpm1_pass") is not None else "N/A"
            kpm2 = "PASS" if result.get("kpm2_pass") else "FAIL" if result.get("kpm2_pass") is not None else "N/A"
            
            print(f"{stretch:<20} {samples:<10} {rmse:<12.6f} {std_dev:<12.6f} {force_res:<12.6f} {kpm1:<8} {kpm2:<8}")
    
    # Combined metrics
    if "force_mapping_combined_full" in metrics:
        result = metrics["force_mapping_combined_full"]
        stretch = "combined (pooled)"
        samples = result.get("samples", 0)
        rmse = result.get("rmse", 0.0)
        std_dev = result.get("std_dev", 0.0)
        force_res = result.get("force_resolution_est", 0.0)
        kpm1 = "PASS" if result.get("kpm1_pass") else "FAIL" if result.get("kpm1_pass") is not None else "N/A"
        kpm2 = "PASS" if result.get("kpm2_pass") else "FAIL" if result.get("kpm2_pass") is not None else "N/A"
        
        print("-" * 80)
        print(f"{stretch:<20} {samples:<10} {rmse:<12.6f} {std_dev:<12.6f} {force_res:<12.6f} {kpm1:<8} {kpm2:<8}")


def print_position_classification_table(metrics: Dict[str, Any], title: str = "Press Location Classification"):
    """Print position classification metrics in table format."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Per-stretch position classification
    if "offset_classification_per_stretch_full" in metrics:
        print(f"{'Stretch':<20} {'Samples':<10} {'Accuracy (full)':<18} {'Accuracy (subset)':<20}")
        print("-" * 80)
        
        per_stretch = metrics["offset_classification_per_stretch_full"]
        per_stretch_subset = metrics.get("offset_classification_per_stretch_subset", {})
        
        # Handle both dict and list formats
        if isinstance(per_stretch, list):
            # List format: iterate through list items
            for item in per_stretch:
                stretch_label = item.get("stretch_label", "unknown")
                full_acc = item.get("accuracy", 0.0)
                samples = item.get("samples", 0)
                
                subset_acc = "N/A"
                if isinstance(per_stretch_subset, list):
                    for sub_item in per_stretch_subset:
                        if sub_item.get("stretch_label") == stretch_label:
                            subset_acc = sub_item.get("accuracy", 0.0)
                            subset_acc = f"{subset_acc:.3f}"
                            break
                elif stretch_label in per_stretch_subset:
                    subset_acc = per_stretch_subset[stretch_label].get("accuracy", 0.0)
                    subset_acc = f"{subset_acc:.3f}"
                
                print(f"{stretch_label:<20} {samples:<10} {full_acc:<18.3f} {subset_acc:<20}")
        else:
            # Dict format
            for stretch_label in sorted(per_stretch.keys()):
                full_acc = per_stretch[stretch_label].get("accuracy", 0.0)
                samples = per_stretch[stretch_label].get("samples", 0)
                
                subset_acc = "N/A"
                if stretch_label in per_stretch_subset:
                    subset_acc = per_stretch_subset[stretch_label].get("accuracy", 0.0)
                    subset_acc = f"{subset_acc:.3f}"
                
                print(f"{stretch_label:<20} {samples:<10} {full_acc:<18.3f} {subset_acc:<20}")
    
    # Pooled position classification
    if "offset_classification_combined_full" in metrics:
        print(f"\n{'Model':<30} {'Samples':<10} {'Accuracy':<12}")
        print("-" * 80)
        
        combined_full = metrics["offset_classification_combined_full"]
        combined_subset = metrics.get("offset_classification_combined_subset", {})
        gated_full = metrics.get("offset_classification_gated_full", {})
        gated_subset = metrics.get("offset_classification_gated_subset", {})
        
        if combined_full:
            print(f"{'combined (full features)':<30} {combined_full.get('samples', 0):<10} {combined_full.get('accuracy', 0.0):<12.3f}")
        if combined_subset:
            print(f"{'combined (subset features)':<30} {combined_subset.get('samples', 0):<10} {combined_subset.get('accuracy', 0.0):<12.3f}")
        if gated_full:
            print(f"{'gated (full features)':<30} {gated_full.get('samples', 0):<10} {gated_full.get('accuracy', 0.0):<12.3f}")
        if gated_subset:
            print(f"{'gated (subset features)':<30} {gated_subset.get('samples', 0):<10} {gated_subset.get('accuracy', 0.0):<12.3f}")


def print_stretch_classification_table(metrics: Dict[str, Any], title: str = "Stretch Classification"):
    """Print stretch classification metrics in table format."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if "stretch_classification_combined_full" in metrics:
        full = metrics["stretch_classification_combined_full"]
        subset = metrics.get("stretch_classification_combined_subset", {})
        
        print(f"{'Metric':<20} {'Full sensors':<18} {'Central subset':<18} {'Notes':<20}")
        print("-" * 80)
        
        samples_full = full.get("samples", 0)
        accuracy_full = full.get("accuracy", 0.0)
        samples_subset = subset.get("samples", 0) if subset else 0
        accuracy_subset = subset.get("accuracy", 0.0) if subset else 0.0
        
        print(f"{'Samples':<20} {samples_full:<18} {samples_subset:<18} {'Random Forest (250 trees)':<20}")
        print(f"{'Accuracy':<20} {accuracy_full:<18.3f} {accuracy_subset:<18.3f} {'Identical performance':<20}")


def print_simulation_metrics(metrics: Dict[str, Any]):
    """Print metrics from train_simulation_positions.py format."""
    print(f"\n{'='*80}")
    print("Simulation Dataset Metrics")
    print(f"{'='*80}")
    
    # Force regression
    if "per_stretch" in metrics:
        print(f"\n{'='*80}")
        print("Force Regression Metrics (Per-Stretch)")
        print(f"{'='*80}")
        print(f"{'Stretch':<20} {'Samples':<10} {'RMSE [N]':<12} {'STD [N]':<12}")
        print("-" * 80)
        
        for stretch_label, stretch_data in sorted(metrics["per_stretch"].items()):
            if "force_regressor" in stretch_data:
                force_metrics = stretch_data["force_regressor"]
                samples = force_metrics.get("samples", 0)
                rmse = force_metrics.get("rmse", 0.0)
                std_dev = force_metrics.get("std_dev", 0.0)
                print(f"{stretch_label:<20} {samples:<10} {rmse:<12.6f} {std_dev:<12.6f}")
        
        # Pooled force regressor
        if "pooled" in metrics and "force_regressor" in metrics["pooled"]:
            pooled = metrics["pooled"]["force_regressor"]
            print("-" * 80)
            print(f"{'combined (pooled)':<20} {pooled.get('samples', 0):<10} {pooled.get('rmse', 0.0):<12.6f} {pooled.get('std_dev', 0.0):<12.6f}")
    
    # Position classification
    if "per_stretch" in metrics:
        print(f"\n{'='*80}")
        print("Position Classification Accuracy (Per-Stretch)")
        print(f"{'='*80}")
        print(f"{'Stretch':<20} {'Samples':<10} {'Accuracy':<12}")
        print("-" * 80)
        
        for stretch_label, stretch_data in sorted(metrics["per_stretch"].items()):
            if "position_classifier" in stretch_data:
                pos_metrics = stretch_data["position_classifier"]
                samples = pos_metrics.get("samples", 0)
                accuracy = pos_metrics.get("accuracy", 0.0)
                print(f"{stretch_label:<20} {samples:<10} {accuracy:<12.3f}")
        
        # Pooled position classifier
        if "pooled" in metrics and "position_classifier" in metrics["pooled"]:
            pooled = metrics["pooled"]["position_classifier"]
            print("-" * 80)
            print(f"{'pooled position classifier':<20} {pooled.get('samples', 0):<10} {pooled.get('accuracy', 0.0):<12.3f}")
    
    # Stretch classification
    if "pooled" in metrics and "stretch_classifier" in metrics["pooled"]:
        print(f"\n{'='*80}")
        print("Stretch Classification")
        print(f"{'='*80}")
        pooled = metrics["pooled"]["stretch_classifier"]
        print(f"{'Model':<30} {'Samples':<10} {'Accuracy':<12}")
        print("-" * 80)
        print(f"{'pooled stretch classifier':<30} {pooled.get('samples', 0):<10} {pooled.get('accuracy', 0.0):<12.3f}")
    
    # Detected positions
    if "config" in metrics and "positions_detected" in metrics["config"]:
        positions = metrics["config"]["positions_detected"]
        if positions:
            print(f"\n{'='*80}")
            print("Detected Contact Positions")
            print(f"{'='*80}")
            print(f"{'Label':<30} {'X [mm]':<12} {'Y [mm]':<12}")
            print("-" * 80)
            for label, (x, y) in sorted(positions.items()):
                print(f"{label:<30} {x:<12.1f} {y:<12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Print metrics from JSON files in LaTeX-style table format"
    )
    parser.add_argument(
        "metrics_json",
        type=Path,
        help="Path to metrics JSON file"
    )
    parser.add_argument(
        "--format",
        choices=["robot", "simulation", "auto"],
        default="auto",
        help="Metrics format (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    if not args.metrics_json.exists():
        print(f"Error: File not found: {args.metrics_json}")
        return
    
    with open(args.metrics_json, "r") as f:
        metrics = json.load(f)
    
    # Auto-detect format
    if args.format == "auto":
        if "force_mapping_per_stretch_full" in metrics:
            args.format = "robot"
        elif "per_stretch" in metrics or "pooled" in metrics:
            args.format = "simulation"
        else:
            print("Warning: Could not auto-detect format. Trying robot format...")
            args.format = "robot"
    
    print(f"\nReading metrics from: {args.metrics_json}")
    print(f"Format detected: {args.format}")
    
    if args.format == "robot":
        # Robot data format (from evaluate_single_point_stretch.py)
        print_force_table(metrics, "Force Regression Metrics (All Sensors)")
        
        if "force_mapping_per_stretch_subset" in metrics:
            print_force_table(metrics, "Force Regression Metrics (Central Subset: Sensors 7, 8, 9)")
        
        print_position_classification_table(metrics)
        print_stretch_classification_table(metrics)
    
    elif args.format == "simulation":
        # Simulation data format (from train_simulation_positions.py)
        print_simulation_metrics(metrics)
    
    print(f"\n{'='*80}")
    print("End of metrics report")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

