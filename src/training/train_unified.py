#!/usr/bin/env python3
"""
Unified training script for both real and simulation data.

This script automatically detects whether the input data is from real robot experiments
or simulation, and applies the same training pipeline, preprocessing, and output structure
to ensure consistency across all datasets.

Usage:
    # Real data (from cleaned directory)
    python3 src/training/train_unified.py \
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
        --run-label test50
    
    # Simulation data
    python3 src/training/train_unified.py \
        --h5-files data/simulation/test5/*.h5 \
        --run-label simulation_test5
    
    # With feature method selection
    python3 src/training/train_unified.py \
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
        --run-label test50 \
        --feature-method raw

Features:
    - Automatic detection of data type (real vs simulation)
    - Same preprocessing pipeline for both types
    - Same output structure (cleaned/{feature_method}/models/, plots/, data/)
    - Same plots generated (confusion matrices, scatter, residuals, raw data)
    - Same JSON metrics format
    - Consistent model naming and organization
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
from sklearn.metrics import ConfusionMatrixDisplay, r2_score

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import training functions
from training.train_single_point_models import (
    balance_sequences,
    load_sequences_from_h5,
    prepare_training_data,
    remove_outliers,
    train_combined_model,
    train_models_for_stretch,
    FT_SENSOR_PRECISION,
    FT_SENSOR_DECIMALS,
)

# Import simulation functions
from training.train_simulation_positions import (
    convert_to_sequences,
    infer_stretch_label,
    load_simulation_file,
    position_key,
)


def detect_data_type(h5_path: Path) -> str:
    """
    Detect if HDF5 file contains real or simulation data.
    
    Real data structure:
        - Has 'presses/' group or 'forces' and 'stretchmagtec' datasets
        - Has 'timestamps' dataset
    
    Simulation data structure:
        - Has 'MagneticField' dataset
        - Has 'IdenterPosition' dataset
        - May have 'forcesTest' dataset
    
    Returns:
        'real' or 'simulation'
    """
    with h5py.File(h5_path, 'r') as f:
        # Check for simulation markers
        if 'MagneticField' in f and 'IdenterPosition' in f:
            return 'simulation'
        
        # Check for real data markers
        if 'presses' in f or ('forces' in f and 'stretchmagtec' in f):
            return 'real'
        
        # If neither, try to infer from attributes
        if 'source' in f.attrs:
            source = str(f.attrs['source']).lower()
            if 'sim' in source:
                return 'simulation'
            elif 'real' in source or 'robot' in source:
                return 'real'
    
    # Default: assume real if unclear
    return 'real'


def load_data_unified(
    h5_files: Dict[str, Path],  # Changed: now expects dict mapping stretch to file path
    is_simulation: bool,
    remove_outliers_flag: bool = False,
    z_threshold: float = 3.0,
) -> Dict[str, List[Dict]]:
    """
    Load sequences from HDF5 files, handling both real and simulation data.
    
    Args:
        h5_files: Dictionary mapping stretch labels to HDF5 file paths
        is_simulation: True if simulation data, False if real data
        remove_outliers_flag: Whether to remove outliers (default: False for simulation, True for real)
        z_threshold: Z-score threshold for outlier detection
    
    Returns:
        Dictionary mapping stretch labels to lists of sequences
    """
    sequences_by_stretch = {}
    
    if is_simulation:
        # Load simulation data
        print(f"\n{'='*80}")
        print("LOADING SIMULATION DATA")
        print(f"{'='*80}")
        
        # For simulation, iterate over files
        for stretch_label, h5_path in h5_files.items():
            print(f"\nLoading: {h5_path.name}")
            
            # Load simulation file
            sim_data = load_simulation_file(h5_path)
            stretch_label = sim_data.get('stretch_label', 'unknown')
            
            # Convert to sequences (automatic cycle segmentation)
            if sim_data.get('indenter') is not None:
                indenter = sim_data['indenter']
                # Create position labels from XY coordinates
                position_labels = []
                for i in range(len(indenter)):
                    key, (x_mm, y_mm) = position_key(indenter[i, :2])
                    position_labels.append(key)
            else:
                position_labels = ["x+00.0mm_y+00.0mm"] * len(sim_data['magnetic'])
            
            sequences = convert_to_sequences(
                magnetic=sim_data['magnetic'],
                forces=sim_data['forces'],
                indenter=sim_data.get('indenter'),
                position_labels=position_labels,
                stretch_label=stretch_label,
            )
            
            print(f"  Converted to {len(sequences)} sequences")
            
            # Store sequences by stretch
            if stretch_label not in sequences_by_stretch:
                sequences_by_stretch[stretch_label] = []
            sequences_by_stretch[stretch_label].extend(sequences)
        
        # Remove outliers if requested
        if remove_outliers_flag:
            print(f"\n{'='*80}")
            print("REMOVING OUTLIERS")
            print(f"{'='*80}")
            for stretch in list(sequences_by_stretch.keys()):
                sequences = sequences_by_stretch[stretch]
                print(f"\n{stretch}: {len(sequences)} sequences before outlier removal")
                cleaned_sequences, outlier_indices = remove_outliers(
                    sequences, z_threshold=z_threshold, remove_per_offset=2
                )
                print(f"{stretch}: {len(cleaned_sequences)} sequences after outlier removal")
                sequences_by_stretch[stretch] = cleaned_sequences
        else:
            print(f"\n{'='*80}")
            print("SKIPPING OUTLIER REMOVAL (using all sequences)")
            print(f"{'='*80}")
    
    else:
        # Load real data - EXACT SAME LOGIC AS train_single_point_models.py
        print(f"\n{'='*80}")
        print("LOADING REAL DATA")
        print(f"{'='*80}")
        
        # Load sequences from each file (same as train_single_point_models.py)
        for stretch, h5_file in h5_files.items():
            print(f"\nLoading {stretch} from {h5_file.name}...")
            sequences = load_sequences_from_h5(h5_file)
            print(f"  Loaded {len(sequences)} sequences")
            
            if sequences:
                # Check if this file is already cleaned (has _cleaned.h5 suffix or is in cleaned/ directory)
                file_is_cleaned = "_cleaned.h5" in h5_file.name or "cleaned" in str(h5_file.parent)
                
                if file_is_cleaned:
                    print(f"  ✓ File is already cleaned (detected _cleaned.h5 suffix or cleaned/ directory)")
                    print(f"  ✓ Using sequences as-is (no additional cleaning)")
                    # Count sequences per offset for info
                    offset_counts = {}
                    for seq in sequences:
                        offset = seq.get('offset', 'unknown')
                        offset_counts[offset] = offset_counts.get(offset, 0) + 1
                    print(f"  Sequences per offset: {offset_counts}")
                    sequences_by_stretch[stretch] = sequences
                else:
                    # File is NOT cleaned - apply cleaning steps
                    # NOTE: First sequence per offset is already removed during data collection (franka_skin_test.py)
                    # So we only need to remove outliers here
                    # Count sequences per offset before any removal
                    initial_offset_counts = {}
                    for seq in sequences:
                        offset = seq.get('offset', 'unknown')
                        initial_offset_counts[offset] = initial_offset_counts.get(offset, 0) + 1
                    print(f"  Initial sequences per offset: {initial_offset_counts}")
                    print(f"  Total initial sequences: {len(sequences)}")
                    print(f"  NOTE: First sequence per offset was already removed during data collection")
                    
                    # Remove outliers (2 per offset, independently)
                    # Calculate expected number of offsets for the message
                    n_offsets = len(initial_offset_counts)
                    print(f"  Removing outliers (2 per offset, independently)...")
                    print(f"  Expected after outlier removal: {len(sequences)} - ({n_offsets} offsets * 2 outliers) = {len(sequences) - (n_offsets * 2)} sequences")
                    cleaned_sequences, outlier_indices = remove_outliers(sequences, z_threshold=z_threshold, remove_per_offset=2)
                    print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
                    
                    # Count sequences per offset to verify
                    offset_counts = {}
                    for seq in cleaned_sequences:
                        offset = seq.get('offset', 'unknown')
                        offset_counts[offset] = offset_counts.get(offset, 0) + 1
                    print(f"  Sequences per offset after cleaning: {offset_counts}")
                    print(f"  Expected per offset: ~{len(sequences_after_first_removal) // 5 - 2} sequences")
                    sequences_by_stretch[stretch] = cleaned_sequences
    
    return sequences_by_stretch


def generate_plots_unified(
    trained_models: Dict,
    sequences_by_stretch: Dict[str, List[Dict]],
    plots_dir: Path,
    feature_method: str,
    is_simulation: bool,
    h5_files: Optional[Dict[str, Path]] = None,
):
    """
    Generate all plots (confusion matrices, scatter, residuals, raw data).
    
    Args:
        trained_models: Dictionary of trained models
        sequences_by_stretch: Sequences grouped by stretch
        plots_dir: Directory to save plots
        feature_method: Feature method used ('raw', 'normalized', 'raw_labelled')
        is_simulation: True if simulation data
        h5_files: Dictionary mapping stretch labels to HDF5 file paths (for raw data plots)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # For simulation data, create a "sim" subdirectory and add "_sim" suffix to filenames
    if is_simulation:
        plots_dir = plots_dir / "sim"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_suffix = "_sim"
    else:
        plot_suffix = ""
    
    # 1. Confusion matrices
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRICES")
    print(f"{'='*80}")
    
    offset_names = ['center', 'ne', 'nw', 'se', 'sw']
    confusion_plots_saved = []
    
    for model_name, result in trained_models.items():
        # Offset confusion matrix
        if 'offset_confusion_matrix' in result and result['offset_confusion_matrix'] is not None:
            cm = result['offset_confusion_matrix']
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=offset_names)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(
                f'Offset Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_offset_{model_name}_{feature_method}{plot_suffix}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {confusion_plot_path.name}")
            confusion_plots_saved.append(confusion_plot_path)
        
        # Stretch confusion matrix
        if 'stretch_confusion_matrix' in result and result['stretch_confusion_matrix'] is not None:
            cm = result['stretch_confusion_matrix']
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            # Dynamically determine stretch labels from the confusion matrix size
            # The confusion matrix is created in train_combined_model with:
            #   stretch_map = {label: i for i, label in enumerate(sorted(set(all_stretches)))}
            # So the labels are sorted alphabetically
            n_stretches = cm_normalized.shape[0]
            
            # Get actual stretch labels from sequences_by_stretch keys (sorted to match train_combined_model)
            available_stretches = sorted([s for s in sequences_by_stretch.keys() if s != 'combined'])
            
            # Use available stretches if they match the matrix size exactly
            if len(available_stretches) == n_stretches:
                stretch_labels = available_stretches
            elif len(available_stretches) > n_stretches:
                # Take first n_stretches (shouldn't happen, but handle it)
                stretch_labels = available_stretches[:n_stretches]
            else:
                # Matrix has more classes than available stretches (shouldn't happen, but handle it)
                # Use available stretches and pad with generic labels
                stretch_labels = available_stretches.copy()
                while len(stretch_labels) < n_stretches:
                    stretch_labels.append(f"{len(stretch_labels)*10:03d}pct")
            
            # Final safety check: ensure we have exactly n_stretches labels
            if len(stretch_labels) != n_stretches:
                # Last resort: generate labels based on matrix size
                stretch_labels = [f"{i*10:03d}pct" for i in range(n_stretches)]
            
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=stretch_labels)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(
                f'Stretch Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_stretch_{model_name}_{feature_method}{plot_suffix}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {confusion_plot_path.name}")
            confusion_plots_saved.append(confusion_plot_path)
    
    # 2. Prediction scatter and residual plots
    print(f"\n{'='*80}")
    print("GENERATING PREDICTION PLOTS")
    print(f"{'='*80}")
    
    prediction_plots_saved = []
    
    for model_name, result in trained_models.items():
        if 'y_fz_test' in result and 'y_fz_pred' in result:
            y_test = result['y_fz_test']
            y_pred = result['y_fz_pred']
            
            if len(y_test) > 0 and len(y_pred) > 0:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.scatter(y_test, y_pred, alpha=0.5, s=20)
                
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
                
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Predicted Fz (N)', fontsize=12)
                ax.set_title(
                    f'Force Prediction - {model_name}\nRMSE: {result["force_rmse"]:.4f} N',
                    fontsize=14, fontweight='bold'
                )
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                r2 = r2_score(y_test, y_pred)
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
                
                plt.tight_layout()
                scatter_path = plots_dir / f"prediction_scatter_{model_name}_{feature_method}{plot_suffix}.png"
                plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {scatter_path.name}")
                prediction_plots_saved.append(scatter_path)
                
                # Residual plot
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, residuals, alpha=0.5, s=20)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Residuals (Actual - Predicted) (N)', fontsize=12)
                ax.set_title(
                    f'Residual Plot - {model_name}\nStd Dev: {result["force_std_dev"]:.4f} N',
                    fontsize=14, fontweight='bold'
                )
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                residual_path = plots_dir / f"prediction_residuals_{model_name}_{feature_method}{plot_suffix}.png"
                plt.savefig(residual_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {residual_path.name}")
                prediction_plots_saved.append(residual_path)
    
    # 3. Raw data plots (if HDF5 files available) - Generate for both real and simulation data
    if h5_files:
        print(f"\n{'='*80}")
        print("GENERATING RAW DATA PLOTS")
        print(f"{'='*80}")
        
        try:
            from training.plot_raw_data import main as plot_main
            import sys as plot_sys
            
            original_argv = plot_sys.argv.copy()
            
            # Raw data plots should be saved in plots_dir (cleaned/{feature_method}/plots/)
            # Generate plots for each stretch level
            for stretch, h5_file in h5_files.items():
                if h5_file and h5_file.exists():
                    stretch_output_dir = plots_dir / f"stretch_{stretch}"
                    stretch_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    plot_sys.argv = [
                        'plot_raw_data.py',
                        '--h5-file', str(h5_file),
                        '--output-dir', str(stretch_output_dir)
                    ]
                    
                    try:
                        plot_main()
                        print(f"  ✓ Raw plots generated for {stretch}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not generate raw plots for {stretch}: {e}")
            
            # Generate average plots
            if len(h5_files) >= 2:
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-files'
                ] + [str(h5_file) for h5_file in h5_files.values()] + [
                    '--output-dir', str(plots_dir)
                ]
                try:
                    plot_main()
                    print(f"  ✓ Average plots generated")
                    print(f"    Location: {plots_dir}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not generate average plots: {e}")
            
            plot_sys.argv = original_argv
        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate raw data plots: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  Total plots saved: {len(confusion_plots_saved) + len(prediction_plots_saved)}")
    print(f"  Location: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for real and simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Real data (from cleaned directory)
    python3 src/training/train_unified.py \\
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \\
        --run-label test50
    
    # Simulation data
    python3 src/training/train_unified.py \\
        --h5-files data/simulation/test5/*.h5 \\
        --run-label simulation_test5
    
    # With feature method and outlier removal
    python3 src/training/train_unified.py \\
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \\
        --run-label test50 \\
        --feature-method raw \\
        --remove-outliers
        """
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--data-dir',
        type=Path,
        help='Directory containing cleaned HDF5 files (real data)'
    )
    data_group.add_argument(
        '--h5-files',
        nargs='+',
        type=Path,
        help='List of HDF5 files (simulation or real data)'
    )
    
    parser.add_argument(
        '--run-label',
        type=str,
        required=True,
        help='Label for this training run (e.g., "test50", "simulation_test5")'
    )
    
    parser.add_argument(
        '--feature-method',
        type=str,
        choices=['raw', 'normalized', 'raw_labelled'],
        default='raw',
        help='Feature extraction method (default: raw)'
    )
    
    parser.add_argument(
        '--use-advanced-features',
        action='store_true',
        help='Use advanced feature engineering (only for raw_labelled method)'
    )
    
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove outliers (default: True for real data, False for simulation)'
    )
    
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: auto-detect based on data-dir or create cleaned/)'
    )
    
    args = parser.parse_args()
    
    # Determine data type and files
    if args.data_dir:
        # Real data: find HDF5 files in data_dir
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Error: Directory not found: {data_dir}")
            return 1
        
        # Detect data type - try to find at least one file first
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            test_files = list(data_subdir.glob("*.h5"))
        else:
            test_files = list(data_dir.glob("*.h5"))
        
        if not test_files:
            print(f"❌ Error: No HDF5 files found in {data_dir}")
            return 1
        
        # Detect data type from first file found
        data_type = detect_data_type(test_files[0])
        is_simulation = (data_type == 'simulation')
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif "cleaned" in str(data_dir):
            output_dir = data_dir
        else:
            output_dir = data_dir / "cleaned"
        
        # Group files by stretch - EXACT SAME LOGIC AS train_single_point_models.py
        h5_files_dict = {}
        
        # Check if we're in a cleaned directory structure (with data/ subdirectory)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            search_dir = data_subdir
        else:
            search_dir = data_dir
        
        for stretch in ['000pct', '010pct', '020pct']:
            # Try to find file with new naming: test_XX_YYY_cleaned.h5
            # Extract stretch number: 000pct -> 000
            stretch_num = stretch.replace('pct', '')
            pattern_new = f"test_*_{stretch_num}_cleaned.h5"
            files = list(search_dir.glob(pattern_new))
            
            # Also try old naming patterns
            if not files:
                pattern1 = f"*stretch_{stretch}.h5"
                pattern2 = f"*stretch_{stretch}_cleaned.h5"
                files = list(search_dir.glob(pattern1)) + list(search_dir.glob(pattern2))
            
            if not files:
                # Try in subdirectory
                stretch_dir = search_dir / f"stretch_{stretch}"
                if stretch_dir.exists():
                    files = list(stretch_dir.glob("*.h5"))
            
            if files:
                h5_files_dict[stretch] = sorted(files)[-1]  # Use latest file
                print(f"\nFound {stretch}: {h5_files_dict[stretch]}")
            else:
                print(f"\n⚠️  No file found for {stretch}")
        
        if not h5_files_dict:
            print("\n❌ No HDF5 files found!")
            return 1
        
    else:
        # Simulation or explicit file list
        h5_files_list = [Path(f) for f in args.h5_files]
        
        # Detect data type from first file
        data_type = detect_data_type(h5_files_list[0])
        is_simulation = (data_type == 'simulation')
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Create cleaned/ directory in parent of first file
            output_dir = h5_files_list[0].parent.parent / "cleaned"
        
        h5_files_dict = {}
        for h5_file in h5_files_list:
            # Infer stretch label
            stretch_label = infer_stretch_label(h5_file, {})
            # Remove 'stretch_' prefix if present
            stretch = stretch_label.replace('stretch_', '') if stretch_label.startswith('stretch_') else stretch_label
            h5_files_dict[stretch] = h5_file
    
    # Set default outlier removal behavior
    remove_outliers_flag = args.remove_outliers
    if not args.remove_outliers and not is_simulation:
        # Real data: remove outliers by default
        remove_outliers_flag = True
    
    print("="*80)
    print("UNIFIED TRAINING PIPELINE")
    print("="*80)
    print(f"Data type: {data_type}")
    print(f"Run label: {args.run_label}")
    print(f"Feature method: {args.feature_method}")
    print(f"Outlier removal: {remove_outliers_flag}")
    print(f"Output directory: {output_dir}")
    if args.data_dir:
        print(f"HDF5 files found: {len(h5_files_dict)}")
        for stretch, h5_file in h5_files_dict.items():
            print(f"  {stretch}: {h5_file.name}")
    else:
        print(f"HDF5 files: {len(h5_files_list)}")
    print("="*80)
    
    # Create output structure
    feature_dir = output_dir / args.feature_method
    models_dir = feature_dir / "models"
    plots_dir = feature_dir / "plots"
    data_dir_out = feature_dir / "data"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # For real data, pass h5_files_dict (mapping stretch to file)
    # For simulation, convert to list
    if is_simulation:
        h5_files_for_loading = {stretch: f for stretch, f in zip(['000pct', '010pct', '020pct'], h5_files_list) if f}
    else:
        h5_files_for_loading = h5_files_dict
    
    sequences_by_stretch = load_data_unified(
        h5_files_for_loading,
        is_simulation=is_simulation,
        remove_outliers_flag=remove_outliers_flag,
        z_threshold=args.z_threshold,
    )
    
    if not sequences_by_stretch:
        print("❌ Error: No sequences loaded!")
        return 1
    
    # Balance sequences
    print(f"\n{'='*80}")
    print("BALANCING SEQUENCES")
    print(f"{'='*80}")
    print("Before balancing:")
    for stretch, sequences in sequences_by_stretch.items():
        print(f"  {stretch}: {len(sequences)} sequences")
    sequences_by_stretch = balance_sequences(sequences_by_stretch)
    print("\nAfter balancing:")
    for stretch, sequences in sequences_by_stretch.items():
        print(f"  {stretch}: {len(sequences)} sequences")
    
    # Train models
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"{'='*80}")
    
    trained_models = {}
    gpu_mapping = {'000pct': 0, '010pct': 0, '020pct': 1}
    
    # Train per-stretch models (sort to ensure consistent order and all stretches are trained)
    for stretch_label in sorted(sequences_by_stretch.keys()):
        sequences = sequences_by_stretch[stretch_label]
        if len(sequences) == 0:
            print(f"⚠️  Skipping {stretch_label}: no sequences")
            continue
        
        gpu_id = gpu_mapping.get(stretch_label, 0)
        result = train_models_for_stretch(
            sequences,
            stretch_label,
            train_ratio=0.7,
            use_gpu=args.use_gpu,
            gpu_id=gpu_id,
            feature_method=args.feature_method,
            use_advanced_features=args.use_advanced_features,
        )
        trained_models[stretch_label] = result
        print(f"  ✓ Trained model for {stretch_label}: {len(sequences)} sequences")
    
    # Train combined model
    if len(sequences_by_stretch) > 1:
        combined_result = train_combined_model(
            sequences_by_stretch,
            train_ratio=0.7,
            use_gpu=args.use_gpu,
            feature_method=args.feature_method,
            use_advanced_features=args.use_advanced_features,
        )
        trained_models['combined'] = combined_result
    
    # Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")
    
    for model_name, result in trained_models.items():
        if result.get('force_model') is not None:
            force_path = models_dir / f"force_regressor_{model_name}.joblib"
            joblib.dump(result['force_model'], force_path)
            print(f"  ✓ Saved: {force_path.name}")
            
            if result.get('scaler') is not None:
                scaler_path = models_dir / f"scaler_{model_name}.joblib"
                joblib.dump(result['scaler'], scaler_path)
            
            if result.get('fz_scaler') is not None:
                fz_scaler_path = models_dir / f"fz_scaler_{model_name}.joblib"
                joblib.dump(result['fz_scaler'], fz_scaler_path)
        
        if result.get('offset_model') is not None:
            offset_path = models_dir / f"offset_classifier_{model_name}.joblib"
            joblib.dump(result['offset_model'], offset_path)
            print(f"  ✓ Saved: {offset_path.name}")
        
        if result.get('stretch_model') is not None:
            stretch_path = models_dir / f"stretch_classifier_{model_name}.joblib"
            joblib.dump(result['stretch_model'], stretch_path)
            print(f"  ✓ Saved: {stretch_path.name}")
    
    # Generate plots
    # For simulation, use h5_files_dict (mapping stretch to file path)
    # For real data, also use h5_files_dict
    generate_plots_unified(
        trained_models,
        sequences_by_stretch,
        plots_dir,
        args.feature_method,
        is_simulation,
        h5_files_dict,  # Pass h5_files_dict for both real and simulation data
    )
    
    # Save JSON metrics
    print(f"\n{'='*80}")
    print("SAVING METRICS JSON")
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
                'force_resolution_est': result.get('force_resolution_est', np.nan),
                'kpm1_pass': result.get('kpm1_pass', None),
                'kpm2_pass': result.get('kpm2_pass', None),
                'fz_min_actual': result.get('fz_min_actual', np.nan),
                'fz_max_actual': result.get('fz_max_actual', np.nan),
            })
            offset_results.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'accuracy': result['offset_accuracy'],
            })
    
    combined_force = {}
    combined_offset = {}
    combined_stretch = {}
    if 'combined' in trained_models:
        result = trained_models['combined']
        combined_force = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'rmse': result.get('force_rmse', np.nan),
            'std_dev': result.get('force_std_dev', np.nan),
        }
        combined_offset = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'accuracy': result.get('offset_accuracy', 0.0),
        }
        combined_stretch = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'accuracy': result.get('stretch_accuracy', 0.0),
        }
    
    def json_encoder(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    metrics_payload = {
        'force_mapping_per_stretch_full': force_results,
        'force_mapping_combined_full': combined_force,
        'offset_classification_per_stretch_full': offset_results,
        'offset_classification_combined_full': combined_offset,
        'stretch_classification_combined_full': combined_stretch,
        'config': {
            'data_type': data_type,
            'run_label': args.run_label,
            'feature_method': args.feature_method,
            'use_advanced_features': args.use_advanced_features,
            'outliers_removed': remove_outliers_flag,
            'z_threshold': args.z_threshold,
        },
    }
    
    metrics_path = feature_dir / f"{args.run_label}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2, default=json_encoder)
    
    print(f"  ✓ Saved: {metrics_path}")
    
    # Print summary (same format as train_single_point_models.py)
    print(f"\n{'='*100}")
    print("TRAINING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'Sequences':<12} {'Train Seq':<12} {'Test Seq':<12} {'Samples':<12} {'Train Samples':<15} {'Test Samples':<15} {'RMSE [N]':<12} {'STD [N]':<12} {'ΔF_min [N]':<14} {'KPM1':<6} {'KPM2':<6} {'Offset Acc':<12}")
    print("-"*100)
    
    # Sort models to show stretch levels in order, then combined
    sorted_model_names = []
    for stretch in ['000pct', '010pct', '020pct']:
        if stretch in trained_models:
            sorted_model_names.append(stretch)
    if 'combined' in trained_models:
        sorted_model_names.append('combined')
    # Add any other models that weren't in the standard list
    for model_name in trained_models.keys():
        if model_name not in sorted_model_names:
            sorted_model_names.append(model_name)
    
    for model_name in sorted_model_names:
        if model_name not in trained_models:
            continue
            
        result = trained_models[model_name]
        
        # Get metrics
        rmse = result.get('force_rmse', np.nan)
        std_dev = result.get('force_std_dev', np.nan)
        force_resolution = result.get('force_resolution_est', np.nan)
        kpm1_pass = result.get('kpm1_pass', None)
        kpm2_pass = result.get('kpm2_pass', None)
        offset_acc = result.get('offset_accuracy', np.nan)
        
        # Format values (round to 2 decimal places for display and KPM checks)
        rmse_rounded = round(rmse, 2) if not np.isnan(rmse) else np.nan
        std_dev_rounded = round(std_dev, 2) if not np.isnan(std_dev) else np.nan
        rmse_str = f"{rmse_rounded:.2f}" if not np.isnan(rmse_rounded) else "N/A"
        std_str = f"{std_dev_rounded:.2f}" if not np.isnan(std_dev_rounded) else "N/A"
        delta_f_str = f"{force_resolution:.4f}" if not np.isnan(force_resolution) else "N/A"
        kpm1_str = "PASS" if kpm1_pass is True else "FAIL" if kpm1_pass is False else "N/A"
        kpm2_str = "PASS" if kpm2_pass is True else "FAIL" if kpm2_pass is False else "N/A"
        offset_acc_str = f"{offset_acc:.4f}" if not np.isnan(offset_acc) else "N/A"
        
        print(f"{model_name:<15} {result['n_sequences']:<12} {result['n_train_sequences']:<12} "
              f"{result['n_test_sequences']:<12} {result['n_samples']:<12} "
              f"{result['n_train_samples']:<15} {result['n_test_samples']:<15} "
              f"{rmse_str:<12} {std_str:<12} {delta_f_str:<14} {kpm1_str:<6} {kpm2_str:<6} {offset_acc_str:<12}")
    
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
    print(f"\n✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Metrics: {metrics_path}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


This script automatically detects whether the input data is from real robot experiments
or simulation, and applies the same training pipeline, preprocessing, and output structure
to ensure consistency across all datasets.

Usage:
    # Real data (from cleaned directory)
    python3 src/training/train_unified.py \
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
        --run-label test50
    
    # Simulation data
    python3 src/training/train_unified.py \
        --h5-files data/simulation/test5/*.h5 \
        --run-label simulation_test5
    
    # With feature method selection
    python3 src/training/train_unified.py \
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
        --run-label test50 \
        --feature-method raw

Features:
    - Automatic detection of data type (real vs simulation)
    - Same preprocessing pipeline for both types
    - Same output structure (cleaned/{feature_method}/models/, plots/, data/)
    - Same plots generated (confusion matrices, scatter, residuals, raw data)
    - Same JSON metrics format
    - Consistent model naming and organization
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
from sklearn.metrics import ConfusionMatrixDisplay, r2_score

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import training functions
from training.train_single_point_models import (
    balance_sequences,
    load_sequences_from_h5,
    prepare_training_data,
    remove_outliers,
    train_combined_model,
    train_models_for_stretch,
    FT_SENSOR_PRECISION,
    FT_SENSOR_DECIMALS,
)

# Import simulation functions
from training.train_simulation_positions import (
    convert_to_sequences,
    infer_stretch_label,
    load_simulation_file,
    position_key,
)


def detect_data_type(h5_path: Path) -> str:
    """
    Detect if HDF5 file contains real or simulation data.
    
    Real data structure:
        - Has 'presses/' group or 'forces' and 'stretchmagtec' datasets
        - Has 'timestamps' dataset
    
    Simulation data structure:
        - Has 'MagneticField' dataset
        - Has 'IdenterPosition' dataset
        - May have 'forcesTest' dataset
    
    Returns:
        'real' or 'simulation'
    """
    with h5py.File(h5_path, 'r') as f:
        # Check for simulation markers
        if 'MagneticField' in f and 'IdenterPosition' in f:
            return 'simulation'
        
        # Check for real data markers
        if 'presses' in f or ('forces' in f and 'stretchmagtec' in f):
            return 'real'
        
        # If neither, try to infer from attributes
        if 'source' in f.attrs:
            source = str(f.attrs['source']).lower()
            if 'sim' in source:
                return 'simulation'
            elif 'real' in source or 'robot' in source:
                return 'real'
    
    # Default: assume real if unclear
    return 'real'


def load_data_unified(
    h5_files: Dict[str, Path],  # Changed: now expects dict mapping stretch to file path
    is_simulation: bool,
    remove_outliers_flag: bool = False,
    z_threshold: float = 3.0,
) -> Dict[str, List[Dict]]:
    """
    Load sequences from HDF5 files, handling both real and simulation data.
    
    Args:
        h5_files: Dictionary mapping stretch labels to HDF5 file paths
        is_simulation: True if simulation data, False if real data
        remove_outliers_flag: Whether to remove outliers (default: False for simulation, True for real)
        z_threshold: Z-score threshold for outlier detection
    
    Returns:
        Dictionary mapping stretch labels to lists of sequences
    """
    sequences_by_stretch = {}
    
    if is_simulation:
        # Load simulation data
        print(f"\n{'='*80}")
        print("LOADING SIMULATION DATA")
        print(f"{'='*80}")
        
        # For simulation, iterate over files
        for stretch_label, h5_path in h5_files.items():
            print(f"\nLoading: {h5_path.name}")
            
            # Load simulation file
            sim_data = load_simulation_file(h5_path)
            stretch_label = sim_data.get('stretch_label', 'unknown')
            
            # Convert to sequences (automatic cycle segmentation)
            if sim_data.get('indenter') is not None:
                indenter = sim_data['indenter']
                # Create position labels from XY coordinates
                position_labels = []
                for i in range(len(indenter)):
                    key, (x_mm, y_mm) = position_key(indenter[i, :2])
                    position_labels.append(key)
            else:
                position_labels = ["x+00.0mm_y+00.0mm"] * len(sim_data['magnetic'])
            
            sequences = convert_to_sequences(
                magnetic=sim_data['magnetic'],
                forces=sim_data['forces'],
                indenter=sim_data.get('indenter'),
                position_labels=position_labels,
                stretch_label=stretch_label,
            )
            
            print(f"  Converted to {len(sequences)} sequences")
            
            # Store sequences by stretch
            if stretch_label not in sequences_by_stretch:
                sequences_by_stretch[stretch_label] = []
            sequences_by_stretch[stretch_label].extend(sequences)
        
        # Remove outliers if requested
        if remove_outliers_flag:
            print(f"\n{'='*80}")
            print("REMOVING OUTLIERS")
            print(f"{'='*80}")
            for stretch in list(sequences_by_stretch.keys()):
                sequences = sequences_by_stretch[stretch]
                print(f"\n{stretch}: {len(sequences)} sequences before outlier removal")
                cleaned_sequences, outlier_indices = remove_outliers(
                    sequences, z_threshold=z_threshold, remove_per_offset=2
                )
                print(f"{stretch}: {len(cleaned_sequences)} sequences after outlier removal")
                sequences_by_stretch[stretch] = cleaned_sequences
        else:
            print(f"\n{'='*80}")
            print("SKIPPING OUTLIER REMOVAL (using all sequences)")
            print(f"{'='*80}")
    
    else:
        # Load real data - EXACT SAME LOGIC AS train_single_point_models.py
        print(f"\n{'='*80}")
        print("LOADING REAL DATA")
        print(f"{'='*80}")
        
        # Load sequences from each file (same as train_single_point_models.py)
        for stretch, h5_file in h5_files.items():
            print(f"\nLoading {stretch} from {h5_file.name}...")
            sequences = load_sequences_from_h5(h5_file)
            print(f"  Loaded {len(sequences)} sequences")
            
            if sequences:
                # Check if this file is already cleaned (has _cleaned.h5 suffix or is in cleaned/ directory)
                file_is_cleaned = "_cleaned.h5" in h5_file.name or "cleaned" in str(h5_file.parent)
                
                if file_is_cleaned:
                    print(f"  ✓ File is already cleaned (detected _cleaned.h5 suffix or cleaned/ directory)")
                    print(f"  ✓ Using sequences as-is (no additional cleaning)")
                    # Count sequences per offset for info
                    offset_counts = {}
                    for seq in sequences:
                        offset = seq.get('offset', 'unknown')
                        offset_counts[offset] = offset_counts.get(offset, 0) + 1
                    print(f"  Sequences per offset: {offset_counts}")
                    sequences_by_stretch[stretch] = sequences
                else:
                    # File is NOT cleaned - apply cleaning steps
                    # NOTE: First sequence per offset is already removed during data collection (franka_skin_test.py)
                    # So we only need to remove outliers here
                    # Count sequences per offset before any removal
                    initial_offset_counts = {}
                    for seq in sequences:
                        offset = seq.get('offset', 'unknown')
                        initial_offset_counts[offset] = initial_offset_counts.get(offset, 0) + 1
                    print(f"  Initial sequences per offset: {initial_offset_counts}")
                    print(f"  Total initial sequences: {len(sequences)}")
                    print(f"  NOTE: First sequence per offset was already removed during data collection")
                    
                    # Remove outliers (2 per offset, independently)
                    # Calculate expected number of offsets for the message
                    n_offsets = len(initial_offset_counts)
                    print(f"  Removing outliers (2 per offset, independently)...")
                    print(f"  Expected after outlier removal: {len(sequences)} - ({n_offsets} offsets * 2 outliers) = {len(sequences) - (n_offsets * 2)} sequences")
                    cleaned_sequences, outlier_indices = remove_outliers(sequences, z_threshold=z_threshold, remove_per_offset=2)
                    print(f"  After outlier removal: {len(cleaned_sequences)} sequences (removed {len(outlier_indices)} outliers)")
                    
                    # Count sequences per offset to verify
                    offset_counts = {}
                    for seq in cleaned_sequences:
                        offset = seq.get('offset', 'unknown')
                        offset_counts[offset] = offset_counts.get(offset, 0) + 1
                    print(f"  Sequences per offset after cleaning: {offset_counts}")
                    print(f"  Expected per offset: ~{len(sequences_after_first_removal) // 5 - 2} sequences")
                    sequences_by_stretch[stretch] = cleaned_sequences
    
    return sequences_by_stretch


def generate_plots_unified(
    trained_models: Dict,
    sequences_by_stretch: Dict[str, List[Dict]],
    plots_dir: Path,
    feature_method: str,
    is_simulation: bool,
    h5_files: Optional[Dict[str, Path]] = None,
):
    """
    Generate all plots (confusion matrices, scatter, residuals, raw data).
    
    Args:
        trained_models: Dictionary of trained models
        sequences_by_stretch: Sequences grouped by stretch
        plots_dir: Directory to save plots
        feature_method: Feature method used ('raw', 'normalized', 'raw_labelled')
        is_simulation: True if simulation data
        h5_files: Dictionary mapping stretch labels to HDF5 file paths (for raw data plots)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # For simulation data, create a "sim" subdirectory and add "_sim" suffix to filenames
    if is_simulation:
        plots_dir = plots_dir / "sim"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_suffix = "_sim"
    else:
        plot_suffix = ""
    
    # 1. Confusion matrices
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRICES")
    print(f"{'='*80}")
    
    offset_names = ['center', 'ne', 'nw', 'se', 'sw']
    confusion_plots_saved = []
    
    for model_name, result in trained_models.items():
        # Offset confusion matrix
        if 'offset_confusion_matrix' in result and result['offset_confusion_matrix'] is not None:
            cm = result['offset_confusion_matrix']
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=offset_names)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(
                f'Offset Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_offset_{model_name}_{feature_method}{plot_suffix}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {confusion_plot_path.name}")
            confusion_plots_saved.append(confusion_plot_path)
        
        # Stretch confusion matrix
        if 'stretch_confusion_matrix' in result and result['stretch_confusion_matrix'] is not None:
            cm = result['stretch_confusion_matrix']
            cm_normalized = cm.astype(float)
            row_sums = cm_normalized.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_normalized = cm_normalized / row_sums
            
            # Dynamically determine stretch labels from the confusion matrix size
            # The confusion matrix is created in train_combined_model with:
            #   stretch_map = {label: i for i, label in enumerate(sorted(set(all_stretches)))}
            # So the labels are sorted alphabetically
            n_stretches = cm_normalized.shape[0]
            
            # Get actual stretch labels from sequences_by_stretch keys (sorted to match train_combined_model)
            available_stretches = sorted([s for s in sequences_by_stretch.keys() if s != 'combined'])
            
            # Use available stretches if they match the matrix size exactly
            if len(available_stretches) == n_stretches:
                stretch_labels = available_stretches
            elif len(available_stretches) > n_stretches:
                # Take first n_stretches (shouldn't happen, but handle it)
                stretch_labels = available_stretches[:n_stretches]
            else:
                # Matrix has more classes than available stretches (shouldn't happen, but handle it)
                # Use available stretches and pad with generic labels
                stretch_labels = available_stretches.copy()
                while len(stretch_labels) < n_stretches:
                    stretch_labels.append(f"{len(stretch_labels)*10:03d}pct")
            
            # Final safety check: ensure we have exactly n_stretches labels
            if len(stretch_labels) != n_stretches:
                # Last resort: generate labels based on matrix size
                stretch_labels = [f"{i*10:03d}pct" for i in range(n_stretches)]
            
            fig, ax = plt.subplots(figsize=(8, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=stretch_labels)
            disp.plot(ax=ax, cmap='Blues', values_format='.2f')
            ax.set_title(
                f'Stretch Classification Confusion Matrix - {model_name}\n(Normalized by row: values = percentage)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            confusion_plot_path = plots_dir / f"confusion_matrix_stretch_{model_name}_{feature_method}{plot_suffix}.png"
            plt.savefig(confusion_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {confusion_plot_path.name}")
            confusion_plots_saved.append(confusion_plot_path)
    
    # 2. Prediction scatter and residual plots
    print(f"\n{'='*80}")
    print("GENERATING PREDICTION PLOTS")
    print(f"{'='*80}")
    
    prediction_plots_saved = []
    
    for model_name, result in trained_models.items():
        if 'y_fz_test' in result and 'y_fz_pred' in result:
            y_test = result['y_fz_test']
            y_pred = result['y_fz_pred']
            
            if len(y_test) > 0 and len(y_pred) > 0:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.scatter(y_test, y_pred, alpha=0.5, s=20)
                
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
                
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Predicted Fz (N)', fontsize=12)
                ax.set_title(
                    f'Force Prediction - {model_name}\nRMSE: {result["force_rmse"]:.4f} N',
                    fontsize=14, fontweight='bold'
                )
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                r2 = r2_score(y_test, y_pred)
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
                
                plt.tight_layout()
                scatter_path = plots_dir / f"prediction_scatter_{model_name}_{feature_method}{plot_suffix}.png"
                plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {scatter_path.name}")
                prediction_plots_saved.append(scatter_path)
                
                # Residual plot
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, residuals, alpha=0.5, s=20)
                ax.axhline(y=0, color='r', linestyle='--', lw=2)
                ax.set_xlabel('Actual Fz (N)', fontsize=12)
                ax.set_ylabel('Residuals (Actual - Predicted) (N)', fontsize=12)
                ax.set_title(
                    f'Residual Plot - {model_name}\nStd Dev: {result["force_std_dev"]:.4f} N',
                    fontsize=14, fontweight='bold'
                )
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                residual_path = plots_dir / f"prediction_residuals_{model_name}_{feature_method}{plot_suffix}.png"
                plt.savefig(residual_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {residual_path.name}")
                prediction_plots_saved.append(residual_path)
    
    # 3. Raw data plots (if HDF5 files available) - Generate for both real and simulation data
    if h5_files:
        print(f"\n{'='*80}")
        print("GENERATING RAW DATA PLOTS")
        print(f"{'='*80}")
        
        try:
            from training.plot_raw_data import main as plot_main
            import sys as plot_sys
            
            original_argv = plot_sys.argv.copy()
            
            # Raw data plots should be saved in plots_dir (cleaned/{feature_method}/plots/)
            # Generate plots for each stretch level
            for stretch, h5_file in h5_files.items():
                if h5_file and h5_file.exists():
                    stretch_output_dir = plots_dir / f"stretch_{stretch}"
                    stretch_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    plot_sys.argv = [
                        'plot_raw_data.py',
                        '--h5-file', str(h5_file),
                        '--output-dir', str(stretch_output_dir)
                    ]
                    
                    try:
                        plot_main()
                        print(f"  ✓ Raw plots generated for {stretch}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not generate raw plots for {stretch}: {e}")
            
            # Generate average plots
            if len(h5_files) >= 2:
                plot_sys.argv = [
                    'plot_raw_data.py',
                    '--h5-files'
                ] + [str(h5_file) for h5_file in h5_files.values()] + [
                    '--output-dir', str(plots_dir)
                ]
                try:
                    plot_main()
                    print(f"  ✓ Average plots generated")
                    print(f"    Location: {plots_dir}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not generate average plots: {e}")
            
            plot_sys.argv = original_argv
        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate raw data plots: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  Total plots saved: {len(confusion_plots_saved) + len(prediction_plots_saved)}")
    print(f"  Location: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for real and simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Real data (from cleaned directory)
    python3 src/training/train_unified.py \\
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \\
        --run-label test50
    
    # Simulation data
    python3 src/training/train_unified.py \\
        --h5-files data/simulation/test5/*.h5 \\
        --run-label simulation_test5
    
    # With feature method and outlier removal
    python3 src/training/train_unified.py \\
        --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \\
        --run-label test50 \\
        --feature-method raw \\
        --remove-outliers
        """
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--data-dir',
        type=Path,
        help='Directory containing cleaned HDF5 files (real data)'
    )
    data_group.add_argument(
        '--h5-files',
        nargs='+',
        type=Path,
        help='List of HDF5 files (simulation or real data)'
    )
    
    parser.add_argument(
        '--run-label',
        type=str,
        required=True,
        help='Label for this training run (e.g., "test50", "simulation_test5")'
    )
    
    parser.add_argument(
        '--feature-method',
        type=str,
        choices=['raw', 'normalized', 'raw_labelled'],
        default='raw',
        help='Feature extraction method (default: raw)'
    )
    
    parser.add_argument(
        '--use-advanced-features',
        action='store_true',
        help='Use advanced feature engineering (only for raw_labelled method)'
    )
    
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        help='Remove outliers (default: True for real data, False for simulation)'
    )
    
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection (default: 3.0)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: auto-detect based on data-dir or create cleaned/)'
    )
    
    args = parser.parse_args()
    
    # Determine data type and files
    if args.data_dir:
        # Real data: find HDF5 files in data_dir
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Error: Directory not found: {data_dir}")
            return 1
        
        # Detect data type - try to find at least one file first
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            test_files = list(data_subdir.glob("*.h5"))
        else:
            test_files = list(data_dir.glob("*.h5"))
        
        if not test_files:
            print(f"❌ Error: No HDF5 files found in {data_dir}")
            return 1
        
        # Detect data type from first file found
        data_type = detect_data_type(test_files[0])
        is_simulation = (data_type == 'simulation')
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif "cleaned" in str(data_dir):
            output_dir = data_dir
        else:
            output_dir = data_dir / "cleaned"
        
        # Group files by stretch - EXACT SAME LOGIC AS train_single_point_models.py
        h5_files_dict = {}
        
        # Check if we're in a cleaned directory structure (with data/ subdirectory)
        data_subdir = data_dir / "data"
        if data_subdir.exists():
            search_dir = data_subdir
        else:
            search_dir = data_dir
        
        for stretch in ['000pct', '010pct', '020pct']:
            # Try to find file with new naming: test_XX_YYY_cleaned.h5
            # Extract stretch number: 000pct -> 000
            stretch_num = stretch.replace('pct', '')
            pattern_new = f"test_*_{stretch_num}_cleaned.h5"
            files = list(search_dir.glob(pattern_new))
            
            # Also try old naming patterns
            if not files:
                pattern1 = f"*stretch_{stretch}.h5"
                pattern2 = f"*stretch_{stretch}_cleaned.h5"
                files = list(search_dir.glob(pattern1)) + list(search_dir.glob(pattern2))
            
            if not files:
                # Try in subdirectory
                stretch_dir = search_dir / f"stretch_{stretch}"
                if stretch_dir.exists():
                    files = list(stretch_dir.glob("*.h5"))
            
            if files:
                h5_files_dict[stretch] = sorted(files)[-1]  # Use latest file
                print(f"\nFound {stretch}: {h5_files_dict[stretch]}")
            else:
                print(f"\n⚠️  No file found for {stretch}")
        
        if not h5_files_dict:
            print("\n❌ No HDF5 files found!")
            return 1
        
    else:
        # Simulation or explicit file list
        h5_files_list = [Path(f) for f in args.h5_files]
        
        # Detect data type from first file
        data_type = detect_data_type(h5_files_list[0])
        is_simulation = (data_type == 'simulation')
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Create cleaned/ directory in parent of first file
            output_dir = h5_files_list[0].parent.parent / "cleaned"
        
        h5_files_dict = {}
        for h5_file in h5_files_list:
            # Infer stretch label
            stretch_label = infer_stretch_label(h5_file, {})
            # Remove 'stretch_' prefix if present
            stretch = stretch_label.replace('stretch_', '') if stretch_label.startswith('stretch_') else stretch_label
            h5_files_dict[stretch] = h5_file
    
    # Set default outlier removal behavior
    remove_outliers_flag = args.remove_outliers
    if not args.remove_outliers and not is_simulation:
        # Real data: remove outliers by default
        remove_outliers_flag = True
    
    print("="*80)
    print("UNIFIED TRAINING PIPELINE")
    print("="*80)
    print(f"Data type: {data_type}")
    print(f"Run label: {args.run_label}")
    print(f"Feature method: {args.feature_method}")
    print(f"Outlier removal: {remove_outliers_flag}")
    print(f"Output directory: {output_dir}")
    if args.data_dir:
        print(f"HDF5 files found: {len(h5_files_dict)}")
        for stretch, h5_file in h5_files_dict.items():
            print(f"  {stretch}: {h5_file.name}")
    else:
        print(f"HDF5 files: {len(h5_files_list)}")
    print("="*80)
    
    # Create output structure
    feature_dir = output_dir / args.feature_method
    models_dir = feature_dir / "models"
    plots_dir = feature_dir / "plots"
    data_dir_out = feature_dir / "data"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # For real data, pass h5_files_dict (mapping stretch to file)
    # For simulation, convert to list
    if is_simulation:
        h5_files_for_loading = {stretch: f for stretch, f in zip(['000pct', '010pct', '020pct'], h5_files_list) if f}
    else:
        h5_files_for_loading = h5_files_dict
    
    sequences_by_stretch = load_data_unified(
        h5_files_for_loading,
        is_simulation=is_simulation,
        remove_outliers_flag=remove_outliers_flag,
        z_threshold=args.z_threshold,
    )
    
    if not sequences_by_stretch:
        print("❌ Error: No sequences loaded!")
        return 1
    
    # Balance sequences
    print(f"\n{'='*80}")
    print("BALANCING SEQUENCES")
    print(f"{'='*80}")
    print("Before balancing:")
    for stretch, sequences in sequences_by_stretch.items():
        print(f"  {stretch}: {len(sequences)} sequences")
    sequences_by_stretch = balance_sequences(sequences_by_stretch)
    print("\nAfter balancing:")
    for stretch, sequences in sequences_by_stretch.items():
        print(f"  {stretch}: {len(sequences)} sequences")
    
    # Train models
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"{'='*80}")
    
    trained_models = {}
    gpu_mapping = {'000pct': 0, '010pct': 0, '020pct': 1}
    
    # Train per-stretch models (sort to ensure consistent order and all stretches are trained)
    for stretch_label in sorted(sequences_by_stretch.keys()):
        sequences = sequences_by_stretch[stretch_label]
        if len(sequences) == 0:
            print(f"⚠️  Skipping {stretch_label}: no sequences")
            continue
        
        gpu_id = gpu_mapping.get(stretch_label, 0)
        result = train_models_for_stretch(
            sequences,
            stretch_label,
            train_ratio=0.7,
            use_gpu=args.use_gpu,
            gpu_id=gpu_id,
            feature_method=args.feature_method,
            use_advanced_features=args.use_advanced_features,
        )
        trained_models[stretch_label] = result
        print(f"  ✓ Trained model for {stretch_label}: {len(sequences)} sequences")
    
    # Train combined model
    if len(sequences_by_stretch) > 1:
        combined_result = train_combined_model(
            sequences_by_stretch,
            train_ratio=0.7,
            use_gpu=args.use_gpu,
            feature_method=args.feature_method,
            use_advanced_features=args.use_advanced_features,
        )
        trained_models['combined'] = combined_result
    
    # Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")
    
    for model_name, result in trained_models.items():
        if result.get('force_model') is not None:
            force_path = models_dir / f"force_regressor_{model_name}.joblib"
            joblib.dump(result['force_model'], force_path)
            print(f"  ✓ Saved: {force_path.name}")
            
            if result.get('scaler') is not None:
                scaler_path = models_dir / f"scaler_{model_name}.joblib"
                joblib.dump(result['scaler'], scaler_path)
            
            if result.get('fz_scaler') is not None:
                fz_scaler_path = models_dir / f"fz_scaler_{model_name}.joblib"
                joblib.dump(result['fz_scaler'], fz_scaler_path)
        
        if result.get('offset_model') is not None:
            offset_path = models_dir / f"offset_classifier_{model_name}.joblib"
            joblib.dump(result['offset_model'], offset_path)
            print(f"  ✓ Saved: {offset_path.name}")
        
        if result.get('stretch_model') is not None:
            stretch_path = models_dir / f"stretch_classifier_{model_name}.joblib"
            joblib.dump(result['stretch_model'], stretch_path)
            print(f"  ✓ Saved: {stretch_path.name}")
    
    # Generate plots
    # For simulation, use h5_files_dict (mapping stretch to file path)
    # For real data, also use h5_files_dict
    generate_plots_unified(
        trained_models,
        sequences_by_stretch,
        plots_dir,
        args.feature_method,
        is_simulation,
        h5_files_dict,  # Pass h5_files_dict for both real and simulation data
    )
    
    # Save JSON metrics
    print(f"\n{'='*80}")
    print("SAVING METRICS JSON")
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
                'force_resolution_est': result.get('force_resolution_est', np.nan),
                'kpm1_pass': result.get('kpm1_pass', None),
                'kpm2_pass': result.get('kpm2_pass', None),
                'fz_min_actual': result.get('fz_min_actual', np.nan),
                'fz_max_actual': result.get('fz_max_actual', np.nan),
            })
            offset_results.append({
                'stretch_label': f'stretch_{stretch_label}',
                'samples': result['n_samples'],
                'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
                'accuracy': result['offset_accuracy'],
            })
    
    combined_force = {}
    combined_offset = {}
    combined_stretch = {}
    if 'combined' in trained_models:
        result = trained_models['combined']
        combined_force = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'rmse': result.get('force_rmse', np.nan),
            'std_dev': result.get('force_std_dev', np.nan),
        }
        combined_offset = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'accuracy': result.get('offset_accuracy', 0.0),
        }
        combined_stretch = {
            'stretch_label': 'combined',
            'samples': result['n_samples'],
            'sequences': result.get('n_test_sequences', result.get('n_sequences', 0)),
            'accuracy': result.get('stretch_accuracy', 0.0),
        }
    
    def json_encoder(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    metrics_payload = {
        'force_mapping_per_stretch_full': force_results,
        'force_mapping_combined_full': combined_force,
        'offset_classification_per_stretch_full': offset_results,
        'offset_classification_combined_full': combined_offset,
        'stretch_classification_combined_full': combined_stretch,
        'config': {
            'data_type': data_type,
            'run_label': args.run_label,
            'feature_method': args.feature_method,
            'use_advanced_features': args.use_advanced_features,
            'outliers_removed': remove_outliers_flag,
            'z_threshold': args.z_threshold,
        },
    }
    
    metrics_path = feature_dir / f"{args.run_label}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2, default=json_encoder)
    
    print(f"  ✓ Saved: {metrics_path}")
    
    # Print summary (same format as train_single_point_models.py)
    print(f"\n{'='*100}")
    print("TRAINING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'Sequences':<12} {'Train Seq':<12} {'Test Seq':<12} {'Samples':<12} {'Train Samples':<15} {'Test Samples':<15} {'RMSE [N]':<12} {'STD [N]':<12} {'ΔF_min [N]':<14} {'KPM1':<6} {'KPM2':<6} {'Offset Acc':<12}")
    print("-"*100)
    
    # Sort models to show stretch levels in order, then combined
    sorted_model_names = []
    for stretch in ['000pct', '010pct', '020pct']:
        if stretch in trained_models:
            sorted_model_names.append(stretch)
    if 'combined' in trained_models:
        sorted_model_names.append('combined')
    # Add any other models that weren't in the standard list
    for model_name in trained_models.keys():
        if model_name not in sorted_model_names:
            sorted_model_names.append(model_name)
    
    for model_name in sorted_model_names:
        if model_name not in trained_models:
            continue
            
        result = trained_models[model_name]
        
        # Get metrics
        rmse = result.get('force_rmse', np.nan)
        std_dev = result.get('force_std_dev', np.nan)
        force_resolution = result.get('force_resolution_est', np.nan)
        kpm1_pass = result.get('kpm1_pass', None)
        kpm2_pass = result.get('kpm2_pass', None)
        offset_acc = result.get('offset_accuracy', np.nan)
        
        # Format values (round to 2 decimal places for display and KPM checks)
        rmse_rounded = round(rmse, 2) if not np.isnan(rmse) else np.nan
        std_dev_rounded = round(std_dev, 2) if not np.isnan(std_dev) else np.nan
        rmse_str = f"{rmse_rounded:.2f}" if not np.isnan(rmse_rounded) else "N/A"
        std_str = f"{std_dev_rounded:.2f}" if not np.isnan(std_dev_rounded) else "N/A"
        delta_f_str = f"{force_resolution:.4f}" if not np.isnan(force_resolution) else "N/A"
        kpm1_str = "PASS" if kpm1_pass is True else "FAIL" if kpm1_pass is False else "N/A"
        kpm2_str = "PASS" if kpm2_pass is True else "FAIL" if kpm2_pass is False else "N/A"
        offset_acc_str = f"{offset_acc:.4f}" if not np.isnan(offset_acc) else "N/A"
        
        print(f"{model_name:<15} {result['n_sequences']:<12} {result['n_train_sequences']:<12} "
              f"{result['n_test_sequences']:<12} {result['n_samples']:<12} "
              f"{result['n_train_samples']:<15} {result['n_test_samples']:<15} "
              f"{rmse_str:<12} {std_str:<12} {delta_f_str:<14} {kpm1_str:<6} {kpm2_str:<6} {offset_acc_str:<12}")
    
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
    print(f"\n✓ Training complete!")
    print(f"  Models: {models_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Metrics: {metrics_path}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())