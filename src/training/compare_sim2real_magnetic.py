#!/usr/bin/env python3
"""
Compare simulated magnetic field data with real physical data.

This script loads real robot data (FT sensor + magnetic field) and simulated data
(forcesTest + MagneticField) and compares ONLY the magnetic field measurements
across different stretch levels (0%, 10%, 20%).

The comparison focuses on magnetic field similarity, not force prediction accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import load_sequences_from_h5, remove_outliers
from training.train_simulation_positions import load_simulation_file, convert_to_sequences
from sklearn.preprocessing import MinMaxScaler


def load_real_data(h5_path: Path, stretch_label: str) -> List[Dict]:
    """Load real robot data sequences from HDF5 file.
    
    Args:
        h5_path: Path to real data HDF5 file
        stretch_label: Expected stretch label (e.g., '000pct', '010pct', '020pct')
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec', 'forces', 'fz', 'timestamps', etc.
    """
    sequences = load_sequences_from_h5(h5_path)
    
    # Filter by stretch label
    filtered = [s for s in sequences if s.get('stretch_label', '').endswith(stretch_label)]
    
    if not filtered:
        print(f"⚠️  Warning: No sequences found for stretch {stretch_label} in {h5_path}")
        return []
    
    print(f"✅ Loaded {len(filtered)} real sequences for stretch {stretch_label}")
    return filtered


def load_sim_data(h5_path: Path) -> List[Dict]:
    """Load simulated data sequences from HDF5 file.
    
    Args:
        h5_path: Path to simulation HDF5 file
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec' (magnetic), 'fz', etc.
    """
    sim_data = load_simulation_file(h5_path)
    
    # Convert continuous data to sequences
    # We need position labels - use IdenterPosition if available
    position_labels = None
    if sim_data.get('indenter') is not None:
        indenter = sim_data['indenter']
        # Create position labels from XY coordinates (round to 0.1mm)
        position_labels = []
        for i in range(len(indenter)):
            x_mm = round(indenter[i, 0] * 1000, 1)
            y_mm = round(indenter[i, 1] * 1000, 1)
            position_labels.append(f"x+{x_mm:+.1f}mm_y+{y_mm:+.1f}mm")
    else:
        # Fallback: single position
        position_labels = ["x+00.0mm_y+00.0mm"] * len(sim_data['magnetic'])
    
    stretch_label = sim_data.get('stretch_label', 'unknown')
    sequences = convert_to_sequences(
        magnetic=sim_data['magnetic'],
        forces=sim_data['forces'],
        indenter=sim_data.get('indenter'),
        position_labels=position_labels,
        stretch_label=stretch_label,
    )
    
    print(f"✅ Loaded {len(sequences)} simulated sequences")
    return sequences


def interpolate_to_common_force(
    magnetic_real: np.ndarray,
    fz_real: np.ndarray,
    magnetic_sim: np.ndarray,
    fz_sim: np.ndarray,
    force_range: Tuple[float, float] = (0.8, 3.0),
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate both real and simulated magnetic data to a common force axis.
    
    Args:
        magnetic_real: Real magnetic data [samples, 15, 3]
        fz_real: Real force values [samples]
        magnetic_sim: Simulated magnetic data [samples, 15, 3]
        fz_sim: Simulated force values [samples]
        force_range: (min_force, max_force) for interpolation
        n_points: Number of interpolation points
    
    Returns:
        Tuple of (magnetic_real_interp, magnetic_sim_interp) both [n_points, 15, 3]
    """
    force_axis = np.linspace(force_range[0], force_range[1], n_points)
    
    # Interpolate real data
    magnetic_real_interp = np.zeros((n_points, 15, 3))
    for sensor in range(15):
        for channel in range(3):
            # Remove duplicates and sort
            unique_mask = np.concatenate(([True], np.diff(fz_real) != 0))
            fz_unique = fz_real[unique_mask]
            mag_unique = magnetic_real[unique_mask, sensor, channel]
            
            if len(fz_unique) > 1:
                interp_func = interp1d(
                    fz_unique,
                    mag_unique,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(mag_unique[0], mag_unique[-1])
                )
                magnetic_real_interp[:, sensor, channel] = interp_func(force_axis)
            else:
                magnetic_real_interp[:, sensor, channel] = mag_unique[0]
    
    # Interpolate simulated data
    magnetic_sim_interp = np.zeros((n_points, 15, 3))
    for sensor in range(15):
        for channel in range(3):
            # Remove duplicates and sort
            unique_mask = np.concatenate(([True], np.diff(fz_sim) != 0))
            fz_unique = fz_sim[unique_mask]
            mag_unique = magnetic_sim[unique_mask, sensor, channel]
            
            if len(fz_unique) > 1:
                interp_func = interp1d(
                    fz_unique,
                    mag_unique,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(mag_unique[0], mag_unique[-1])
                )
                magnetic_sim_interp[:, sensor, channel] = interp_func(force_axis)
            else:
                magnetic_sim_interp[:, sensor, channel] = mag_unique[0]
    
    return magnetic_real_interp, magnetic_sim_interp


def compare_magnetic_fields(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    stretch_label: str,
    output_dir: Path,
    z_threshold: float = 3.0,
) -> Dict:
    """Compare real and simulated magnetic field data.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        stretch_label: Stretch level label (e.g., '000pct')
        output_dir: Directory to save plots and metrics
        z_threshold: Threshold for outlier removal (MAD-based)
    
    Returns:
        Dictionary with comparison metrics
    """
    if not real_sequences or not sim_sequences:
        print(f"⚠️  Skipping comparison for {stretch_label}: missing data")
        return {}
    
    # STEP 1: Remove outliers from real sequences
    print(f"  Removing outliers from real data (z_threshold={z_threshold})...")
    real_sequences_cleaned, outlier_indices = remove_outliers(real_sequences, z_threshold)
    if outlier_indices:
        print(f"    Removed {len(outlier_indices)} outlier sequences: {outlier_indices}")
    print(f"    Real sequences: {len(real_sequences)} → {len(real_sequences_cleaned)}")
    
    # STEP 2: Aggregate all sequences
    all_real_magnetic = []
    all_real_fz = []
    all_sim_magnetic = []
    all_sim_fz = []
    
    for seq in real_sequences_cleaned:
        # Use absolute Fz values
        fz = np.abs(seq['fz'])
        # Filter to force range [0.8, 3.0] N
        mask = (fz >= 0.8) & (fz <= 3.0)
        if np.sum(mask) > 10:  # Need at least 10 samples
            all_real_magnetic.append(seq['stretchmagtec'][mask])
            all_real_fz.append(fz[mask])
    
    for seq in sim_sequences:
        # Use absolute Fz values
        fz = np.abs(seq['fz'])
        # Filter to force range [0.8, 3.0] N
        mask = (fz >= 0.8) & (fz <= 3.0)
        if np.sum(mask) > 10:  # Need at least 10 samples
            all_sim_magnetic.append(seq['stretchmagtec'][mask])
            all_sim_fz.append(fz[mask])
    
    if not all_real_magnetic or not all_sim_magnetic:
        print(f"⚠️  Skipping comparison for {stretch_label}: insufficient data after filtering")
        return {}
    
    # Concatenate all sequences
    magnetic_real = np.concatenate(all_real_magnetic, axis=0)
    fz_real = np.concatenate(all_real_fz)
    magnetic_sim = np.concatenate(all_sim_magnetic, axis=0)
    fz_sim = np.concatenate(all_sim_fz)
    
    # STEP 3: Interpolate to common force axis
    magnetic_real_interp, magnetic_sim_interp = interpolate_to_common_force(
        magnetic_real, fz_real, magnetic_sim, fz_sim,
        force_range=(0.8, 3.0), n_points=50
    )
    
    # STEP 4: Normalize separately using MinMaxScaler
    # Normalize real data (after outlier removal)
    print(f"  Normalizing real data (separate minmax)...")
    real_scaler = MinMaxScaler()
    n_samples, n_sensors, n_channels = magnetic_real_interp.shape
    magnetic_real_flat = magnetic_real_interp.reshape(n_samples, -1)  # [samples, 45]
    magnetic_real_normalized_flat = real_scaler.fit_transform(magnetic_real_flat)
    magnetic_real_normalized = magnetic_real_normalized_flat.reshape(n_samples, n_sensors, n_channels)
    
    # Normalize simulated data (separate minmax)
    print(f"  Normalizing simulated data (separate minmax)...")
    sim_scaler = MinMaxScaler()
    n_samples_sim, n_sensors_sim, n_channels_sim = magnetic_sim_interp.shape
    magnetic_sim_flat = magnetic_sim_interp.reshape(n_samples_sim, -1)  # [samples, 45]
    magnetic_sim_normalized_flat = sim_scaler.fit_transform(magnetic_sim_flat)
    magnetic_sim_normalized = magnetic_sim_normalized_flat.reshape(n_samples_sim, n_sensors_sim, n_channels_sim)
    
    # Use normalized data for comparison
    magnetic_real_interp = magnetic_real_normalized
    magnetic_sim_interp = magnetic_sim_normalized
    
    # Compute metrics per sensor and channel
    metrics = {
        'stretch_label': stretch_label,
        'n_real_sequences': len(real_sequences),
        'n_real_sequences_after_outlier_removal': len(real_sequences_cleaned),
        'n_outliers_removed': len(outlier_indices),
        'n_sim_sequences': len(sim_sequences),
        'n_real_samples': len(magnetic_real),
        'n_sim_samples': len(magnetic_sim),
        'normalization': {
            'real_min': float(np.min(magnetic_real)),
            'real_max': float(np.max(magnetic_real)),
            'sim_min': float(np.min(magnetic_sim)),
            'sim_max': float(np.max(magnetic_sim)),
        },
        'sensor_metrics': {},
    }
    
    # Per-sensor metrics
    rmse_per_sensor = np.zeros(15)
    correlation_per_sensor = np.zeros(15)
    
    for sensor in range(15):
        sensor_rmse = []
        sensor_corr = []
        
        for channel in range(3):
            real_ch = magnetic_real_interp[:, sensor, channel]
            sim_ch = magnetic_sim_interp[:, sensor, channel]
            
            # RMSE
            rmse = np.sqrt(np.mean((real_ch - sim_ch)**2))
            sensor_rmse.append(rmse)
            
            # Correlation
            if np.std(real_ch) > 0 and np.std(sim_ch) > 0:
                corr, _ = stats.pearsonr(real_ch, sim_ch)
                sensor_corr.append(corr)
            else:
                sensor_corr.append(0.0)
        
        rmse_per_sensor[sensor] = np.mean(sensor_rmse)
        correlation_per_sensor[sensor] = np.mean(sensor_corr)
        
        metrics['sensor_metrics'][f'sensor_{sensor+1}'] = {
            'rmse': float(np.mean(sensor_rmse)),
            'correlation': float(np.mean(sensor_corr)),
        }
    
    # Overall metrics
    metrics['overall_rmse'] = float(np.mean(rmse_per_sensor))
    metrics['overall_correlation'] = float(np.mean(correlation_per_sensor))
    metrics['best_sensor'] = int(np.argmin(rmse_per_sensor)) + 1
    metrics['worst_sensor'] = int(np.argmax(rmse_per_sensor)) + 1
    
    # Create comparison plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sim2real_comparison_{stretch_label}.png"
    create_comparison_plot(
        magnetic_real_interp,
        magnetic_sim_interp,
        force_axis=np.linspace(0.8, 3.0, 50),
        metrics=metrics,
        stretch_label=stretch_label,
        output_path=plot_path,
    )
    
    print(f"✅ Comparison for {stretch_label}:")
    print(f"   Overall RMSE: {metrics['overall_rmse']:.4f}")
    print(f"   Overall Correlation: {metrics['overall_correlation']:.4f}")
    print(f"   Best sensor: {metrics['best_sensor']} (RMSE: {rmse_per_sensor[metrics['best_sensor']-1]:.4f})")
    print(f"   Plot saved: {plot_path}")
    
    return metrics


def create_comparison_plot(
    magnetic_real: np.ndarray,
    magnetic_sim: np.ndarray,
    force_axis: np.ndarray,
    metrics: Dict,
    stretch_label: str,
    output_path: Path,
):
    """Create comparison plot showing real vs simulated magnetic fields."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Real vs Simulated Magnetic Field Comparison - {stretch_label}\n'
                 f'Overall RMSE: {metrics["overall_rmse"]:.4f}, '
                 f'Correlation: {metrics["overall_correlation"]:.4f}',
                 fontsize=14, fontweight='bold')
    
    channel_names = ['Bx', 'By', 'Bz']
    colors = ['r', 'g', 'b']
    
    for sensor in range(15):
        row = sensor // 5
        col = sensor % 5
        ax = axes[row, col]
        
        for channel in range(3):
            real_ch = magnetic_real[:, sensor, channel]
            sim_ch = magnetic_sim[:, sensor, channel]
            
            ax.plot(force_axis, real_ch, color=colors[channel], linestyle='-', 
                   linewidth=2, alpha=0.7, label=f'Real {channel_names[channel]}')
            ax.plot(force_axis, sim_ch, color=colors[channel], linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Sim {channel_names[channel]}')
        
        sensor_metrics = metrics['sensor_metrics'][f'sensor_{sensor+1}']
        ax.set_title(f'Sensor {sensor+1}\nRMSE: {sensor_metrics["rmse"]:.3f}, '
                    f'Corr: {sensor_metrics["correlation"]:.3f}',
                    fontsize=10)
        ax.set_xlabel('Force [N]', fontsize=9)
        ax.set_ylabel('Magnetic Field [a.u.]', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare simulated magnetic field data with real physical data'
    )
    parser.add_argument(
        '--real-data-dir',
        type=Path,
        required=True,
        help='Directory containing real data HDF5 files (e.g., data/Single_Point/force_0.0to3.0N_step0.1N_single_test1/)'
    )
    parser.add_argument(
        '--sim-data-dir',
        type=Path,
        required=True,
        help='Directory containing simulation HDF5 files (e.g., data/simulation/test2/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('plots/comparison/sim2real'),
        help='Output directory for plots and metrics JSON'
    )
    parser.add_argument(
        '--stretches',
        nargs='+',
        default=['000pct', '010pct', '020pct'],
        help='Stretch labels to compare (default: 000pct 010pct 020pct)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-threshold for outlier removal using MAD (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    # Find real data files
    real_files = {}
    for stretch in args.stretches:
        pattern = f"*stretch_{stretch}.h5"
        matches = list(args.real_data_dir.glob(pattern))
        if matches:
            real_files[stretch] = matches[0]
        else:
            print(f"⚠️  Warning: No real data file found for {stretch}")
    
    # Find simulation files
    sim_files = {}
    for stretch in args.stretches:
        # Try different naming conventions
        patterns = [
            f"*stretch_{stretch}.h5",
            f"*stretch_{stretch.replace('pct', '')}.h5",
            f"*{stretch.replace('pct', '')}.h5",
        ]
        for pattern in patterns:
            matches = list(args.sim_data_dir.glob(pattern))
            if matches:
                sim_files[stretch] = matches[0]
                break
        if stretch not in sim_files:
            print(f"⚠️  Warning: No simulation file found for {stretch}")
    
    # Compare each stretch level
    all_metrics = {}
    for stretch in args.stretches:
        if stretch not in real_files or stretch not in sim_files:
            continue
        
        print(f"\n{'='*80}")
        print(f"Comparing stretch level: {stretch}")
        print(f"{'='*80}")
        
        real_sequences = load_real_data(real_files[stretch], stretch)
        sim_sequences = load_sim_data(sim_files[stretch])
        
        metrics = compare_magnetic_fields(
            real_sequences,
            sim_sequences,
            stretch,
            args.output_dir,
            z_threshold=args.z_threshold,
        )
        
        if metrics:
            all_metrics[stretch] = metrics
    
    # Save combined metrics
    if all_metrics:
        metrics_path = args.output_dir / 'sim2real_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✅ Saved metrics to: {metrics_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for stretch, metrics in all_metrics.items():
            print(f"{stretch}: RMSE={metrics['overall_rmse']:.4f}, "
                  f"Correlation={metrics['overall_correlation']:.4f}")
    else:
        print("\n⚠️  No comparisons completed. Check data file paths.")


if __name__ == '__main__':
    main()

