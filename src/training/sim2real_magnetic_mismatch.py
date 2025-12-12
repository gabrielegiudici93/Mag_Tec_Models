#!/usr/bin/env python3
"""
Sim2Real Magnetic Mismatch Analysis

This script compares magnetic sensor readings from real robot experiments with
simulated magnetic field data. The analysis focuses on:
1. Offset differences (bias between real and simulated)
2. Scaling differences (gain/amplitude mismatch)
3. Normalization to account for unitless sensors
4. Force-matched comparison (same force levels)

The goal is to identify systematic differences that could be corrected to improve
simulation-to-reality transfer.

Usage:
    python3 src/training/sim2real_magnetic_mismatch.py \
        --real-data data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data/test_50_000_cleaned.h5 \
        --sim-data data/simulation/test5/*_stretch_0.h5 \
        --output-dir analysis/sim2real_mismatch
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import load_sequences_from_h5, remove_outliers
from training.train_simulation_positions import load_simulation_file, convert_to_sequences, position_key


def load_real_sequences(h5_path: Path, stretch_label: Optional[str] = None) -> List[Dict]:
    """Load real robot data sequences from HDF5 file.
    
    Args:
        h5_path: Path to real data HDF5 file
        stretch_label: Optional stretch label to filter (e.g., '000pct')
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec', 'forces', 'fz', etc.
    """
    sequences = load_sequences_from_h5(h5_path)
    
    if stretch_label:
        # Filter by stretch label
        filtered = [s for s in sequences if s.get('stretch_label', '').endswith(stretch_label)]
        if not filtered:
            print(f"⚠️  Warning: No sequences found for stretch {stretch_label} in {h5_path}")
            return []
        sequences = filtered
    
    print(f"✅ Loaded {len(sequences)} real sequences from {h5_path.name}")
    return sequences


def load_sim_sequences(h5_path: Path) -> List[Dict]:
    """Load simulated data sequences from HDF5 file.
    
    Args:
        h5_path: Path to simulation HDF5 file
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec' (magnetic), 'fz', etc.
    """
    sim_data = load_simulation_file(h5_path)
    
    # Convert continuous data to sequences
    position_labels = None
    if sim_data.get('indenter') is not None:
        indenter = sim_data['indenter']
        # Create position labels from XY coordinates
        position_labels = []
        for i in range(len(indenter)):
            key, (x_mm, y_mm) = position_key(indenter[i, :2])
            position_labels.append(key)
    else:
        position_labels = ["x+00.0mm_y+00.0mm"] * len(sim_data['magnetic'])
    
    stretch_label = sim_data.get('stretch_label', 'unknown')
    sequences = convert_to_sequences(
        magnetic=sim_data['magnetic'],
        forces=sim_data['forces'],
        indenter=sim_data.get('indenter'),
        position_labels=position_labels,
        stretch_label=stretch_label,
    )
    
    print(f"✅ Loaded {len(sequences)} simulated sequences from {h5_path.name}")
    return sequences


def match_sequences_by_force_profile(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    force_tolerance: float = 1.0,
    match_by_order: bool = True,
) -> List[Tuple[Dict, Dict, float]]:
    """Match real and simulated sequences based on similar force profiles.
    
    The simulation test replicated the force profiles from the real test,
    so we can match sequences by comparing their force profiles or by order.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        force_tolerance: Maximum allowed difference in force profiles for matching (N)
        match_by_order: If True, match sequences in order (same offset, same order)
                        If False, find best match by force profile similarity
    
    Returns:
        List of tuples (real_seq, sim_seq, match_score) where match_score is the
        RMSE between the two force profiles (lower is better)
    """
    matched_pairs = []
    
    # Group sequences by offset
    real_by_offset = {}
    for seq in real_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in real_by_offset:
            real_by_offset[offset] = []
        real_by_offset[offset].append(seq)
    
    sim_by_offset = {}
    for seq in sim_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in sim_by_offset:
            sim_by_offset[offset] = []
        sim_by_offset[offset].append(seq)
    
    print(f"  Matching sequences...")
    print(f"  Real offsets: {list(real_by_offset.keys())}")
    print(f"  Sim offsets: {list(sim_by_offset.keys())}")
    print(f"  Match method: {'by order' if match_by_order else 'by force profile'}")
    
    # Match offset by offset
    for offset in real_by_offset.keys():
        if offset not in sim_by_offset:
            print(f"  ⚠️  Warning: Offset '{offset}' not found in sim data, skipping")
            continue
        
        real_seqs = real_by_offset[offset]
        sim_seqs = sim_by_offset[offset]
        
        print(f"  Matching offset '{offset}': {len(real_seqs)} real, {len(sim_seqs)} sim sequences")
        
        if match_by_order:
            # Match sequences in order (assuming they're already in the same order)
            n_pairs = min(len(real_seqs), len(sim_seqs))
            for i in range(n_pairs):
                real_seq = real_seqs[i]
                sim_seq = sim_seqs[i]
                
                # Calculate force profile match score for reference
                real_fz = np.abs(real_seq['fz'])
                sim_fz = np.abs(sim_seq['fz'])
                min_len = min(len(real_fz), len(sim_fz))
                if min_len >= 10:
                    real_fz_short = real_fz[:min_len]
                    sim_fz_short = sim_fz[:min_len]
                    match_score = np.sqrt(np.mean((real_fz_short - sim_fz_short)**2))
                else:
                    match_score = 0.0
                
                matched_pairs.append((real_seq, sim_seq, match_score))
        else:
            # Find best match by force profile
            used_sim_indices = set()
            for real_seq in real_seqs:
                real_fz = np.abs(real_seq['fz'])
                
                best_match = None
                best_score = float('inf')
                best_idx = -1
                
                for idx, sim_seq in enumerate(sim_seqs):
                    if idx in used_sim_indices:
                        continue
                    
                    sim_fz = np.abs(sim_seq['fz'])
                    min_len = min(len(real_fz), len(sim_fz))
                    if min_len < 10:
                        continue
                    
                    real_fz_short = real_fz[:min_len]
                    sim_fz_short = sim_fz[:min_len]
                    score = np.sqrt(np.mean((real_fz_short - sim_fz_short)**2))
                    
                    if score < best_score:
                        best_score = score
                        best_match = sim_seq
                        best_idx = idx
                
                if best_match is not None and best_score < force_tolerance:
                    matched_pairs.append((real_seq, best_match, best_score))
                    used_sim_indices.add(best_idx)
                else:
                    if len(matched_pairs) < 5:  # Only print first few warnings
                        print(f"    ⚠️  No good match found (best score: {best_score:.4f} N)")
    
    print(f"  Matched {len(matched_pairs)} sequence pairs")
    return matched_pairs


def extract_force_matched_samples(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    force_range: Tuple[float, float] = (0.5, 3.0),
    force_bins: int = 50,
    min_samples_per_bin: int = 5,
    match_by_offset: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Extract magnetic samples matched by force level.
    
    DEPRECATED: Use match_sequences_by_force_profile instead.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        force_range: (min_force, max_force) range to consider
        force_bins: Number of force bins for matching
        min_samples_per_bin: Minimum samples required per bin
        match_by_offset: If True, match sequences by offset (center vs center, etc.)
    
    Returns:
        Tuple of (real_magnetic, real_fz, sim_magnetic, sim_fz, info_dict)
        All arrays are [total_samples, 15, 3] for magnetic, [total_samples] for fz
        info_dict contains debug information about what was compared
    """
    info = {
        'real_offsets': {},
        'sim_offsets': {},
        'matched_offsets': [],
    }
    
    # Group sequences by offset
    real_by_offset = {}
    for seq in real_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in real_by_offset:
            real_by_offset[offset] = []
        real_by_offset[offset].append(seq)
        info['real_offsets'][offset] = info['real_offsets'].get(offset, 0) + 1
    
    sim_by_offset = {}
    for seq in sim_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in sim_by_offset:
            sim_by_offset[offset] = []
        sim_by_offset[offset].append(seq)
        info['sim_offsets'][offset] = info['sim_offsets'].get(offset, 0) + 1
    
    print(f"  Real data offsets: {info['real_offsets']}")
    print(f"  Sim data offsets: {info['sim_offsets']}")
    
    # Collect all samples from real sequences
    all_real_magnetic = []
    all_real_fz = []
    all_sim_magnetic = []
    all_sim_fz = []
    
    if match_by_offset:
        # Match offset by offset
        for offset in real_by_offset.keys():
            if offset not in sim_by_offset:
                print(f"  ⚠️  Warning: Offset '{offset}' not found in sim data, skipping")
                continue
            
            info['matched_offsets'].append(offset)
            print(f"  Matching offset: {offset} (real: {len(real_by_offset[offset])} seqs, sim: {len(sim_by_offset[offset])} seqs)")
            
            real_samples_count = 0
            sim_samples_count = 0
            
            for seq in real_by_offset[offset]:
                fz = np.abs(seq['fz'])
                mask = (fz >= force_range[0]) & (fz <= force_range[1])
                if np.sum(mask) > min_samples_per_bin:
                    all_real_magnetic.append(seq['stretchmagtec'][mask])
                    all_real_fz.append(fz[mask])
                    real_samples_count += np.sum(mask)
            
            for seq in sim_by_offset[offset]:
                fz = np.abs(seq['fz'])
                mask = (fz >= force_range[0]) & (fz <= force_range[1])
                if np.sum(mask) > min_samples_per_bin:
                    all_sim_magnetic.append(seq['stretchmagtec'][mask])
                    all_sim_fz.append(fz[mask])
                    sim_samples_count += np.sum(mask)
            
            print(f"    Offset {offset}: collected {real_samples_count} real samples, {sim_samples_count} sim samples")
    else:
        # Collect all samples without offset matching
        for seq in real_sequences:
            fz = np.abs(seq['fz'])
            mask = (fz >= force_range[0]) & (fz <= force_range[1])
            if np.sum(mask) > min_samples_per_bin:
                all_real_magnetic.append(seq['stretchmagtec'][mask])
                all_real_fz.append(fz[mask])
    
    # Collect all samples from sim sequences (if not already done in match_by_offset)
    if not match_by_offset:
        for seq in sim_sequences:
            fz = np.abs(seq['fz'])
            mask = (fz >= force_range[0]) & (fz <= force_range[1])
            if np.sum(mask) > min_samples_per_bin:
                all_sim_magnetic.append(seq['stretchmagtec'][mask])
                all_sim_fz.append(fz[mask])
    
    if not all_real_magnetic or not all_sim_magnetic:
        print(f"  ⚠️  Error: Insufficient data after filtering")
        print(f"     Real samples: {len(all_real_magnetic)} sequences")
        print(f"     Sim samples: {len(all_sim_magnetic)} sequences")
        raise ValueError("Insufficient data after filtering")
    
    # Concatenate
    real_magnetic = np.concatenate(all_real_magnetic, axis=0)  # [N, 15, 3]
    real_fz = np.concatenate(all_real_fz)  # [N]
    sim_magnetic = np.concatenate(all_sim_magnetic, axis=0)  # [M, 15, 3]
    sim_fz = np.concatenate(all_sim_fz)  # [M]
    
    # Create force bins and match samples
    force_edges = np.linspace(force_range[0], force_range[1], force_bins + 1)
    
    matched_real_magnetic = []
    matched_real_fz = []
    matched_sim_magnetic = []
    matched_sim_fz = []
    
    for i in range(force_bins):
        f_min, f_max = force_edges[i], force_edges[i + 1]
        
        # Find samples in this bin for both datasets
        real_mask = (real_fz >= f_min) & (real_fz < f_max)
        sim_mask = (sim_fz >= f_min) & (sim_fz < f_max)
        
        if np.sum(real_mask) >= min_samples_per_bin and np.sum(sim_mask) >= min_samples_per_bin:
            # Take mean per bin to reduce noise
            matched_real_magnetic.append(np.mean(real_magnetic[real_mask], axis=0))
            matched_real_fz.append((f_min + f_max) / 2)
            matched_sim_magnetic.append(np.mean(sim_magnetic[sim_mask], axis=0))
            matched_sim_fz.append((f_min + f_max) / 2)
    
    if not matched_real_magnetic:
        raise ValueError("No matched samples found after binning")
    
    info['n_matched_bins'] = len(matched_real_magnetic)
    info['real_magnetic_stats'] = {
        'min': float(np.min(matched_real_magnetic)),
        'max': float(np.max(matched_real_magnetic)),
        'mean': float(np.mean(matched_real_magnetic)),
        'std': float(np.std(matched_real_magnetic)),
    }
    info['sim_magnetic_stats'] = {
        'min': float(np.min(matched_sim_magnetic)),
        'max': float(np.max(matched_sim_magnetic)),
        'mean': float(np.mean(matched_sim_magnetic)),
        'std': float(np.std(matched_sim_magnetic)),
    }
    
    print(f"  Real magnetic stats: min={info['real_magnetic_stats']['min']:.2f}, max={info['real_magnetic_stats']['max']:.2f}, mean={info['real_magnetic_stats']['mean']:.2f}, std={info['real_magnetic_stats']['std']:.2f}")
    print(f"  Sim magnetic stats: min={info['sim_magnetic_stats']['min']:.2f}, max={info['sim_magnetic_stats']['max']:.2f}, mean={info['sim_magnetic_stats']['mean']:.2f}, std={info['sim_magnetic_stats']['std']:.2f}")
    
    return (
        np.array(matched_real_magnetic),  # [n_bins, 15, 3]
        np.array(matched_real_fz),  # [n_bins]
        np.array(matched_sim_magnetic),  # [n_bins, 15, 3]
        np.array(matched_sim_fz),  # [n_bins]
        info,
    )


def compute_offset_and_scaling(
    real_magnetic: np.ndarray,  # [n_samples, 15, 3]
    sim_magnetic: np.ndarray,  # [n_samples, 15, 3]
) -> Dict:
    """Compute offset (bias) and scaling (gain) between real and simulated magnetic data.
    
    For each sensor and channel, we fit:
        sim = scale * real + offset
    
    Using least squares: sim = A @ [scale, offset]^T
    
    Args:
        real_magnetic: Real magnetic data [n_samples, 15, 3]
        sim_magnetic: Simulated magnetic data [n_samples, 15, 3]
    
    Returns:
        Dictionary with offset and scaling per sensor/channel
    """
    n_samples, n_sensors, n_channels = real_magnetic.shape
    
    results = {
        'sensor_metrics': {},
        'overall_offset': {},
        'overall_scaling': {},
    }
    
    # Per-sensor, per-channel analysis
    all_offsets = []
    all_scalings = []
    all_correlations = []
    
    for sensor in range(n_sensors):
        sensor_results = {}
        
        for channel in range(n_channels):
            real_ch = real_magnetic[:, sensor, channel]
            sim_ch = sim_magnetic[:, sensor, channel]
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(real_ch) & np.isfinite(sim_ch)
            if np.sum(valid_mask) < 10:
                continue
            
            real_ch = real_ch[valid_mask]
            sim_ch = sim_ch[valid_mask]
            
            # Fit linear model: sim = scale * real + offset
            # Using least squares: [sim] = [real, 1] @ [scale, offset]^T
            A = np.vstack([real_ch, np.ones(len(real_ch))]).T
            scale, offset = np.linalg.lstsq(A, sim_ch, rcond=None)[0]
            
            # Compute correlation
            if np.std(real_ch) > 0 and np.std(sim_ch) > 0:
                corr, _ = stats.pearsonr(real_ch, sim_ch)
            else:
                corr = 0.0
            
            # Compute RMSE
            sim_predicted = scale * real_ch + offset
            rmse = np.sqrt(np.mean((sim_ch - sim_predicted)**2))
            
            sensor_results[f'channel_{channel}'] = {
                'offset': float(offset),
                'scaling': float(scale),
                'correlation': float(corr),
                'rmse': float(rmse),
            }
            
            all_offsets.append(offset)
            all_scalings.append(scale)
            all_correlations.append(corr)
        
        results['sensor_metrics'][f'sensor_{sensor+1}'] = sensor_results
    
    # Overall statistics
    if all_offsets:
        results['overall_offset'] = {
            'mean': float(np.mean(all_offsets)),
            'std': float(np.std(all_offsets)),
            'median': float(np.median(all_offsets)),
            'min': float(np.min(all_offsets)),
            'max': float(np.max(all_offsets)),
        }
    
    if all_scalings:
        results['overall_scaling'] = {
            'mean': float(np.mean(all_scalings)),
            'std': float(np.std(all_scalings)),
            'median': float(np.median(all_scalings)),
            'min': float(np.min(all_scalings)),
            'max': float(np.max(all_scalings)),
        }
    
    if all_correlations:
        results['overall_correlation'] = {
            'mean': float(np.mean(all_correlations)),
            'std': float(np.std(all_correlations)),
            'median': float(np.median(all_correlations)),
        }
    
    return results


def normalize_magnetic_data(
    real_magnetic: np.ndarray,
    sim_magnetic: np.ndarray,
    method: str = 'standardize',
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize magnetic data to account for unitless sensors.
    
    Args:
        real_magnetic: Real magnetic data [n_samples, 15, 3]
        sim_magnetic: Simulated magnetic data [n_samples, 15, 3]
        method: Normalization method ('standardize', 'minmax', 'separate_minmax')
    
    Returns:
        Tuple of (normalized_real, normalized_sim, scaler_info)
    """
    n_samples_real, n_sensors, n_channels = real_magnetic.shape
    n_samples_sim = sim_magnetic.shape[0]
    
    # Flatten to [samples, 45] for normalization
    real_flat = real_magnetic.reshape(n_samples_real, -1)  # [n_samples, 45]
    sim_flat = sim_magnetic.reshape(n_samples_sim, -1)  # [n_samples, 45]
    
    scaler_info = {}
    
    if method == 'standardize':
        # StandardScaler: zero mean, unit variance (applied jointly)
        combined = np.vstack([real_flat, sim_flat])
        scaler = StandardScaler()
        scaler.fit(combined)
        
        real_normalized = scaler.transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = scaler.transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'standardize',
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
        }
    
    elif method == 'minmax':
        # MinMaxScaler: scale to [0, 1] (applied jointly)
        combined = np.vstack([real_flat, sim_flat])
        scaler = MinMaxScaler()
        scaler.fit(combined)
        
        real_normalized = scaler.transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = scaler.transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'minmax',
            'min': scaler.data_min_.tolist(),
            'max': scaler.data_max_.tolist(),
        }
    
    elif method == 'separate_minmax':
        # Separate MinMaxScaler for real and sim (preserves relative differences)
        real_scaler = MinMaxScaler()
        sim_scaler = MinMaxScaler()
        
        real_normalized = real_scaler.fit_transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = sim_scaler.fit_transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'separate_minmax',
            'real_min': real_scaler.data_min_.tolist(),
            'real_max': real_scaler.data_max_.tolist(),
            'sim_min': sim_scaler.data_min_.tolist(),
            'sim_max': sim_scaler.data_max_.tolist(),
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return real_normalized, sim_normalized, scaler_info


def compute_sequence_error(
    real_seq: Dict,
    sim_seq: Dict,
    normalize: bool = True,
) -> Dict:
    """Compute error between a real and simulated sequence.
    
    Args:
        real_seq: Real sequence dictionary with 'stretchmagtec' and 'fz'
        sim_seq: Simulated sequence dictionary with 'stretchmagtec' and 'fz'
        normalize: Whether to normalize magnetic data before comparison
    
    Returns:
        Dictionary with error metrics for this sequence pair
    """
    real_mag = real_seq['stretchmagtec']  # [samples, 15, 3]
    sim_mag = sim_seq['stretchmagtec']  # [samples, 15, 3]
    real_fz = np.abs(real_seq['fz'])
    sim_fz = np.abs(sim_seq['fz'])
    
    # Interpolate to same length (use shorter length)
    min_len = min(len(real_mag), len(sim_mag))
    if min_len < 10:
        return None
    
    real_mag = real_mag[:min_len]
    sim_mag = sim_mag[:min_len]
    real_fz = real_fz[:min_len]
    sim_fz = sim_fz[:min_len]
    
    # Normalize if requested
    if normalize:
        # Flatten for normalization
        real_flat = real_mag.reshape(-1, 45)
        sim_flat = sim_mag.reshape(-1, 45)
        
        # Use joint standardization
        combined = np.vstack([real_flat, sim_flat])
        scaler = StandardScaler()
        scaler.fit(combined)
        
        real_norm = scaler.transform(real_flat).reshape(min_len, 15, 3)
        sim_norm = scaler.transform(sim_flat).reshape(min_len, 15, 3)
    else:
        real_norm = real_mag
        sim_norm = sim_mag
    
    # Compute per-sensor RMSE
    sensor_rmse = []
    for sensor in range(15):
        real_sensor = real_norm[:, sensor, :].reshape(-1)
        sim_sensor = sim_norm[:, sensor, :].reshape(-1)
        rmse = np.sqrt(np.mean((real_sensor - sim_sensor)**2))
        sensor_rmse.append(rmse)
    
    # Overall RMSE
    overall_rmse = np.sqrt(np.mean((real_norm - sim_norm)**2))
    
    # Mean absolute error
    mae = np.mean(np.abs(real_norm - sim_norm))
    
    return {
        'offset': real_seq.get('offset', 'unknown'),
        'sequence_length': min_len,
        'overall_rmse': float(overall_rmse),
        'mae': float(mae),
        'sensor_rmse': [float(x) for x in sensor_rmse],
        'real_fz_range': [float(np.min(real_fz)), float(np.max(real_fz))],
        'sim_fz_range': [float(np.min(sim_fz)), float(np.max(sim_fz))],
    }


def analyze_mismatch(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    stretch_label: str,
    output_dir: Path,
    normalize: bool = True,
    normalization_method: str = 'standardize',
    remove_outliers_flag: bool = True,
    z_threshold: float = 3.0,
    force_tolerance: float = 0.1,
) -> Dict:
    """Analyze mismatch between real and simulated magnetic data.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        stretch_label: Stretch level label (e.g., '000pct')
        output_dir: Directory to save plots and metrics
        normalize: Whether to normalize data before comparison
        normalization_method: Normalization method ('standardize', 'minmax', 'separate_minmax')
        remove_outliers_flag: Whether to remove outliers from real data
        z_threshold: Z-score threshold for outlier removal
    
    Returns:
        Dictionary with analysis results
    """
    if not real_sequences or not sim_sequences:
        print(f"⚠️  Skipping analysis for {stretch_label}: missing data")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Analyzing mismatch for stretch: {stretch_label}")
    print(f"{'='*80}")
    
    # Step 1: Remove outliers from real data
    if remove_outliers_flag:
        print("  Removing outliers from real data...")
        real_sequences_cleaned, outlier_indices = remove_outliers(real_sequences, z_threshold, remove_per_offset=2)
        print(f"    Removed {len(outlier_indices)} outlier sequences")
        real_sequences = real_sequences_cleaned
    
    # Step 2: Match sequences by force profile
    print("  Matching sequences...")
    matched_pairs = match_sequences_by_force_profile(
        real_sequences, sim_sequences,
        force_tolerance=force_tolerance,
        match_by_order=True,  # Match by order since sim replicates real force profiles
    )
    
    if not matched_pairs:
        print(f"⚠️  No sequences matched for {stretch_label}")
        return {}
    
    print(f"  Matched {len(matched_pairs)} sequence pairs")
    
    # Step 3: Compute error for each sequence pair
    print("  Computing error for each sequence pair...")
    sequence_errors = []
    for real_seq, sim_seq, match_score in matched_pairs:
        error = compute_sequence_error(real_seq, sim_seq, normalize=normalize)
        if error is not None:
            error['force_profile_match_score'] = float(match_score)
            sequence_errors.append(error)
    
    if not sequence_errors:
        print(f"⚠️  No valid sequence errors computed for {stretch_label}")
        return {}
    
    print(f"  Computed errors for {len(sequence_errors)} sequence pairs")
    
    # Step 4: Aggregate results
    # Average RMSE across all sequences
    overall_rmse_mean = np.mean([e['overall_rmse'] for e in sequence_errors])
    overall_rmse_std = np.std([e['overall_rmse'] for e in sequence_errors])
    
    # Average MAE across all sequences
    mae_mean = np.mean([e['mae'] for e in sequence_errors])
    mae_std = np.std([e['mae'] for e in sequence_errors])
    
    # Per-sensor average RMSE
    sensor_rmse_means = []
    sensor_rmse_stds = []
    for sensor in range(15):
        sensor_rmses = [e['sensor_rmse'][sensor] for e in sequence_errors]
        sensor_rmse_means.append(np.mean(sensor_rmses))
        sensor_rmse_stds.append(np.std(sensor_rmses))
    
    # Group by offset
    errors_by_offset = {}
    for error in sequence_errors:
        offset = error['offset']
        if offset not in errors_by_offset:
            errors_by_offset[offset] = []
        errors_by_offset[offset].append(error)
    
    # Compile results
    results = {
        'stretch_label': stretch_label,
        'n_real_sequences': len(real_sequences),
        'n_sim_sequences': len(sim_sequences),
        'n_matched_pairs': len(matched_pairs),
        'n_valid_errors': len(sequence_errors),
        'normalization': {'method': 'standardize' if normalize else 'none'},
        'overall_rmse_mean': float(overall_rmse_mean),
        'overall_rmse_std': float(overall_rmse_std),
        'mae_mean': float(mae_mean),
        'mae_std': float(mae_std),
        'sensor_rmse_mean': [float(x) for x in sensor_rmse_means],
        'sensor_rmse_std': [float(x) for x in sensor_rmse_stds],
        'best_sensor': int(np.argmin(sensor_rmse_means)) + 1,
        'worst_sensor': int(np.argmax(sensor_rmse_means)) + 1,
        'errors_by_offset': {
            offset: {
                'count': len(errors),
                'mean_rmse': float(np.mean([e['overall_rmse'] for e in errors])),
                'std_rmse': float(np.std([e['overall_rmse'] for e in errors])),
            }
            for offset, errors in errors_by_offset.items()
        },
        'sequence_errors': sequence_errors,  # Detailed per-sequence errors
    }
    
    # Step 5: Create plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"mismatch_analysis_{stretch_label}.png"
    create_mismatch_plots_sequence_based(
        matched_pairs,
        results,
        stretch_label,
        plot_path,
        normalize=normalize,
    )
    
    # Step 6: Create matching verification plots
    verification_plot_path = output_dir / f"matching_verification_{stretch_label}.png"
    create_matching_verification_plots(
        matched_pairs,
        stretch_label,
        verification_plot_path,
        n_examples=6,
    )
    
    print(f"✅ Analysis complete for {stretch_label}")
    print(f"   Matched pairs: {len(matched_pairs)}")
    print(f"   Mean RMSE: {overall_rmse_mean:.4f} ± {overall_rmse_std:.4f}")
    print(f"   Mean MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"   Best sensor: {results['best_sensor']} (RMSE: {sensor_rmse_means[results['best_sensor']-1]:.4f})")
    print(f"   Worst sensor: {results['worst_sensor']} (RMSE: {sensor_rmse_means[results['worst_sensor']-1]:.4f})")
    print(f"   Plot saved: {plot_path}")
    
    return results


def create_mismatch_plots_sequence_based(
    matched_pairs: List[Tuple[Dict, Dict, float]],
    results: Dict,
    stretch_label: str,
    output_path: Path,
    normalize: bool = True,
):
    """Create plots based on sequence-by-sequence comparison.
    
    Shows DIFFERENCE (Real - Sim) for each channel (Bx, By, Bz) separately.
    
    If normalize=True: Uses StandardScaler (zero mean, unit variance) on combined real+sim data.
                       RMSE is in standard deviations. RMSE ~1.89 = ~1.89 std dev difference.
    If normalize=False: Uses raw magnetic units. RMSE is in original sensor units.
    """
    # Aggregate all magnetic data (both normalized and raw for comparison)
    all_real_mag_norm = []
    all_sim_mag_norm = []
    all_real_mag_raw = []
    all_sim_mag_raw = []
    all_fz = []
    
    for real_seq, sim_seq, _ in matched_pairs:
        real_mag = real_seq['stretchmagtec']
        sim_mag = sim_seq['stretchmagtec']
        real_fz = np.abs(real_seq['fz'])
        
        min_len = min(len(real_mag), len(sim_mag))
        if min_len < 10:
            continue
        
        real_mag = real_mag[:min_len]
        sim_mag = sim_mag[:min_len]
        real_fz = real_fz[:min_len]
        
        # Store raw data
        all_real_mag_raw.append(real_mag)
        all_sim_mag_raw.append(sim_mag)
        
        # Normalize if requested
        if normalize:
            real_flat = real_mag.reshape(-1, 45)
            sim_flat = sim_mag.reshape(-1, 45)
            combined = np.vstack([real_flat, sim_flat])
            scaler = StandardScaler()
            scaler.fit(combined)
            real_norm = scaler.transform(real_flat).reshape(min_len, 15, 3)
            sim_norm = scaler.transform(sim_flat).reshape(min_len, 15, 3)
        else:
            real_norm = real_mag
            sim_norm = sim_mag
        
        all_real_mag_norm.append(real_norm)
        all_sim_mag_norm.append(sim_norm)
        all_fz.append(real_fz)
    
    # Create time axis (normalized to 0-1)
    n_samples = len(all_real_mag_norm[0])
    time_axis = np.linspace(0, 1, n_samples)
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
    
    # Main title with RMSE explanation
    if normalize:
        rmse_units = "standard deviations"
        rmse_explanation = (
            f"Normalization: StandardScaler (zero mean, unit variance) applied to combined real+sim data.\n"
            f"RMSE ~{results['overall_rmse_mean']:.2f} = average difference of ~{results['overall_rmse_mean']:.2f} std dev (relatively high mismatch).\n"
            f"Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately."
        )
    else:
        rmse_units = "raw magnetic units"
        # Calculate raw stats for explanation
        real_raw_all = np.concatenate(all_real_mag_raw, axis=0)
        sim_raw_all = np.concatenate(all_sim_mag_raw, axis=0)
        real_mean = np.mean(real_raw_all)
        sim_mean = np.mean(sim_raw_all)
        real_std = np.std(real_raw_all)
        sim_std = np.std(sim_raw_all)
        rmse_explanation = (
            f"No normalization: Raw magnetic sensor units.\n"
            f"Real data: mean={real_mean:.1f}, std={real_std:.1f} | "
            f"Sim data: mean={sim_mean:.1f}, std={sim_std:.1f}\n"
            f"Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately."
        )
    
    fig.suptitle(
        f'Sim2Real Magnetic Mismatch Analysis - {stretch_label}\n'
        f'Mean RMSE: {results["overall_rmse_mean"]:.4f} ± {results["overall_rmse_std"]:.4f} ({rmse_units})\n'
        f'{rmse_explanation}',
        fontsize=12, fontweight='bold'
    )
    
    # Plots 1-15: Per-sensor comparison showing DIFFERENCE (real - sim) for each channel
    channel_names = ['Bx', 'By', 'Bz']
    colors = ['r', 'g', 'b']
    
    for sensor in range(15):
        row = sensor // 3
        col = sensor % 3
        ax = fig.add_subplot(gs[row, col])
        
        for channel in range(3):
            # Average across all sequences for this sensor/channel
            real_ch = np.mean([mag[:, sensor, channel] for mag in all_real_mag_norm], axis=0)
            sim_ch = np.mean([mag[:, sensor, channel] for mag in all_sim_mag_norm], axis=0)
            
            # Calculate DIFFERENCE (real - sim)
            diff_ch = real_ch - sim_ch
            
            # Plot the difference
            ax.plot(time_axis, diff_ch, color=colors[channel], linestyle='-',
                   linewidth=2, alpha=0.8, label=f'{channel_names[channel]} diff (Real-Sim)')
        
        # Add zero line for reference
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        sensor_rmse = results['sensor_rmse_mean'][sensor]
        ax.set_title(f'Sensor {sensor+1}\nRMSE: {sensor_rmse:.3f} std dev', fontsize=10, fontweight='bold')
        ax.set_xlabel('Normalized Time [0-1]', fontsize=9)
        ax.set_ylabel('Difference (Real - Sim) [normalized]', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Plot 16: RMSE distribution across sequences
    ax_rmse = fig.add_subplot(gs[5, 0])
    rmses = [e['overall_rmse'] for e in results['sequence_errors']]
    ax_rmse.hist(rmses, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax_rmse.axvline(results['overall_rmse_mean'], color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {results["overall_rmse_mean"]:.4f}')
    ax_rmse.set_xlabel(f'RMSE per Sequence [{rmse_units}]', fontsize=11)
    ax_rmse.set_ylabel('Frequency', fontsize=11)
    ax_rmse.set_title('RMSE Distribution Across Sequences', fontsize=12, fontweight='bold')
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3, axis='y')
    
    # Plot 17: Per-sensor average RMSE
    ax_sensor = fig.add_subplot(gs[5, 1])
    sensor_ids = range(1, 16)
    ax_sensor.bar(sensor_ids, results['sensor_rmse_mean'], yerr=results['sensor_rmse_std'],
                  color='orange', alpha=0.7, capsize=5)
    ax_sensor.axhline(results['overall_rmse_mean'], color='r', linestyle='--', linewidth=2,
                      label=f'Overall Mean: {results["overall_rmse_mean"]:.4f}')
    ax_sensor.set_xlabel('Sensor ID', fontsize=11)
    ax_sensor.set_ylabel(f'Mean RMSE [{rmse_units}]', fontsize=11)
    ax_sensor.set_title('Per-Sensor Average RMSE', fontsize=12, fontweight='bold')
    ax_sensor.legend()
    ax_sensor.grid(True, alpha=0.3, axis='y')
    
    # Plot 18: RMSE by offset
    ax_offset = fig.add_subplot(gs[5, 2])
    offsets = list(results['errors_by_offset'].keys())
    offset_means = [results['errors_by_offset'][o]['mean_rmse'] for o in offsets]
    offset_stds = [results['errors_by_offset'][o]['std_rmse'] for o in offsets]
    ax_offset.bar(offsets, offset_means, yerr=offset_stds, color='green', alpha=0.7, capsize=5)
    ax_offset.set_xlabel('Offset', fontsize=11)
    ax_offset.set_ylabel(f'Mean RMSE [{rmse_units}]', fontsize=11)
    ax_offset.set_title('RMSE by Offset', fontsize=12, fontweight='bold')
    ax_offset.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {output_path}")
    
    # Also create a plot with RAW (non-normalized) data for comparison
    if normalize:
        raw_output_path = output_path.parent / f"{output_path.stem}_raw{output_path.suffix}"
        fig_raw = plt.figure(figsize=(24, 20))
        gs_raw = fig_raw.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        fig_raw.suptitle(
            f'Sim2Real Magnetic Mismatch Analysis - {stretch_label} (RAW DATA, NO NORMALIZATION)\n'
            f'Shows absolute difference in original sensor units.\n'
            f'Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately.',
            fontsize=12, fontweight='bold'
        )
        
        for sensor in range(15):
            row = sensor // 3
            col = sensor % 3
            ax = fig_raw.add_subplot(gs_raw[row, col])
            
            for channel in range(3):
                # Average across all sequences for this sensor/channel (RAW data)
                real_ch_raw = np.mean([mag[:, sensor, channel] for mag in all_real_mag_raw], axis=0)
                sim_ch_raw = np.mean([mag[:, sensor, channel] for mag in all_sim_mag_raw], axis=0)
                
                # Calculate DIFFERENCE (real - sim) in RAW units
                diff_ch_raw = real_ch_raw - sim_ch_raw
                
                ax.plot(time_axis, diff_ch_raw, color=colors[channel], linestyle='-',
                       linewidth=2, alpha=0.8, label=f'{channel_names[channel]} diff (Real-Sim)')
            
            ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_title(f'Sensor {sensor+1} (RAW)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Normalized Time [0-1]', fontsize=9)
            ax.set_ylabel('Difference (Real - Sim) [raw units]', fontsize=9)
            ax.grid(True, alpha=0.3)
            if sensor == 0:
                ax.legend(fontsize=8, loc='best')
        
        # Summary stats for raw data
        real_raw_all = np.concatenate(all_real_mag_raw, axis=0)
        sim_raw_all = np.concatenate(all_sim_mag_raw, axis=0)
        diff_raw_all = real_raw_all - sim_raw_all
        raw_rmse = np.sqrt(np.mean(diff_raw_all**2))
        raw_mae = np.mean(np.abs(diff_raw_all))
        
        # Add summary text
        ax_summary = fig_raw.add_subplot(gs_raw[5, :])
        ax_summary.axis('off')
        summary_text = (
            f"Raw Data Statistics:\n"
            f"  Real data: mean={np.mean(real_raw_all):.1f}, std={np.std(real_raw_all):.1f}, range=[{np.min(real_raw_all):.1f}, {np.max(real_raw_all):.1f}]\n"
            f"  Sim data: mean={np.mean(sim_raw_all):.1f}, std={np.std(sim_raw_all):.1f}, range=[{np.min(sim_raw_all):.1f}, {np.max(sim_raw_all):.1f}]\n"
            f"  Difference (Real-Sim): RMSE={raw_rmse:.1f}, MAE={raw_mae:.1f}"
        )
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', 
                       fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.savefig(raw_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved RAW plot: {raw_output_path}")


def create_matching_verification_plots(
    matched_pairs: List[Tuple[Dict, Dict, float]],
    stretch_label: str,
    output_path: Path,
    n_examples: int = 6,
):
    """Create plots showing matched sequences to verify correct pairing.
    
    Shows force profiles and magnetic data for a few matched pairs to verify
    that the matching algorithm is working correctly.
    """
    # Select a few examples from different offsets
    examples_by_offset = {}
    for real_seq, sim_seq, match_score in matched_pairs:
        offset = real_seq.get('offset', 'unknown')
        if offset not in examples_by_offset:
            examples_by_offset[offset] = []
        if len(examples_by_offset[offset]) < 2:  # 2 examples per offset
            examples_by_offset[offset].append((real_seq, sim_seq, match_score))
    
    # Flatten and take first n_examples
    all_examples = []
    for offset_examples in examples_by_offset.values():
        all_examples.extend(offset_examples)
    examples = all_examples[:n_examples]
    
    print(f"   Selected {len(examples)} examples for matching verification plot")
    print(f"   Examples by offset: {[(e[0].get('offset', 'unknown'), e[2]) for e in examples]}")
    
    if not examples:
        print(f"   ⚠️  No examples to plot for matching verification")
        return
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Matching Verification: Real vs Simulated Sequences - {stretch_label}\n'
        f'Shows {len(examples)} matched pairs to verify correct pairing',
        fontsize=14, fontweight='bold'
    )
    
    for idx, (real_seq, sim_seq, match_score) in enumerate(examples):
        offset = real_seq.get('offset', 'unknown')
        real_fz = np.abs(real_seq['fz'])
        sim_fz = np.abs(sim_seq['fz'])
        real_mag = real_seq['stretchmagtec']
        sim_mag = sim_seq['stretchmagtec']
        
        # Align lengths
        min_len = min(len(real_fz), len(sim_fz))
        real_fz = real_fz[:min_len]
        sim_fz = sim_fz[:min_len]
        real_mag = real_mag[:min_len]
        sim_mag = sim_mag[:min_len]
        
        time_axis = np.arange(min_len) / 100.0  # Assuming 100 Hz
        
        # Plot 1: Force profiles
        ax1 = axes[idx, 0]
        ax1.plot(time_axis, real_fz, 'b-', linewidth=2, alpha=0.7, label='Real Fz')
        ax1.plot(time_axis, sim_fz, 'r--', linewidth=2, alpha=0.7, label='Sim Fz')
        ax1.set_xlabel('Time [s]', fontsize=10)
        ax1.set_ylabel('Force Fz [N]', fontsize=10)
        ax1.set_title(f'Example {idx+1} - {offset}\nForce Match Score: {match_score:.4f} N', 
                      fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Magnetic channel 8 Z (most sensitive)
        ax2 = axes[idx, 1]
        real_ch8_z = real_mag[:, 7, 2]  # Sensor 8 (index 7), channel Z (index 2)
        sim_ch8_z = sim_mag[:, 7, 2]
        ax2.plot(time_axis, real_ch8_z, 'b-', linewidth=2, alpha=0.7, label='Real Mag Ch8 Z')
        ax2.plot(time_axis, sim_ch8_z, 'r--', linewidth=2, alpha=0.7, label='Sim Mag Ch8 Z')
        ax2.set_xlabel('Time [s]', fontsize=10)
        ax2.set_ylabel('Magnetic Field [raw units]', fontsize=10)
        ax2.set_title(f'Magnetic Sensor 8 (Z-component)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difference in magnetic channel 8 Z
        ax3 = axes[idx, 2]
        diff_ch8_z = real_ch8_z - sim_ch8_z
        ax3.plot(time_axis, diff_ch8_z, 'g-', linewidth=2, alpha=0.8, label='Real - Sim')
        ax3.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Time [s]', fontsize=10)
        ax3.set_ylabel('Difference [raw units]', fontsize=10)
        rmse_ch8_z = np.sqrt(np.mean(diff_ch8_z**2))
        ax3.set_title(f'Difference (Real - Sim)\nRMSE: {rmse_ch8_z:.1f}', 
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved matching verification plot: {output_path}")


def create_mismatch_plots(
    real_magnetic: np.ndarray,  # [n_samples, 15, 3]
    sim_magnetic: np.ndarray,  # [n_samples, 15, 3]
    force_axis: np.ndarray,  # [n_samples]
    results: Dict,
    stretch_label: str,
    output_path: Path,
):
    """Create comprehensive mismatch analysis plots."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # Main title
    mean_offset = results["offset_and_scaling"]["overall_offset"].get("mean", 0)
    mean_scaling = results["offset_and_scaling"]["overall_scaling"].get("mean", 1.0)
    fig.suptitle(
        f'Sim2Real Magnetic Mismatch Analysis - {stretch_label}\n'
        f'Overall RMSE: {results["overall_rmse"]:.4f}, '
        f'Mean Offset: {mean_offset:.4f}, '
        f'Mean Scaling: {mean_scaling:.4f}',
        fontsize=14, fontweight='bold'
    )
    
    channel_names = ['Bx', 'By', 'Bz']
    colors = ['r', 'g', 'b']
    
    # Plot 1-15: Per-sensor comparison
    for sensor in range(15):
        row = sensor // 5
        col = sensor % 5
        ax = fig.add_subplot(gs[row, col])
        
        for channel in range(3):
            real_ch = real_magnetic[:, sensor, channel]
            sim_ch = sim_magnetic[:, sensor, channel]
            
            ax.plot(force_axis, real_ch, color=colors[channel], linestyle='-',
                   linewidth=2, alpha=0.7, label=f'Real {channel_names[channel]}')
            ax.plot(force_axis, sim_ch, color=colors[channel], linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Sim {channel_names[channel]}')
        
        sensor_rmse = results['sensor_rmse'][sensor]
        ax.set_title(f'Sensor {sensor+1}\nRMSE: {sensor_rmse:.3f}', fontsize=10)
        ax.set_xlabel('Force [N]', fontsize=9)
        ax.set_ylabel('Magnetic Field [normalized]', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Plot 16: RMSE per sensor
    ax_rmse = fig.add_subplot(gs[3, 0])
    ax_rmse.bar(range(1, 16), results['sensor_rmse'], color='steelblue', alpha=0.7)
    ax_rmse.axhline(results['overall_rmse'], color='r', linestyle='--', linewidth=2, label='Overall RMSE')
    ax_rmse.set_xlabel('Sensor ID', fontsize=10)
    ax_rmse.set_ylabel('RMSE', fontsize=10)
    ax_rmse.set_title('RMSE per Sensor', fontsize=11, fontweight='bold')
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3, axis='y')
    
    # Plot 17: Offset distribution
    ax_offset = fig.add_subplot(gs[3, 1])
    offsets = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        for ch_key, ch_data in sensor_data.items():
            offsets.append(ch_data['offset'])
    if offsets:
        ax_offset.hist(offsets, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax_offset.axvline(np.mean(offsets), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(offsets):.4f}')
        ax_offset.set_xlabel('Offset', fontsize=10)
        ax_offset.set_ylabel('Frequency', fontsize=10)
        ax_offset.set_title('Offset Distribution', fontsize=11, fontweight='bold')
        ax_offset.legend()
        ax_offset.grid(True, alpha=0.3, axis='y')
    
    # Plot 18: Scaling distribution
    ax_scale = fig.add_subplot(gs[3, 2])
    scalings = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        for ch_key, ch_data in sensor_data.items():
            scalings.append(ch_data['scaling'])
    if scalings:
        ax_scale.hist(scalings, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax_scale.axvline(np.mean(scalings), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scalings):.4f}')
        ax_scale.axvline(1.0, color='k', linestyle=':', linewidth=2, label='Ideal: 1.0')
        ax_scale.set_xlabel('Scaling Factor', fontsize=10)
        ax_scale.set_ylabel('Frequency', fontsize=10)
        ax_scale.set_title('Scaling Factor Distribution', fontsize=11, fontweight='bold')
        ax_scale.legend()
        ax_scale.grid(True, alpha=0.3, axis='y')
    
    # Plot 19: Correlation per sensor
    ax_corr = fig.add_subplot(gs[3, 3])
    correlations = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        sensor_corrs = []
        for ch_key, ch_data in sensor_data.items():
            sensor_corrs.append(ch_data['correlation'])
        if sensor_corrs:
            correlations.append(np.mean(sensor_corrs))
    if correlations:
        ax_corr.bar(range(1, len(correlations) + 1), correlations, color='purple', alpha=0.7)
        ax_corr.set_xlabel('Sensor ID', fontsize=10)
        ax_corr.set_ylabel('Correlation', fontsize=10)
        ax_corr.set_title('Correlation per Sensor', fontsize=11, fontweight='bold')
        ax_corr.set_ylim([0, 1])
        ax_corr.grid(True, alpha=0.3, axis='y')
    
    # Plot 20: Scatter plot (real vs sim) for best sensor
    ax_scatter = fig.add_subplot(gs[3, 4])
    best_sensor_idx = results['best_sensor'] - 1
    real_best = real_magnetic[:, best_sensor_idx, :].reshape(-1)
    sim_best = sim_magnetic[:, best_sensor_idx, :].reshape(-1)
    ax_scatter.scatter(real_best, sim_best, alpha=0.5, s=20)
    # Add diagonal line
    min_val = min(np.min(real_best), np.min(sim_best))
    max_val = max(np.max(real_best), np.max(sim_best))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax_scatter.set_xlabel('Real Magnetic Field', fontsize=10)
    ax_scatter.set_ylabel('Simulated Magnetic Field', fontsize=10)
    ax_scatter.set_title(f'Best Sensor {results["best_sensor"]} Scatter', fontsize=11, fontweight='bold')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze mismatch between real and simulated magnetic sensor data'
    )
    parser.add_argument(
        '--real-data',
        type=Path,
        required=True,
        help='Path to real data HDF5 file (cleaned)'
    )
    parser.add_argument(
        '--sim-data',
        type=Path,
        required=True,
        help='Path to simulation HDF5 file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/sim2real_mismatch'),
        help='Output directory for plots and metrics JSON'
    )
    parser.add_argument(
        '--stretch-label',
        type=str,
        default=None,
        help='Stretch label to filter (e.g., 000pct). If None, uses all sequences.'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize data before comparison (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Do not normalize data'
    )
    parser.add_argument(
        '--normalization-method',
        type=str,
        choices=['standardize', 'minmax', 'separate_minmax'],
        default='standardize',
        help='Normalization method (default: standardize)'
    )
    parser.add_argument(
        '--force-tolerance',
        type=float,
        default=0.1,
        help='Maximum allowed difference in force profiles for matching (default: 0.1)'
    )
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        default=True,
        help='Remove outliers from real data (default: True)'
    )
    parser.add_argument(
        '--no-remove-outliers',
        dest='remove_outliers',
        action='store_false',
        help='Do not remove outliers'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier removal (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    real_sequences = load_real_sequences(args.real_data, args.stretch_label)
    sim_sequences = load_sim_sequences(args.sim_data)
    
    if not real_sequences or not sim_sequences:
        print("❌ Error: Could not load sufficient data")
        return
    
    # Infer stretch label if not provided
    if args.stretch_label is None:
        # Try to infer from real sequences
        if real_sequences:
            stretch_label = real_sequences[0].get('stretch_label', 'unknown')
            if stretch_label != 'unknown':
                args.stretch_label = stretch_label.split('_')[-1] if '_' in stretch_label else stretch_label
        if args.stretch_label is None:
            args.stretch_label = 'unknown'
    
    # Analyze mismatch
    results = analyze_mismatch(
        real_sequences,
        sim_sequences,
        args.stretch_label,
        args.output_dir,
        normalize=args.normalize,
        normalization_method=args.normalization_method,
        remove_outliers_flag=args.remove_outliers,
        z_threshold=args.z_threshold,
        force_tolerance=args.force_tolerance,
    )
    
    # Save results
    if results:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = output_dir / f'mismatch_metrics_{args.stretch_label}.json'
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved metrics to: {metrics_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Stretch: {args.stretch_label}")
        print(f"Matched pairs: {results['n_matched_pairs']}")
        print(f"Mean RMSE: {results['overall_rmse_mean']:.4f} ± {results['overall_rmse_std']:.4f}")
        print(f"Mean MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"Best sensor: {results['best_sensor']} (RMSE: {results['sensor_rmse_mean'][results['best_sensor']-1]:.4f})")
        print(f"Worst sensor: {results['worst_sensor']} (RMSE: {results['sensor_rmse_mean'][results['worst_sensor']-1]:.4f})")


if __name__ == '__main__':
    main()


Sim2Real Magnetic Mismatch Analysis

This script compares magnetic sensor readings from real robot experiments with
simulated magnetic field data. The analysis focuses on:
1. Offset differences (bias between real and simulated)
2. Scaling differences (gain/amplitude mismatch)
3. Normalization to account for unitless sensors
4. Force-matched comparison (same force levels)

The goal is to identify systematic differences that could be corrected to improve
simulation-to-reality transfer.

Usage:
    python3 src/training/sim2real_magnetic_mismatch.py \
        --real-data data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/data/test_50_000_cleaned.h5 \
        --sim-data data/simulation/test5/*_stretch_0.h5 \
        --output-dir analysis/sim2real_mismatch
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train_single_point_models import load_sequences_from_h5, remove_outliers
from training.train_simulation_positions import load_simulation_file, convert_to_sequences, position_key


def load_real_sequences(h5_path: Path, stretch_label: Optional[str] = None) -> List[Dict]:
    """Load real robot data sequences from HDF5 file.
    
    Args:
        h5_path: Path to real data HDF5 file
        stretch_label: Optional stretch label to filter (e.g., '000pct')
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec', 'forces', 'fz', etc.
    """
    sequences = load_sequences_from_h5(h5_path)
    
    if stretch_label:
        # Filter by stretch label
        filtered = [s for s in sequences if s.get('stretch_label', '').endswith(stretch_label)]
        if not filtered:
            print(f"⚠️  Warning: No sequences found for stretch {stretch_label} in {h5_path}")
            return []
        sequences = filtered
    
    print(f"✅ Loaded {len(sequences)} real sequences from {h5_path.name}")
    return sequences


def load_sim_sequences(h5_path: Path) -> List[Dict]:
    """Load simulated data sequences from HDF5 file.
    
    Args:
        h5_path: Path to simulation HDF5 file
    
    Returns:
        List of sequence dictionaries with 'stretchmagtec' (magnetic), 'fz', etc.
    """
    sim_data = load_simulation_file(h5_path)
    
    # Convert continuous data to sequences
    position_labels = None
    if sim_data.get('indenter') is not None:
        indenter = sim_data['indenter']
        # Create position labels from XY coordinates
        position_labels = []
        for i in range(len(indenter)):
            key, (x_mm, y_mm) = position_key(indenter[i, :2])
            position_labels.append(key)
    else:
        position_labels = ["x+00.0mm_y+00.0mm"] * len(sim_data['magnetic'])
    
    stretch_label = sim_data.get('stretch_label', 'unknown')
    sequences = convert_to_sequences(
        magnetic=sim_data['magnetic'],
        forces=sim_data['forces'],
        indenter=sim_data.get('indenter'),
        position_labels=position_labels,
        stretch_label=stretch_label,
    )
    
    print(f"✅ Loaded {len(sequences)} simulated sequences from {h5_path.name}")
    return sequences


def match_sequences_by_force_profile(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    force_tolerance: float = 1.0,
    match_by_order: bool = True,
) -> List[Tuple[Dict, Dict, float]]:
    """Match real and simulated sequences based on similar force profiles.
    
    The simulation test replicated the force profiles from the real test,
    so we can match sequences by comparing their force profiles or by order.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        force_tolerance: Maximum allowed difference in force profiles for matching (N)
        match_by_order: If True, match sequences in order (same offset, same order)
                        If False, find best match by force profile similarity
    
    Returns:
        List of tuples (real_seq, sim_seq, match_score) where match_score is the
        RMSE between the two force profiles (lower is better)
    """
    matched_pairs = []
    
    # Group sequences by offset
    real_by_offset = {}
    for seq in real_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in real_by_offset:
            real_by_offset[offset] = []
        real_by_offset[offset].append(seq)
    
    sim_by_offset = {}
    for seq in sim_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in sim_by_offset:
            sim_by_offset[offset] = []
        sim_by_offset[offset].append(seq)
    
    print(f"  Matching sequences...")
    print(f"  Real offsets: {list(real_by_offset.keys())}")
    print(f"  Sim offsets: {list(sim_by_offset.keys())}")
    print(f"  Match method: {'by order' if match_by_order else 'by force profile'}")
    
    # Match offset by offset
    for offset in real_by_offset.keys():
        if offset not in sim_by_offset:
            print(f"  ⚠️  Warning: Offset '{offset}' not found in sim data, skipping")
            continue
        
        real_seqs = real_by_offset[offset]
        sim_seqs = sim_by_offset[offset]
        
        print(f"  Matching offset '{offset}': {len(real_seqs)} real, {len(sim_seqs)} sim sequences")
        
        if match_by_order:
            # Match sequences in order (assuming they're already in the same order)
            n_pairs = min(len(real_seqs), len(sim_seqs))
            for i in range(n_pairs):
                real_seq = real_seqs[i]
                sim_seq = sim_seqs[i]
                
                # Calculate force profile match score for reference
                real_fz = np.abs(real_seq['fz'])
                sim_fz = np.abs(sim_seq['fz'])
                min_len = min(len(real_fz), len(sim_fz))
                if min_len >= 10:
                    real_fz_short = real_fz[:min_len]
                    sim_fz_short = sim_fz[:min_len]
                    match_score = np.sqrt(np.mean((real_fz_short - sim_fz_short)**2))
                else:
                    match_score = 0.0
                
                matched_pairs.append((real_seq, sim_seq, match_score))
        else:
            # Find best match by force profile
            used_sim_indices = set()
            for real_seq in real_seqs:
                real_fz = np.abs(real_seq['fz'])
                
                best_match = None
                best_score = float('inf')
                best_idx = -1
                
                for idx, sim_seq in enumerate(sim_seqs):
                    if idx in used_sim_indices:
                        continue
                    
                    sim_fz = np.abs(sim_seq['fz'])
                    min_len = min(len(real_fz), len(sim_fz))
                    if min_len < 10:
                        continue
                    
                    real_fz_short = real_fz[:min_len]
                    sim_fz_short = sim_fz[:min_len]
                    score = np.sqrt(np.mean((real_fz_short - sim_fz_short)**2))
                    
                    if score < best_score:
                        best_score = score
                        best_match = sim_seq
                        best_idx = idx
                
                if best_match is not None and best_score < force_tolerance:
                    matched_pairs.append((real_seq, best_match, best_score))
                    used_sim_indices.add(best_idx)
                else:
                    if len(matched_pairs) < 5:  # Only print first few warnings
                        print(f"    ⚠️  No good match found (best score: {best_score:.4f} N)")
    
    print(f"  Matched {len(matched_pairs)} sequence pairs")
    return matched_pairs


def extract_force_matched_samples(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    force_range: Tuple[float, float] = (0.5, 3.0),
    force_bins: int = 50,
    min_samples_per_bin: int = 5,
    match_by_offset: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Extract magnetic samples matched by force level.
    
    DEPRECATED: Use match_sequences_by_force_profile instead.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        force_range: (min_force, max_force) range to consider
        force_bins: Number of force bins for matching
        min_samples_per_bin: Minimum samples required per bin
        match_by_offset: If True, match sequences by offset (center vs center, etc.)
    
    Returns:
        Tuple of (real_magnetic, real_fz, sim_magnetic, sim_fz, info_dict)
        All arrays are [total_samples, 15, 3] for magnetic, [total_samples] for fz
        info_dict contains debug information about what was compared
    """
    info = {
        'real_offsets': {},
        'sim_offsets': {},
        'matched_offsets': [],
    }
    
    # Group sequences by offset
    real_by_offset = {}
    for seq in real_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in real_by_offset:
            real_by_offset[offset] = []
        real_by_offset[offset].append(seq)
        info['real_offsets'][offset] = info['real_offsets'].get(offset, 0) + 1
    
    sim_by_offset = {}
    for seq in sim_sequences:
        offset = seq.get('offset', 'unknown')
        if offset not in sim_by_offset:
            sim_by_offset[offset] = []
        sim_by_offset[offset].append(seq)
        info['sim_offsets'][offset] = info['sim_offsets'].get(offset, 0) + 1
    
    print(f"  Real data offsets: {info['real_offsets']}")
    print(f"  Sim data offsets: {info['sim_offsets']}")
    
    # Collect all samples from real sequences
    all_real_magnetic = []
    all_real_fz = []
    all_sim_magnetic = []
    all_sim_fz = []
    
    if match_by_offset:
        # Match offset by offset
        for offset in real_by_offset.keys():
            if offset not in sim_by_offset:
                print(f"  ⚠️  Warning: Offset '{offset}' not found in sim data, skipping")
                continue
            
            info['matched_offsets'].append(offset)
            print(f"  Matching offset: {offset} (real: {len(real_by_offset[offset])} seqs, sim: {len(sim_by_offset[offset])} seqs)")
            
            real_samples_count = 0
            sim_samples_count = 0
            
            for seq in real_by_offset[offset]:
                fz = np.abs(seq['fz'])
                mask = (fz >= force_range[0]) & (fz <= force_range[1])
                if np.sum(mask) > min_samples_per_bin:
                    all_real_magnetic.append(seq['stretchmagtec'][mask])
                    all_real_fz.append(fz[mask])
                    real_samples_count += np.sum(mask)
            
            for seq in sim_by_offset[offset]:
                fz = np.abs(seq['fz'])
                mask = (fz >= force_range[0]) & (fz <= force_range[1])
                if np.sum(mask) > min_samples_per_bin:
                    all_sim_magnetic.append(seq['stretchmagtec'][mask])
                    all_sim_fz.append(fz[mask])
                    sim_samples_count += np.sum(mask)
            
            print(f"    Offset {offset}: collected {real_samples_count} real samples, {sim_samples_count} sim samples")
    else:
        # Collect all samples without offset matching
        for seq in real_sequences:
            fz = np.abs(seq['fz'])
            mask = (fz >= force_range[0]) & (fz <= force_range[1])
            if np.sum(mask) > min_samples_per_bin:
                all_real_magnetic.append(seq['stretchmagtec'][mask])
                all_real_fz.append(fz[mask])
    
    # Collect all samples from sim sequences (if not already done in match_by_offset)
    if not match_by_offset:
        for seq in sim_sequences:
            fz = np.abs(seq['fz'])
            mask = (fz >= force_range[0]) & (fz <= force_range[1])
            if np.sum(mask) > min_samples_per_bin:
                all_sim_magnetic.append(seq['stretchmagtec'][mask])
                all_sim_fz.append(fz[mask])
    
    if not all_real_magnetic or not all_sim_magnetic:
        print(f"  ⚠️  Error: Insufficient data after filtering")
        print(f"     Real samples: {len(all_real_magnetic)} sequences")
        print(f"     Sim samples: {len(all_sim_magnetic)} sequences")
        raise ValueError("Insufficient data after filtering")
    
    # Concatenate
    real_magnetic = np.concatenate(all_real_magnetic, axis=0)  # [N, 15, 3]
    real_fz = np.concatenate(all_real_fz)  # [N]
    sim_magnetic = np.concatenate(all_sim_magnetic, axis=0)  # [M, 15, 3]
    sim_fz = np.concatenate(all_sim_fz)  # [M]
    
    # Create force bins and match samples
    force_edges = np.linspace(force_range[0], force_range[1], force_bins + 1)
    
    matched_real_magnetic = []
    matched_real_fz = []
    matched_sim_magnetic = []
    matched_sim_fz = []
    
    for i in range(force_bins):
        f_min, f_max = force_edges[i], force_edges[i + 1]
        
        # Find samples in this bin for both datasets
        real_mask = (real_fz >= f_min) & (real_fz < f_max)
        sim_mask = (sim_fz >= f_min) & (sim_fz < f_max)
        
        if np.sum(real_mask) >= min_samples_per_bin and np.sum(sim_mask) >= min_samples_per_bin:
            # Take mean per bin to reduce noise
            matched_real_magnetic.append(np.mean(real_magnetic[real_mask], axis=0))
            matched_real_fz.append((f_min + f_max) / 2)
            matched_sim_magnetic.append(np.mean(sim_magnetic[sim_mask], axis=0))
            matched_sim_fz.append((f_min + f_max) / 2)
    
    if not matched_real_magnetic:
        raise ValueError("No matched samples found after binning")
    
    info['n_matched_bins'] = len(matched_real_magnetic)
    info['real_magnetic_stats'] = {
        'min': float(np.min(matched_real_magnetic)),
        'max': float(np.max(matched_real_magnetic)),
        'mean': float(np.mean(matched_real_magnetic)),
        'std': float(np.std(matched_real_magnetic)),
    }
    info['sim_magnetic_stats'] = {
        'min': float(np.min(matched_sim_magnetic)),
        'max': float(np.max(matched_sim_magnetic)),
        'mean': float(np.mean(matched_sim_magnetic)),
        'std': float(np.std(matched_sim_magnetic)),
    }
    
    print(f"  Real magnetic stats: min={info['real_magnetic_stats']['min']:.2f}, max={info['real_magnetic_stats']['max']:.2f}, mean={info['real_magnetic_stats']['mean']:.2f}, std={info['real_magnetic_stats']['std']:.2f}")
    print(f"  Sim magnetic stats: min={info['sim_magnetic_stats']['min']:.2f}, max={info['sim_magnetic_stats']['max']:.2f}, mean={info['sim_magnetic_stats']['mean']:.2f}, std={info['sim_magnetic_stats']['std']:.2f}")
    
    return (
        np.array(matched_real_magnetic),  # [n_bins, 15, 3]
        np.array(matched_real_fz),  # [n_bins]
        np.array(matched_sim_magnetic),  # [n_bins, 15, 3]
        np.array(matched_sim_fz),  # [n_bins]
        info,
    )


def compute_offset_and_scaling(
    real_magnetic: np.ndarray,  # [n_samples, 15, 3]
    sim_magnetic: np.ndarray,  # [n_samples, 15, 3]
) -> Dict:
    """Compute offset (bias) and scaling (gain) between real and simulated magnetic data.
    
    For each sensor and channel, we fit:
        sim = scale * real + offset
    
    Using least squares: sim = A @ [scale, offset]^T
    
    Args:
        real_magnetic: Real magnetic data [n_samples, 15, 3]
        sim_magnetic: Simulated magnetic data [n_samples, 15, 3]
    
    Returns:
        Dictionary with offset and scaling per sensor/channel
    """
    n_samples, n_sensors, n_channels = real_magnetic.shape
    
    results = {
        'sensor_metrics': {},
        'overall_offset': {},
        'overall_scaling': {},
    }
    
    # Per-sensor, per-channel analysis
    all_offsets = []
    all_scalings = []
    all_correlations = []
    
    for sensor in range(n_sensors):
        sensor_results = {}
        
        for channel in range(n_channels):
            real_ch = real_magnetic[:, sensor, channel]
            sim_ch = sim_magnetic[:, sensor, channel]
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(real_ch) & np.isfinite(sim_ch)
            if np.sum(valid_mask) < 10:
                continue
            
            real_ch = real_ch[valid_mask]
            sim_ch = sim_ch[valid_mask]
            
            # Fit linear model: sim = scale * real + offset
            # Using least squares: [sim] = [real, 1] @ [scale, offset]^T
            A = np.vstack([real_ch, np.ones(len(real_ch))]).T
            scale, offset = np.linalg.lstsq(A, sim_ch, rcond=None)[0]
            
            # Compute correlation
            if np.std(real_ch) > 0 and np.std(sim_ch) > 0:
                corr, _ = stats.pearsonr(real_ch, sim_ch)
            else:
                corr = 0.0
            
            # Compute RMSE
            sim_predicted = scale * real_ch + offset
            rmse = np.sqrt(np.mean((sim_ch - sim_predicted)**2))
            
            sensor_results[f'channel_{channel}'] = {
                'offset': float(offset),
                'scaling': float(scale),
                'correlation': float(corr),
                'rmse': float(rmse),
            }
            
            all_offsets.append(offset)
            all_scalings.append(scale)
            all_correlations.append(corr)
        
        results['sensor_metrics'][f'sensor_{sensor+1}'] = sensor_results
    
    # Overall statistics
    if all_offsets:
        results['overall_offset'] = {
            'mean': float(np.mean(all_offsets)),
            'std': float(np.std(all_offsets)),
            'median': float(np.median(all_offsets)),
            'min': float(np.min(all_offsets)),
            'max': float(np.max(all_offsets)),
        }
    
    if all_scalings:
        results['overall_scaling'] = {
            'mean': float(np.mean(all_scalings)),
            'std': float(np.std(all_scalings)),
            'median': float(np.median(all_scalings)),
            'min': float(np.min(all_scalings)),
            'max': float(np.max(all_scalings)),
        }
    
    if all_correlations:
        results['overall_correlation'] = {
            'mean': float(np.mean(all_correlations)),
            'std': float(np.std(all_correlations)),
            'median': float(np.median(all_correlations)),
        }
    
    return results


def normalize_magnetic_data(
    real_magnetic: np.ndarray,
    sim_magnetic: np.ndarray,
    method: str = 'standardize',
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize magnetic data to account for unitless sensors.
    
    Args:
        real_magnetic: Real magnetic data [n_samples, 15, 3]
        sim_magnetic: Simulated magnetic data [n_samples, 15, 3]
        method: Normalization method ('standardize', 'minmax', 'separate_minmax')
    
    Returns:
        Tuple of (normalized_real, normalized_sim, scaler_info)
    """
    n_samples_real, n_sensors, n_channels = real_magnetic.shape
    n_samples_sim = sim_magnetic.shape[0]
    
    # Flatten to [samples, 45] for normalization
    real_flat = real_magnetic.reshape(n_samples_real, -1)  # [n_samples, 45]
    sim_flat = sim_magnetic.reshape(n_samples_sim, -1)  # [n_samples, 45]
    
    scaler_info = {}
    
    if method == 'standardize':
        # StandardScaler: zero mean, unit variance (applied jointly)
        combined = np.vstack([real_flat, sim_flat])
        scaler = StandardScaler()
        scaler.fit(combined)
        
        real_normalized = scaler.transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = scaler.transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'standardize',
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
        }
    
    elif method == 'minmax':
        # MinMaxScaler: scale to [0, 1] (applied jointly)
        combined = np.vstack([real_flat, sim_flat])
        scaler = MinMaxScaler()
        scaler.fit(combined)
        
        real_normalized = scaler.transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = scaler.transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'minmax',
            'min': scaler.data_min_.tolist(),
            'max': scaler.data_max_.tolist(),
        }
    
    elif method == 'separate_minmax':
        # Separate MinMaxScaler for real and sim (preserves relative differences)
        real_scaler = MinMaxScaler()
        sim_scaler = MinMaxScaler()
        
        real_normalized = real_scaler.fit_transform(real_flat).reshape(n_samples_real, n_sensors, n_channels)
        sim_normalized = sim_scaler.fit_transform(sim_flat).reshape(n_samples_sim, n_sensors, n_channels)
        
        scaler_info = {
            'method': 'separate_minmax',
            'real_min': real_scaler.data_min_.tolist(),
            'real_max': real_scaler.data_max_.tolist(),
            'sim_min': sim_scaler.data_min_.tolist(),
            'sim_max': sim_scaler.data_max_.tolist(),
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return real_normalized, sim_normalized, scaler_info


def compute_sequence_error(
    real_seq: Dict,
    sim_seq: Dict,
    normalize: bool = True,
) -> Dict:
    """Compute error between a real and simulated sequence.
    
    Args:
        real_seq: Real sequence dictionary with 'stretchmagtec' and 'fz'
        sim_seq: Simulated sequence dictionary with 'stretchmagtec' and 'fz'
        normalize: Whether to normalize magnetic data before comparison
    
    Returns:
        Dictionary with error metrics for this sequence pair
    """
    real_mag = real_seq['stretchmagtec']  # [samples, 15, 3]
    sim_mag = sim_seq['stretchmagtec']  # [samples, 15, 3]
    real_fz = np.abs(real_seq['fz'])
    sim_fz = np.abs(sim_seq['fz'])
    
    # Interpolate to same length (use shorter length)
    min_len = min(len(real_mag), len(sim_mag))
    if min_len < 10:
        return None
    
    real_mag = real_mag[:min_len]
    sim_mag = sim_mag[:min_len]
    real_fz = real_fz[:min_len]
    sim_fz = sim_fz[:min_len]
    
    # Normalize if requested
    if normalize:
        # Flatten for normalization
        real_flat = real_mag.reshape(-1, 45)
        sim_flat = sim_mag.reshape(-1, 45)
        
        # Use joint standardization
        combined = np.vstack([real_flat, sim_flat])
        scaler = StandardScaler()
        scaler.fit(combined)
        
        real_norm = scaler.transform(real_flat).reshape(min_len, 15, 3)
        sim_norm = scaler.transform(sim_flat).reshape(min_len, 15, 3)
    else:
        real_norm = real_mag
        sim_norm = sim_mag
    
    # Compute per-sensor RMSE
    sensor_rmse = []
    for sensor in range(15):
        real_sensor = real_norm[:, sensor, :].reshape(-1)
        sim_sensor = sim_norm[:, sensor, :].reshape(-1)
        rmse = np.sqrt(np.mean((real_sensor - sim_sensor)**2))
        sensor_rmse.append(rmse)
    
    # Overall RMSE
    overall_rmse = np.sqrt(np.mean((real_norm - sim_norm)**2))
    
    # Mean absolute error
    mae = np.mean(np.abs(real_norm - sim_norm))
    
    return {
        'offset': real_seq.get('offset', 'unknown'),
        'sequence_length': min_len,
        'overall_rmse': float(overall_rmse),
        'mae': float(mae),
        'sensor_rmse': [float(x) for x in sensor_rmse],
        'real_fz_range': [float(np.min(real_fz)), float(np.max(real_fz))],
        'sim_fz_range': [float(np.min(sim_fz)), float(np.max(sim_fz))],
    }


def analyze_mismatch(
    real_sequences: List[Dict],
    sim_sequences: List[Dict],
    stretch_label: str,
    output_dir: Path,
    normalize: bool = True,
    normalization_method: str = 'standardize',
    remove_outliers_flag: bool = True,
    z_threshold: float = 3.0,
    force_tolerance: float = 0.1,
) -> Dict:
    """Analyze mismatch between real and simulated magnetic data.
    
    Args:
        real_sequences: List of real data sequences
        sim_sequences: List of simulated data sequences
        stretch_label: Stretch level label (e.g., '000pct')
        output_dir: Directory to save plots and metrics
        normalize: Whether to normalize data before comparison
        normalization_method: Normalization method ('standardize', 'minmax', 'separate_minmax')
        remove_outliers_flag: Whether to remove outliers from real data
        z_threshold: Z-score threshold for outlier removal
    
    Returns:
        Dictionary with analysis results
    """
    if not real_sequences or not sim_sequences:
        print(f"⚠️  Skipping analysis for {stretch_label}: missing data")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Analyzing mismatch for stretch: {stretch_label}")
    print(f"{'='*80}")
    
    # Step 1: Remove outliers from real data
    if remove_outliers_flag:
        print("  Removing outliers from real data...")
        real_sequences_cleaned, outlier_indices = remove_outliers(real_sequences, z_threshold, remove_per_offset=2)
        print(f"    Removed {len(outlier_indices)} outlier sequences")
        real_sequences = real_sequences_cleaned
    
    # Step 2: Match sequences by force profile
    print("  Matching sequences...")
    matched_pairs = match_sequences_by_force_profile(
        real_sequences, sim_sequences,
        force_tolerance=force_tolerance,
        match_by_order=True,  # Match by order since sim replicates real force profiles
    )
    
    if not matched_pairs:
        print(f"⚠️  No sequences matched for {stretch_label}")
        return {}
    
    print(f"  Matched {len(matched_pairs)} sequence pairs")
    
    # Step 3: Compute error for each sequence pair
    print("  Computing error for each sequence pair...")
    sequence_errors = []
    for real_seq, sim_seq, match_score in matched_pairs:
        error = compute_sequence_error(real_seq, sim_seq, normalize=normalize)
        if error is not None:
            error['force_profile_match_score'] = float(match_score)
            sequence_errors.append(error)
    
    if not sequence_errors:
        print(f"⚠️  No valid sequence errors computed for {stretch_label}")
        return {}
    
    print(f"  Computed errors for {len(sequence_errors)} sequence pairs")
    
    # Step 4: Aggregate results
    # Average RMSE across all sequences
    overall_rmse_mean = np.mean([e['overall_rmse'] for e in sequence_errors])
    overall_rmse_std = np.std([e['overall_rmse'] for e in sequence_errors])
    
    # Average MAE across all sequences
    mae_mean = np.mean([e['mae'] for e in sequence_errors])
    mae_std = np.std([e['mae'] for e in sequence_errors])
    
    # Per-sensor average RMSE
    sensor_rmse_means = []
    sensor_rmse_stds = []
    for sensor in range(15):
        sensor_rmses = [e['sensor_rmse'][sensor] for e in sequence_errors]
        sensor_rmse_means.append(np.mean(sensor_rmses))
        sensor_rmse_stds.append(np.std(sensor_rmses))
    
    # Group by offset
    errors_by_offset = {}
    for error in sequence_errors:
        offset = error['offset']
        if offset not in errors_by_offset:
            errors_by_offset[offset] = []
        errors_by_offset[offset].append(error)
    
    # Compile results
    results = {
        'stretch_label': stretch_label,
        'n_real_sequences': len(real_sequences),
        'n_sim_sequences': len(sim_sequences),
        'n_matched_pairs': len(matched_pairs),
        'n_valid_errors': len(sequence_errors),
        'normalization': {'method': 'standardize' if normalize else 'none'},
        'overall_rmse_mean': float(overall_rmse_mean),
        'overall_rmse_std': float(overall_rmse_std),
        'mae_mean': float(mae_mean),
        'mae_std': float(mae_std),
        'sensor_rmse_mean': [float(x) for x in sensor_rmse_means],
        'sensor_rmse_std': [float(x) for x in sensor_rmse_stds],
        'best_sensor': int(np.argmin(sensor_rmse_means)) + 1,
        'worst_sensor': int(np.argmax(sensor_rmse_means)) + 1,
        'errors_by_offset': {
            offset: {
                'count': len(errors),
                'mean_rmse': float(np.mean([e['overall_rmse'] for e in errors])),
                'std_rmse': float(np.std([e['overall_rmse'] for e in errors])),
            }
            for offset, errors in errors_by_offset.items()
        },
        'sequence_errors': sequence_errors,  # Detailed per-sequence errors
    }
    
    # Step 5: Create plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"mismatch_analysis_{stretch_label}.png"
    create_mismatch_plots_sequence_based(
        matched_pairs,
        results,
        stretch_label,
        plot_path,
        normalize=normalize,
    )
    
    # Step 6: Create matching verification plots
    verification_plot_path = output_dir / f"matching_verification_{stretch_label}.png"
    create_matching_verification_plots(
        matched_pairs,
        stretch_label,
        verification_plot_path,
        n_examples=6,
    )
    
    print(f"✅ Analysis complete for {stretch_label}")
    print(f"   Matched pairs: {len(matched_pairs)}")
    print(f"   Mean RMSE: {overall_rmse_mean:.4f} ± {overall_rmse_std:.4f}")
    print(f"   Mean MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"   Best sensor: {results['best_sensor']} (RMSE: {sensor_rmse_means[results['best_sensor']-1]:.4f})")
    print(f"   Worst sensor: {results['worst_sensor']} (RMSE: {sensor_rmse_means[results['worst_sensor']-1]:.4f})")
    print(f"   Plot saved: {plot_path}")
    
    return results


def create_mismatch_plots_sequence_based(
    matched_pairs: List[Tuple[Dict, Dict, float]],
    results: Dict,
    stretch_label: str,
    output_path: Path,
    normalize: bool = True,
):
    """Create plots based on sequence-by-sequence comparison.
    
    Shows DIFFERENCE (Real - Sim) for each channel (Bx, By, Bz) separately.
    
    If normalize=True: Uses StandardScaler (zero mean, unit variance) on combined real+sim data.
                       RMSE is in standard deviations. RMSE ~1.89 = ~1.89 std dev difference.
    If normalize=False: Uses raw magnetic units. RMSE is in original sensor units.
    """
    # Aggregate all magnetic data (both normalized and raw for comparison)
    all_real_mag_norm = []
    all_sim_mag_norm = []
    all_real_mag_raw = []
    all_sim_mag_raw = []
    all_fz = []
    
    for real_seq, sim_seq, _ in matched_pairs:
        real_mag = real_seq['stretchmagtec']
        sim_mag = sim_seq['stretchmagtec']
        real_fz = np.abs(real_seq['fz'])
        
        min_len = min(len(real_mag), len(sim_mag))
        if min_len < 10:
            continue
        
        real_mag = real_mag[:min_len]
        sim_mag = sim_mag[:min_len]
        real_fz = real_fz[:min_len]
        
        # Store raw data
        all_real_mag_raw.append(real_mag)
        all_sim_mag_raw.append(sim_mag)
        
        # Normalize if requested
        if normalize:
            real_flat = real_mag.reshape(-1, 45)
            sim_flat = sim_mag.reshape(-1, 45)
            combined = np.vstack([real_flat, sim_flat])
            scaler = StandardScaler()
            scaler.fit(combined)
            real_norm = scaler.transform(real_flat).reshape(min_len, 15, 3)
            sim_norm = scaler.transform(sim_flat).reshape(min_len, 15, 3)
        else:
            real_norm = real_mag
            sim_norm = sim_mag
        
        all_real_mag_norm.append(real_norm)
        all_sim_mag_norm.append(sim_norm)
        all_fz.append(real_fz)
    
    # Create time axis (normalized to 0-1)
    n_samples = len(all_real_mag_norm[0])
    time_axis = np.linspace(0, 1, n_samples)
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
    
    # Main title with RMSE explanation
    if normalize:
        rmse_units = "standard deviations"
        rmse_explanation = (
            f"Normalization: StandardScaler (zero mean, unit variance) applied to combined real+sim data.\n"
            f"RMSE ~{results['overall_rmse_mean']:.2f} = average difference of ~{results['overall_rmse_mean']:.2f} std dev (relatively high mismatch).\n"
            f"Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately."
        )
    else:
        rmse_units = "raw magnetic units"
        # Calculate raw stats for explanation
        real_raw_all = np.concatenate(all_real_mag_raw, axis=0)
        sim_raw_all = np.concatenate(all_sim_mag_raw, axis=0)
        real_mean = np.mean(real_raw_all)
        sim_mean = np.mean(sim_raw_all)
        real_std = np.std(real_raw_all)
        sim_std = np.std(sim_raw_all)
        rmse_explanation = (
            f"No normalization: Raw magnetic sensor units.\n"
            f"Real data: mean={real_mean:.1f}, std={real_std:.1f} | "
            f"Sim data: mean={sim_mean:.1f}, std={sim_std:.1f}\n"
            f"Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately."
        )
    
    fig.suptitle(
        f'Sim2Real Magnetic Mismatch Analysis - {stretch_label}\n'
        f'Mean RMSE: {results["overall_rmse_mean"]:.4f} ± {results["overall_rmse_std"]:.4f} ({rmse_units})\n'
        f'{rmse_explanation}',
        fontsize=12, fontweight='bold'
    )
    
    # Plots 1-15: Per-sensor comparison showing DIFFERENCE (real - sim) for each channel
    channel_names = ['Bx', 'By', 'Bz']
    colors = ['r', 'g', 'b']
    
    for sensor in range(15):
        row = sensor // 3
        col = sensor % 3
        ax = fig.add_subplot(gs[row, col])
        
        for channel in range(3):
            # Average across all sequences for this sensor/channel
            real_ch = np.mean([mag[:, sensor, channel] for mag in all_real_mag_norm], axis=0)
            sim_ch = np.mean([mag[:, sensor, channel] for mag in all_sim_mag_norm], axis=0)
            
            # Calculate DIFFERENCE (real - sim)
            diff_ch = real_ch - sim_ch
            
            # Plot the difference
            ax.plot(time_axis, diff_ch, color=colors[channel], linestyle='-',
                   linewidth=2, alpha=0.8, label=f'{channel_names[channel]} diff (Real-Sim)')
        
        # Add zero line for reference
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        sensor_rmse = results['sensor_rmse_mean'][sensor]
        ax.set_title(f'Sensor {sensor+1}\nRMSE: {sensor_rmse:.3f} std dev', fontsize=10, fontweight='bold')
        ax.set_xlabel('Normalized Time [0-1]', fontsize=9)
        ax.set_ylabel('Difference (Real - Sim) [normalized]', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Plot 16: RMSE distribution across sequences
    ax_rmse = fig.add_subplot(gs[5, 0])
    rmses = [e['overall_rmse'] for e in results['sequence_errors']]
    ax_rmse.hist(rmses, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax_rmse.axvline(results['overall_rmse_mean'], color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {results["overall_rmse_mean"]:.4f}')
    ax_rmse.set_xlabel(f'RMSE per Sequence [{rmse_units}]', fontsize=11)
    ax_rmse.set_ylabel('Frequency', fontsize=11)
    ax_rmse.set_title('RMSE Distribution Across Sequences', fontsize=12, fontweight='bold')
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3, axis='y')
    
    # Plot 17: Per-sensor average RMSE
    ax_sensor = fig.add_subplot(gs[5, 1])
    sensor_ids = range(1, 16)
    ax_sensor.bar(sensor_ids, results['sensor_rmse_mean'], yerr=results['sensor_rmse_std'],
                  color='orange', alpha=0.7, capsize=5)
    ax_sensor.axhline(results['overall_rmse_mean'], color='r', linestyle='--', linewidth=2,
                      label=f'Overall Mean: {results["overall_rmse_mean"]:.4f}')
    ax_sensor.set_xlabel('Sensor ID', fontsize=11)
    ax_sensor.set_ylabel(f'Mean RMSE [{rmse_units}]', fontsize=11)
    ax_sensor.set_title('Per-Sensor Average RMSE', fontsize=12, fontweight='bold')
    ax_sensor.legend()
    ax_sensor.grid(True, alpha=0.3, axis='y')
    
    # Plot 18: RMSE by offset
    ax_offset = fig.add_subplot(gs[5, 2])
    offsets = list(results['errors_by_offset'].keys())
    offset_means = [results['errors_by_offset'][o]['mean_rmse'] for o in offsets]
    offset_stds = [results['errors_by_offset'][o]['std_rmse'] for o in offsets]
    ax_offset.bar(offsets, offset_means, yerr=offset_stds, color='green', alpha=0.7, capsize=5)
    ax_offset.set_xlabel('Offset', fontsize=11)
    ax_offset.set_ylabel(f'Mean RMSE [{rmse_units}]', fontsize=11)
    ax_offset.set_title('RMSE by Offset', fontsize=12, fontweight='bold')
    ax_offset.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {output_path}")
    
    # Also create a plot with RAW (non-normalized) data for comparison
    if normalize:
        raw_output_path = output_path.parent / f"{output_path.stem}_raw{output_path.suffix}"
        fig_raw = plt.figure(figsize=(24, 20))
        gs_raw = fig_raw.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        fig_raw.suptitle(
            f'Sim2Real Magnetic Mismatch Analysis - {stretch_label} (RAW DATA, NO NORMALIZATION)\n'
            f'Shows absolute difference in original sensor units.\n'
            f'Plots show DIFFERENCE (Real - Sim) for each channel (Bx=red, By=green, Bz=blue) separately.',
            fontsize=12, fontweight='bold'
        )
        
        for sensor in range(15):
            row = sensor // 3
            col = sensor % 3
            ax = fig_raw.add_subplot(gs_raw[row, col])
            
            for channel in range(3):
                # Average across all sequences for this sensor/channel (RAW data)
                real_ch_raw = np.mean([mag[:, sensor, channel] for mag in all_real_mag_raw], axis=0)
                sim_ch_raw = np.mean([mag[:, sensor, channel] for mag in all_sim_mag_raw], axis=0)
                
                # Calculate DIFFERENCE (real - sim) in RAW units
                diff_ch_raw = real_ch_raw - sim_ch_raw
                
                ax.plot(time_axis, diff_ch_raw, color=colors[channel], linestyle='-',
                       linewidth=2, alpha=0.8, label=f'{channel_names[channel]} diff (Real-Sim)')
            
            ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_title(f'Sensor {sensor+1} (RAW)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Normalized Time [0-1]', fontsize=9)
            ax.set_ylabel('Difference (Real - Sim) [raw units]', fontsize=9)
            ax.grid(True, alpha=0.3)
            if sensor == 0:
                ax.legend(fontsize=8, loc='best')
        
        # Summary stats for raw data
        real_raw_all = np.concatenate(all_real_mag_raw, axis=0)
        sim_raw_all = np.concatenate(all_sim_mag_raw, axis=0)
        diff_raw_all = real_raw_all - sim_raw_all
        raw_rmse = np.sqrt(np.mean(diff_raw_all**2))
        raw_mae = np.mean(np.abs(diff_raw_all))
        
        # Add summary text
        ax_summary = fig_raw.add_subplot(gs_raw[5, :])
        ax_summary.axis('off')
        summary_text = (
            f"Raw Data Statistics:\n"
            f"  Real data: mean={np.mean(real_raw_all):.1f}, std={np.std(real_raw_all):.1f}, range=[{np.min(real_raw_all):.1f}, {np.max(real_raw_all):.1f}]\n"
            f"  Sim data: mean={np.mean(sim_raw_all):.1f}, std={np.std(sim_raw_all):.1f}, range=[{np.min(sim_raw_all):.1f}, {np.max(sim_raw_all):.1f}]\n"
            f"  Difference (Real-Sim): RMSE={raw_rmse:.1f}, MAE={raw_mae:.1f}"
        )
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', 
                       fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.savefig(raw_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved RAW plot: {raw_output_path}")


def create_matching_verification_plots(
    matched_pairs: List[Tuple[Dict, Dict, float]],
    stretch_label: str,
    output_path: Path,
    n_examples: int = 6,
):
    """Create plots showing matched sequences to verify correct pairing.
    
    Shows force profiles and magnetic data for a few matched pairs to verify
    that the matching algorithm is working correctly.
    """
    # Select a few examples from different offsets
    examples_by_offset = {}
    for real_seq, sim_seq, match_score in matched_pairs:
        offset = real_seq.get('offset', 'unknown')
        if offset not in examples_by_offset:
            examples_by_offset[offset] = []
        if len(examples_by_offset[offset]) < 2:  # 2 examples per offset
            examples_by_offset[offset].append((real_seq, sim_seq, match_score))
    
    # Flatten and take first n_examples
    all_examples = []
    for offset_examples in examples_by_offset.values():
        all_examples.extend(offset_examples)
    examples = all_examples[:n_examples]
    
    print(f"   Selected {len(examples)} examples for matching verification plot")
    print(f"   Examples by offset: {[(e[0].get('offset', 'unknown'), e[2]) for e in examples]}")
    
    if not examples:
        print(f"   ⚠️  No examples to plot for matching verification")
        return
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 4*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Matching Verification: Real vs Simulated Sequences - {stretch_label}\n'
        f'Shows {len(examples)} matched pairs to verify correct pairing',
        fontsize=14, fontweight='bold'
    )
    
    for idx, (real_seq, sim_seq, match_score) in enumerate(examples):
        offset = real_seq.get('offset', 'unknown')
        real_fz = np.abs(real_seq['fz'])
        sim_fz = np.abs(sim_seq['fz'])
        real_mag = real_seq['stretchmagtec']
        sim_mag = sim_seq['stretchmagtec']
        
        # Align lengths
        min_len = min(len(real_fz), len(sim_fz))
        real_fz = real_fz[:min_len]
        sim_fz = sim_fz[:min_len]
        real_mag = real_mag[:min_len]
        sim_mag = sim_mag[:min_len]
        
        time_axis = np.arange(min_len) / 100.0  # Assuming 100 Hz
        
        # Plot 1: Force profiles
        ax1 = axes[idx, 0]
        ax1.plot(time_axis, real_fz, 'b-', linewidth=2, alpha=0.7, label='Real Fz')
        ax1.plot(time_axis, sim_fz, 'r--', linewidth=2, alpha=0.7, label='Sim Fz')
        ax1.set_xlabel('Time [s]', fontsize=10)
        ax1.set_ylabel('Force Fz [N]', fontsize=10)
        ax1.set_title(f'Example {idx+1} - {offset}\nForce Match Score: {match_score:.4f} N', 
                      fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Magnetic channel 8 Z (most sensitive)
        ax2 = axes[idx, 1]
        real_ch8_z = real_mag[:, 7, 2]  # Sensor 8 (index 7), channel Z (index 2)
        sim_ch8_z = sim_mag[:, 7, 2]
        ax2.plot(time_axis, real_ch8_z, 'b-', linewidth=2, alpha=0.7, label='Real Mag Ch8 Z')
        ax2.plot(time_axis, sim_ch8_z, 'r--', linewidth=2, alpha=0.7, label='Sim Mag Ch8 Z')
        ax2.set_xlabel('Time [s]', fontsize=10)
        ax2.set_ylabel('Magnetic Field [raw units]', fontsize=10)
        ax2.set_title(f'Magnetic Sensor 8 (Z-component)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difference in magnetic channel 8 Z
        ax3 = axes[idx, 2]
        diff_ch8_z = real_ch8_z - sim_ch8_z
        ax3.plot(time_axis, diff_ch8_z, 'g-', linewidth=2, alpha=0.8, label='Real - Sim')
        ax3.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Time [s]', fontsize=10)
        ax3.set_ylabel('Difference [raw units]', fontsize=10)
        rmse_ch8_z = np.sqrt(np.mean(diff_ch8_z**2))
        ax3.set_title(f'Difference (Real - Sim)\nRMSE: {rmse_ch8_z:.1f}', 
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved matching verification plot: {output_path}")


def create_mismatch_plots(
    real_magnetic: np.ndarray,  # [n_samples, 15, 3]
    sim_magnetic: np.ndarray,  # [n_samples, 15, 3]
    force_axis: np.ndarray,  # [n_samples]
    results: Dict,
    stretch_label: str,
    output_path: Path,
):
    """Create comprehensive mismatch analysis plots."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # Main title
    mean_offset = results["offset_and_scaling"]["overall_offset"].get("mean", 0)
    mean_scaling = results["offset_and_scaling"]["overall_scaling"].get("mean", 1.0)
    fig.suptitle(
        f'Sim2Real Magnetic Mismatch Analysis - {stretch_label}\n'
        f'Overall RMSE: {results["overall_rmse"]:.4f}, '
        f'Mean Offset: {mean_offset:.4f}, '
        f'Mean Scaling: {mean_scaling:.4f}',
        fontsize=14, fontweight='bold'
    )
    
    channel_names = ['Bx', 'By', 'Bz']
    colors = ['r', 'g', 'b']
    
    # Plot 1-15: Per-sensor comparison
    for sensor in range(15):
        row = sensor // 5
        col = sensor % 5
        ax = fig.add_subplot(gs[row, col])
        
        for channel in range(3):
            real_ch = real_magnetic[:, sensor, channel]
            sim_ch = sim_magnetic[:, sensor, channel]
            
            ax.plot(force_axis, real_ch, color=colors[channel], linestyle='-',
                   linewidth=2, alpha=0.7, label=f'Real {channel_names[channel]}')
            ax.plot(force_axis, sim_ch, color=colors[channel], linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Sim {channel_names[channel]}')
        
        sensor_rmse = results['sensor_rmse'][sensor]
        ax.set_title(f'Sensor {sensor+1}\nRMSE: {sensor_rmse:.3f}', fontsize=10)
        ax.set_xlabel('Force [N]', fontsize=9)
        ax.set_ylabel('Magnetic Field [normalized]', fontsize=9)
        ax.grid(True, alpha=0.3)
        if sensor == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Plot 16: RMSE per sensor
    ax_rmse = fig.add_subplot(gs[3, 0])
    ax_rmse.bar(range(1, 16), results['sensor_rmse'], color='steelblue', alpha=0.7)
    ax_rmse.axhline(results['overall_rmse'], color='r', linestyle='--', linewidth=2, label='Overall RMSE')
    ax_rmse.set_xlabel('Sensor ID', fontsize=10)
    ax_rmse.set_ylabel('RMSE', fontsize=10)
    ax_rmse.set_title('RMSE per Sensor', fontsize=11, fontweight='bold')
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3, axis='y')
    
    # Plot 17: Offset distribution
    ax_offset = fig.add_subplot(gs[3, 1])
    offsets = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        for ch_key, ch_data in sensor_data.items():
            offsets.append(ch_data['offset'])
    if offsets:
        ax_offset.hist(offsets, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax_offset.axvline(np.mean(offsets), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(offsets):.4f}')
        ax_offset.set_xlabel('Offset', fontsize=10)
        ax_offset.set_ylabel('Frequency', fontsize=10)
        ax_offset.set_title('Offset Distribution', fontsize=11, fontweight='bold')
        ax_offset.legend()
        ax_offset.grid(True, alpha=0.3, axis='y')
    
    # Plot 18: Scaling distribution
    ax_scale = fig.add_subplot(gs[3, 2])
    scalings = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        for ch_key, ch_data in sensor_data.items():
            scalings.append(ch_data['scaling'])
    if scalings:
        ax_scale.hist(scalings, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax_scale.axvline(np.mean(scalings), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scalings):.4f}')
        ax_scale.axvline(1.0, color='k', linestyle=':', linewidth=2, label='Ideal: 1.0')
        ax_scale.set_xlabel('Scaling Factor', fontsize=10)
        ax_scale.set_ylabel('Frequency', fontsize=10)
        ax_scale.set_title('Scaling Factor Distribution', fontsize=11, fontweight='bold')
        ax_scale.legend()
        ax_scale.grid(True, alpha=0.3, axis='y')
    
    # Plot 19: Correlation per sensor
    ax_corr = fig.add_subplot(gs[3, 3])
    correlations = []
    for sensor_key, sensor_data in results['offset_and_scaling']['sensor_metrics'].items():
        sensor_corrs = []
        for ch_key, ch_data in sensor_data.items():
            sensor_corrs.append(ch_data['correlation'])
        if sensor_corrs:
            correlations.append(np.mean(sensor_corrs))
    if correlations:
        ax_corr.bar(range(1, len(correlations) + 1), correlations, color='purple', alpha=0.7)
        ax_corr.set_xlabel('Sensor ID', fontsize=10)
        ax_corr.set_ylabel('Correlation', fontsize=10)
        ax_corr.set_title('Correlation per Sensor', fontsize=11, fontweight='bold')
        ax_corr.set_ylim([0, 1])
        ax_corr.grid(True, alpha=0.3, axis='y')
    
    # Plot 20: Scatter plot (real vs sim) for best sensor
    ax_scatter = fig.add_subplot(gs[3, 4])
    best_sensor_idx = results['best_sensor'] - 1
    real_best = real_magnetic[:, best_sensor_idx, :].reshape(-1)
    sim_best = sim_magnetic[:, best_sensor_idx, :].reshape(-1)
    ax_scatter.scatter(real_best, sim_best, alpha=0.5, s=20)
    # Add diagonal line
    min_val = min(np.min(real_best), np.min(sim_best))
    max_val = max(np.max(real_best), np.max(sim_best))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax_scatter.set_xlabel('Real Magnetic Field', fontsize=10)
    ax_scatter.set_ylabel('Simulated Magnetic Field', fontsize=10)
    ax_scatter.set_title(f'Best Sensor {results["best_sensor"]} Scatter', fontsize=11, fontweight='bold')
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze mismatch between real and simulated magnetic sensor data'
    )
    parser.add_argument(
        '--real-data',
        type=Path,
        required=True,
        help='Path to real data HDF5 file (cleaned)'
    )
    parser.add_argument(
        '--sim-data',
        type=Path,
        required=True,
        help='Path to simulation HDF5 file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/sim2real_mismatch'),
        help='Output directory for plots and metrics JSON'
    )
    parser.add_argument(
        '--stretch-label',
        type=str,
        default=None,
        help='Stretch label to filter (e.g., 000pct). If None, uses all sequences.'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize data before comparison (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Do not normalize data'
    )
    parser.add_argument(
        '--normalization-method',
        type=str,
        choices=['standardize', 'minmax', 'separate_minmax'],
        default='standardize',
        help='Normalization method (default: standardize)'
    )
    parser.add_argument(
        '--force-tolerance',
        type=float,
        default=0.1,
        help='Maximum allowed difference in force profiles for matching (default: 0.1)'
    )
    parser.add_argument(
        '--remove-outliers',
        action='store_true',
        default=True,
        help='Remove outliers from real data (default: True)'
    )
    parser.add_argument(
        '--no-remove-outliers',
        dest='remove_outliers',
        action='store_false',
        help='Do not remove outliers'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier removal (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    real_sequences = load_real_sequences(args.real_data, args.stretch_label)
    sim_sequences = load_sim_sequences(args.sim_data)
    
    if not real_sequences or not sim_sequences:
        print("❌ Error: Could not load sufficient data")
        return
    
    # Infer stretch label if not provided
    if args.stretch_label is None:
        # Try to infer from real sequences
        if real_sequences:
            stretch_label = real_sequences[0].get('stretch_label', 'unknown')
            if stretch_label != 'unknown':
                args.stretch_label = stretch_label.split('_')[-1] if '_' in stretch_label else stretch_label
        if args.stretch_label is None:
            args.stretch_label = 'unknown'
    
    # Analyze mismatch
    results = analyze_mismatch(
        real_sequences,
        sim_sequences,
        args.stretch_label,
        args.output_dir,
        normalize=args.normalize,
        normalization_method=args.normalization_method,
        remove_outliers_flag=args.remove_outliers,
        z_threshold=args.z_threshold,
        force_tolerance=args.force_tolerance,
    )
    
    # Save results
    if results:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = output_dir / f'mismatch_metrics_{args.stretch_label}.json'
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved metrics to: {metrics_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Stretch: {args.stretch_label}")
        print(f"Matched pairs: {results['n_matched_pairs']}")
        print(f"Mean RMSE: {results['overall_rmse_mean']:.4f} ± {results['overall_rmse_std']:.4f}")
        print(f"Mean MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"Best sensor: {results['best_sensor']} (RMSE: {results['sensor_rmse_mean'][results['best_sensor']-1]:.4f})")
        print(f"Worst sensor: {results['worst_sensor']} (RMSE: {results['sensor_rmse_mean'][results['worst_sensor']-1]:.4f})")


if __name__ == '__main__':
    main()