#!/usr/bin/env python3
"""
Plot raw force data from HDF5 file - no model needed.
Shows measured forces over time for each press sequence.
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Import outlier removal functions from train_single_point_models
try:
    from training.train_single_point_models import identify_outliers, remove_outliers
except ImportError:
    # Fallback if import fails
    def identify_outliers(sequences, z_threshold=3.0):
        return []
    def remove_outliers(sequences, z_threshold=3.0, remove_per_offset=2):
        return sequences, []

def load_raw_sequences(h5_path: Path, complete_only: bool = False, group_by_offset: bool = False) -> Dict[str, Dict]:
    """Load raw sequences grouped by press_id.
    
    Args:
        h5_path: Path to HDF5 file
        complete_only: If True, only load complete sequences (from _sequence_start to _sequence_end)
        group_by_offset: If True, return sequences grouped by offset (center, ne, nw, se, sw)
    """
    sequences = {}
    sequences_by_offset = defaultdict(list)  # {offset: [seq1, seq2, ...]}
    
    with h5py.File(h5_path, 'r') as hf:
        # Detect if this is simulation data
        is_simulation = 'MagneticField' in hf and 'forcesTest' in hf
        
        if is_simulation:
            # Load simulation data format
            magnetic = hf['MagneticField'][:]  # [samples, 15, 3]
            forces_test = hf['forcesTest'][:]  # [samples, 3] - Fx, Fy, Fz
            
            # Convert to real data format: forces [samples, 6] (Fx, Fy, Fz, Tx, Ty, Tz)
            # For simulation, we only have Fx, Fy, Fz, so set torques to 0
            forces = np.zeros((len(forces_test), 6), dtype=forces_test.dtype)
            forces[:, :3] = forces_test  # Fx, Fy, Fz
            # Tx, Ty, Tz remain 0
            
            # Convert magnetic [samples, 15, 3] to stretchmagtec format (same)
            stretchmagtec = magnetic
            
            # Create labels from position if available
            if 'IdenterPosition' in hf:
                indenter = hf['IdenterPosition'][:]  # [samples, 3]
                # Create position-based labels (simplified)
                labels = np.array([f"pos_{i}" for i in range(len(indenter))])
            else:
                labels = np.array([f"seq_{i}" for i in range(len(forces))])
        else:
            # Load real data format
            forces = hf['forces'][:]  # [samples, 6]
            labels = hf['labels'][:]  # [samples]
            
            # Load magnetic sensor data if available
            if 'stretchmagtec' in hf:
                stretchmagtec = hf['stretchmagtec'][:]  # [samples, 15, 3]
            else:
                stretchmagtec = None
        
        # Get timestamps if available
        if is_simulation:
            # For simulation, create timestamps from sample index (assume 100 Hz)
            time_relative = np.arange(len(forces)) / 100.0
        elif 'timestamps' in hf:
            timestamps = hf['timestamps'][:]
            # Convert to relative time in seconds
            if timestamps.dtype.kind in {'S', 'O'}:
                # String timestamps - convert to relative time
                from datetime import datetime
                ts_str = [ts.decode('utf-8') if isinstance(ts, bytes) else str(ts) for ts in timestamps]
                ts_dt = [datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'T' in ts else datetime.fromisoformat(ts) for ts in ts_str]
                time_relative = np.array([(t - ts_dt[0]).total_seconds() for t in ts_dt])
            else:
                # Numeric timestamps
                time_relative = timestamps - timestamps[0]
        else:
            # Fallback: assume 100 Hz sampling
            time_relative = np.arange(len(forces)) / 100.0
        
        if complete_only:
            # For simulation data, segment by position and map to offsets
            if is_simulation:
                # Import position_key function for mapping positions to offsets
                try:
                    # Try relative import first (when called as module)
                    try:
                        from training.train_simulation_positions import position_key, convert_to_sequences
                    except ImportError:
                        # Try absolute import (when called as script)
                        import sys
                        from pathlib import Path
                        src_root = Path(__file__).parent.parent
                        if str(src_root) not in sys.path:
                            sys.path.insert(0, str(src_root))
                        from training.train_simulation_positions import position_key, convert_to_sequences
                except ImportError as e:
                    # Fallback: treat as single sequence
                    print(f"  Warning: Could not import position_key/convert_to_sequences: {e}")
                    position_key = None
                    convert_to_sequences = None
                
                # Check if indenter was loaded (it should be if is_simulation is True)
                if 'IdenterPosition' in hf:
                    indenter_for_segmentation = hf['IdenterPosition'][:]
                else:
                    indenter_for_segmentation = None
                
                if position_key is not None and convert_to_sequences is not None and indenter_for_segmentation is not None:
                    # Segment simulation data by position and map to offsets
                    # indenter was already loaded above
                    
                    # Create position labels
                    position_labels = []
                    for i in range(len(indenter_for_segmentation)):
                        key, (x_mm, y_mm) = position_key(indenter_for_segmentation[i, :2])
                        position_labels.append(key)
                    
                    # Convert to sequences (this will map positions to offsets)
                    sim_sequences = convert_to_sequences(
                        magnetic=stretchmagtec,
                        forces=forces_test,
                        indenter=indenter_for_segmentation,
                        position_labels=position_labels,
                        stretch_label='unknown',
                    )
                    
                    print(f"  Converted simulation data to {len(sim_sequences)} sequences")
                    
                    # Convert simulation sequences to plot format
                    for seq_idx, sim_seq in enumerate(sim_sequences):
                        seq_id = f"sim_seq_{seq_idx}"
                        offset = sim_seq.get('offset', 'unknown')
                        
                        # Extract forces components
                        # convert_to_sequences returns 'fz' as array and 'forces' as [samples, 3]
                        fz = sim_seq.get('fz', np.array([]))
                        seq_forces = sim_seq.get('forces')
                        if seq_forces is not None and len(seq_forces.shape) == 2 and seq_forces.shape[1] >= 3:
                            fx = seq_forces[:, 0]
                            fy = seq_forces[:, 1]
                            fz_from_forces = seq_forces[:, 2]
                            # Use fz from forces if fz is empty
                            if len(fz) == 0:
                                fz = fz_from_forces
                        else:
                            # Fallback: create forces array from fz
                            fx = np.zeros(len(fz))
                            fy = np.zeros(len(fz))
                            if seq_forces is None:
                                seq_forces = np.zeros((len(fz), 3))
                                seq_forces[:, 2] = fz
                        
                        # Create time array for this sequence
                        seq_time = np.arange(len(fz)) / 100.0  # Assume 100 Hz
                        
                        seq_data = {
                            'time': seq_time,
                            'fx': fx,
                            'fy': fy,
                            'fz': fz,
                            'tx': np.zeros(len(fz)),
                            'ty': np.zeros(len(fz)),
                            'tz': np.zeros(len(fz)),
                            'forces': seq_forces if seq_forces is not None else np.zeros((len(fz), 6)),
                            'labels': np.array([f"sim_{seq_idx}_{i}" for i in range(len(fz))]),
                            'indices': np.arange(len(fz)),
                            'offset': offset,
                            'press_id': seq_id,
                        }
                        
                        # Add magnetic sensor data
                        seq_mag = sim_seq.get('stretchmagtec')
                        if seq_mag is not None and len(seq_mag.shape) == 3:
                            seq_data['mag_ch8_z'] = seq_mag[:, 7, 2]  # Channel 8, Z component
                            seq_data['mag_ch8'] = seq_mag[:, 7, :]  # Channel 8, all components
                            seq_data['mag_all_channels'] = seq_mag  # All channels
                        
                        sequences[seq_id] = seq_data
                        if group_by_offset:
                            sequences_by_offset[offset].append(seq_data)
                    
                    return sequences if not group_by_offset else sequences_by_offset
                else:
                    # Fallback: treat entire dataset as a single sequence
                    fx = forces[:, 0]  # Fx
                    fy = forces[:, 1]  # Fy
                    fz = forces[:, 2]  # Fz
                    tx = forces[:, 3] if forces.shape[1] > 3 else np.zeros(len(fz))
                    ty = forces[:, 4] if forces.shape[1] > 4 else np.zeros(len(fz))
                    tz = forces[:, 5] if forces.shape[1] > 5 else np.zeros(len(fz))
                    
                    seq_id = "sim_sequence_0"
                    seq_data = {
                        'time': time_relative,
                        'fx': fx,
                        'fy': fy,
                        'fz': fz,
                        'tx': tx,
                        'ty': ty,
                        'tz': tz,
                        'forces': forces,
                        'labels': labels,
                        'indices': np.arange(len(forces)),
                        'offset': 'unknown',
                        'press_id': seq_id,
                    }
                    
                    if stretchmagtec is not None:
                        seq_data['mag_ch8_z'] = stretchmagtec[:, 7, 2]
                        seq_data['mag_ch8'] = stretchmagtec[:, 7, :]
                        seq_data['mag_all_channels'] = stretchmagtec
                    
                    sequences[seq_id] = seq_data
                    if group_by_offset:
                        sequences_by_offset['unknown'].append(seq_data)
                    
                    return sequences if not group_by_offset else sequences_by_offset
            
            # Check if data is in presses group (new format) - for real data
            if 'presses' in hf:
                presses_group = hf['presses']
                for press_key in sorted(presses_group.keys()):
                    press_group = presses_group[press_key]
                    
                    # Get label from attributes or first label in dataset
                    label_str = ""
                    if 'label' in press_group.attrs:
                        label_attr = press_group.attrs['label']
                        label_str = label_attr.decode('utf-8') if isinstance(label_attr, bytes) else str(label_attr)
                    elif 'labels' in press_group:
                        labels_arr = press_group['labels'][:]
                        if len(labels_arr) > 0:
                            label_str = labels_arr[0].decode('utf-8') if isinstance(labels_arr[0], bytes) else str(labels_arr[0])
                    
                    # Extract press_id and offset
                    press_id = None
                    offset = None
                    
                    # Extract press_id (e.g., "pos_32_ne_press_B_sequence_start" -> "B")
                    match = re.search(r'press_([A-Za-z0-9_]+)_sequence', label_str)
                    if match:
                        press_id = match.group(1)
                    else:
                        match = re.search(r'press([A-Za-z0-9]+)', label_str)
                        if match:
                            press_id = match.group(1)
                    
                    # Extract offset (center, ne, nw, se, sw)
                    # Note: 'sw' must come before 'se' to avoid false matches (sw contains 'se')
                    for offset_key in ['center', 'ne', 'nw', 'sw', 'se']:
                        if offset_key in label_str.lower():
                            offset = offset_key
                            break
                    
                    if not offset:
                        offset = 'unknown'
                    
                    # Load data from press group
                    press_forces = press_group['forces'][:]  # [samples, 6]
                    if 'timestamps' in press_group:
                        press_timestamps = press_group['timestamps'][:]
                        # Timestamps in press groups are already normalized (starting from 0)
                        if press_timestamps.dtype.kind in {'S', 'O'}:
                            # String timestamps - convert to relative time
                            from datetime import datetime
                            ts_str = [ts.decode('utf-8') if isinstance(ts, bytes) else str(ts) for ts in press_timestamps]
                            ts_dt = [datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'T' in ts else datetime.fromisoformat(ts) for ts in ts_str]
                            press_time = np.array([(t - ts_dt[0]).total_seconds() for t in ts_dt])
                        else:
                            # Numeric timestamps - should already be relative (starting from 0)
                            press_time = press_timestamps.astype(float)
                            # Ensure it starts from 0 (should already be, but double-check)
                            if len(press_time) > 0 and press_time[0] != 0:
                                press_time = press_time - press_time[0]
                    else:
                        # Fallback: use index-based time (100 Hz sampling)
                        press_time = np.arange(len(press_forces)) / 100.0
                    
                    # Time should already start from 0 (normalized during save), but ensure it
                    seq_time = press_time
                    if len(seq_time) > 0 and seq_time[0] != 0:
                        seq_time = seq_time - seq_time[0]
                    
                    # Cut sequence at 3N: keep only samples where Fz <= 3.0N
                    fz_data = press_forces[:, 2]  # Fz component
                    fz_cut_mask = np.abs(fz_data) <= 3.0
                    cut_idx = len(press_forces)  # Default: no cut
                    if np.any(~fz_cut_mask):
                        # Find first index where Fz exceeds 3N
                        first_exceed_idx = np.where(~fz_cut_mask)[0]
                        if len(first_exceed_idx) > 0:
                            cut_idx = first_exceed_idx[0]
                            press_forces = press_forces[:cut_idx]
                            seq_time = seq_time[:cut_idx]
                            if 'stretchmagtec' in press_group:
                                press_mag = press_group['stretchmagtec'][:cut_idx]
                            else:
                                press_mag = None
                        else:
                            if 'stretchmagtec' in press_group:
                                press_mag = press_group['stretchmagtec'][:]
                            else:
                                press_mag = None
                    else:
                        if 'stretchmagtec' in press_group:
                            press_mag = press_group['stretchmagtec'][:]
                        else:
                            press_mag = None
                    
                    seq_data = {
                        'indices': list(range(len(press_forces))),
                        'time': seq_time,
                        'fz': press_forces[:, 2],  # Fz component
                        'fx': press_forces[:, 0],
                        'fy': press_forces[:, 1],
                        'press_id': press_id if press_id else press_key,
                        'offset': offset
                    }
                    
                    # Add magnetic sensor channel 8
                    if press_mag is not None:
                        seq_data['mag_ch8_z'] = press_mag[:, 7, 2]
                        seq_data['mag_ch8'] = press_mag[:, 7, :]
                        seq_data['mag_all_channels'] = press_mag[:, :, :]
                    elif stretchmagtec is not None:
                        # Fallback to root level data if available
                        start_idx = press_group.attrs.get('start_idx', 0)
                        end_idx = press_group.attrs.get('end_idx', len(press_forces))
                        indices = list(range(start_idx, end_idx + 1))
                        seq_data['mag_ch8_z'] = stretchmagtec[indices, 7, 2]
                        seq_data['mag_ch8'] = stretchmagtec[indices, 7, :]
                        seq_data['mag_all_channels'] = stretchmagtec[indices, :, :]
                    
                    # Add indentation data if available (must be cut to same length)
                    if 'indentation' in press_group:
                        indentation_data = press_group['indentation'][:]
                        seq_data['indentation'] = indentation_data[:cut_idx] if cut_idx < len(indentation_data) else indentation_data
                    elif 'positions' in press_group:
                        # Calculate indentation from positions if not directly available
                        positions = press_group['positions'][:]  # [samples, 3]
                        if len(positions) > 0:
                            initial_z = positions[0, 2]
                            indentation_data = positions[:, 2] - initial_z
                            seq_data['indentation'] = indentation_data[:cut_idx] if cut_idx < len(indentation_data) else indentation_data
                    
                    # Use press_id as key, but handle duplicates by appending number
                    key = press_id if press_id else press_key
                    counter = 1
                    while key in sequences:
                        key = f"{press_id if press_id else press_key}_{counter}"
                        counter += 1
                    sequences[key] = seq_data
                    
                    # Also group by offset if requested
                    if group_by_offset:
                        sequences_by_offset[offset].append(seq_data)
            else:
                # Old format: find sequences from root level labels
                press_sequences = []
                current_sequence_start = None
                current_press_id = None
                current_offset = None
                
                for idx, label in enumerate(labels):
                    label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
                    
                    if '_sequence_start' in label_str:
                        # Extract press_id and offset from sequence_start label
                        press_id = None
                        offset = None
                        
                        # Extract press_id
                        match = re.search(r'press_([A-Za-z0-9_]+)_sequence_start', label_str)
                        if match:
                            press_id = match.group(1)
                        else:
                            # Try alternative pattern
                            match = re.search(r'press([A-Za-z0-9]+)', label_str)
                            if match:
                                press_id = match.group(1)
                        
                        # Extract offset (center, ne, nw, se, sw)
                        # Note: 'sw' must come before 'se' to avoid false matches (sw contains 'se')
                        for offset_key in ['center', 'ne', 'nw', 'sw', 'se']:
                            if offset_key in label_str.lower():
                                offset = offset_key
                                break
                        
                        if press_id:
                            current_sequence_start = idx
                            current_press_id = press_id
                            current_offset = offset if offset else 'unknown'
                    elif '_sequence_end' in label_str and current_sequence_start is not None:
                        # End of current press sequence
                        if current_press_id:
                            press_sequences.append({
                                'press_id': current_press_id,
                                'offset': current_offset,
                                'start_idx': current_sequence_start,
                                'end_idx': idx
                            })
                        current_sequence_start = None
                        current_press_id = None
                        current_offset = None
                
                # Create sequences from complete press sequences
                for seq_info in press_sequences:
                    press_id = seq_info['press_id']
                    offset = seq_info['offset']
                    start_idx = seq_info['start_idx']
                    end_idx = seq_info['end_idx']
                    indices = list(range(start_idx, end_idx + 1))
                    
                    # Normalize time to start from 0
                    seq_time = time_relative[indices] - time_relative[indices[0]]
                    
                    # Cut sequence at 3N: keep only samples where Fz <= 3.0N
                    fz_data = forces[indices, 2]  # Fz component
                    fz_cut_mask = np.abs(fz_data) <= 3.0
                    cut_idx = len(indices)  # Default: no cut
                    if np.any(~fz_cut_mask):
                        # Find first index where Fz exceeds 3N
                        first_exceed_idx = np.where(~fz_cut_mask)[0]
                        if len(first_exceed_idx) > 0:
                            cut_idx = first_exceed_idx[0]
                            indices = indices[:cut_idx]
                            seq_time = seq_time[:cut_idx]
                    
                    seq_data = {
                        'indices': indices,
                        'time': seq_time,
                        'fz': forces[indices, 2],  # Fz component
                        'fx': forces[indices, 0],
                        'fy': forces[indices, 1],
                        'press_id': press_id,
                        'offset': offset
                    }
                    
                    # Add indentation if available (must be cut to same length)
                    # Note: indentation is not in the old format sequences, but we check anyway
                    if 'indentation' in locals() and len(indentation) > 0:
                        if len(indentation) >= len(indices):
                            seq_data['indentation'] = indentation[:cut_idx] if cut_idx < len(indentation) else indentation
                    
                    # Add magnetic sensor channel 8
                    if stretchmagtec is not None:
                        seq_data['mag_ch8_z'] = stretchmagtec[indices, 7, 2]
                        seq_data['mag_ch8'] = stretchmagtec[indices, 7, :]
                        seq_data['mag_all_channels'] = stretchmagtec[indices, :, :]
                    
                    # Use press_id as key, but handle duplicates by appending number
                    key = press_id
                    counter = 1
                    while key in sequences:
                        key = f"{press_id}_{counter}"
                        counter += 1
                    sequences[key] = seq_data
                    
                    # Also group by offset if requested
                    if group_by_offset:
                        sequences_by_offset[offset].append(seq_data)
        else:
            # Original logic: group by press_id (all samples with same press_id)
            press_groups = defaultdict(list)
            
            for idx, label in enumerate(labels):
                label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
                
                # Extract press_id from label
                press_id = None
                # Pattern 1: press_<ID>_...
                match = re.search(r'press_([A-Za-z0-9_]+)', label_str)
                if match:
                    press_id_raw = match.group(1)
                    # Remove common suffixes
                    press_id = re.sub(r'_(step|lift|contact|release|sequence_start|sequence_end).*', '', press_id_raw, flags=re.I)
                    if not press_id:
                        press_id = press_id_raw
                else:
                    # Pattern 2: press<ID>
                    match = re.search(r'press([A-Za-z0-9]+)', label_str)
                    if match:
                        press_id = match.group(1)
                
                if press_id:
                    press_groups[press_id].append(idx)
            
            # Create sequences
            for press_id, indices in press_groups.items():
                indices = sorted(indices)
                # Normalize time to start from 0
                seq_time = time_relative[indices] - time_relative[indices[0]]
                
                seq_data = {
                    'indices': indices,
                    'time': seq_time,
                    'fz': forces[indices, 2],  # Fz component
                    'fx': forces[indices, 0],
                    'fy': forces[indices, 1],
                    'press_id': press_id
                }
                
                # Add magnetic sensor channel 8
                if stretchmagtec is not None:
                    seq_data['mag_ch8_z'] = stretchmagtec[indices, 7, 2]
                    seq_data['mag_ch8'] = stretchmagtec[indices, 7, :]
                    seq_data['mag_all_channels'] = stretchmagtec[indices, :, :]
                
                sequences[press_id] = seq_data
    
    if group_by_offset:
        return dict(sequences_by_offset)
    return sequences

def plot_all_sequences(sequences: Dict[str, Dict], output_path: Path, max_sequences: int = 20):
    """Plot all sequences overlapped in one figure, all starting from 0s."""
    # Sort by press_id for consistent ordering
    sorted_press_ids = sorted(sequences.keys())
    
    # Limit number of sequences to plot
    if len(sorted_press_ids) > max_sequences:
        print(f"  Plotting first {max_sequences} sequences (out of {len(sorted_press_ids)})")
        sorted_press_ids = sorted_press_ids[:max_sequences]
    
    n_sequences = len(sorted_press_ids)
    if n_sequences == 0:
        print("  Warning: No sequences to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_sequences, 20)))
    
    for i, press_id in enumerate(sorted_press_ids):
        seq = sequences[press_id]
        color = colors[i % len(colors)]
        ax.plot(seq['time'], seq['fz'], label=f"{press_id} ({len(seq['indices'])} samples)", 
                color=color, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Fz (N)', fontsize=12)
    ax.set_title(f'Raw Force Data - All Sequences Overlapped ({n_sequences} sequences)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

def plot_sequences_by_type(sequences: Dict[str, Dict], output_dir: Path):
    """Plot sequences grouped by type (force_1, force_2, force_3, lift, etc.)."""
    # Group by type
    by_type = defaultdict(list)
    for press_id, seq in sequences.items():
        # Extract type from press_id (e.g., "A_force_1" -> "force_1")
        if '_' in press_id:
            parts = press_id.split('_')
            if len(parts) >= 2:
                seq_type = '_'.join(parts[1:])  # Everything after first underscore
            else:
                seq_type = 'other'
        else:
            seq_type = 'other'
        by_type[seq_type].append((press_id, seq))
    
    # Plot each type
    for seq_type, seq_list in sorted(by_type.items()):
        if not seq_list:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(seq_list)))
        
        for i, (press_id, seq) in enumerate(seq_list):
            ax.plot(seq['time'], seq['fz'], label=f"{press_id}", 
                   color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Fz (N)', fontsize=12)
        ax.set_title(f'Raw Force Data - Type: {seq_type} ({len(seq_list)} sequences)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        output_path = output_dir / f"raw_data_{seq_type}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")

def plot_single_sequence(sequences: Dict[str, Dict], output_path: Path, press_id: str = None):
    """Plot a single sequence in detail."""
    if press_id is None:
        # Find first sequence that looks like a continuous press (not lift, start, etc.)
        candidate_ids = [pid for pid in sorted(sequences.keys()) 
                        if not any(x in pid.lower() for x in ['lift', 'start', 'controlled'])]
        if candidate_ids:
            press_id = candidate_ids[0]
        else:
            press_id = sorted(sequences.keys())[0]
    
    if press_id not in sequences:
        print(f"  Warning: Press ID '{press_id}' not found, using first available")
        press_id = sorted(sequences.keys())[0]
    
    seq = sequences[press_id]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Fz over time
    ax = axes[0]
    ax.plot(seq['time'], seq['fz'], 'b-', linewidth=2, label='Fz')
    ax.set_ylabel('Fz (N)', fontsize=12)
    ax.set_title(f'Single Sequence: {press_id} - Force Z', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 2: Fx and Fy
    ax = axes[1]
    ax.plot(seq['time'], seq['fx'], 'r-', linewidth=1.5, label='Fx', alpha=0.7)
    ax.plot(seq['time'], seq['fy'], 'g-', linewidth=1.5, label='Fy', alpha=0.7)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title('Forces X and Y', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 3: Force magnitude
    ax = axes[2]
    force_mag = np.sqrt(seq['fx']**2 + seq['fy']**2 + seq['fz']**2)
    ax.plot(seq['time'], force_mag, 'm-', linewidth=2, label='|F|')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Force Magnitude (N)', fontsize=12)
    ax.set_title('Force Magnitude', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Add text box with statistics
    stats_text = f"""Statistics:
Samples: {len(seq['indices'])}
Duration: {seq['time'][-1] - seq['time'][0]:.2f} s
Fz range: [{np.min(seq['fz']):.2f}, {np.max(seq['fz']):.2f}] N
Fz mean: {np.mean(seq['fz']):.2f} N
Fz std: {np.std(seq['fz']):.2f} N"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} (Press ID: {press_id})")

def plot_summary_statistics(sequences: Dict[str, Dict], output_path: Path):
    """Plot summary statistics for all sequences."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sorted_press_ids = sorted(sequences.keys())
    
    # Extract statistics
    durations = []
    fz_max = []
    fz_min = []
    fz_mean = []
    n_samples = []
    
    for press_id in sorted_press_ids:
        seq = sequences[press_id]
        durations.append(seq['time'][-1] - seq['time'][0] if len(seq['time']) > 1 else 0)
        fz_max.append(np.max(seq['fz']))
        fz_min.append(np.min(seq['fz']))
        fz_mean.append(np.mean(seq['fz']))
        n_samples.append(len(seq['indices']))
    
    # Plot 1: Duration vs Press ID
    ax = axes[0, 0]
    ax.bar(range(len(sorted_press_ids)), durations, alpha=0.7)
    ax.set_xlabel('Press ID (index)')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Sequence Duration')
    ax.set_xticks(range(len(sorted_press_ids)))
    ax.set_xticklabels(sorted_press_ids, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Fz range
    ax = axes[0, 1]
    x_pos = np.arange(len(sorted_press_ids))
    ax.bar(x_pos, fz_max, alpha=0.7, label='Max Fz', color='red')
    ax.bar(x_pos, fz_min, alpha=0.7, label='Min Fz', color='blue')
    ax.set_xlabel('Press ID (index)')
    ax.set_ylabel('Fz (N)')
    ax.set_title('Fz Range per Sequence')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_press_ids, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 3: Mean Fz
    ax = axes[1, 0]
    ax.bar(range(len(sorted_press_ids)), fz_mean, alpha=0.7, color='green')
    ax.set_xlabel('Press ID (index)')
    ax.set_ylabel('Mean Fz (N)')
    ax.set_title('Mean Fz per Sequence')
    ax.set_xticks(range(len(sorted_press_ids)))
    ax.set_xticklabels(sorted_press_ids, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 4: Number of samples
    ax = axes[1, 1]
    ax.bar(range(len(sorted_press_ids)), n_samples, alpha=0.7, color='orange')
    ax.set_xlabel('Press ID (index)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Samples per Sequence')
    ax.set_xticks(range(len(sorted_press_ids)))
    ax.set_xticklabels(sorted_press_ids, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Raw Data Summary Statistics', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

def plot_sequences_by_offset(sequences_by_offset: Dict[str, List[Dict]], output_dir: Path, stretch_label: str = ""):
    """Plot 10 presses for each offset point, with Fz above and magnetic X, Y, Z below.
    
    Args:
        sequences_by_offset: Dict mapping offset (center, ne, nw, se, sw) to list of sequences
        output_dir: Output directory for plots
        stretch_label: Stretch level label (e.g., "0%", "10%", "20%")
    """
    # Expected offsets
    expected_offsets = ['center', 'ne', 'nw', 'se', 'sw']
    
    for offset in expected_offsets:
        if offset not in sequences_by_offset or len(sequences_by_offset[offset]) == 0:
            print(f"  Warning: No sequences found for offset '{offset}'")
            continue
        
        seqs = sequences_by_offset[offset]
        print(f"  Plotting {len(seqs)} sequences for offset '{offset}'")
        
        # Create figure with 5 subplots: Fz, Indentation, X, Y, Z
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(seqs), 10)))
        
        # Calculate common time range - use median to focus on typical duration
        # This ensures all sequences are visible in the same time window
        import statistics
        durations = [seq['time'][-1] if len(seq['time']) > 0 else 0 for seq in seqs]
        if durations:
            # Use median duration to focus on typical press duration
            # This helps see all sequences overlapped in the same time window
            common_duration = statistics.median(durations) if len(durations) > 1 else durations[0]
            # But also show up to max to see longer sequences
            max_duration = max(durations)
        else:
            common_duration = 10.0  # Default
            max_duration = 10.0
        
        # Plot Fz (top subplot)
        ax_fz = axes[0]
        for i, seq in enumerate(seqs):
            color = colors[i % len(colors)]
            press_id = seq.get('press_id', f'press_{i}')
            # Verify time starts from 0
            if len(seq['time']) > 0 and seq['time'][0] != 0:
                print(f"    Warning: Sequence {press_id} does not start from 0s (starts at {seq['time'][0]:.3f}s)")
            # Plot sequence - all should start from 0s
            ax_fz.plot(seq['time'], seq['fz'], 
                      label=f"{press_id}", 
                      color=color, alpha=0.7, linewidth=1.5)
        ax_fz.set_ylabel('Fz (N)', fontsize=12)
        ax_fz.set_title(f'Offset: {offset.upper()} - {len(seqs)} Presses (all starting from 0s)' + (f' ({stretch_label})' if stretch_label else ''), fontsize=14)
        ax_fz.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax_fz.grid(True, alpha=0.3)
        ax_fz.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        # Set x-axis limit to show all sequences clearly overlapped
        # Use max duration to show all data, but could use common_duration to focus on typical
        ax_fz.set_xlim(0, max_duration * 1.05)  # Add 5% margin
        
        # Plot indentation (second subplot)
        ax_ind = axes[1]
        has_indentation = False
        for i, seq in enumerate(seqs):
            if 'indentation' in seq:
                has_indentation = True
                color = colors[i % len(colors)]
                press_id = seq.get('press_id', f'press_{i}')
                ax_ind.plot(seq['time'], seq['indentation'], 
                          label=f"{press_id}", 
                          color=color, alpha=0.7, linewidth=1.5)
        if has_indentation:
            ax_ind.set_ylabel('Indentation (m)', fontsize=12)
            ax_ind.set_title('Indentation Z Position', fontsize=12)
            ax_ind.grid(True, alpha=0.3)
            ax_ind.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax_ind.set_xlim(0, max_duration * 1.05)
            ax_ind.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        else:
            ax_ind.set_ylabel('Indentation (m)', fontsize=12)
            ax_ind.text(0.5, 0.5, 'No indentation data available', 
                       ha='center', va='center', transform=ax_ind.transAxes)
            ax_ind.set_xlim(0, max_duration * 1.05)
        
        # Plot magnetic sensor X, Y, Z (bottom 3 subplots)
        if 'mag_ch8' in seqs[0]:
            for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes[2:])):
                for i, seq in enumerate(seqs):
                    if 'mag_ch8' in seq:
                        color = colors[i % len(colors)]
                        press_id = seq.get('press_id', f'press_{i}')
                        mag_data = seq['mag_ch8'][:, comp_idx]  # X, Y, or Z component
                        ax.plot(seq['time'], mag_data, 
                               label=f"{press_id}" if comp_idx == 0 and i < 5 else "", 
                               color=color, alpha=0.7, linewidth=1.5)
                ax.set_ylabel(f'Mag Ch8 {comp_name} [digits]', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, max_duration * 1.05)  # Same x-axis limit as Fz
                if comp_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        
        output_path = output_dir / f"raw_data_offset_{offset}_{stretch_label.replace('%', 'pct')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path.name}")

def plot_average_by_offset_and_stretch(h5_files: List[Path], output_dir: Path):
    """Plot average for each offset point for each stretch level."""
    # Group files by stretch level
    stretch_files = defaultdict(list)
    
    for h5_file in h5_files:
        stretch_label = "unknown"
        if "000pct" in h5_file.name or "stretch_0" in h5_file.name:
            stretch_label = "0%"
        elif "010pct" in h5_file.name or "stretch_10" in h5_file.name:
            stretch_label = "10%"
        elif "020pct" in h5_file.name or "stretch_20" in h5_file.name:
            stretch_label = "20%"
        
        # Try to get from file attributes
        try:
            with h5py.File(h5_file, 'r') as hf:
                if 'stretch_label' in hf.attrs:
                    stretch_label = hf.attrs['stretch_label'].decode('utf-8') if isinstance(hf.attrs['stretch_label'], bytes) else str(hf.attrs['stretch_label'])
        except:
            pass
        
        stretch_files[stretch_label].append(h5_file)
    
    # Expected offsets
    expected_offsets = ['center', 'ne', 'nw', 'se', 'sw']
    
    # Color mapping for stretch levels (normalize labels to percentage format)
    def normalize_stretch_label(label):
        """Normalize stretch label to percentage format (e.g., 'stretch_000pct' -> '0%')."""
        label_str = str(label).lower()
        if '000' in label_str or '0%' in label_str:
            return '0%'
        elif '010' in label_str or '10%' in label_str:
            return '10%'
        elif '020' in label_str or '20%' in label_str:
            return '20%'
        return label
    
    # For each offset, create a plot with average for each stretch level
    for offset in expected_offsets:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        colors = {'0%': '#1f77b4', '10%': '#ff7f0e', '20%': '#2ca02c'}  # Blue, Orange, Green
        
        for stretch_label in sorted(stretch_files.keys()):
            if stretch_label == "unknown":
                continue
            
            # Load all sequences for this stretch level
            all_seqs_for_offset = []
            for h5_file in stretch_files[stretch_label]:
                sequences_by_offset = load_raw_sequences(h5_file, complete_only=True, group_by_offset=True)
                if offset in sequences_by_offset:
                    all_seqs_for_offset.extend(sequences_by_offset[offset])
            
            if not all_seqs_for_offset:
                continue
            
            # Interpolate to common time grid
            max_duration = max(seq['time'][-1] if len(seq['time']) > 0 else 0 for seq in all_seqs_for_offset)
            common_time = np.linspace(0, max_duration, int(max_duration * 100) + 1)
            
            # Interpolate Fz
            interpolated_fz = []
            for seq in all_seqs_for_offset:
                if len(seq['time']) < 2:
                    continue
                fz_interp = np.interp(common_time, seq['time'], seq['fz'])
                interpolated_fz.append(fz_interp)
            
            if interpolated_fz:
                interpolated_fz = np.array(interpolated_fz)
                avg_fz = np.mean(interpolated_fz, axis=0)
                std_fz = np.std(interpolated_fz, axis=0)
                
                normalized_label = normalize_stretch_label(stretch_label)
                color = colors.get(normalized_label, '#d62728')  # Red as fallback instead of gray
                axes[0].plot(common_time, avg_fz, label=f'{stretch_label} (n={len(all_seqs_for_offset)})', 
                           color=color, linewidth=2, alpha=0.8)
                axes[0].fill_between(common_time, avg_fz - std_fz, avg_fz + std_fz, 
                                    color=color, alpha=0.2)
            
            # Interpolate magnetic X, Y, Z
            if 'mag_ch8' in all_seqs_for_offset[0]:
                for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes[1:])):
                    interpolated_mag = []
                    for seq in all_seqs_for_offset:
                        if 'mag_ch8' in seq and len(seq['time']) >= 2:
                            mag_data = seq['mag_ch8'][:, comp_idx]
                            mag_interp = np.interp(common_time, seq['time'], mag_data)
                            interpolated_mag.append(mag_interp)
                    
                    if interpolated_mag:
                        interpolated_mag = np.array(interpolated_mag)
                        avg_mag = np.mean(interpolated_mag, axis=0)
                        std_mag = np.std(interpolated_mag, axis=0)
                        
                        normalized_label = normalize_stretch_label(stretch_label)
                        color = colors.get(normalized_label, '#d62728')  # Red as fallback instead of gray
                        ax.plot(common_time, avg_mag, label=f'{stretch_label} (n={len(all_seqs_for_offset)})', 
                               color=color, linewidth=2, alpha=0.8)
                        ax.fill_between(common_time, avg_mag - std_mag, avg_mag + std_mag, 
                                      color=color, alpha=0.2)
        
        axes[0].set_ylabel('Fz (N)', fontsize=12)
        axes[0].set_title(f'Average for Offset: {offset.upper()} - All Stretch Levels', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes[1:])):
            ax.set_ylabel(f'Mag Ch8 {comp_name} [digits]', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            if comp_idx == 0:
                ax.legend(fontsize=9)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        
        output_path = output_dir / f"raw_data_average_offset_{offset}_all_stretches.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")

def plot_mean_std_by_position(sequences_by_offset: Dict[str, List[Dict]], output_dir: Path, stretch_label: str = ""):
    """Plot mean and std deviation for each position separately (Fz, Indentation, Mag Ch8 X/Y/Z).
    
    Args:
        sequences_by_offset: Dict mapping offset (center, ne, nw, se, sw) to list of sequences
        output_dir: Output directory for plots
        stretch_label: Stretch level label (e.g., "0%", "10%", "20%")
    """
    expected_offsets = ['center', 'ne', 'nw', 'se', 'sw']
    
    for offset in expected_offsets:
        if offset not in sequences_by_offset or len(sequences_by_offset[offset]) == 0:
            print(f"  Warning: No sequences found for offset '{offset}'")
            continue
        
        seqs = sequences_by_offset[offset]
        print(f"  Generating mean/std plots for offset '{offset}' ({len(seqs)} sequences)")
        
        # Find max duration for common time grid
        durations = [seq['time'][-1] if len(seq['time']) > 0 else 0 for seq in seqs]
        if not durations:
            continue
        max_duration = max(durations)
        common_time = np.linspace(0, max_duration, int(max_duration * 100) + 1)
        
        # Create figure with 6 subplots: Fz, Indentation, Mag Ch8 X, Y, Z, and Raw Data (all channels)
        fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
        
        # 1. Fz - Mean and Std
        interpolated_fz = []
        for seq in seqs:
            if len(seq['time']) >= 2:
                fz_interp = np.interp(common_time, seq['time'], seq['fz'])
                interpolated_fz.append(fz_interp)
        
        if interpolated_fz:
            interpolated_fz = np.array(interpolated_fz)
            mean_fz = np.mean(interpolated_fz, axis=0)
            std_fz = np.std(interpolated_fz, axis=0)
            
            axes[0].plot(common_time, mean_fz, 'b-', label='Mean', linewidth=2)
            axes[0].fill_between(common_time, mean_fz - std_fz, mean_fz + std_fz, 
                                color='blue', alpha=0.3, label='±1 Std Dev')
            axes[0].set_ylabel('Fz (N)', fontsize=12)
            axes[0].set_title(f'Mean ± Std Dev - Offset: {offset.upper()} (n={len(seqs)})' + (f' ({stretch_label})' if stretch_label else ''), fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # 2. Indentation - Mean and Std
        interpolated_ind = []
        for seq in seqs:
            if 'indentation' in seq and len(seq['time']) >= 2:
                ind_interp = np.interp(common_time, seq['time'], seq['indentation'])
                interpolated_ind.append(ind_interp)
        
        if interpolated_ind:
            interpolated_ind = np.array(interpolated_ind)
            mean_ind = np.mean(interpolated_ind, axis=0)
            std_ind = np.std(interpolated_ind, axis=0)
            
            axes[1].plot(common_time, mean_ind, 'g-', label='Mean', linewidth=2)
            axes[1].fill_between(common_time, mean_ind - std_ind, mean_ind + std_ind, 
                                 color='green', alpha=0.3, label='±1 Std Dev')
            axes[1].set_ylabel('Indentation (m)', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        else:
            axes[1].text(0.5, 0.5, 'No indentation data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_ylabel('Indentation (m)', fontsize=12)
        
        # 3-5. Magnetic Ch8 X, Y, Z - Mean and Std
        if 'mag_ch8' in seqs[0]:
            for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes[2:5])):
                interpolated_mag = []
                for seq in seqs:
                    if 'mag_ch8' in seq and len(seq['time']) >= 2:
                        mag_data = seq['mag_ch8'][:, comp_idx]
                        mag_interp = np.interp(common_time, seq['time'], mag_data)
                        interpolated_mag.append(mag_interp)
                
                if interpolated_mag:
                    interpolated_mag = np.array(interpolated_mag)
                    mean_mag = np.mean(interpolated_mag, axis=0)
                    std_mag = np.std(interpolated_mag, axis=0)
                    
                    ax.plot(common_time, mean_mag, 'r-', label='Mean', linewidth=2)
                    ax.fill_between(common_time, mean_mag - std_mag, mean_mag + std_mag, 
                                   color='red', alpha=0.3, label='±1 Std Dev')
                    ax.set_ylabel(f'Mag Ch8 {comp_name} [digits]', fontsize=11)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # 6. Raw Data - Mean and Std for all magnetic channels (flattened)
        if 'mag_all_channels' in seqs[0]:
            interpolated_all_mag = []
            for seq in seqs:
                if 'mag_all_channels' in seq and len(seq['time']) >= 2:
                    # Flatten all channels: [samples, 15, 3] -> [samples, 45]
                    mag_flat = seq['mag_all_channels'].reshape(len(seq['time']), -1)
                    # Interpolate each channel
                    mag_interp = np.zeros((len(common_time), mag_flat.shape[1]))
                    for ch in range(mag_flat.shape[1]):
                        mag_interp[:, ch] = np.interp(common_time, seq['time'], mag_flat[:, ch])
                    interpolated_all_mag.append(mag_interp)
            
            if interpolated_all_mag:
                interpolated_all_mag = np.array(interpolated_all_mag)  # [n_seqs, n_time, 45]
                # Compute mean and std across all channels (flattened)
                mean_all_mag = np.mean(interpolated_all_mag, axis=(0, 2))  # Mean across sequences and channels
                std_all_mag = np.std(interpolated_all_mag, axis=(0, 2))    # Std across sequences and channels
                
                axes[5].plot(common_time, mean_all_mag, 'm-', label='Mean (all channels)', linewidth=2)
                axes[5].fill_between(common_time, mean_all_mag - std_all_mag, mean_all_mag + std_all_mag, 
                                    color='magenta', alpha=0.3, label='±1 Std Dev')
                axes[5].set_ylabel('All Mag Channels [digits]', fontsize=11)
                axes[5].legend(fontsize=9)
                axes[5].grid(True, alpha=0.3)
                axes[5].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        
        output_path = output_dir / f"mean_std_offset_{offset}_{stretch_label.replace('%', 'pct')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path.name}")

def plot_average_all_points_by_stretch(h5_files: List[Path], output_path: Path):
    """Plot average of all points (all offsets combined) for each stretch level (0%, 10%, 20%) in the same plot."""
    stretch_data = {}  # {stretch_label: list of sequences}
    
    # Load sequences from all files
    for h5_file in h5_files:
        # Extract stretch level from filename or file attributes
        stretch_label = "unknown"
        if "000pct" in h5_file.name or "stretch_0" in h5_file.name:
            stretch_label = "0%"
        elif "010pct" in h5_file.name or "stretch_10" in h5_file.name:
            stretch_label = "10%"
        elif "020pct" in h5_file.name or "stretch_20" in h5_file.name:
            stretch_label = "20%"
        
        # Try to get from file attributes
        try:
            with h5py.File(h5_file, 'r') as hf:
                if 'stretch_label' in hf.attrs:
                    stretch_label = hf.attrs['stretch_label'].decode('utf-8') if isinstance(hf.attrs['stretch_label'], bytes) else str(hf.attrs['stretch_label'])
        except:
            pass
        
        # Load complete sequences only (all offsets combined)
        sequences = load_raw_sequences(h5_file, complete_only=True)
        
        if stretch_label not in stretch_data:
            stretch_data[stretch_label] = []
        stretch_data[stretch_label].extend(sequences.values())
    
    if not stretch_data:
        print("  Warning: No stretch data found")
        return
    
    # Compute average for each stretch level
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color mapping for stretch levels (normalize labels to percentage format)
    def normalize_stretch_label(label):
        """Normalize stretch label to percentage format (e.g., 'stretch_000pct' -> '0%')."""
        label_str = str(label).lower()
        if '000' in label_str or '0%' in label_str:
            return '0%'
        elif '010' in label_str or '10%' in label_str:
            return '10%'
        elif '020' in label_str or '20%' in label_str:
            return '20%'
        return label
    
    colors = {'0%': '#1f77b4', '10%': '#ff7f0e', '20%': '#2ca02c'}  # Blue, Orange, Green
    
    for stretch_label in sorted(stretch_data.keys()):
        if stretch_label == "unknown":
            continue
        seqs = stretch_data[stretch_label]
        if not seqs:
            continue
        
        # Interpolate all sequences to a common time grid
        # Find max duration - use 95th percentile to avoid outliers
        durations = [seq['time'][-1] if len(seq['time']) > 0 else 0 for seq in seqs]
        if durations:
            # Use 95th percentile instead of max to avoid very long sequences
            max_duration = np.percentile(durations, 95)
            # But also cap at reasonable maximum (e.g., 10 seconds)
            max_duration = min(max_duration, 10.0)
        else:
            max_duration = 10.0
        
        # Create common time grid (100 Hz sampling)
        common_time = np.linspace(0, max_duration, int(max_duration * 100) + 1)
        
        # Interpolate each sequence to common grid
        interpolated_fz = []
        for seq in seqs:
            if len(seq['time']) < 2:
                continue
            fz_interp = np.interp(common_time, seq['time'], seq['fz'])
            interpolated_fz.append(fz_interp)
        
        if not interpolated_fz:
            continue
        
        # Compute average and std
        interpolated_fz = np.array(interpolated_fz)  # [n_sequences, n_samples]
        avg_fz = np.mean(interpolated_fz, axis=0)
        std_fz = np.std(interpolated_fz, axis=0)
        
        normalized_label = normalize_stretch_label(stretch_label)
        color = colors.get(normalized_label, '#d62728')  # Red as fallback instead of gray
        ax.plot(common_time, avg_fz, label=f'{stretch_label} stretch (avg, n={len(seqs)})', 
                color=color, linewidth=2, alpha=0.8)
        ax.fill_between(common_time, avg_fz - std_fz, avg_fz + std_fz, 
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Fz (N)', fontsize=12)
    ax.set_title('Average Force Sequences - All Points Combined by Stretch Level', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Plot raw force data from HDF5 file")
    parser.add_argument("--h5-file", type=Path, required=False, help="HDF5 file to plot")
    parser.add_argument("--h5-files", type=Path, nargs='+', required=False, help="Multiple HDF5 files to plot (for average by stretch)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single file or multiple files
    if args.h5_files:
        h5_files = [Path(f) for f in args.h5_files]
        for h5_file in h5_files:
            if not h5_file.exists():
                print(f"Error: File not found: {h5_file}")
                return
        
        # Group files by stretch level
        stretch_files = defaultdict(list)
        for h5_file in h5_files:
            stretch_label = "unknown"
            if "000pct" in h5_file.name or "stretch_0" in h5_file.name:
                stretch_label = "0%"
            elif "010pct" in h5_file.name or "stretch_10" in h5_file.name:
                stretch_label = "10%"
            elif "020pct" in h5_file.name or "stretch_20" in h5_file.name:
                stretch_label = "20%"
            
            # Try to get from file attributes
            try:
                with h5py.File(h5_file, 'r') as hf:
                    if 'stretch_label' in hf.attrs:
                        stretch_label = hf.attrs['stretch_label'].decode('utf-8') if isinstance(hf.attrs['stretch_label'], bytes) else str(hf.attrs['stretch_label'])
            except:
                pass
            
            stretch_files[stretch_label].append(h5_file)
        
        # For each stretch level, create subfolder and generate plots
        for stretch_label, files in sorted(stretch_files.items()):
            if stretch_label == "unknown":
                continue
            
            stretch_dir = args.output_dir / stretch_label.replace('%', 'pct')
            stretch_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"Processing stretch level: {stretch_label}")
            print(f"Output directory: {stretch_dir}")
            print(f"{'='*80}")
            
            # Load sequences grouped by offset for this stretch level
            all_sequences_by_offset = defaultdict(list)
            for h5_file in files:
                print(f"\nLoading: {h5_file.name}")
                sequences_by_offset = load_raw_sequences(h5_file, complete_only=True, group_by_offset=True)
                for offset, seqs in sequences_by_offset.items():
                    all_sequences_by_offset[offset].extend(seqs)
            
            # Remove first sequence per offset (warm-up/calibration sequence) BEFORE outlier removal
            print(f"\nRemoving first sequence per offset...")
            for offset, seqs in all_sequences_by_offset.items():
                if len(seqs) > 1:  # Only remove if there's more than one sequence
                    all_sequences_by_offset[offset] = seqs[1:]  # Remove first sequence
                    print(f"  Offset {offset}: Removed first sequence, {len(seqs)} -> {len(seqs) - 1} sequences")
            
            # Remove 2 outliers per offset before plotting
            print(f"\nRemoving 2 outliers per offset before plotting...")
            for offset, seqs in all_sequences_by_offset.items():
                if len(seqs) > 2:
                    # Convert sequences to format expected by remove_outliers
                    seqs_dict = []
                    for seq in seqs:
                        seq_dict = {
                            'offset': offset,
                            'duration': seq['time'][-1] - seq['time'][0] if len(seq['time']) > 1 else len(seq['fz']) / 100.0,
                            'max_fz': float(np.max(seq['fz'])),
                            'num_samples': len(seq['fz']),
                            'fz_max': float(np.max(seq['fz'])),
                        }
                        seqs_dict.append(seq_dict)
                    
                    # Remove outliers
                    cleaned_seqs_dict, outlier_indices = remove_outliers(seqs_dict, z_threshold=3.0, remove_per_offset=2)
                    if outlier_indices:
                        # Remove corresponding sequences
                        cleaned_seqs = [seq for i, seq in enumerate(seqs) if i not in outlier_indices]
                        all_sequences_by_offset[offset] = cleaned_seqs
                        print(f"  Offset {offset}: Removed {len(outlier_indices)} outliers, {len(seqs)} -> {len(cleaned_seqs)} sequences")
                    else:
                        all_sequences_by_offset[offset] = seqs
                else:
                    all_sequences_by_offset[offset] = seqs
            
            # Plot sequences by offset (10 presses per point)
            print(f"\nGenerating plots for each offset point...")
            plot_sequences_by_offset(all_sequences_by_offset, stretch_dir, stretch_label)
            
            # Plot mean and std deviation for each position
            print(f"\nGenerating mean/std plots for each offset point...")
            plot_mean_std_by_position(all_sequences_by_offset, stretch_dir, stretch_label)
        
        # Generate average plots
        print(f"\n{'='*80}")
        print("Generating average plots...")
        print(f"{'='*80}")
        
        # Average for each offset across all stretch levels
        plot_average_by_offset_and_stretch(h5_files, args.output_dir)
        
        # Average of all points for each stretch level
        plot_average_all_points_by_stretch(h5_files, args.output_dir / "raw_data_average_all_points_by_stretch.png")
    else:
        if not args.h5_file or not args.h5_file.exists():
            print(f"Error: File not found: {args.h5_file}")
            return
        
        print(f"Loading raw data from: {args.h5_file}")
        # Load sequences grouped by offset (needed for offset-based plots)
        sequences_by_offset = load_raw_sequences(args.h5_file, complete_only=True, group_by_offset=True)
        
        # Flatten to get all sequences
        sequences = {}
        for offset, seqs in sequences_by_offset.items():
            for seq in seqs:
                sequences[seq['press_id']] = seq
        
        print(f"Found {len(sequences)} complete sequences:")
        offset_counts = {}
        for press_id, seq in sorted(sequences.items()):
            duration = seq['time'][-1] - seq['time'][0] if len(seq['time']) > 1 else 0
            offset = seq.get('offset', 'unknown')
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
            if len(sequences) <= 10:  # Only print details if few sequences
                print(f"  {press_id}: {len(seq['indices'])} samples, duration: {duration:.2f}s, Fz range: [{np.min(seq['fz']):.2f}, {np.max(seq['fz']):.2f}] N, offset: {offset}")
        
        if len(sequences) > 10:
            print(f"  ... and {len(sequences) - 10} more sequences")
        
        print(f"Sequences per offset: {offset_counts}")
        
        if len(sequences) == 0:
            print("  Warning: No complete sequences found. Skipping plots.")
            return
        
        print("\nGenerating plots...")
        
        # Remove 2 outliers per offset before plotting
        print(f"\nRemoving 2 outliers per offset before plotting...")
        for offset, seqs in sequences_by_offset.items():
            if len(seqs) > 2:
                # Convert sequences to format expected by remove_outliers
                seqs_dict = []
                for seq in seqs:
                    seq_dict = {
                        'offset': offset,
                        'duration': seq['time'][-1] - seq['time'][0] if len(seq['time']) > 1 else len(seq['fz']) / 100.0,
                        'max_fz': float(np.max(seq['fz'])),
                        'num_samples': len(seq['fz']),
                        'fz_max': float(np.max(seq['fz'])),
                    }
                    seqs_dict.append(seq_dict)
                
                # Remove outliers
                cleaned_seqs_dict, outlier_indices = remove_outliers(seqs_dict, z_threshold=3.0, remove_per_offset=2)
                if outlier_indices:
                    # Remove corresponding sequences
                    cleaned_seqs = [seq for i, seq in enumerate(seqs) if i not in outlier_indices]
                    sequences_by_offset[offset] = cleaned_seqs
                    print(f"  Offset {offset}: Removed {len(outlier_indices)} outliers, {len(seqs)} -> {len(cleaned_seqs)} sequences")
        
        # Extract stretch label from filename or file attributes
        stretch_label = "unknown"
        if "000pct" in args.h5_file.name or "stretch_0" in args.h5_file.name:
            stretch_label = "0%"
        elif "010pct" in args.h5_file.name or "stretch_10" in args.h5_file.name:
            stretch_label = "10%"
        elif "020pct" in args.h5_file.name or "stretch_20" in args.h5_file.name:
            stretch_label = "20%"
        
        # Try to get from file attributes
        try:
            with h5py.File(args.h5_file, 'r') as hf:
                if 'stretch_label' in hf.attrs:
                    stretch_label = hf.attrs['stretch_label'].decode('utf-8') if isinstance(hf.attrs['stretch_label'], bytes) else str(hf.attrs['stretch_label'])
        except:
            pass
        
        # Plot sequences by offset (separate plot for each of the 5 positions)
        print(f"\nGenerating plots for each offset point (5 separate plots)...")
        plot_sequences_by_offset(sequences_by_offset, args.output_dir, stretch_label)
        
        # Plot mean and std deviation for each position
        print(f"\nGenerating mean/std plots for each offset point...")
        plot_mean_std_by_position(sequences_by_offset, args.output_dir, stretch_label)
        
        # Plot all sequences
        plot_all_sequences(sequences, args.output_dir / "raw_data_all_sequences.png", max_sequences=50)
        
        # Plot single sequence (first continuous press sequence found)
        if len(sequences) > 0:
            plot_single_sequence(sequences, args.output_dir / "raw_data_single_sequence.png")
            
            # Plot Fz FT vs Magnetic sensor channel 8
            plot_fz_ft_vs_magnetic(sequences, args.output_dir / "raw_data_fz_ft_vs_magnetic_ch8.png")
            
            # Plot only Magnetic sensor channel 8
            plot_magnetic_only(sequences, args.output_dir / "raw_data_magnetic_ch8_only.png")
        
        # Plot all sequences of magnetic sensor channel 8
        plot_all_magnetic_sequences(sequences, args.output_dir / "raw_data_magnetic_ch8_all_sequences.png")
        
        # Plot each magnetic sensor channel separately
        plot_all_magnetic_channels(sequences, args.output_dir)
        
        # Summary statistics
        if len(sequences) > 0:
            plot_summary_statistics(sequences, args.output_dir / "raw_data_summary.png")
    
    print(f"\nAll plots saved to: {args.output_dir}")

def plot_fz_ft_vs_magnetic(sequences: Dict[str, Dict], output_path: Path, press_id: str = None):
    """Plot Fz from FT sensor vs magnetic sensor channel 8 (absolute value, first sequence only)."""
    if press_id is None:
        # Use the first sequence (sorted by key)
        press_id = sorted(sequences.keys())[0]
    
    if press_id not in sequences:
        print(f"  Warning: Press ID '{press_id}' not found, using first available")
        press_id = sorted(sequences.keys())[0]
    
    seq = sequences[press_id]
    
    if 'mag_ch8_z' not in seq:
        print(f"  Warning: Magnetic sensor data not available for {press_id}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Fz from FT sensor on left y-axis (N) - USE ABSOLUTE VALUE
    fz_abs = np.abs(seq['fz'])
    ax.plot(seq['time'], fz_abs, 'b-', linewidth=2, label='Fz FT Sensor |Fz|', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Fz FT Sensor |Fz| (N)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='b', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot magnetic sensor channel 8, Z component on right y-axis (digits) - USE ABSOLUTE VALUE
    mag_ch8_abs = np.abs(seq['mag_ch8_z'])
    ax2 = ax.twinx()
    ax2.plot(seq['time'], mag_ch8_abs, 'r-', linewidth=2, label='Magnetic Sensor Ch8 |Z|', alpha=0.8)
    ax2.set_ylabel('Magnetic Sensor Ch8 |Z| [digits]', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
    ax.set_title(f'Fz FT Sensor |Fz| (N) vs Magnetic Sensor Channel 8 |Z| [digits] - First Sequence: {press_id}', fontsize=14, fontweight='bold')
    
    # Add text box with statistics
    stats_text = f"""Statistics (First Sequence):
Samples: {len(seq['indices'])}
Duration: {seq['time'][-1] - seq['time'][0]:.2f} s
Fz FT |Fz| range: [{np.min(fz_abs):.2f}, {np.max(fz_abs):.2f}] N
Fz FT |Fz| mean: {np.mean(fz_abs):.2f} N
Mag Ch8 |Z| range: [{np.min(mag_ch8_abs):.2f}, {np.max(mag_ch8_abs):.2f}]
Mag Ch8 |Z| mean: {np.mean(mag_ch8_abs):.2f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} (First Sequence: {press_id}, using |Z|)")

def plot_magnetic_only(sequences: Dict[str, Dict], output_path: Path, press_id: str = None):
    """Plot only magnetic sensor channel 8 (Z component)."""
    if press_id is None:
        # Find first sequence that looks like a continuous press (not lift, start, etc.)
        candidate_ids = [pid for pid in sorted(sequences.keys()) 
                        if not any(x in pid.lower() for x in ['lift', 'start', 'controlled'])]
        if candidate_ids:
            press_id = candidate_ids[0]
        else:
            press_id = sorted(sequences.keys())[0]
    
    if press_id not in sequences:
        print(f"  Warning: Press ID '{press_id}' not found, using first available")
        press_id = sorted(sequences.keys())[0]
    
    seq = sequences[press_id]
    
    if 'mag_ch8_z' not in seq:
        print(f"  Warning: Magnetic sensor data not available for {press_id}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot magnetic sensor channel 8, Z component
    ax.plot(seq['time'], seq['mag_ch8_z'], 'r-', linewidth=2, label='Magnetic Sensor Ch8 (Z)', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Magnetic Sensor Ch8 (Z) [digits]', fontsize=12)
    ax.set_title(f'Magnetic Sensor Channel 8 (Z) - Sequence: {press_id}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Add text box with statistics
    stats_text = f"""Statistics:
Samples: {len(seq['indices'])}
Duration: {seq['time'][-1] - seq['time'][0]:.2f} s
Mag Ch8 Z range: [{np.min(seq['mag_ch8_z']):.2f}, {np.max(seq['mag_ch8_z']):.2f}] digits
Mag Ch8 Z mean: {np.mean(seq['mag_ch8_z']):.2f} digits
Mag Ch8 Z std: {np.std(seq['mag_ch8_z']):.2f} digits"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} (Press ID: {press_id})")

def plot_all_magnetic_sequences(sequences: Dict[str, Dict], output_path: Path, max_sequences: int = 50):
    """Plot all sequences of magnetic sensor channel 8 (Z component) overlapped, all starting from 0s."""
    # Filter sequences that have magnetic data
    sequences_with_mag = {pid: seq for pid, seq in sequences.items() if 'mag_ch8_z' in seq}
    
    if not sequences_with_mag:
        print("  Warning: No sequences with magnetic sensor data found")
        return
    
    # Sort by press_id for consistent ordering
    sorted_press_ids = sorted(sequences_with_mag.keys())
    
    # Limit number of sequences to plot
    if len(sorted_press_ids) > max_sequences:
        print(f"  Plotting first {max_sequences} sequences (out of {len(sorted_press_ids)})")
        sorted_press_ids = sorted_press_ids[:max_sequences]
    
    n_sequences = len(sorted_press_ids)
    if n_sequences == 0:
        print("  Warning: No sequences to plot")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_sequences, 20)))
    
    for i, press_id in enumerate(sorted_press_ids):
        seq = sequences_with_mag[press_id]
        color = colors[i % len(colors)]
        ax.plot(seq['time'], seq['mag_ch8_z'], 
                label=f"{press_id} ({len(seq['indices'])} samples)", 
                color=color, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Magnetic Sensor Ch8 (Z) [digits]', fontsize=12)
    ax.set_title(f'Magnetic Sensor Channel 8 (Z) - All Sequences Overlapped ({n_sequences} sequences)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} ({n_sequences} sequences)")

def plot_all_magnetic_channels(sequences: Dict[str, Dict], output_dir: Path, max_sequences: int = 50):
    """Plot all sequences for each magnetic sensor channel (0-14), overlapped like FT plots."""
    # Filter sequences that have magnetic data
    sequences_with_mag = {pid: seq for pid, seq in sequences.items() if 'mag_all_channels' in seq}
    
    if not sequences_with_mag:
        print("  Warning: No sequences with magnetic sensor data found")
        return
    
    # Get number of channels (should be 15)
    first_seq = list(sequences_with_mag.values())[0]
    n_channels = first_seq['mag_all_channels'].shape[1]  # Should be 15
    
    print(f"  Generating plots for {n_channels} magnetic sensor channels...")
    
    # Sort by press_id for consistent ordering
    sorted_press_ids = sorted(sequences_with_mag.keys())
    
    # Limit number of sequences to plot
    if len(sorted_press_ids) > max_sequences:
        print(f"  Plotting first {max_sequences} sequences (out of {len(sorted_press_ids)})")
        sorted_press_ids = sorted_press_ids[:max_sequences]
    
    # Calculate max duration across all sequences for x-axis limit
    max_duration = 0
    for press_id in sorted_press_ids:
        seq = sequences_with_mag[press_id]
        if len(seq['time']) > 0:
            max_duration = max(max_duration, seq['time'][-1])
    
    # Color palette for different sequences
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_press_ids)))
    
    # Create a plot for each channel
    for ch in range(n_channels):
        # Create figure with 3 subplots (X, Y, Z) side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        
        # Plot all sequences overlapped for each component (X, Y, Z)
        for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes)):
            for seq_idx, press_id in enumerate(sorted_press_ids):
                seq = sequences_with_mag[press_id]
                color = colors[seq_idx % len(colors)]
                
                # Check if sequence starts from 0s
                if len(seq['time']) > 0 and abs(seq['time'][0]) > 0.01:
                    print(f"    Warning: Sequence {press_id} does not start from 0s (starts at {seq['time'][0]:.3f}s)")
                
                # Plot sequence - all should start from 0s
                mag_data = seq['mag_all_channels'][:, ch, comp_idx]  # [samples] for channel ch, component comp_idx
                ax.plot(seq['time'], mag_data, 
                       label=f"{press_id}" if comp_idx == 0 and seq_idx < 10 else "", 
                       color=color, alpha=0.7, linewidth=1.5)
            
            ax.set_ylabel(f'{comp_name} [digits]', fontsize=12)
            ax.set_title(f'Channel {ch+1} - {comp_name} Component', fontsize=12)
            if comp_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            # Set x-axis limit to show all sequences clearly overlapped
            ax.set_xlim(0, max_duration * 1.05)  # Add 5% margin
            ax.set_xlabel('Time (s)', fontsize=11)
        
        plt.suptitle(f'Magnetic Sensor Channel {ch+1} - All Sequences Overlapped ({len(sorted_press_ids)} sequences)', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_path = output_dir / f"raw_data_magnetic_ch{ch+1}_all_sequences.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path.name}")
    
    print(f"  Generated {n_channels} plots (one per channel)")

if __name__ == "__main__":
    main()
            offset = seq.get('offset', 'unknown')
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
            if len(sequences) <= 10:  # Only print details if few sequences
                print(f"  {press_id}: {len(seq['indices'])} samples, duration: {duration:.2f}s, Fz range: [{np.min(seq['fz']):.2f}, {np.max(seq['fz']):.2f}] N, offset: {offset}")
        
        if len(sequences) > 10:
            print(f"  ... and {len(sequences) - 10} more sequences")
        
        print(f"Sequences per offset: {offset_counts}")
        
        if len(sequences) == 0:
            print("  Warning: No complete sequences found. Skipping plots.")
            return
        
        print("\nGenerating plots...")
        
        # Remove 2 outliers per offset before plotting
        print(f"\nRemoving 2 outliers per offset before plotting...")
        for offset, seqs in sequences_by_offset.items():
            if len(seqs) > 2:
                # Convert sequences to format expected by remove_outliers
                seqs_dict = []
                for seq in seqs:
                    seq_dict = {
                        'offset': offset,
                        'duration': seq['time'][-1] - seq['time'][0] if len(seq['time']) > 1 else len(seq['fz']) / 100.0,
                        'max_fz': float(np.max(seq['fz'])),
                        'num_samples': len(seq['fz']),
                        'fz_max': float(np.max(seq['fz'])),
                    }
                    seqs_dict.append(seq_dict)
                
                # Remove outliers
                cleaned_seqs_dict, outlier_indices = remove_outliers(seqs_dict, z_threshold=3.0, remove_per_offset=2)
                if outlier_indices:
                    # Remove corresponding sequences
                    cleaned_seqs = [seq for i, seq in enumerate(seqs) if i not in outlier_indices]
                    sequences_by_offset[offset] = cleaned_seqs
                    print(f"  Offset {offset}: Removed {len(outlier_indices)} outliers, {len(seqs)} -> {len(cleaned_seqs)} sequences")
        
        # Extract stretch label from filename or file attributes
        stretch_label = "unknown"
        if "000pct" in args.h5_file.name or "stretch_0" in args.h5_file.name:
            stretch_label = "0%"
        elif "010pct" in args.h5_file.name or "stretch_10" in args.h5_file.name:
            stretch_label = "10%"
        elif "020pct" in args.h5_file.name or "stretch_20" in args.h5_file.name:
            stretch_label = "20%"
        
        # Try to get from file attributes
        try:
            with h5py.File(args.h5_file, 'r') as hf:
                if 'stretch_label' in hf.attrs:
                    stretch_label = hf.attrs['stretch_label'].decode('utf-8') if isinstance(hf.attrs['stretch_label'], bytes) else str(hf.attrs['stretch_label'])
        except:
            pass
        
        # Plot sequences by offset (separate plot for each of the 5 positions)
        print(f"\nGenerating plots for each offset point (5 separate plots)...")
        plot_sequences_by_offset(sequences_by_offset, args.output_dir, stretch_label)
        
        # Plot mean and std deviation for each position
        print(f"\nGenerating mean/std plots for each offset point...")
        plot_mean_std_by_position(sequences_by_offset, args.output_dir, stretch_label)
        
        # Plot all sequences
        plot_all_sequences(sequences, args.output_dir / "raw_data_all_sequences.png", max_sequences=50)
        
        # Plot single sequence (first continuous press sequence found)
        if len(sequences) > 0:
            plot_single_sequence(sequences, args.output_dir / "raw_data_single_sequence.png")
            
            # Plot Fz FT vs Magnetic sensor channel 8
            plot_fz_ft_vs_magnetic(sequences, args.output_dir / "raw_data_fz_ft_vs_magnetic_ch8.png")
            
            # Plot only Magnetic sensor channel 8
            plot_magnetic_only(sequences, args.output_dir / "raw_data_magnetic_ch8_only.png")
        
        # Plot all sequences of magnetic sensor channel 8
        plot_all_magnetic_sequences(sequences, args.output_dir / "raw_data_magnetic_ch8_all_sequences.png")
        
        # Plot each magnetic sensor channel separately
        plot_all_magnetic_channels(sequences, args.output_dir)
        
        # Summary statistics
        if len(sequences) > 0:
            plot_summary_statistics(sequences, args.output_dir / "raw_data_summary.png")
    
    print(f"\nAll plots saved to: {args.output_dir}")

def plot_fz_ft_vs_magnetic(sequences: Dict[str, Dict], output_path: Path, press_id: str = None):
    """Plot Fz from FT sensor vs magnetic sensor channel 8 (absolute value, first sequence only)."""
    if press_id is None:
        # Use the first sequence (sorted by key)
        press_id = sorted(sequences.keys())[0]
    
    if press_id not in sequences:
        print(f"  Warning: Press ID '{press_id}' not found, using first available")
        press_id = sorted(sequences.keys())[0]
    
    seq = sequences[press_id]
    
    if 'mag_ch8_z' not in seq:
        print(f"  Warning: Magnetic sensor data not available for {press_id}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Fz from FT sensor on left y-axis (N) - USE ABSOLUTE VALUE
    fz_abs = np.abs(seq['fz'])
    ax.plot(seq['time'], fz_abs, 'b-', linewidth=2, label='Fz FT Sensor |Fz|', alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Fz FT Sensor |Fz| (N)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='b', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot magnetic sensor channel 8, Z component on right y-axis (digits) - USE ABSOLUTE VALUE
    mag_ch8_abs = np.abs(seq['mag_ch8_z'])
    ax2 = ax.twinx()
    ax2.plot(seq['time'], mag_ch8_abs, 'r-', linewidth=2, label='Magnetic Sensor Ch8 |Z|', alpha=0.8)
    ax2.set_ylabel('Magnetic Sensor Ch8 |Z| [digits]', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
    ax.set_title(f'Fz FT Sensor |Fz| (N) vs Magnetic Sensor Channel 8 |Z| [digits] - First Sequence: {press_id}', fontsize=14, fontweight='bold')
    
    # Add text box with statistics
    stats_text = f"""Statistics (First Sequence):
Samples: {len(seq['indices'])}
Duration: {seq['time'][-1] - seq['time'][0]:.2f} s
Fz FT |Fz| range: [{np.min(fz_abs):.2f}, {np.max(fz_abs):.2f}] N
Fz FT |Fz| mean: {np.mean(fz_abs):.2f} N
Mag Ch8 |Z| range: [{np.min(mag_ch8_abs):.2f}, {np.max(mag_ch8_abs):.2f}]
Mag Ch8 |Z| mean: {np.mean(mag_ch8_abs):.2f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} (First Sequence: {press_id}, using |Z|)")

def plot_magnetic_only(sequences: Dict[str, Dict], output_path: Path, press_id: str = None):
    """Plot only magnetic sensor channel 8 (Z component)."""
    if press_id is None:
        # Find first sequence that looks like a continuous press (not lift, start, etc.)
        candidate_ids = [pid for pid in sorted(sequences.keys()) 
                        if not any(x in pid.lower() for x in ['lift', 'start', 'controlled'])]
        if candidate_ids:
            press_id = candidate_ids[0]
        else:
            press_id = sorted(sequences.keys())[0]
    
    if press_id not in sequences:
        print(f"  Warning: Press ID '{press_id}' not found, using first available")
        press_id = sorted(sequences.keys())[0]
    
    seq = sequences[press_id]
    
    if 'mag_ch8_z' not in seq:
        print(f"  Warning: Magnetic sensor data not available for {press_id}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot magnetic sensor channel 8, Z component
    ax.plot(seq['time'], seq['mag_ch8_z'], 'r-', linewidth=2, label='Magnetic Sensor Ch8 (Z)', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Magnetic Sensor Ch8 (Z) [digits]', fontsize=12)
    ax.set_title(f'Magnetic Sensor Channel 8 (Z) - Sequence: {press_id}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Add text box with statistics
    stats_text = f"""Statistics:
Samples: {len(seq['indices'])}
Duration: {seq['time'][-1] - seq['time'][0]:.2f} s
Mag Ch8 Z range: [{np.min(seq['mag_ch8_z']):.2f}, {np.max(seq['mag_ch8_z']):.2f}] digits
Mag Ch8 Z mean: {np.mean(seq['mag_ch8_z']):.2f} digits
Mag Ch8 Z std: {np.std(seq['mag_ch8_z']):.2f} digits"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} (Press ID: {press_id})")

def plot_all_magnetic_sequences(sequences: Dict[str, Dict], output_path: Path, max_sequences: int = 50):
    """Plot all sequences of magnetic sensor channel 8 (Z component) overlapped, all starting from 0s."""
    # Filter sequences that have magnetic data
    sequences_with_mag = {pid: seq for pid, seq in sequences.items() if 'mag_ch8_z' in seq}
    
    if not sequences_with_mag:
        print("  Warning: No sequences with magnetic sensor data found")
        return
    
    # Sort by press_id for consistent ordering
    sorted_press_ids = sorted(sequences_with_mag.keys())
    
    # Limit number of sequences to plot
    if len(sorted_press_ids) > max_sequences:
        print(f"  Plotting first {max_sequences} sequences (out of {len(sorted_press_ids)})")
        sorted_press_ids = sorted_press_ids[:max_sequences]
    
    n_sequences = len(sorted_press_ids)
    if n_sequences == 0:
        print("  Warning: No sequences to plot")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_sequences, 20)))
    
    for i, press_id in enumerate(sorted_press_ids):
        seq = sequences_with_mag[press_id]
        color = colors[i % len(colors)]
        ax.plot(seq['time'], seq['mag_ch8_z'], 
                label=f"{press_id} ({len(seq['indices'])} samples)", 
                color=color, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Magnetic Sensor Ch8 (Z) [digits]', fontsize=12)
    ax.set_title(f'Magnetic Sensor Channel 8 (Z) - All Sequences Overlapped ({n_sequences} sequences)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name} ({n_sequences} sequences)")

def plot_all_magnetic_channels(sequences: Dict[str, Dict], output_dir: Path, max_sequences: int = 50):
    """Plot all sequences for each magnetic sensor channel (0-14), overlapped like FT plots."""
    # Filter sequences that have magnetic data
    sequences_with_mag = {pid: seq for pid, seq in sequences.items() if 'mag_all_channels' in seq}
    
    if not sequences_with_mag:
        print("  Warning: No sequences with magnetic sensor data found")
        return
    
    # Get number of channels (should be 15)
    first_seq = list(sequences_with_mag.values())[0]
    n_channels = first_seq['mag_all_channels'].shape[1]  # Should be 15
    
    print(f"  Generating plots for {n_channels} magnetic sensor channels...")
    
    # Sort by press_id for consistent ordering
    sorted_press_ids = sorted(sequences_with_mag.keys())
    
    # Limit number of sequences to plot
    if len(sorted_press_ids) > max_sequences:
        print(f"  Plotting first {max_sequences} sequences (out of {len(sorted_press_ids)})")
        sorted_press_ids = sorted_press_ids[:max_sequences]
    
    # Calculate max duration across all sequences for x-axis limit
    max_duration = 0
    for press_id in sorted_press_ids:
        seq = sequences_with_mag[press_id]
        if len(seq['time']) > 0:
            max_duration = max(max_duration, seq['time'][-1])
    
    # Color palette for different sequences
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_press_ids)))
    
    # Create a plot for each channel
    for ch in range(n_channels):
        # Create figure with 3 subplots (X, Y, Z) side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        
        # Plot all sequences overlapped for each component (X, Y, Z)
        for comp_idx, (comp_name, ax) in enumerate(zip(['X', 'Y', 'Z'], axes)):
            for seq_idx, press_id in enumerate(sorted_press_ids):
                seq = sequences_with_mag[press_id]
                color = colors[seq_idx % len(colors)]
                
                # Check if sequence starts from 0s
                if len(seq['time']) > 0 and abs(seq['time'][0]) > 0.01:
                    print(f"    Warning: Sequence {press_id} does not start from 0s (starts at {seq['time'][0]:.3f}s)")
                
                # Plot sequence - all should start from 0s
                mag_data = seq['mag_all_channels'][:, ch, comp_idx]  # [samples] for channel ch, component comp_idx
                ax.plot(seq['time'], mag_data, 
                       label=f"{press_id}" if comp_idx == 0 and seq_idx < 10 else "", 
                       color=color, alpha=0.7, linewidth=1.5)
            
            ax.set_ylabel(f'{comp_name} [digits]', fontsize=12)
            ax.set_title(f'Channel {ch+1} - {comp_name} Component', fontsize=12)
            if comp_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            # Set x-axis limit to show all sequences clearly overlapped
            ax.set_xlim(0, max_duration * 1.05)  # Add 5% margin
            ax.set_xlabel('Time (s)', fontsize=11)
        
        plt.suptitle(f'Magnetic Sensor Channel {ch+1} - All Sequences Overlapped ({len(sorted_press_ids)} sequences)', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        
        output_path = output_dir / f"raw_data_magnetic_ch{ch+1}_all_sequences.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_path.name}")
    
    print(f"  Generated {n_channels} plots (one per channel)")

if __name__ == "__main__":
    main()