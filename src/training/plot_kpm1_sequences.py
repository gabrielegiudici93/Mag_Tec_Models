#!/usr/bin/env python3
"""
Plot KPM1 sequences: Continuous indentation from Fmin to Fmax.

This script creates plots showing measured force (F) and estimated force (F^) 
during continuous indentation sequences, and computes effective resolution.

For each dataset (simulation and physical):
- One plot with a random sequence
- One plot with average across all sequences
- Table showing step, RMSE, and effective resolution
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import DATA_DIR  # noqa: E402

# Central neighbourhood sensor indices (0-based: corresponds to sensors 7,8,9 in 1-based)
CENTRAL_SENSOR_INDICES = [6, 7, 8]  # Sensors 7, 8, 9 (1-based): top, center, bottom


def extract_continuous_sequence(
    magnetic: np.ndarray,
    forces: np.ndarray,
    fz_index: int = 2,
    fmin: Optional[float] = None,
    fmax: float = 3.0,
    min_step: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a continuous indentation sequence from Fmin to Fmax.
    
    Args:
        magnetic: [samples, 15, 3] magnetic field data
        forces: [samples, 3] force data (Fx, Fy, Fz)
        fz_index: Index of Fz in forces array (default 2)
        fmin: Minimum force threshold (if None, uses first non-zero)
        fmax: Maximum force threshold
        min_step: Minimum force step to consider (to filter noise)
    
    Returns:
        Tuple of (magnetic_features, measured_Fz, sample_indices)
        where sample_indices are the original indices in the full dataset
    """
    fz = forces[:, fz_index]
    
    # Handle negative forces (simulation convention: forces are negative)
    # We'll work with absolute values for sequence detection
    fz_abs = np.abs(fz)
    
    # Find Fmin (first non-zero force in absolute terms)
    if fmin is None:
        non_zero_mask = fz_abs > 0.001  # 1mN threshold
        if np.any(non_zero_mask):
            fmin_abs = np.min(fz_abs[non_zero_mask])
            fmin = -fmin_abs if np.mean(fz[non_zero_mask]) < 0 else fmin_abs
        else:
            return np.array([]), np.array([]), np.array([])
    else:
        fmin_abs = abs(fmin)
    
    # Determine if forces are negative (simulation) or positive (physical)
    is_negative = np.mean(fz[np.abs(fz) > 0.001]) < 0 if np.any(np.abs(fz) > 0.001) else False
    
    if is_negative:
        # For negative forces: we want increasing magnitude (more negative = larger force)
        # So we look for decreasing fz values (from 0 to -fmax)
        # As force becomes more negative, absolute value increases
        fmax_abs = abs(fmax) if fmax > 0 else abs(fmin)
        valid_mask = (fz_abs >= fmin_abs) & (fz_abs <= fmax_abs) & (fz <= 0)
        # For negative forces, "increasing magnitude" means fz_abs increases (fz becomes more negative)
        direction_check = lambda curr_abs, prev_abs: curr_abs >= prev_abs  # Absolute value increases
    else:
        # For positive forces: normal increasing
        valid_mask = (fz >= fmin) & (fz <= fmax) & (fz >= 0)
        direction_check = lambda curr_abs, prev_abs: curr_abs >= prev_abs  # Absolute value increases
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    # Find continuous sequences
    valid_indices = np.where(valid_mask)[0]
    sequences = []
    current_seq = [valid_indices[0]]
    
    for i in range(1, len(valid_indices)):
        idx = valid_indices[i]
        prev_idx = valid_indices[i-1]
        
        # Check if force is changing in the right direction and step is meaningful
        force_step_abs = abs(fz[idx] - fz[prev_idx])
        if direction_check(fz_abs[idx], fz_abs[prev_idx]) and force_step_abs >= min_step and idx == prev_idx + 1:
            current_seq.append(idx)
        else:
            if len(current_seq) >= 5:  # Minimum sequence length
                sequences.append(current_seq)
            current_seq = [idx]
    
    if len(current_seq) >= 5:
        sequences.append(current_seq)
    
    if not sequences:
        return np.array([]), np.array([]), np.array([])
    
    # Select the longest sequence
    longest_seq = max(sequences, key=len)
    seq_indices = np.array(longest_seq)
    
    # Extract data for this sequence
    seq_magnetic = magnetic[seq_indices]  # [n, 15, 3]
    seq_fz = fz[seq_indices]  # [n]
    
    # Flatten magnetic data to features
    seq_features = seq_magnetic.reshape(len(seq_indices), -1)  # [n, 45]
    
    return seq_features, seq_fz, seq_indices


def load_physical_sequence(h5_path: Path, model_path: Path) -> Optional[Dict]:
    """Load continuous sequence from physical robot data."""
    try:
        with h5py.File(h5_path, "r") as f:
            # Prefer press_summaries (what model was trained on) if available
            use_press_summaries = "press_summaries/sensors" in f and "press_summaries/forces" in f
            
            if use_press_summaries:
                # Use press summaries (snapshots at max indentation)
                magnetic = f["press_summaries/sensors"][:]  # [n_press, 15, 3]
                forces = f["press_summaries/forces"][:]  # [n_press, 6]
                # Sort by force to create a sequence from low to high
                fz = forces[:, 2]
                sort_idx = np.argsort(np.abs(fz))  # Sort by absolute force magnitude
                magnetic = magnetic[sort_idx]
                forces = forces[sort_idx]
            elif "stretchmagtec" in f and "forces" in f:
                # Use continuous data, but filter to high-force region (where model was trained)
                magnetic = f["stretchmagtec"][:]  # [samples, 15, 3]
                forces = f["forces"][:]  # [samples, 6]
                fz = forces[:, 2]
                # Filter to high-force samples (model was trained on forces around -4 to -3 N)
                high_force_mask = np.abs(fz) > 2.0  # Use samples with |Fz| > 2N
                if np.sum(high_force_mask) < 5:
                    # If not enough high-force samples, use all
                    high_force_mask = np.abs(fz) > 0.5
                magnetic = magnetic[high_force_mask]
                forces = forces[high_force_mask]
                # Sort by force magnitude
                fz_filtered = forces[:, 2]
                sort_idx = np.argsort(np.abs(fz_filtered))
                magnetic = magnetic[sort_idx]
                forces = forces[sort_idx]
            else:
                return None
            
        # Load model
        model = joblib.load(model_path)
        
        # Check if model expects subset features
        is_subset_model = "subset" in model_path.name.lower()
        
        # Flatten magnetic data
        features = magnetic.reshape(len(magnetic), -1)  # [n, 45]
        fz_measured = forces[:, 2]  # [n]
        
        if len(features) == 0:
            return None
        
        # Apply feature selection if needed
        if is_subset_model:
            # Reshape to [n, 15, 3], select subset, then flatten to [n, 9]
            features_reshaped = features.reshape(len(features), 15, 3)
            features_subset = features_reshaped[:, CENTRAL_SENSOR_INDICES, :]
            features = features_subset.reshape(len(features), -1)
        
        # Predict forces
        fz_predicted = model.predict(features)
        
        # Compute metrics
        errors = fz_predicted - fz_measured
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_error = float(np.mean(errors))
        
        # Compute force steps (use absolute value for negative forces)
        force_steps = np.diff(fz_measured)
        force_steps_abs = np.abs(force_steps)
        force_steps_abs = force_steps_abs[force_steps_abs > 0.001]  # Filter meaningful steps
        avg_step = float(np.mean(force_steps_abs)) if len(force_steps_abs) > 0 else 0.0
        
        return {
            "features": features,
            "fz_measured": fz_measured,
            "fz_predicted": fz_predicted,
            "sample_indices": np.arange(len(fz_measured)),  # Sequential indices
            "rmse": rmse,
            "mean_error": mean_error,
            "avg_step": avg_step,
            "steps": force_steps_abs,
            "source": "physical",
            "file": h5_path.name,
        }
    except Exception as e:
        print(f"Error loading physical sequence from {h5_path}: {e}")
        return None


def load_simulation_sequence(h5_path: Path, model_path: Path) -> Optional[Dict]:
    """Load continuous sequence from simulation data."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "MagneticField" not in f or "forcesTest" not in f:
                return None
            
            magnetic = f["MagneticField"][:]  # [samples, 15, 3]
            forces = f["forcesTest"][:]  # [samples, 3]
            
        # Load model
        model = joblib.load(model_path)
        
        # Extract sequence
        features, fz_measured, indices = extract_continuous_sequence(magnetic, forces, fz_index=2)
        
        if len(features) == 0:
            return None
        
        # Predict forces
        fz_predicted = model.predict(features)
        
        # Compute metrics
        errors = fz_predicted - fz_measured
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_error = float(np.mean(errors))
        
        # Compute force steps (should be more uniform in simulation)
        # For negative forces, steps are negative, so take absolute value
        force_steps = np.diff(fz_measured)
        force_steps_abs = np.abs(force_steps)
        force_steps_abs = force_steps_abs[force_steps_abs > 0.001]
        avg_step = float(np.mean(force_steps_abs)) if len(force_steps_abs) > 0 else 0.0
        step_std = float(np.std(force_steps_abs)) if len(force_steps_abs) > 0 else 0.0
        
        return {
            "features": features,
            "fz_measured": fz_measured,
            "fz_predicted": fz_predicted,
            "sample_indices": indices,
            "rmse": rmse,
            "mean_error": mean_error,
            "avg_step": avg_step,
            "step_std": step_std,
            "steps": force_steps,
            "source": "simulation",
            "file": h5_path.name,
        }
    except Exception as e:
        print(f"Error loading simulation sequence from {h5_path}: {e}")
        return None


def plot_sequence(
    seq_data: Dict,
    title: str,
    output_path: Path,
    is_average: bool = False,
):
    """Plot a single sequence or average sequence."""
    fz_measured = seq_data["fz_measured"]
    fz_predicted = seq_data["fz_predicted"]
    sample_ids = np.arange(len(fz_measured))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sample_ids, fz_measured, "b-", label="Measured F", linewidth=2, marker="o", markersize=4)
    ax.plot(sample_ids, fz_predicted, "r--", label="Predicted F^", linewidth=2, marker="s", markersize=4)
    
    ax.set_xlabel("Sample ID", fontsize=12)
    ax.set_ylabel("Force Fz [N]", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    rmse = seq_data["rmse"]
    mean_err = seq_data["mean_error"]
    avg_step = seq_data["avg_step"]
    effective_res = avg_step + rmse if not is_average else avg_step + rmse
    
    textstr = f"RMSE: {rmse:.4f} N\nMean Error: {mean_err:.4f} N\n"
    if "step_std" in seq_data and seq_data["step_std"] is not None:
        textstr += f"Step: {avg_step:.4f} ± {seq_data['step_std']:.4f} N\n"
    else:
        textstr += f"Avg Step: {avg_step:.4f} N\n"
    textstr += f"Effective Resolution: {effective_res:.4f} N"
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def compute_average_sequence(sequences: List[Dict]) -> Dict:
    """Compute average sequence across multiple sequences."""
    if not sequences:
        return None
    
    # Find minimum length
    min_len = min(len(s["fz_measured"]) for s in sequences)
    
    # Interpolate all sequences to same length
    sample_ids = np.arange(min_len)
    fz_measured_avg = []
    fz_predicted_avg = []
    
    for seq in sequences:
        orig_ids = np.arange(len(seq["fz_measured"]))
        fz_m_interp = np.interp(sample_ids, orig_ids, seq["fz_measured"])
        fz_p_interp = np.interp(sample_ids, orig_ids, seq["fz_predicted"])
        fz_measured_avg.append(fz_m_interp)
        fz_predicted_avg.append(fz_p_interp)
    
    fz_measured_avg = np.mean(fz_measured_avg, axis=0)
    fz_predicted_avg = np.mean(fz_predicted_avg, axis=0)
    
    # Compute average metrics
    all_rmse = [s["rmse"] for s in sequences]
    all_mean_err = [s["mean_error"] for s in sequences]
    all_avg_steps = [s["avg_step"] for s in sequences]
    
    return {
        "fz_measured": fz_measured_avg,
        "fz_predicted": fz_predicted_avg,
        "rmse": float(np.mean(all_rmse)),
        "mean_error": float(np.mean(all_mean_err)),
        "avg_step": float(np.mean(all_avg_steps)),
        "step_std": float(np.std(all_avg_steps)) if "step_std" in sequences[0] else None,
        "source": sequences[0]["source"],
        "n_sequences": len(sequences),
    }


def create_summary_table(sequences: List[Dict], output_path: Path, source_type: str):
    """Create a summary table with metrics."""
    if not sequences:
        return
    
    # Collect metrics
    all_rmse = [s["rmse"] for s in sequences]
    all_avg_steps = [s["avg_step"] for s in sequences]
    all_effective_res = [s["avg_step"] + s["rmse"] for s in sequences]
    
    avg_rmse = np.mean(all_rmse)
    avg_step = np.mean(all_avg_steps)
    avg_eff_res = np.mean(all_effective_res)
    
    if source_type == "simulation" and sequences[0].get("step_std") is not None:
        step_std = np.mean([s.get("step_std", 0) for s in sequences])
        step_str = f"{avg_step:.4f} ± {step_std:.4f}"
    else:
        step_std = np.std(all_avg_steps)
        step_str = f"{avg_step:.4f} ± {step_std:.4f}"
    
    # Create table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    
    table_data = [
        ["Metric", "Value"],
        ["Number of Sequences", f"{len(sequences)}"],
        ["Average Step", f"{step_str} N"],
        ["Average RMSE", f"{avg_rmse:.4f} N"],
        ["Effective Resolution", f"{avg_eff_res:.4f} N"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    ax.set_title(f"KPM1 Sequence Summary - {source_type.capitalize()}", fontsize=12, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved table: {output_path}")


def find_physical_data_and_model(data_root: Path) -> Tuple[List[Path], Optional[Path]]:
    """Find physical HDF5 files and corresponding model."""
    h5_files = sorted(data_root.glob("**/*.h5"))
    physical_files = [f for f in h5_files if "stretch_" in f.name or "test" in f.name]
    
    # Look for model in models directory
    models_dir = data_root / "models"
    if not models_dir.exists():
        models_dir = data_root.parent / "models"
    
    model_file = None
    if models_dir.exists():
        # Prefer full model (all sensors), fallback to subset
        full_models = list(models_dir.glob("*FT_MAPPING_FZ_MODEL*.joblib"))
        if not full_models:
            full_models = [m for m in models_dir.glob("*force_regressor*.joblib") if "subset" not in m.name.lower()]
        subset_models = [m for m in models_dir.glob("*force_regressor*.joblib") if "subset" in m.name.lower()]
        
        # Prefer full model, but use subset if that's all we have
        if full_models:
            model_file = full_models[0]
        elif subset_models:
            model_file = subset_models[0]
    
    return physical_files[:10], model_file  # Limit to 10 files


def find_simulation_data_and_model(data_root: Path) -> Tuple[List[Path], Optional[Path]]:
    """Find simulation HDF5 files and corresponding model."""
    h5_files = sorted(data_root.glob("*.h5"))
    # Check which files actually have MagneticField dataset
    sim_files = []
    for f in h5_files:
        try:
            with h5py.File(f, "r") as hf:
                if "MagneticField" in hf:
                    sim_files.append(f)
        except Exception:
            pass
    
    # Look for model in models directory
    models_dir = data_root / "models"
    if not models_dir.exists():
        models_dir = data_root.parent / "models"
    
    model_file = None
    if models_dir.exists():
        # Try to find per-stretch or combined model
        candidates = list(models_dir.glob("*force_regressor*.joblib"))
        if candidates:
            # Prefer combined model, otherwise use first
            combined = [c for c in candidates if "stretch" not in c.name]
            model_file = combined[0] if combined else candidates[0]
    
    return sim_files[:10], model_file  # Limit to 10 files


def main():
    parser = argparse.ArgumentParser(description="Plot KPM1 continuous indentation sequences")
    parser.add_argument("--physical-data-dir", type=Path, help="Physical data directory (auto-finds HDF5 files)")
    parser.add_argument("--physical-data", type=Path, help="Physical HDF5 file (single file)")
    parser.add_argument("--physical-model", type=Path, help="Physical force regressor model")
    parser.add_argument("--simulation-data-dir", type=Path, help="Simulation data directory (auto-finds HDF5 files)")
    parser.add_argument("--simulation-data", type=Path, help="Simulation HDF5 file (single file)")
    parser.add_argument("--simulation-model", type=Path, help="Simulation force regressor model")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "plots", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process physical data
    physical_sequences = []
    if args.physical_data_dir:
        print(f"Searching for physical data in {args.physical_data_dir}...")
        physical_files, model_file = find_physical_data_and_model(args.physical_data_dir)
        if model_file:
            print(f"Found {len(physical_files)} HDF5 files, using model: {model_file.name}")
            for h5_file in physical_files:
                seq = load_physical_sequence(h5_file, model_file)
                if seq:
                    physical_sequences.append(seq)
        else:
            print("⚠️  No model found for physical data")
    elif args.physical_data and args.physical_model:
        print("Processing single physical data file...")
        seq = load_physical_sequence(args.physical_data, args.physical_model)
        if seq:
            physical_sequences.append(seq)
    
    if physical_sequences:
        print(f"Loaded {len(physical_sequences)} physical sequences")
        # Random sequence plot
        random_seq = random.choice(physical_sequences)
        plot_sequence(
            random_seq,
            f"Physical Data - Random Sequence\n{random_seq['file']}",
            output_dir / "physical_random_sequence.png",
            is_average=False,
        )
        # Average sequence plot
        if len(physical_sequences) > 1:
            avg_seq = compute_average_sequence(physical_sequences)
            if avg_seq:
                plot_sequence(
                    avg_seq,
                    f"Physical Data - Average Across {avg_seq['n_sequences']} Sequences",
                    output_dir / "physical_average_sequence.png",
                    is_average=True,
                )
        # Summary table
        create_summary_table(
            physical_sequences,
            output_dir / "physical_summary_table.png",
            "physical",
        )
    
    # Process simulation data
    sim_sequences = []
    if args.simulation_data_dir:
        print(f"Searching for simulation data in {args.simulation_data_dir}...")
        sim_files, model_file = find_simulation_data_and_model(args.simulation_data_dir)
        if model_file:
            print(f"Found {len(sim_files)} HDF5 files, using model: {model_file.name}")
            for h5_file in sim_files:
                seq = load_simulation_sequence(h5_file, model_file)
                if seq:
                    sim_sequences.append(seq)
        else:
            print("⚠️  No model found for simulation data")
    elif args.simulation_data and args.simulation_model:
        print("Processing single simulation data file...")
        seq = load_simulation_sequence(args.simulation_data, args.simulation_model)
        if seq:
            sim_sequences.append(seq)
    
    if sim_sequences:
        print(f"Loaded {len(sim_sequences)} simulation sequences")
        # Random sequence plot
        random_seq = random.choice(sim_sequences)
        plot_sequence(
            random_seq,
            f"Simulation Data - Random Sequence\n{random_seq['file']}",
            output_dir / "simulation_random_sequence.png",
            is_average=False,
        )
        # Average sequence plot
        if len(sim_sequences) > 1:
            avg_seq = compute_average_sequence(sim_sequences)
            if avg_seq:
                plot_sequence(
                    avg_seq,
                    f"Simulation Data - Average Across {avg_seq['n_sequences']} Sequences",
                    output_dir / "simulation_average_sequence.png",
                    is_average=True,
                )
        # Summary table
        create_summary_table(
            sim_sequences,
            output_dir / "simulation_summary_table.png",
            "simulation",
        )
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

