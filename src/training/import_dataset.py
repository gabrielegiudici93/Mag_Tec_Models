#!/usr/bin/env python3
"""Import external HDF5 datasets and evaluate them with the standard pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import h5py
import numpy as np

# Add src root to sys.path so we can reuse the project modules
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from franka_controller.config import DATA_DIR  # noqa: E402


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def infer_stretch_label(h5_path: Path) -> str | None:
    """Infer stretch label from file attributes, if present."""
    try:
        with h5py.File(h5_path, "r") as f:
            for key in ("stretch_label", "stretch", "stretch_name"):
                if key in f.attrs:
                    return _decode(f.attrs[key])
            for key in ("stretch_level", "stretch_value"):
                if key in f.attrs:
                    value = float(f.attrs[key])
                    percent = int(round(value * 100))
                    return f"stretch_{percent:03d}pct"
    except Exception:
        pass
    return None


def validate_file(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        if "press_summaries/sensors" in f:
            return
        if {"MagneticField", "forcesTest"}.issubset(f.keys()):
            return
        raise RuntimeError(
            f"{h5_path} does not contain the expected datasets "
            "(press_summaries/sensors or MagneticField + forcesTest)."
        )


def convert_simulation_file(src: Path, dest: Path, stretch_label: str) -> None:
    percent = 0
    try:
        percent = int("".join(filter(str.isdigit, stretch_label)))
    except Exception:
        pass
    stretch_value = percent / 100.0 if percent else np.nan

    with h5py.File(src, "r") as fin, h5py.File(dest, "w") as fout:
        magnetic = fin["MagneticField"][()]  # (samples, 15, 3)
        forces = fin["forcesTest"][()]       # (samples, 3)
        samples = magnetic.shape[0]

        # Create press summaries (one entry per time sample)
        fout.create_dataset("press_summaries/sensors", data=magnetic.astype(float))

        forces_full = np.zeros((samples, 6), dtype=float)
        forces_full[:, : forces.shape[1]] = forces
        fout.create_dataset("press_summaries/forces", data=forces_full)

        str_dtype = h5py.string_dtype(encoding="utf-8")
        metadata = []
        for idx in range(samples):
            metadata.append(
                json.dumps(
                    {
                        "press_id": f"sim_{idx:04d}",
                        "offset_key": "sim_center",
                        "stretch_label": stretch_label,
                        "stretch_level": stretch_value,
                        "press_depth_m": 0.0025,
                    }
                )
            )
        fout.create_dataset("press_summaries/metadata", data=np.array(metadata, dtype=str_dtype))

        # Store raw sequences for completeness
        fout.create_dataset("forces", data=forces_full)
        fout.create_dataset("magnetic_field", data=magnetic.astype(float))

        fout.attrs["source"] = "simulation"
        fout.attrs["stretch_label"] = stretch_label
        fout.attrs["stretch_level"] = stretch_value


def import_dataset(inputs: List[Path], run_label: str, overwrite: bool = False, evaluate: bool = True) -> Path:
    dest_root = DATA_DIR / "Imported" / run_label
    if dest_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination {dest_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    inferred_labels: List[str] = []
    for idx, src in enumerate(inputs, 1):
        validate_file(src)
        stretch_label = infer_stretch_label(src)
        if stretch_label is None:
            stretch_label = f"stretch_{idx:03d}"
        inferred_labels.append(stretch_label)

    for idx in range(len(inputs)):
        src = inputs[idx]
        label = inferred_labels[idx]
        target = dest_root / f"{run_label}_{label}.h5"
        with h5py.File(src, "r") as f:
            if "press_summaries/sensors" in f:
                shutil.copy2(src, target)
                print(f"Copied {src} -> {target}")
            elif {"MagneticField", "forcesTest"}.issubset(f.keys()):
                convert_simulation_file(src, target, label)
                print(f"Converted simulation file {src} -> {target}")
            else:
                raise RuntimeError(f"Unsupported structure for {src}")

    if evaluate:
        eval_script = SRC_ROOT / "training" / "evaluate_single_point_stretch.py"
        metrics_path = dest_root / f"{run_label}_metrics.json"
        cmd = [
            sys.executable,
            str(eval_script),
            "--data-root",
            str(dest_root),
            "--report",
            str(metrics_path),
        ]
        print("Running evaluation pipeline...")
        process = __import__("subprocess").run(cmd, cwd=REPO_ROOT)
        if process.returncode != 0:
            raise RuntimeError("Evaluation pipeline failed. See console output above.")
        else:
            print(f"Metrics written to {metrics_path}")

    return dest_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import external HDF5 datasets (e.g., simulation runs) into the standard run structure "
            "and evaluate them with the MagTec pipeline."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input HDF5 files to import")
    parser.add_argument("--run-label", type=str, default=None, help="Name for the imported run")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination folder")
    parser.add_argument("--no-eval", action="store_true", help="Skip automatic evaluation after import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    run_label = args.run_label
    if run_label is None:
        if len(inputs) == 1:
            run_label = inputs[0].stem
        else:
            run_label = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dest = import_dataset(inputs, run_label, overwrite=args.overwrite, evaluate=not args.no_eval)
    print(f"Imported dataset available at {dest}")


if __name__ == "__main__":
    main()
