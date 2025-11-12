# Mag_Tec_Models

Standalone repository for the MagTec single-point and multi-point skin characterisation pipelines.
The repository mirrors the structure used on the robot workstation and can be published directly on
GitHub.

## Layout

- `src/franka_controller/`
  - `config.py` – central configuration and path management
  - `franka_skin_test.py` – full-grid data collection controller
  - `franka_skin_test_single_point.py` – single-point collection routine (multi-stretch)
  - `franka_skin_test_multiple_points.py` – centre + neighbourhood collection with automatic training
  - `teleop_franka_keyboard.py` – keyboard teleoperation with live visualisation
- `src/validation_tests/`
  - `real_time_predictor.py` – GUI for live sensor monitoring and model predictions
  - `franka_validation_test.py` – scripted validation trajectory without logging
- `src/training/`
  - `evaluate_single_point_stretch.py` – training + evaluation pipeline for the single-point dataset
  - `import_dataset.py` – converts external HDF5 recordings into the standard run structure
  - `train_simulation_dataset.py` – trains baseline models straight from simulation exports (no robot data)
  - `train_simulation_positions.py` – same as above but also learns contact-location classes when multiple probe points are present
- `doc/`
  - `single_point_validation.tex` – scientific report describing the pipeline

## 1. Data-Collection Workflow (robot + GUI)

1. Adjust `src/franka_controller/config.py` (robot IP, serial ports, reference pose, press parameters).
2. Activate the `franka_interface` environment and run one of:
   - **Single point** (centre + NW/NE/SE/SW offsets):
     ```bash
     cd /home/gabriele/franka_interface/Mag_Tec_Models
     conda activate franka_interface
     python3 src/franka_controller/franka_skin_test_single_point.py
     ```
   - **Multi point** (centre + offsets 4–12, optional exploration step):
     ```bash
     python3 src/franka_controller/franka_skin_test_multiple_points.py
     ```
3. Each acquisition auto-generates a run directory (e.g. `data/2.5mm_single_test1/` or
   `data/Multiple_Points/2.5mm_single_test1/`) containing:
   - Three HDF5 files (`…_stretch_000pct.h5`, `…_stretch_010pct.h5`, `…_stretch_020pct.h5`).
   - `models/` with the trained Random Forest artefacts (per-stretch, pooled, gated).
   - `<run_label>_metrics.json` summarising RMSE/STD, classification accuracy, and KPM pass/fail flags.
4. The scripts launch the training pipeline automatically after all stretches finish. To rerun later:
   ```bash
   python3 src/training/evaluate_single_point_stretch.py \
       --data-root data/2.5mm_single_test1 \
       --report data/2.5mm_single_test1/2.5mm_single_test1_metrics.json
   ```
5. Optional live visualisation: `python3 src/validation_tests/real_time_predictor.py`. It shares the
   same calibration pipeline as the data-collection scripts.

## 2. Training & KPIs from Existing HDF5 Datasets

Teams that already have HDF5 recordings (e.g., simulation exports) can reuse the analysis pipeline
without running the robot:

1. Use the import helper to copy the external datasets into the standard run structure and trigger
   evaluation:
   ```bash
   python3 src/training/import_dataset.py \
       path/to/sim_stretch_000pct.h5 path/to/sim_stretch_010pct.h5 path/to/sim_stretch_020pct.h5 \
       --run-label sim_test1
   ```
   - The script validates the input files (`press_summaries/sensors`, etc.), infers stretch labels
     from attributes when available, and writes the results to `data/Imported/sim_test1/`.
   - A metrics JSON and the trained models are stored alongside the copied HDF5 files.
   - Add `--no-eval` if you only want to stage the data for later analysis or `--overwrite` to replace
     an existing import.
2. If you only have the raw simulation exports (with the original `MagneticField` arrays and optional
   `forcesTest`), you can run the lightweight trainer directly:
   ```bash
   python3 src/training/train_simulation_dataset.py \
       path/to/sim_experiment_data_strecht_0.h5 \
       path/to/sim_experiment_data_strecht_10.h5 \
       path/to/sim_experiment_data_strecht_20.h5 \
       --run-label sim_raw_test1
   ```
   - The script flattens the 15×3 sensor grid into features, learns per-stretch force regressors when
     the `forcesTest` target is available, and always fits a stretch classifier.
   - Artefacts and metrics are written to `data/Imported/<run_label>/`.
3. If the simulation covers several probe points per stretch (e.g. centre plus four offsets), use the
   position-aware trainer:
   ```bash
   python3 src/training/train_simulation_positions.py \
       path/to/sim_experiment_data_strecht_0.h5 \
       path/to/sim_experiment_data_strecht_10.h5 \
       path/to/sim_experiment_data_strecht_20.h5 \
       --run-label simulation_points_test1 --overwrite
   ```
   - Outputs live under `data/Imported/<run_label>/`; expect a `models/` folder with per-stretch force
     regressors, position classifiers, and pooled stretch/position models.
   - The generated `<run_label>_metrics.json` summarises RMSE, residual STD, confusion matrices, and the
     rounded XY coordinates the trainer inferred from `IdenterPosition`.
4. To run the evaluation manually on an imported folder (or any compatible dataset):
   ```bash
   python3 src/training/evaluate_single_point_stretch.py \
       --data-root data/Imported/sim_test1 \
       --report data/Imported/sim_test1/sim_test1_metrics.json
   ```
3. The scientific report (`doc/single_point_validation.tex`) already contains tables for both
   single-point and multi-point runs; recompiling the document after training will capture the latest
   metrics.

If imported data lack `press_summaries` or stretch metadata, the helper script now fabricates them from
the raw sensor timelines. When possible, please ensure the simulation export matches the structure
generated by `franka_skin_test.py` (press snapshots plus continuous logging) to keep feature extraction
consistent. See Section “Simulation Dataset Evaluation” in `doc/single_point_validation.tex` for the
simulation run (`data/Imported/simulation_points_test1/`) and the resulting metrics.

## 3. Export Guidelines for the Simulation Team

To plug simulated runs into the same pipeline used on the robot, please structure each HDF5 file as
follows:

- `MagneticField` – required; shape `[samples, 15, 3]` with channels ordered (Bx, By, Bz). Sampling
  should remain uniform through each indentation cycle.
- `IdenterPosition` – required for position-aware training; shape `[samples, 3]` with probe position in
  metres (X, Y, Z). The pipeline rounds X/Y coordinates to 0.1mm to derive discrete contact labels.
- `forcesTest` – optional but recommended when the simulator can provide the ground-truth normal force;
  shape `[samples, 3]` (Fx, Fy, Fz). The training script falls back to stretch-only models if this
  dataset is absent or constant.
- `attrs/stretch` – optional string/int identifying the stretch percentage (e.g., `"10"`). If missing,
  the tooling infers the label from the filename (`stretch_010pct`, etc.).
- Time segmentation – if feasible, provide either `press_summaries/` groups or a dataset with press
  boundaries (start / end indices). The import helper can synthesise summaries automatically, but
  supplying them makes the statistics identical to the robot logs.

For best compatibility, mirror the naming convention already used on the robot:
`<run_label>_stretch_000pct.h5`, `<run_label>_stretch_010pct.h5`, `<run_label>_stretch_020pct.h5`, with
each file containing a single stretch condition.

### Example Usage: Testing Center + 4 Offsets

**Step 1: Data Collection Structure**

Export HDF5 files with the following structure for each stretch level:

**Required datasets:**
- `MagneticField` [samples, 15, 3]: Raw Bx, By, Bz readings from the 3×5 sensor grid
- `IdenterPosition` [samples, 3]: Probe position in metres (X, Y, Z) - used to automatically derive position labels
- `forcesTest` [samples, 3]: (Optional but recommended) Ground-truth forces Fx, Fy, Fz

**What to simulate:**
- Center position (0, 0)
- 4 offsets: NW, NE, SE, SW (or positions 4, 5, 6, 7, 9, 10, 11, 12)
- All at 3 stretch levels: 0%, 10%, 20%

**File naming examples:**
```
data/simulation/test1/
├── sim_experiment_data_stretch_0.h5
├── sim_experiment_data_stretch_10.h5
└── sim_experiment_data_stretch_20.h5
```

Or using the standard naming convention:
```
data/simulation/test1/
├── sim_test1_stretch_000pct.h5
├── sim_test1_stretch_010pct.h5
└── sim_test1_stretch_020pct.h5
```

**Step 2: Run Training Script**

Use `train_simulation_positions.py` when you have multiple probe points (center + offsets):

```bash
python3 src/training/train_simulation_positions.py \
    data/simulation/test1/sim_stretch_0.h5 \
    data/simulation/test1/sim_stretch_10.h5 \
    data/simulation/test1/sim_stretch_20.h5 \
    --run-label simulation_points_test1 \
    --overwrite
```

**What the script does automatically:**
1. Loads the HDF5 files
2. Infers stretch labels from filenames or attributes
3. Derives position labels by rounding `IdenterPosition` X/Y coordinates to 0.1mm
4. Trains per-stretch force regressors (if `forcesTest` is provided)
5. Trains per-stretch position classifiers (center + 4 offsets)
6. Trains pooled stretch and position classifiers
7. Saves all models and metrics

**Step 3: Where Results Are Collected**

All results are stored in:
```
data/Imported/<run_label>/
```

**Complete structure:**
```
data/Imported/simulation_points_test1/
├── simulation_points_test1_stretch_000pct.h5    # Original data (preserved)
├── simulation_points_test1_stretch_010pct.h5
├── simulation_points_test1_stretch_020pct.h5
├── models/                                       # Trained models
│   ├── simulation_points_test1_stretch_000pct_force_regressor.joblib
│   ├── simulation_points_test1_stretch_000pct_position_classifier.joblib
│   ├── simulation_points_test1_stretch_010pct_force_regressor.joblib
│   ├── simulation_points_test1_stretch_010pct_position_classifier.joblib
│   ├── simulation_points_test1_stretch_020pct_force_regressor.joblib
│   ├── simulation_points_test1_stretch_020pct_position_classifier.joblib
│   ├── simulation_points_test1_pooled_position_classifier.joblib
│   └── simulation_points_test1_pooled_stretch_classifier.joblib
└── simulation_points_test1_metrics.json          # Performance metrics
```

**Step 4: What's in the Metrics JSON**

The `simulation_points_test1_metrics.json` contains:
- Force regression metrics (RMSE, STD) per stretch
- Position classification accuracy per stretch
- Pooled position classifier accuracy
- Pooled stretch classifier accuracy
- Detected contact positions (rounded X/Y coordinates in mm)

**Important Notes:**
- The script automatically detects center and offsets from `IdenterPosition` coordinates by rounding to 0.1mm, so no manual labeling is needed
- Position labels are derived automatically (e.g., "x+0.0mm_y+0.0mm" for center, "x+2.9mm_y+2.0mm" for NE offset)
- If you only have single-point data (no multiple positions), use `train_simulation_dataset.py` instead
- For full KPI suite matching robot data, use `import_dataset.py` followed by `evaluate_single_point_stretch.py`

---

All commands assume the `franka_interface` conda environment with the required dependencies
(`pyfranka`, `pyserial`, `numpy`, `scikit-learn`, `matplotlib`, `h5py`, etc.).
