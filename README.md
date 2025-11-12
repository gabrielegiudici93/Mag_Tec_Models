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
2. **Single point only (no position classification):** If you only have one contact location per stretch
   (e.g., only center, no offsets), use the lightweight trainer:
   ```bash
   python3 src/training/train_simulation_dataset.py \
       path/to/sim_experiment_data_stretch_0.h5 \
       path/to/sim_experiment_data_stretch_10.h5 \
       path/to/sim_experiment_data_stretch_20.h5 \
       --run-label sim_raw_test1
   ```
   - The script flattens the 15×3 sensor grid into features, learns per-stretch force regressors when
     the `forcesTest` target is available, and always fits a stretch classifier.
   - **Note:** This script does NOT train position classifiers (only force and stretch models).
   - Artefacts and metrics are written to `data/Imported/<run_label>/`.

3. **Multiple points (center + offsets):** If the simulation covers multiple probe points per stretch
   (e.g., center + 4 offsets = 5 points, or center + 12 offsets = 13 points), use the position-aware
   trainer:
   ```bash
   python3 src/training/train_simulation_positions.py \
       path/to/sim_experiment_data_stretch_0.h5 \
       path/to/sim_experiment_data_stretch_10.h5 \
       path/to/sim_experiment_data_stretch_20.h5 \
       --run-label simulation_points_test1 --overwrite
   ```
   - **Required:** `IdenterPosition` dataset must be present (used to automatically derive position labels).
   - The script automatically detects all unique contact positions by rounding X/Y coordinates to 0.1mm.
   - Works for any number of positions: 5 points (center + 4 offsets), 13 points (center + 12 offsets), etc.
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

5. **View metrics in table format:** To print metrics in a LaTeX-style table format similar to the
   scientific report:
   
   **For robot datasets:**
   ```bash
   # Single-point dataset
   python3 src/training/print_metrics_tables.py data/2.5mm_single_test1/2.5mm_single_test1_metrics.json
   
   # Multi-point dataset
   python3 src/training/print_metrics_tables.py data/Multiple_Points/2.5mm_single_test1/2.5mm_single_test1_metrics.json
   ```
   
   **For simulation datasets:**
   ```bash
   python3 src/training/print_metrics_tables.py data/Imported/simulation_points_test1/simulation_points_test1_metrics.json
   ```
   
   The script auto-detects the metrics format (robot vs. simulation) and prints force regression,
   position classification, and stretch classification tables. See example outputs below.

The scientific report (`doc/single_point_validation.tex`) already contains tables for both
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

### How Simulation Models Compare to Robot Models

**Similarities:**
- **Same model types**: Both use Random Forest for force regression, position classification, and stretch classification
- **Same input features**: Both use flattened magnetic field data (45 features: 15 sensors × 3 channels)
- **Same training methodology**: 70/30 train/test split, same hyperparameters (200-400 trees, unlimited depth)
- **Same output structure**: Models saved as `.joblib` files, metrics in JSON format

**Key Differences:**

1. **Position Detection Method:**
   - **Robot**: Uses predefined offset keys (center, nw, ne, se, sw, or positions 4-12) from robot configuration
   - **Simulation**: Automatically derives positions by rounding `IdenterPosition` X/Y coordinates to 0.1mm resolution
   - **Result**: Simulation labels look like "x+02.9mm_y+02.0mm" instead of "ne" or "4"

2. **Force Ground Truth:**
   - **Robot**: Uses FT-300S sensor mounted on robot end-effector
   - **Simulation**: Uses `forcesTest` dataset from simulation (if available)
   - **Note**: Simulation may have different noise characteristics

3. **Data Structure:**
   - **Robot**: Data collected with `press_summaries` structure (snapshots at max indentation)
   - **Simulation**: Raw continuous data; `train_simulation_positions.py` processes it directly
   - **Conversion**: `import_dataset.py` can convert simulation data to match robot structure

4. **Position Labels:**
   - **Robot**: Human-readable labels (center, nw, ne, se, sw, 4, 5, 6, etc.)
   - **Simulation**: Coordinate-based labels (x+00.0mm_y+00.0mm, x+02.9mm_y+02.0mm, etc.)
   - **Compatibility**: Both formats work with the same training pipeline

**Workflow Comparison:**

**Robot Workflow:**
```
Data Collection → press_summaries → evaluate_single_point_stretch.py → Models + Metrics
```

**Simulation Workflow:**
```
HDF5 Export → train_simulation_positions.py → Models + Metrics
```

Both workflows produce the same model types and can be evaluated using the same tools (e.g., `print_metrics_tables.py`).

### Example Output: Simulation Metrics Tables

After running `train_simulation_positions.py`, you can view the results using:

```bash
python3 src/training/print_metrics_tables.py data/Imported/simulation_points_test1/simulation_points_test1_metrics.json
```

**Example output (from actual simulation dataset):**

```
Reading metrics from: data/Imported/simulation_points_test1/simulation_points_test1_metrics.json
Format detected: simulation

================================================================================
Simulation Dataset Metrics
================================================================================

================================================================================
Force Regression Metrics (Per-Stretch)
================================================================================
Stretch              Samples    RMSE [N]     STD [N]     
--------------------------------------------------------------------------------
stretch_000pct       594        0.015411     0.015407    
stretch_010pct       594        0.016880     0.016880    
stretch_020pct       594        0.017300     0.017272    
--------------------------------------------------------------------------------
combined (pooled)    1782       0.022885     0.022867    

================================================================================
Position Classification Accuracy (Per-Stretch)
================================================================================
Stretch              Samples    Accuracy    
--------------------------------------------------------------------------------
stretch_000pct       594        1.000       
stretch_010pct       594        1.000       
stretch_020pct       594        1.000       
--------------------------------------------------------------------------------
pooled position classifier    1782       0.996       

================================================================================
Stretch Classification
================================================================================
Model                          Samples    Accuracy    
--------------------------------------------------------------------------------
pooled stretch classifier      1782       1.000       

================================================================================
Detected Contact Positions
================================================================================
Label                          X [mm]       Y [mm]      
--------------------------------------------------------------------------------
x+00.0mm_y+00.0mm              0.0          0.0         
x+00.5mm_y+01.9mm              0.5          1.9         
x+00.5mm_y-01.9mm              0.5          -1.9        
x+01.7mm_y+01.9mm              1.7          1.9         
x+01.7mm_y-01.9mm              1.7          -1.9        
x+02.9mm_y+02.0mm              2.9          2.0         
x+02.9mm_y-02.0mm              2.9          -2.0        
x-01.5mm_y+00.0mm              -1.5         0.0         
x-02.9mm_y+02.0mm              -2.9         2.0         
x-02.9mm_y-02.0mm              -2.9         -2.0        
x-03.0mm_y+00.0mm              -3.0         0.0         
x-04.7mm_y+01.9mm              -4.7         1.9         
x-04.7mm_y-01.9mm              -4.7         -1.9        
x-06.5mm_y+01.9mm              -6.5         1.9         
x-06.5mm_y-01.9mm              -6.5         -1.9        
```

The tables show:
- **Force Regression**: RMSE and STD for each stretch level and combined model
- **Position Classification**: Accuracy for detecting which position was pressed
- **Stretch Classification**: Accuracy for detecting stretch level
- **Detected Positions**: All unique contact positions found (rounded to 0.1mm)

---

All commands assume the `franka_interface` conda environment with the required dependencies
(`pyfranka`, `pyserial`, `numpy`, `scikit-learn`, `matplotlib`, `h5py`, etc.).
