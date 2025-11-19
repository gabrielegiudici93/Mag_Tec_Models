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
4. The scripts launch the training pipeline automatically after all stretches finish. To rerun training manually:
   ```bash
   # For single-point data
   python3 src/training/train_single_point_models.py \
       --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test1 \
       --models-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test1/models \
       --use-gpu \
       --z-threshold 3.0 \
       --report data/Single_Point/force_0.0to3.0N_step0.1N_single_test1/force_0.0to3.0N_step0.1N_single_test1_metrics.json
   ```
   
   Or use the evaluation script (legacy):
   ```bash
   python3 src/training/evaluate_single_point_stretch.py \
       --data-root data/2.5mm_single_test1 \
       --report data/2.5mm_single_test1/2.5mm_single_test1_metrics.json
   ```
5. Optional live visualisation: `python3 src/validation_tests/real_time_predictor.py`. It shares the
   same calibration pipeline as the data-collection scripts.

## 2. Training & KPIs from Existing HDF5 Datasets

### Training Pipeline Overview

The training pipeline (`train_single_point_models.py` and `train_simulation_positions.py`) applies the following preprocessing and training steps:

**Data Processing:**
1. **Outlier Removal**: Sequences with anomalous characteristics are removed using Median Absolute Deviation (MAD) outlier detection based on duration, max Fz, and number of samples (default z-threshold: 3.0)
2. **Dataset Balancing**: Sequences are balanced across stretch levels by taking the minimum count to ensure equal representation
3. **Displacement Filtering**: Samples with high consecutive displacement are filtered out independently for:
   - Magnetic data: Removes samples where consecutive magnetic field readings show displacement above the 95th percentile
   - Force data: Removes samples where consecutive force readings show displacement above the 95th percentile
   - The two sensors are treated independently to avoid cross-contamination
4. **Feature Extraction**: Uses **raw features only** (45 features: 15 sensors × 3 channels). Feature engineering has been removed.
5. **Feature Normalization**: All magnetic features are normalized using `StandardScaler` (zero mean, unit variance) applied globally across all training sequences
6. **Force Normalization**: Fz values are normalized to a target range:
   - Physical data: [0, 3] N
   - Simulation data: [0, 3] N (absolute values, negative sign restored during plotting)
   - Original force range is stored in `fz_scaler` for denormalization

**Model Training:**
- **Train/Test Split**: 70/30 split by **sequence** (not by sample) to avoid data leakage
- **Model Type**: Random Forest Regressor/Classifier with:
  - 200-400 estimators (trees)
  - Unlimited depth (or limited for regularization)
  - GPU acceleration support (using `cuml` or `xgboost` if available)
- **Scaler Persistence**: `StandardScaler` and `fz_scaler` are saved alongside models for consistent preprocessing during inference
- **KPM1 Calculation**: Force resolution (ΔF_min) and KPM1 pass/fail are automatically calculated from test set force values

**Output:**
- Trained models saved as `.joblib` files (force regressor, offset classifier, stretch classifier)
- `scaler.joblib` and `fz_scaler.joblib` saved for consistent preprocessing
- JSON metrics report with RMSE, STD, accuracy, KPM1/KPM2 flags, and sequence counts

### Training Physical Models

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
4. Converts continuous data to sequences (similar to real data structure)
5. Duplicates sequences to match real data count (if needed)
6. Removes outliers using MAD (Median Absolute Deviation)
7. Balances sequences across stretch levels
8. Applies displacement filtering (independent for magnetic and force data)
9. Normalizes features using StandardScaler
10. Normalizes Fz to target range (0-3N for physical, absolute values)
11. Trains per-stretch force regressors (if `forcesTest` is provided)
12. Trains per-stretch position classifiers (center + 4 offsets)
13. Trains pooled stretch and position classifiers
14. Saves all models, scalers, and metrics

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
- **Same input features**: Both use flattened magnetic field data (45 features: 15 sensors × 3 channels) - **raw features only, no feature engineering**
- **Same training methodology**: 
  - 70/30 train/test split **by sequence** (not by sample)
  - Same hyperparameters (200-400 trees, unlimited or limited depth)
  - GPU acceleration support (using `cuml` or `xgboost` if available)
- **Same preprocessing**: 
  - Outlier removal using MAD
  - Dataset balancing across stretch levels
  - Displacement filtering (independent for magnetic and force data)
  - Feature normalization using StandardScaler
  - Fz normalization to target range (0-3N)
- **Same output structure**: Models saved as `.joblib` files, scalers saved separately, metrics in JSON format
- **KPM1 calculation**: Both automatically calculate force resolution (ΔF_min) and KPM1 pass/fail from test set

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

## COMPARISON MODEL SIM2REAL

### Extracting Force Trajectories from Real Data for Simulation Replication

The simulation team can extract force trajectories and magnetic field data from the real robot datasets to replicate experiments. Here's how to extract the information:

**Real Data Structure:**
- Real data is stored in HDF5 files with a `presses/` group
- Each press sequence is in `presses/press_XXX/` with:
  - `forces` [samples, 6]: FT sensor data (Fz is in column 2)
  - `stretchmagtec` [samples, 15, 3]: Magnetic field from 15 sensors, 3 channels (Bx, By, Bz)
  - `timestamps` [samples]: Time stamps (normalized to start at 0)
  - Attributes: `stretch_label` (e.g., '000pct', '010pct', '020pct'), `offset` (e.g., 'center', 'ne', 'nw', 'se', 'sw')

**How to Extract Force Trajectories:**

1. **Load the HDF5 file:**
   ```python
   import h5py
   import numpy as np
   
   # Example using the existing dataset in data/Single_Point/
   h5_path = "data/Single_Point/force_0.0to3.0N_step0.1N_single_test1/force_0.0to3.0N_step0.1N_single_test1_stretch_000pct.h5"
   
   with h5py.File(h5_path, 'r') as f:
       presses = f['presses']
       for press_key in sorted(presses.keys()):
           press = presses[press_key]
           forces = press['forces'][:]  # [samples, 6]
           fz = forces[:, 2]  # Extract Fz component (force in Z direction)
           timestamps = press['timestamps'][:]  # Time axis
           stretch_label = press.attrs.get('stretch_label', 'unknown')
           offset = press.attrs.get('offset', 'unknown')
   ```

2. **Force Trajectory Characteristics:**
   - Each press sequence represents one complete indentation cycle
   - Fz starts from ~0.8N (initial contact) and increases to 3.0N in steps of 0.1N
   - The trajectory is force-controlled: the robot presses until reaching each target force (0.0, 0.1, 0.2, ..., 3.0N)
   - Each sequence typically contains 50-100 samples (duration ~0.5-1.0 seconds at 100Hz)

3. **What to Replicate in Simulation:**
   - **Force trajectory**: Use the same Fz trajectory (from ~0.8N to 3.0N) as input to your simulator
   - **Contact positions**: Test the same 5 positions (center, ne, nw, se, sw) for each stretch level
   - **Stretch levels**: Replicate experiments at 0%, 10%, and 20% stretch separately
   - **Output**: Measure the magnetic field at 15 sensor locations (3×5 grid) with 3 channels (Bx, By, Bz)

4. **Expected Output Format:**
   Your simulation should export HDF5 files with:
   - `MagneticField` [samples, 15, 3]: Magnetic field readings matching the force trajectory
   - `forcesTest` [samples, 3]: The force trajectory used (Fx, Fy, Fz)
   - `IdenterPosition` [samples, 3]: Probe position in metres (X, Y, Z)

**Example Workflow:**
```python
# 1. Load real force trajectory
fz_trajectory = [...]  # From real data, shape [samples]

# 2. Run simulation with this force trajectory
simulated_magnetic = simulate_press(fz_trajectory, stretch_level=0.0, position='center')

# 3. Save to HDF5 matching the expected format
save_simulation_data(simulated_magnetic, fz_trajectory, output_path)
```

### Comparing Real vs Simulated Magnetic Fields

Use the `compare_sim2real_magnetic.py` script to quantitatively compare your simulated magnetic fields with the real measurements:

**Example using the existing Single_Point dataset:**

```bash
python3 src/training/compare_sim2real_magnetic.py \
    --real-data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test1 \
    --sim-data-dir data/Imported/simulation_test2 \
    --output-dir plots/comparison/sim2real \
    --stretches 000pct 010pct 020pct
```

**Note:** The real data directory should point to a folder containing HDF5 files with names like `*_stretch_000pct.h5`, `*_stretch_010pct.h5`, `*_stretch_020pct.h5`. The simulation data directory should contain HDF5 files with the same naming convention or compatible names.

**What the comparison does:**
- Loads real magnetic field data from robot HDF5 files
- **Removes outliers from real data** using Median Absolute Deviation (MAD) with configurable z-threshold (default: 3.0)
- Loads simulated magnetic field data from your HDF5 files
- Interpolates both to a common force axis (0.8N to 3.0N)
- **Normalizes real and simulated data separately** using MinMaxScaler (each dataset gets its own min/max normalization)
- Computes per-sensor metrics:
  - **RMSE**: Root Mean Square Error between real and simulated magnetic fields (after separate normalization)
  - **Correlation**: Pearson correlation coefficient
- Generates comparison plots showing real vs simulated magnetic fields for all 15 sensors and 3 channels (Bx, By, Bz)

**Output:**
- Comparison plots: `plots/comparison/sim2real/sim2real_comparison_XXXpct.png`
- Metrics JSON: `plots/comparison/sim2real/sim2real_metrics.json`

**Important Notes:**
- The comparison focuses **ONLY on magnetic field similarity**, not force prediction accuracy
- **Outlier removal**: Real data sequences are cleaned using MAD-based outlier detection before comparison
- **Separate normalization**: Real and simulated data are normalized independently using MinMaxScaler (each dataset uses its own min/max values)
- Both datasets are interpolated to the same force range (0.8-3.0N) for alignment
- The comparison is done per stretch level separately (0%, 10%, 20%)
- Best matching sensors typically show RMSE < 0.1 and correlation > 0.9

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
