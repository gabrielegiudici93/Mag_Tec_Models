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
  - `clean_sequences.py` – data cleaning script: removes first sequence per offset and outliers, saves cleaned HDF5 files, generates raw data plots
  - `train_unified.py` – **unified training script** for both real and simulation data (automatically detects data type and applies same pipeline)
  - `train_single_point_models.py` – core training functions (used by train_unified.py)
  - `plot_raw_data.py` – raw data visualization: generates plots from HDF5 files, handles both real and simulation data formats
  - `evaluate_single_point_stretch.py` – legacy offline analysis utility (superseded by train_unified.py)
  - `import_dataset.py` – converts external HDF5 recordings into the standard run structure
  - `train_simulation_positions.py` – simulation data processing functions (used by train_unified.py)
- `doc/`
  - `single_point_validation.tex` – scientific report describing the pipeline

## Complete Workflow: Data Collection → Cleaning → Training

### Step 1: Data Collection

1. Adjust `src/franka_controller/config.py` (robot IP, serial ports, reference pose, press parameters).
2. Activate the `franka_interface` environment and run:
   ```bash
   cd /home/gabriele/franka_interface/Mag_Tec_Models
   conda activate franka_interface
   python3 src/franka_controller/franka_skin_test_single_point.py
   ```
3. Each acquisition creates a run directory (e.g., `data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/`) containing:
   - Three HDF5 files (`…_stretch_000pct.h5`, `…_stretch_010pct.h5`, `…_stretch_020pct.h5`).

### Step 2: Data Cleaning

After data acquisition, clean the sequences by removing outliers and the first sequence per offset:

```bash
python3 src/training/clean_sequences.py \
    --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50
```

**What the cleaning script does:**
- Loads all HDF5 files from the specified directory
- Removes the first sequence per offset (warm-up/calibration)
- Removes 2 outliers per offset independently using MAD outlier detection
- Saves cleaned HDF5 files to `cleaned/data/` with simplified naming: `test_XX_YYY_cleaned.h5`
- Generates raw data visualization plots in `cleaned/raw_data/`

**Output structure:**
```
data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/
└── cleaned/
    ├── data/                          # Cleaned HDF5 files
    │   ├── test_50_000_cleaned.h5
    │   ├── test_50_010_cleaned.h5
    │   └── test_50_020_cleaned.h5
    └── raw_data/                      # Raw data plots
        ├── stretch_000pct/
        ├── stretch_010pct/
        └── stretch_020pct/
```

### Step 3: Model Training

Train models using the unified training script, which works for both real and simulation data:

**For real data:**
```bash
python3 src/training/train_unified.py \
    --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
    --run-label test50 \
    --feature-method raw
```

**For simulation data:**
```bash
python3 src/training/train_unified.py \
    --h5-files data/simulation/test5/*.h5 \
    --run-label simulation_test5 \
    --feature-method raw \
    --output-dir data/simulation/test5/cleaned
```

**What the unified training script does:**
- Automatically detects data type (real vs simulation)
- Loads cleaned HDF5 files from `cleaned/data/` (real data) or directly from HDF5 files (simulation)
- Applies preprocessing (feature normalization, displacement filtering, Fz cutting at 3N)
- Removes first sequence per offset and 2 outliers per offset (already done in cleaning for real data, skipped if files are already cleaned)
- For simulation data: automatically segments continuous data into force cycles and maps positions to offsets
- Trains per-stretch and combined models (force regressors, offset classifiers, stretch classifier)
- Saves models to `cleaned/{feature_method}/models/`
- Generates confusion matrix plots and prediction plots in `cleaned/{feature_method}/plots/`
- Generates raw data plots (for real data: `cleaned/raw_data/`, for simulation: `cleaned/{feature_method}/plots/sim/` with `_sim` suffix)
- Generates average plots across all stretch levels
- Saves JSON metrics report in `cleaned/{feature_method}/models/`

**Complete output structure after training:**
```
data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/
└── cleaned/
    ├── data/                          # Cleaned HDF5 files
    ├── raw_data/                      # Raw data plots (real data only)
    └── raw/                           # Raw feature models and plots
        ├── models/                    # Trained models (*.joblib) and JSON metrics
        │   ├── force_regressor_000pct.joblib
        │   ├── offset_classifier_000pct.joblib
        │   ├── stretch_classifier_combined.joblib
        │   ├── scaler_000pct.joblib
        │   └── test50_metrics.json
        └── plots/                     # Training plots
            ├── confusion_matrix_offset_000pct_raw.png
            ├── prediction_scatter_000pct_raw.png
            ├── prediction_residuals_000pct_raw.png
            └── ...
```

## Data Processing Pipeline

The training pipeline applies the following preprocessing steps:

1. **First Sequence Removal**: The first sequence per offset is removed to eliminate warm-up/calibration effects.
2. **Outlier Removal**: Sequences with anomalous characteristics are removed using Median Absolute Deviation (MAD) outlier detection. The process is applied **independently for each offset position** (center, nw, ne, se, sw), removing 2 outliers per offset based on:
   - Sequence duration (weighted 1.5× in the combined score)
   - Maximum force Fz
   - Number of samples
3. **Displacement Filtering**: Samples with high consecutive displacement are filtered out independently for magnetic and force data (95th percentile threshold).
4. **Feature Extraction**: Uses **raw features only** (45 features: 15 sensors × 3 channels). No feature engineering is applied.
5. **Feature Normalization**: All magnetic features are normalized using `StandardScaler` (zero mean, unit variance) applied globally across all training sequences.
6. **Force Processing**: Sequences are **cut at 3N** (all samples where Fz > 3.0N are removed). For simulation data, absolute values are used during training.
7. **Dataset Balancing**: Sequences are balanced across stretch levels to ensure equal representation.
8. **Train/Test Split**: 70/30 split by **sequence** (random, seed=42) to avoid data leakage.

## Model Training Details

**Model Type**: Random Forest Regressor/Classifier with:
- 200-250 estimators (trees)
- Limited depth (to prevent overfitting)
- GPU acceleration support (using `cuml` or `xgboost` if available)

**Output Models:**
- Per-stretch force regressors (3 models: 0%, 10%, 20%)
- Per-stretch press location classifiers (3 models: center + NW/NE/SE/SW)
- Combined force regressor (pooled across all stretch levels)
- Combined press location classifier (pooled)
- Stretch classifier (predicts stretch level from magnetic features)
- Scalers (`scaler_*.joblib`) for consistent preprocessing during inference

**Metrics:**
- RMSE and STD for force regression
- Accuracy for offset and stretch classification
- KPM1 (force resolution ΔF_min) and KPM2 (accuracy & precision) pass/fail flags
- All metrics saved in JSON format: `{run_label}_metrics.json`

## Simulation Data Training

The unified training script automatically handles simulation data:

1. **Automatic Detection**: Detects simulation data by checking for `MagneticField` and `forcesTest` datasets.
2. **Position Detection**: Automatically derives position labels by rounding `IdenterPosition` X/Y coordinates to 0.1mm.
3. **Force Cycle Segmentation**: Automatically segments continuous data into individual force cycles (0N → 3N → 0N).
4. **Offset Mapping**: Maps positions to offsets (center, ne, nw, se, sw) based on relative positions to center.
5. **Outlier Removal**: By default, outliers are **not** removed for simulation data (use `--remove-outliers` flag to enable).

**Example for simulation data:**
```bash
python3 src/training/train_unified.py \
    --h5-files data/simulation/test5/*.h5 \
    --run-label simulation_test5 \
    --feature-method raw \
    --output-dir data/simulation/test5/cleaned
```

**Simulation output structure:**
```
data/simulation/test5/
└── cleaned/
    ├── raw/
    │   ├── models/                    # Trained models and JSON metrics
    │   ├── plots/                     # Training plots
    │   │   ├── sim/                   # Simulation-specific plots
    │   │   │   ├── stretch_000pct/
    │   │   │   ├── stretch_010pct/
    │   │   │   └── stretch_020pct/
    │   │   ├── confusion_matrix_offset_000pct_raw_sim.png
    │   │   └── ...
    │   └── data/                      # Processed data (if saved)
```

## Feature Methods

The unified training script supports three feature methods:

- **`raw`** (default): 45 raw magnetic features (15 sensors × 3 channels)
- **`normalized`**: 15 normalized features (mean of 3 channels per sensor)
- **`raw_labelled`**: 45 raw features + 5 one-hot encoded offset labels (50 features total)

Use the `--feature-method` argument to select the method:
```bash
python3 src/training/train_unified.py \
    --data-dir data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned \
    --run-label test50 \
    --feature-method raw
```

## Viewing Results

**View metrics in table format:**
```bash
python3 src/training/print_metrics_tables.py \
    data/Single_Point/force_0.0to3.0N_step0.1N_single_test50/cleaned/raw/models/test50_metrics.json
```

**View raw data plots:**
All plots are automatically generated and saved in:
- Real data: `cleaned/raw_data/stretch_XXXpct/`
- Simulation data: `cleaned/raw/plots/sim/stretch_XXXpct/`

**Scientific report:**
The scientific report (`doc/single_point_validation.tex`) contains detailed documentation of the pipeline, results, and methodology. Recompile the LaTeX document after training to capture the latest metrics.

## Export Guidelines for Simulation Team

To plug simulated runs into the same pipeline, structure each HDF5 file as follows:

- `MagneticField` – required; shape `[samples, 15, 3]` with channels ordered (Bx, By, Bz)
- `IdenterPosition` – required for position-aware training; shape `[samples, 3]` with probe position in metres (X, Y, Z)
- `forcesTest` – optional but recommended; shape `[samples, 3]` (Fx, Fy, Fz). Forces are typically negative (0 to -3N) in simulation
- `attrs/stretch` – optional string/int identifying the stretch percentage (e.g., `"10"`)

**File naming convention:**
Use the standard naming: `<run_label>_stretch_000pct.h5`, `<run_label>_stretch_010pct.h5`, `<run_label>_stretch_020pct.h5`

The unified training script automatically handles simulation data format conversion and processing.

---

All commands assume the `franka_interface` conda environment with the required dependencies
(`pyfranka`, `pyserial`, `numpy`, `scikit-learn`, `matplotlib`, `h5py`, etc.).
