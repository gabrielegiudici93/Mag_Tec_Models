#!/usr/bin/env python3
"""
MagTecK_PM - Centralized Configuration File

This file contains all configuration parameters for the entire pipeline:
- Robot settings
- Grid configuration
- Sensor settings  
- Press parameters
- Calibration flags
- File paths

Author: Gabriele Giudici
Date: 2025
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"
DOC_DIR = PROJECT_ROOT / "doc"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, DOC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ROBOT CONFIGURATION
# =============================================================================
ROBOT_IP = '192.168.2.10'
ROBOT_SPEED_FACTOR = 0.1  # Speed factor for precision movements (0.0-1.0)

# =============================================================================
# GRID CONFIGURATION (9-POINT SYSTEM)
# =============================================================================
# Grid position offsets (in meters) - 2mm offsets around each grid point
# This creates a 3x3 sub-grid around each main grid position
GRID_OFFSET_DIST = 0.001  # Offset distance in meters (1mm for neighboring grid points)


GRID_OFFSETS = {
    'center': [0.0, 0.0, 0.0],                                      # Exact grid position
    'nw':    [-GRID_OFFSET_DIST,  -GRID_OFFSET_DIST, 0.0],            # North-West (-1mm X, +1mm Y)
    'n':     [-GRID_OFFSET_DIST, 0.0, 0.0],             # North (0mm X, +1mm Y)
    'ne':    [-GRID_OFFSET_DIST,  GRID_OFFSET_DIST, 0.0],            # North-East (+1mm X, +1mm Y)
    'e':     [0.0,               GRID_OFFSET_DIST, 0.0],             # East (0mm X, +1mm Y)
    'se':    [ GRID_OFFSET_DIST,  GRID_OFFSET_DIST, 0.0],            # South-East (+1mm X, +1mm Y)
    's':     [ GRID_OFFSET_DIST, 0.0, 0.0],             # South (0mm X, +1mm Y)
    'sw':    [ GRID_OFFSET_DIST, -GRID_OFFSET_DIST, 0.0],            # South-West (+1mm X, -1mm Y)
    'w':     [0.0,              -GRID_OFFSET_DIST, 0.0],             # West (0mm X, -1mm Y)
}

# =============================================================================
# POSITION AND OFFSET SELECTION
# =============================================================================
# Specify which positions and offsets to test
# Leave empty lists [] to test ALL positions/offsets

# Positions to test (e.g., [11, 22, 33] or [] for all)
SELECTED_POSITIONS = []  # Empty = test all 15 positions
# Offsets to test (e.g., ['center'] or [] for all 9 offsets)
SELECTED_OFFSETS = ['center']  # Empty = test all offsets

# Helper function to get positions to test
def get_positions_to_test():
    """Returns list of position IDs to test based on SELECTED_POSITIONS."""
    if SELECTED_POSITIONS:
        return SELECTED_POSITIONS
    else:
        return sorted(MAIN_GRID_POSITIONS.keys())

# Helper function to get offsets to test
def get_offsets_to_test():
    """Returns list of offset keys to test based on SELECTED_OFFSETS."""
    if SELECTED_OFFSETS:
        return SELECTED_OFFSETS
    else:
        return list(GRID_OFFSETS.keys())

# Main grid configuration
# Physical layout is 5 rows × 3 cols (as before - CORRECT positions)
# But we LABEL them as row-major for consistency
GRID_ROWS = 5  # Number of rows in the main grid (PHYSICAL layout)
GRID_COLS = 3  # Number of columns in the main grid (PHYSICAL layout)
GRID_DX = 0.011/2  # X-axis spacing between main grid points (meters)
GRID_DY = 0.011    # Y-axis spacing between main grid points (meters)
#    ##REFERENCE_POSITION = [0.491276, 0.4175, 0.032811]  # Reference position (1,1) - bottom-left corner
REFERENCE_POSITION = [0.491276, 0.4175, 0.031311]  # Reference position (1,1) - bottom-left corner

# =============================================================================
# PRESS PARAMETERS
# =============================================================================
NUMBER_OF_PRESSES = 1  # Number of complete press cycles per position
STEPS_PER_PRESS = 1    # Number of vertical steps per press (1 step = direct press)
DZ_PRESS = -0.0015      # Press depth (1mm down)
DZ_LIFT = 0.0015        # Lift distance after press (1mm up, back to original position)
PRESS_DELAY = 0.5      # Delay between steps during press (seconds)
LIFT_DELAY = 0.5       # Delay after lifting (seconds)
MOVEMENT_DURATION = 0.5  # Duration for each relative movement (seconds)
ABSOLUTE_MOVEMENT_DURATION = 0.5  # Duration for absolute movements (seconds)

# Press ID system - alphabetical identifiers for each press cycle
PRESS_IDS = [chr(i) for i in range(65, 65 + 50)]  # Support up to 50 press cycles (A, B, ..., AX)

# =============================================================================
# SENSOR IDENTIFICATION
# =============================================================================
SENSOR_NAME = "bulk_skin_std_1"  # Identifier for the sensor being characterized

# =============================================================================
# FT SENSOR CONFIGURATION
# =============================================================================
FT_PORT = '/dev/ttyUSB0'
FT_BAUDRATE = 19200
FT_NOISE_THRESHOLD = 0.0  # Threshold to filter noise

# Calibration settings
FT_INITIAL_CALIBRATION_ENABLED = True  # Initial calibration at start (ALWAYS recommended)
FT_PER_POSITION_CALIBRATION_ENABLED = False  # Recalibrate before/after each position (optional)
FT_CALIBRATION_DURATION = 2.0  # seconds
FT_CALIBRATION_SAMPLES = int(FT_CALIBRATION_DURATION * 100)  # Assuming 100 Hz

# Legacy flag for backwards compatibility
FT_CALIBRATION_ENABLED = FT_PER_POSITION_CALIBRATION_ENABLED

# =============================================================================
# STRETCHMAGTEC 3x5 SENSOR CONFIGURATION
# =============================================================================
STRETCHMAGTEC_PORT = '/dev/ttyACM0'
STRETCHMAGTEC_BAUD = 2000000  # High-speed mode required for StretchMagTec streaming
STRETCHMAGTEC_ROWS = 3
STRETCHMAGTEC_COLS = 5
STRETCHMAGTEC_SENSORS = 15  # 3x5 grid
STRETCHMAGTEC_CHANNELS = 3  # X, Y, Z magnetic field
STRETCHMAGTEC_THRESHOLD = 50.0  # Threshold for magnetic field values

# Calibration settings
STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED = True  # Initial calibration at start (ALWAYS recommended)
STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED = False  # Recalibrate before/after each position (optional)
STRETCHMAGTEC_CALIBRATION_DURATION = 5.0  # seconds

# Legacy flag for backwards compatibility
STRETCHMAGTEC_CALIBRATION_ENABLED = STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================
TARGET_FREQ = 100  # Target sampling frequency (Hz)
PERIOD = 1.0 / TARGET_FREQ

# DEBUG MODE - Limit data collection for testing
DEBUG_MODE = False  # Set to True for testing with limited positions
DEBUG_MAX_POSITIONS = 3  # Only collect first N positions when DEBUG_MODE=True
DEBUG_OFFSETS = ['center', 'n', 'e']  # Only collect these offsets when DEBUG_MODE=True

# Data collection modes
COLLECT_ALL_GRID_POSITIONS = True  # If False, only collect from selected positions
COLLECT_ALL_OFFSETS = True  # If False, only collect from 'center' offset

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
# Contact classifier settings
CONTACT_CLASSIFIER_TEST_SIZE = 0.2
CONTACT_CLASSIFIER_RANDOM_STATE = 42

# Force mapping settings
FT_MAPPING_TEST_SIZE = 0.2
FT_MAPPING_RANDOM_STATE = 42

# Model file names (include sensor name for multi-sensor support)
CONTACT_CLASSIFIER_MODEL = f"{SENSOR_NAME}_contact_classifier.joblib"
CONTACT_SCALER_MODEL = f"{SENSOR_NAME}_contact_scaler.joblib"
FT_MAPPING_FX_MODEL = f"{SENSOR_NAME}_ft_mapping_fx.joblib"
FT_MAPPING_FY_MODEL = f"{SENSOR_NAME}_ft_mapping_fy.joblib"
FT_MAPPING_FZ_MODEL = f"{SENSOR_NAME}_ft_mapping_fz.joblib"
FT_MAPPING_SCALER_FX = f"{SENSOR_NAME}_ft_mapping_scaler_fx.joblib"
FT_MAPPING_SCALER_FY = f"{SENSOR_NAME}_ft_mapping_scaler_fy.joblib"
FT_MAPPING_SCALER_FZ = f"{SENSOR_NAME}_ft_mapping_scaler_fz.joblib"
FT_MAPPING_OUTPUT_SCALER_FX = f"{SENSOR_NAME}_ft_mapping_output_scaler_fx.joblib"
FT_MAPPING_OUTPUT_SCALER_FY = f"{SENSOR_NAME}_ft_mapping_output_scaler_fy.joblib"
FT_MAPPING_OUTPUT_SCALER_FZ = f"{SENSOR_NAME}_ft_mapping_output_scaler_fz.joblib"

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================
VALIDATION_POSITIONS = ['center', 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se']
VALIDATION_PRESS_DEPTH = -0.003  # 3mm press for validation

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
PLOT_SCALE_FACTOR = 500  # Divide sensor values by this for plotting
PLOT_Y_LIMIT = 50  # Fixed Y-axis limits for plots (±PLOT_Y_LIMIT)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_model_path(model_name):
    """Get full path for a model file."""
    return MODELS_DIR / model_name

def get_data_path(filename):
    """Get full path for a data file."""
    return DATA_DIR / filename

def get_plot_path(filename):
    """Get full path for a plot file."""
    return PLOTS_DIR / filename

def get_log_path(filename):
    """Get full path for a log file."""
    return LOGS_DIR / filename

def generate_grid_positions(rows, cols, dx, dy, reference_pos):
    """
    Generate grid positions with row-major labeling.
    
    Physical layout: 5 rows × 3 cols (vertical × horizontal)
    Label layout: Row-major (11-15 in first row, 21-25 in second, etc.)
    
    Args:
        rows: Number of rows in the grid (5 - vertical direction)
        cols: Number of columns in the grid (3 - horizontal direction)
        dx: X-axis spacing between grid points
        dy: Y-axis spacing between grid points
        reference_pos: Reference position [x, y, z] for the bottom-left corner
    
    Returns:
        dict: Dictionary mapping position IDs to [x, y, z] coordinates
              Labels are row-major: 11,12,13,14,15 (first row), 21,22,23,24,25 (second row), etc.
    """
    positions = {}
    
    # Physical grid is 5 rows (vertical) × 3 cols (horizontal)
    # We want labels: 11-15 (first row), 21-25 (second), 31-35 (third)
    # So we iterate through physical positions but assign row-major labels
    
    label_row = 1  # Label row counter (1, 2, 3)
    label_col = 1  # Label column counter (1-5)
    
    for physical_row in range(1, rows + 1):  # 1 to 5 (physical rows)
        for physical_col in range(1, cols + 1):  # 1 to 3 (physical cols)
            # Calculate physical position
            x = reference_pos[0] + (physical_col - 1) * dx
            y = reference_pos[1] + (physical_row - 1) * dy
            z = reference_pos[2]
            
            # Assign row-major label
            position_id = label_row * 10 + label_col
            positions[position_id] = [x, y, z]
            
            # Increment label counters (row-major)
            label_col += 1
            if label_col > 5:  # After 5 columns, go to next row
                label_col = 1
                label_row += 1
    
    return positions

def get_position_with_offset(position_coords, offset_key):
    """
    Get position coordinates with grid offset applied.
    
    Args:
        position_coords: Base position [x, y, z]
        offset_key: Offset type ('center', 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se')
    
    Returns:
        list: [x, y, z] coordinates with offset applied
    """
    if offset_key not in GRID_OFFSETS:
        raise ValueError(f"Invalid offset key: {offset_key}. Available: {list(GRID_OFFSETS.keys())}")
    
    offset = GRID_OFFSETS[offset_key]
    return [
        position_coords[0] + offset[0],
        position_coords[1] + offset[1],
        position_coords[2] + offset[2]
    ]

# Generate main grid positions
MAIN_GRID_POSITIONS = generate_grid_positions(
    GRID_ROWS, GRID_COLS, GRID_DX, GRID_DY, REFERENCE_POSITION
)

# Print configuration summary
def print_config_summary():
    """Print a summary of the configuration."""
    print("\n" + "="*70)
    print("MagTecK_PM CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Main Grid: {GRID_ROWS}x{GRID_COLS} positions")
    print(f"Grid Spacing: dx={GRID_DX:.6f}m, dy={GRID_DY:.6f}m")
    print(f"9-Point Offsets: {list(GRID_OFFSETS.keys())}")
    print(f"Total test positions: {len(MAIN_GRID_POSITIONS) * len(GRID_OFFSETS)}")
    print(f"\nFT Calibration: {'Enabled' if FT_CALIBRATION_ENABLED else 'Disabled'}")
    print(f"StretchMagTec Calibration: {'Enabled' if STRETCHMAGTEC_CALIBRATION_ENABLED else 'Disabled'}")
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Plots Directory: {PLOTS_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config_summary()

