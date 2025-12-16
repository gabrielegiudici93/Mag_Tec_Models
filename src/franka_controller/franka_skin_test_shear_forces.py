#!/usr/bin/env python3
"""
Multiple-point variant of the Franka skin data collection script with shear forces.

This script extends the multiple-point workflow to collect data with shear forces (Fx, Fy)
in addition to normal force (Fz). For each point, it performs:
1. Press to 1N on Fz
2. Move in X direction until 1N on Fx (while maintaining Fz)
3. Return to center
4. Lift
5. Repeat for -1N Fx, +1N Fy, -1N Fy

This is repeated for all 10 points.
"""

import runpy
import subprocess
import shutil
from pathlib import Path
import sys
import threading
import time
from datetime import datetime
from typing import List, Tuple
import copy

import numpy as np
import tkinter as tk
import pyfranka_interface as franka
from scipy.spatial.transform import Rotation as R

# Ensure the parent directory is on the path so we can import the project modules
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import project modules after adjusting sys.path
import franka_controller.config as config  # noqa: E402
import validation_tests.real_time_predictor as predictor  # noqa: E402

# =============================================================================
# SINGLE-POINT CONFIGURATION
# =============================================================================

# =============================================================================
# SINGLE-POINT CONFIGURATION AND TEST PARAMETERS
# =============================================================================
TARGET_POSITION_ID = 32
# New center position from point 1: X,Y from user, Z from previous single point tests
TARGET_POSITION_COORDS = [0.500781, 0.419620, 0.034311]

# Initial joint configuration (set to None to disable joint movement)
# To get current joint positions, run: python3 src/franka_controller/get_current_joints.py
# Then copy the INITIAL_JOINT_POSITIONS value here
INITIAL_JOINT_POSITIONS = [-1.460883997177473, -1.4397968588005559, 1.8498105422813298, -1.680352194797862, 1.4646542101436189, 1.8593807739681665, 0.8594902150722012]

BASE_NS_OFFSET = 0.0025  # 2.5 mm
BASE_EW_OFFSET = 0.0050  # 5.0 mm

BASE_OFFSETS = {
    'center': [0.0, 0.0, 0.0],
    'n': [-BASE_NS_OFFSET, 0.0, 0.0],
    's': [BASE_NS_OFFSET, 0.0, 0.0],
    'e': [0.0, BASE_EW_OFFSET, 0.0],
    'w': [0.0, -BASE_EW_OFFSET, 0.0],
    'ne': [-BASE_NS_OFFSET, BASE_EW_OFFSET, 0.0],
    'nw': [-BASE_NS_OFFSET, -BASE_EW_OFFSET, 0.0],
    'se': [BASE_NS_OFFSET, BASE_EW_OFFSET, 0.0],
    'sw': [BASE_NS_OFFSET, -BASE_EW_OFFSET, 0.0],
}

# Multi-point offsets relative to center (point 1)
# Pattern: delta X or delta Y of 0.01 m between each point
# Center (point 1): (0.500781, 0.419620, 0.032812)
MULTI_POINT_OFFSETS = {
    '1': [0.0, 0.0, 0.0],           # Center
    '2': [-0.01, 0.0, 0.0],          # X-0.01
    '3': [-0.01, 0.01, 0.0],         # X-0.01, Y+0.01
    '4': [0.0, 0.01, 0.0],           # Y+0.01
    '5': [0.0, 0.02, 0.0],           # Y+0.02
    '6': [-0.01, 0.02, 0.0],         # X-0.01, Y+0.02
    '7': [-0.01, 0.03, 0.0],         # X-0.01, Y+0.03
    '8': [0.0, 0.03, 0.0],           # Y+0.03
    '9': [0.0, 0.04, 0.0],           # Y+0.04
    '10': [-0.01, 0.04, 0.0],        # X-0.01, Y+0.04
}

TARGET_OFFSETS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # Only the 10 multi-points

# Stretch levels to test (percentages expressed as decimal fractions)
STRETCH_LEVELS = [0.0, 0.10, 0.20]  # 0%, 10%, 20% stretch
PROMPT_FOR_STRETCH = True  # Prompt operator before each stretch run

# Shear force collection parameters
FZ_TARGET = 2.0  # Target Fz force (N) - will press until Fz = -2.0N
FX_TARGET = 1.0  # Target Fx force (N) - positive and negative
FY_TARGET = 1.0  # Target Fy force (N) - positive and negative
FORCE_TOLERANCE = 0.1  # Tolerance for force control (N)
MAX_INDENTATION = 0.005  # Maximum indentation (m) - 5mm
MAX_SHEAR_DISPLACEMENT = 0.005  # Maximum displacement in shear direction (m) - 5mm
DATA_COLLECTION_DURATION = 0.5  # Duration to collect data after reaching target force (s)
MOVEMENTS_PER_DIRECTION = 33  # Total movements per direction (30 final after removing first + 2 worst)
MOVEMENTS_PER_DIRECTION = 33  # Total movements per direction (30 final after removing first + 2 worst)

# Shear sequences: (direction, target_force)
# direction: 'x+', 'x-', 'y+', 'y-'
# Note: Robot movement direction is inverted relative to FT sensor:
# - Moving in x+ produces negative Fx, so to get Fx = +1N we move in x- (which produces positive Fx)
# - Moving in x- produces positive Fx, so to get Fx = -1N we move in x+ (which produces negative Fx)
# - Moving in y+ produces positive Fy, so to get Fy = +1N we move in y+ (which produces positive Fy)
# - Moving in y- produces negative Fy, so to get Fy = -1N we move in y- (which produces negative Fy)
SHEAR_SEQUENCES = [
    ('x-', FX_TARGET),    # Move in -X direction until Fx = +1N (x- produces positive Fx)
    ('x+', -FX_TARGET),   # Move in +X direction until Fx = -1N (x+ produces negative Fx)
    ('y+', FY_TARGET),    # Move in +Y direction until Fy = +1N (y+ produces positive Fy)
    ('y-', -FY_TARGET),   # Move in -Y direction until Fy = -1N (y- produces negative Fy)
]

# GUI flag (set False to disable visualization)
ENABLE_GUI = True
ENABLE_EXPLORATION = False  # Set to False to skip exploration and go directly to data collection

# Base references for restoring configuration after the test
BASE_DATA_DIR = Path(config.DATA_DIR) / "Multiple_Points"
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Apply single-point selection defaults
config.SELECTED_POSITIONS = [TARGET_POSITION_ID]
config.SELECTED_OFFSETS = TARGET_OFFSETS
config.MAIN_GRID_POSITIONS[TARGET_POSITION_ID] = TARGET_POSITION_COORDS
config.GRID_OFFSETS.update(BASE_OFFSETS)
config.GRID_OFFSETS.update(MULTI_POINT_OFFSETS)

# Set initial joint positions if configured
if INITIAL_JOINT_POSITIONS is not None:
    config.INITIAL_JOINT_POSITIONS = INITIAL_JOINT_POSITIONS
    print(f"Initial joint positions configured: {INITIAL_JOINT_POSITIONS}")
else:
    config.INITIAL_JOINT_POSITIONS = None

print("=" * 70)
print(" MULTI-POINT SKIN TEST (SHEAR FORCES) ")
print("=" * 70)
print(f"Target position ID: {TARGET_POSITION_ID}")
print(f"Offsets under test: {', '.join(TARGET_OFFSETS)}")
print(f"Base coordinates: {TARGET_POSITION_COORDS}")
print(f"Shear sequences per point: {len(SHEAR_SEQUENCES)}")
print(f"Fz target: {FZ_TARGET}N, Fx/Fy target: ±{FX_TARGET}N")
print(f"Stretch levels configured: {', '.join(f'{int(s*100)}%' for s in STRETCH_LEVELS)}")
print("=" * 70 + "\n")

# =============================================================================
# GUI SUPPORT CLASSES (SHARED WITH REAL-TIME PREDICTOR)
# =============================================================================
class SkinTestSensorAdapter:
    """
    Adapts the StretchMagTec data streaming from the main data collection script
    to the interface expected by `RealTimePredictorGUI`.
    """

    def __init__(self, max_buffer_size: int = 1000):
        self.running = False
        self.max_buffer_size = max_buffer_size

        self.ft_buffer: List[np.ndarray] = []
        self.stretch_buffer: List[np.ndarray] = []
        self.time_buffer: List[float] = []

        self.sensor_hz_values = [0.0] * config.STRETCHMAGTEC_SENSORS
        self._sensor_hz_counts = [0] * config.STRETCHMAGTEC_SENSORS
        self._last_hz_time = time.time()

        self.individual_sensor_buffers = {
            sensor_id: {'X': [], 'Y': [], 'Z': [], 'time': []}
            for sensor_id in range(config.STRETCHMAGTEC_SENSORS)
        }

        self._latest_data = np.zeros((config.STRETCHMAGTEC_SENSORS, config.STRETCHMAGTEC_CHANNELS))
        self._latest_ft = np.zeros(6)
        self._lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._poll_interval = config.PERIOD if hasattr(config, "PERIOD") else 0.01

    def start_sensors(self):
        self.running = True
        self.ft_buffer.clear()
        self.stretch_buffer.clear()
        self.time_buffer.clear()
        self._sensor_hz_counts = [0] * config.STRETCHMAGTEC_SENSORS
        self._last_hz_time = time.time()
        for buffer in self.individual_sensor_buffers.values():
            for key in buffer:
                buffer[key].clear()

        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="skin_test_gui_poll")
            self._poll_thread.start()

    def stop_sensors(self):
        self.running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self):
        while self.running:
            module = sys.modules.get('__main__')
            stretch_event = getattr(module, 'stretchmagtec_ready_event', None) if module else None
            stretch_stream_ready = stretch_event.is_set() if stretch_event else False

            data = self._read_latest_data()
            if data is not None and data.size:
                if module:
                    stretch_cal = getattr(module, 'stretchmagtec_calibration', None)
                    if stretch_stream_ready and stretch_cal is not None and getattr(stretch_cal, 'is_calibrated', False):
                        try:
                            data = stretch_cal.compensate_sensors(data)
                        except Exception:
                            pass

                ft_reading = None
                if module and hasattr(module, 'ft_thread'):
                    try:
                        ft_thread = getattr(module, 'ft_thread')
                        if ft_thread is not None and hasattr(ft_thread, 'get_ft'):
                            ft_reading = np.array(ft_thread.get_ft(), dtype=float)
                    except Exception:
                        ft_reading = None
                if ft_reading is None:
                    ft_reading = np.zeros(6)

                current_time = time.time()

                with self._lock:
                    self._latest_data = data.copy()
                    self._latest_ft = ft_reading.copy()

                    self.time_buffer.append(current_time)
                    self.stretch_buffer.append(data.copy())
                    self.ft_buffer.append(ft_reading.copy())

                    if len(self.time_buffer) > self.max_buffer_size:
                        self.time_buffer.pop(0)
                        self.stretch_buffer.pop(0)
                        self.ft_buffer.pop(0)

                    for sensor_id in range(config.STRETCHMAGTEC_SENSORS):
                        sensor_buffer = self.individual_sensor_buffers[sensor_id]
                        sensor_buffer['X'].append(data[sensor_id, 0])
                        sensor_buffer['Y'].append(data[sensor_id, 1])
                        sensor_buffer['Z'].append(data[sensor_id, 2])
                        sensor_buffer['time'].append(current_time)
                        if len(sensor_buffer['X']) > self.max_buffer_size:
                            sensor_buffer['X'].pop(0)
                            sensor_buffer['Y'].pop(0)
                            sensor_buffer['Z'].pop(0)
                            sensor_buffer['time'].pop(0)

                        if not np.allclose(data[sensor_id], 0.0):
                            self._sensor_hz_counts[sensor_id] += 1

                    elapsed = current_time - self._last_hz_time
                    if elapsed >= 1.0:
                        for sensor_id in range(config.STRETCHMAGTEC_SENSORS):
                            self.sensor_hz_values[sensor_id] = self._sensor_hz_counts[sensor_id] / elapsed
                            self._sensor_hz_counts[sensor_id] = 0
                        self._last_hz_time = current_time

            time.sleep(max(0.001, self._poll_interval))

    # --- RealTimePredictorGUI interface -------------------------------------------------
    def get_ft_data(self) -> np.ndarray:
        with self._lock:
            return self._latest_ft.copy()

    def get_stretchmagtec_data(self) -> np.ndarray:
        with self._lock:
            return self._latest_data.copy()

    def get_plot_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        with self._lock:
            if not self.time_buffer:
                return [], [], []
            t0 = self.time_buffer[0]
            relative_time = [t - t0 for t in self.time_buffer]
            return self.ft_buffer.copy(), self.stretch_buffer.copy(), relative_time

    # --- Internal helpers ---------------------------------------------------------------
    @staticmethod
    def _read_latest_data():
        """
        Retrieve the most recent StretchMagTec frame from the running
        `franka_skin_test` script. The module is executed via runpy with
        run_name='__main__', so we expect to find the function there.
        """
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, 'read_stretchmagtec_data'):
            try:
                return getattr(main_module, 'read_stretchmagtec_data')()
            except Exception:
                return None
        return None


class SkinTestModelStub:
    """
    Minimal stand-in for ModelPredictor. We only need the GUI, so all predictions
    are disabled while keeping the interface intact.
    """

    models_loaded = False
    use_spatial_features = False
    use_normalized_features = False

    def load_models(self):
        return False

    def predict_contact_point(self, _):
        return "Visualization-only", 0.0

    def get_contact_probabilities(self, _):
        return {}

    def predict_ft_forces(self, _):
        return {"fx": 0.0, "fy": 0.0, "fz": 0.0}


def format_stretch_label(stretch_value: float) -> str:
    percent = int(round(stretch_value * 100))
    return f"stretch_{percent:03d}pct"


def build_offsets_for_stretch(stretch_value: float) -> dict:
    scale = 1.0 + stretch_value
    offsets = {}
    for key, vec in BASE_OFFSETS.items():
        new_vec = list(vec)
        if key in ('e', 'w', 'ne', 'nw', 'se', 'sw'):
            new_vec[1] = new_vec[1] * scale
        offsets[key] = new_vec
    for key, vec in MULTI_POINT_OFFSETS.items():
        new_vec = list(vec)
        # Apply stretch scaling to Y component for all multi-point offsets (Y direction stretch)
        # Point 1 (center) doesn't need scaling, but all others do
        if key != '1':
            new_vec[1] = new_vec[1] * scale
        offsets[key] = new_vec
    return offsets


def generate_run_name() -> Tuple[str, Path]:
    indent_tag = "shear_forces"
    base_name = f"{indent_tag}_test"

    data_root = BASE_DATA_DIR
    data_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(p.name for p in data_root.glob(f"{base_name}*") if p.is_dir())
    run_id = 1
    for name in existing:
        suffix = name.replace(base_name, "").strip("_")
        try:
            candidate = int(suffix)
            run_id = max(run_id, candidate + 1)
        except ValueError:
            continue

    run_label = f"{base_name}{run_id}"
    return run_label, data_root / run_label


def run_training_pipeline(run_root: Path, run_label: str):
    print("\nStarting training and evaluation pipeline...")
    metrics_path = run_root / f"{run_label}_metrics.json"

    eval_script = SRC_ROOT / "training" / "evaluate_single_point_stretch.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--data-root",
        str(run_root),
        "--report",
        str(metrics_path),
    ]

    process = subprocess.run(cmd, cwd=REPO_ROOT)
    if process.returncode != 0:
        print("⚠️  Training pipeline failed. Check console output for details.")
        return

    models_dest = run_root / "models"
    models_dest.mkdir(parents=True, exist_ok=True)

    model_names = [
        config.CONTACT_CLASSIFIER_MODEL,
        config.CONTACT_SCALER_MODEL,
        config.FT_MAPPING_FX_MODEL,
        config.FT_MAPPING_FY_MODEL,
        config.FT_MAPPING_FZ_MODEL,
        config.FT_MAPPING_SCALER_FX,
        config.FT_MAPPING_SCALER_FY,
        config.FT_MAPPING_SCALER_FZ,
        config.FT_MAPPING_OUTPUT_SCALER_FX,
        config.FT_MAPPING_OUTPUT_SCALER_FY,
        config.FT_MAPPING_OUTPUT_SCALER_FZ,
    ]

    copied = 0
    for name in model_names:
        src = config.MODELS_DIR / name
        if src.exists():
            shutil.copy2(src, models_dest / name)
            copied += 1

    if copied:
        print(f"Models copied to {models_dest} ({copied} files).")
    else:
        print("⚠️  No model files found to copy.")

    if metrics_path.exists():
        print(f"Metrics saved to {metrics_path}")
    else:
        print("⚠️  Metrics file was not generated.")


def explore_offsets():
    if not ENABLE_EXPLORATION:
        return

    print("\n=== EXPLORATION MODE ENABLED ===")
    print("The robot will visit each configured offset without pressing.")
    print("Use this step to verify that the positions are correct.")

    rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
    robot = None

    try:
        print(f"Connecting to robot at {config.ROBOT_IP} for exploration...")
        robot = franka.Robot_(config.ROBOT_IP, False, hand_franka=False,
                              auto_init=True, speed_factor=config.ROBOT_SPEED_FACTOR)
        print("✅ Robot connected for exploration.\n")

        base_position = config.MAIN_GRID_POSITIONS[TARGET_POSITION_ID]

        for offset_key in TARGET_OFFSETS:
            target = config.get_position_with_offset(base_position, offset_key)

            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = target

            print(f"→ Moving to offset '{offset_key}' at "
                  f"[{target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}]")
            robot.move("absolute", pose, config.ABSOLUTE_MOVEMENT_DURATION)

            user_input = input("   Press Enter to continue, or type 'skip' to stop exploration: ").strip()
            if user_input.lower() == "skip":
                print("Exploration interrupted by user.")
                break

        # Return to centre before exiting exploration
        centre_pose = np.eye(4)
        centre_pose[:3, :3] = rotation_matrix
        centre_pose[:3, 3] = base_position
        robot.move("absolute", centre_pose, config.ABSOLUTE_MOVEMENT_DURATION)
        print("Exploration complete. Robot returned to centre.\n")

    except Exception as exc:
        print(f"⚠️  Exploration failed: {exc}")
    finally:
        try:
            if robot is not None:
                robot.stop()
        except Exception:
            pass


def capture_original_config():
    original_positions = list(getattr(config, "SELECTED_POSITIONS", []))
    original_offsets = list(getattr(config, "SELECTED_OFFSETS", []))
    if not original_positions:
        original_positions = list(config.MAIN_GRID_POSITIONS.keys())
    if not original_offsets:
        original_offsets = list(config.GRID_OFFSETS.keys())

    return {
        "DATA_DIR": Path(config.DATA_DIR),
        "GRID_OFFSETS": copy.deepcopy(config.GRID_OFFSETS),
        "NUMBER_OF_PRESSES": config.NUMBER_OF_PRESSES,
        "STEPS_PER_PRESS": config.STEPS_PER_PRESS,
        "DZ_PRESS": config.DZ_PRESS,
        "DZ_LIFT": config.DZ_LIFT,
        "PRESS_DELAY": config.PRESS_DELAY,
        "LIFT_DELAY": config.LIFT_DELAY,
        "SELECTED_POSITIONS": original_positions,
        "SELECTED_OFFSETS": original_offsets,
    }


def restore_original_config(original_state):
    config.DATA_DIR = original_state["DATA_DIR"]
    config.GRID_OFFSETS.clear()
    config.GRID_OFFSETS.update(original_state["GRID_OFFSETS"])
    config.NUMBER_OF_PRESSES = original_state["NUMBER_OF_PRESSES"]
    config.STEPS_PER_PRESS = original_state["STEPS_PER_PRESS"]
    config.DZ_PRESS = original_state["DZ_PRESS"]
    config.DZ_LIFT = original_state["DZ_LIFT"]
    config.PRESS_DELAY = original_state["PRESS_DELAY"]
    config.LIFT_DELAY = original_state["LIFT_DELAY"]
    config.SELECTED_POSITIONS = original_state["SELECTED_POSITIONS"]
    config.SELECTED_OFFSETS = original_state["SELECTED_OFFSETS"]

    for attr in [
        "CURRENT_STRETCH_VALUE",
        "CURRENT_STRETCH_LABEL",
        "CURRENT_PRESS_PROFILE",
        "CURRENT_PRESS_SETTINGS",
        "CURRENT_OUTPUT_PREFIX",
        "CURRENT_STRETCH_INDEX",
        "CURRENT_RUN_LABEL",
        "LAST_OUTPUT_FILE",
    ]:
        if hasattr(config, attr):
            delattr(config, attr)


def configure_for_stretch(stretch_value: float, stretch_label: str, run_root: Path, run_name: str, stretch_idx: int):
    updated_offsets = build_offsets_for_stretch(stretch_value)
    config.GRID_OFFSETS.update(updated_offsets)
    config.SELECTED_POSITIONS = [TARGET_POSITION_ID]
    config.SELECTED_OFFSETS = TARGET_OFFSETS
    config.DATA_DIR = run_root

    config.CURRENT_RUN_LABEL = run_name
    config.CURRENT_STRETCH_VALUE = stretch_value
    config.CURRENT_STRETCH_LABEL = stretch_label
    config.CURRENT_PRESS_PROFILE = "shear_force_3d"
    config.CURRENT_PRESS_SETTINGS = {
        "fz_target": FZ_TARGET,
        "fx_target": FX_TARGET,
        "fy_target": FY_TARGET,
        "shear_sequences": len(SHEAR_SEQUENCES),
    }
    config.CURRENT_STRETCH_INDEX = stretch_idx
    config.CURRENT_OUTPUT_PREFIX = f"{run_name}_shear_{stretch_label}"
    if hasattr(config, "LAST_OUTPUT_FILE"):
        delattr(config, "LAST_OUTPUT_FILE")


def launch_predictor_gui(collection_done_event: threading.Event):
    """
    Launch a RealTimePredictor GUI instance configured for passive visualization.
    This function runs on the main thread (Tkinter requirement).
    """
    adapter = SkinTestSensorAdapter()
    gui = predictor.RealTimePredictorGUI(model_dir=config.MODELS_DIR)

    # Replace heavy components with lightweight adapters
    gui.sensor_reader = adapter
    gui.model_predictor = SkinTestModelStub()

    def load_models_stub():
        gui.status_label.config(text="Status: Visualization only (no models)", foreground="purple")
        return False

    gui.load_models = load_models_stub
    gui.load_models_button.config(state=tk.DISABLED)
    gui.grid_viz_button.config(state=tk.DISABLED)
    gui.start_button.config(state=tk.DISABLED)
    gui.stop_button.config(state=tk.DISABLED)

    gui.contact_label.config(text="Contact: visualization only", foreground="purple")
    gui.confidence_label.config(text="Confidence: --")
    for lbl in gui.force_pred_labels:
        lbl.config(text="Fx/Fy/Fz: --", foreground="gray")

    adapter.start_sensors()
    gui.update_running = True

    def gui_update_loop():
        if not gui.update_running:
            return
        stretchmagtec_data = gui.sensor_reader.get_stretchmagtec_data()
        ft_data = gui.sensor_reader.get_ft_data()

        for i, value in enumerate(ft_data):
            color = "red" if abs(value) > 1.0 else "black"
            gui.ft_labels[i].config(text=f"{['Fx (N)','Fy (N)','Fz (N)','Tx (Nm)','Ty (Nm)','Tz (Nm)'][i]}: {value:7.3f}", foreground=color)

        for sensor_id in range(config.STRETCHMAGTEC_SENSORS):
            for channel_id in range(config.STRETCHMAGTEC_CHANNELS):
                val = stretchmagtec_data[sensor_id, channel_id]
                color = "red" if abs(val) > config.STRETCHMAGTEC_THRESHOLD else "black"
                gui.stretchmagtec_labels[sensor_id][channel_id].config(
                    text=f"{['X','Y','Z'][channel_id]}: {val:6.0f}",
                    foreground=color,
                )

        hz_values = gui.sensor_reader.sensor_hz_values
        for sensor_id in range(config.STRETCHMAGTEC_SENSORS):
            gui.stretchmagtec_labels[sensor_id][3].config(text=f"Hz: {hz_values[sensor_id]:.1f}")

        gui.model_predictor = None
        gui.contact_label.config(text="Contact: visualization only", foreground="purple")
        gui.confidence_label.config(text="Confidence: --", foreground="purple")
        for lbl in gui.force_pred_labels:
            lbl.config(text="Fx/Fy/Fz: --", foreground="gray")

        ft_data_plot, stretch_data_plot, time_data_plot = gui.sensor_reader.get_plot_data()
        if time_data_plot and gui.selected_sensors:
            gui.ax1.clear()
            gui.ax2.clear()
            gui.ax3.clear()
            gui.ax4.clear()

            gui.ax1.set_title("FT Sensor Data")
            gui.ax1.set_ylabel("Force/Torque")
            gui.ax1.set_xlabel("Time (s)")
            gui.ax1.grid(True, alpha=0.3)

            gui.ax2.set_title("StretchMagTec X-Axis")
            gui.ax2.set_ylabel("Magnetic Field")
            gui.ax2.set_xlabel("Time (s)")
            gui.ax2.grid(True, alpha=0.3)

            gui.ax3.set_title("StretchMagTec Y-Axis")
            gui.ax3.set_ylabel("Magnetic Field")
            gui.ax3.set_xlabel("Time (s)")
            gui.ax3.grid(True, alpha=0.3)

            gui.ax4.set_title("StretchMagTec Z-Axis")
            gui.ax4.set_ylabel("Magnetic Field")
            gui.ax4.set_xlabel("Time (s)")
            gui.ax4.grid(True, alpha=0.3)

            if ft_data_plot:
                fx = [entry[0] for entry in ft_data_plot]
                fy = [entry[1] for entry in ft_data_plot]
                fz = [entry[2] for entry in ft_data_plot]
                gui.ax1.plot(time_data_plot, fx, label="Fx")
                gui.ax1.plot(time_data_plot, fy, label="Fy")
                gui.ax1.plot(time_data_plot, fz, label="Fz")
                gui.ax1.legend(loc="upper right")

            for sensor_id in sorted(gui.selected_sensors):
                if sensor_id < len(stretch_data_plot):
                    series = [entry[sensor_id, :] for entry in stretch_data_plot]
                    x_series = [val[0] for val in series]
                    y_series = [val[1] for val in series]
                    z_series = [val[2] for val in series]

                    gui.ax2.plot(time_data_plot, x_series, label=f"S{sensor_id+1}")
                    gui.ax3.plot(time_data_plot, y_series, label=f"S{sensor_id+1}")
                    gui.ax4.plot(time_data_plot, z_series, label=f"S{sensor_id+1}")

            if gui.selected_sensors:
                legend_labels = [f"S{s+1}" for s in sorted(gui.selected_sensors)]
                gui.ax2.set_title(f"StretchMagTec X-Axis: {legend_labels}")
                gui.ax3.set_title(f"StretchMagTec Y-Axis: {legend_labels}")
                gui.ax4.set_title(f"StretchMagTec Z-Axis: {legend_labels}")

                gui.ax2.legend(loc="upper right", fontsize=8)
                gui.ax3.legend(loc="upper right", fontsize=8)
                gui.ax4.legend(loc="upper right", fontsize=8)

            gui.fig.tight_layout()
            gui.canvas.draw_idle()

        gui.root.after(gui.update_interval, gui_update_loop)

    gui.root.after(gui.update_interval, gui_update_loop)

    def refresh_status():
        module = sys.modules.get('__main__')
        ft_ready = not getattr(config, 'FT_INITIAL_CALIBRATION_ENABLED', True)
        stretch_ready = not getattr(config, 'STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED', True)

        if module:
            ft_cal = getattr(module, 'ft_calibration', None)
            if ft_cal is not None and getattr(config, 'FT_INITIAL_CALIBRATION_ENABLED', True):
                if hasattr(ft_cal, 'is_calibrated'):
                    ft_ready = getattr(ft_cal, 'is_calibrated', False)

            stretch_cal = getattr(module, 'stretchmagtec_calibration', None)
            if stretch_cal is not None and getattr(config, 'STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED', True):
                if hasattr(stretch_cal, 'is_calibrated'):
                    stretch_ready = getattr(stretch_cal, 'is_calibrated', False)

        if ft_ready and stretch_ready:
            gui.status_label.config(text="Status: Receiving data (calibrated)", foreground="green")
        else:
            gui.status_label.config(text="Status: Reading data (uncalibrated)", foreground="orange")

        gui.root.after(500, refresh_status)

    gui.root.after(500, refresh_status)

    # Close the GUI automatically when data collection is done
    def check_collection():
        if collection_done_event.is_set():
            adapter.stop_sensors()
            gui.update_running = False
            gui.root.after(250, gui.root.quit)
        else:
            gui.root.after(500, check_collection)

    gui.root.after(500, check_collection)

    gui.root.mainloop()


# Import sensor classes and functions from franka_skin_test
import franka_controller.franka_skin_test as skin_test_module
FTSensorThread = skin_test_module.FTSensorThread
StretchMagTecSerialReader = skin_test_module.StretchMagTecSerialReader
DynamicFTCalibration = skin_test_module.DynamicFTCalibration
StretchMagTecCalibration = skin_test_module.StretchMagTecCalibration
read_stretchmagtec_data = skin_test_module.read_stretchmagtec_data
stretchmagtec_data = skin_test_module.stretchmagtec_data
stretchmagtec_data_lock = skin_test_module.stretchmagtec_data_lock
stretchmagtec_ready_event = skin_test_module.stretchmagtec_ready_event
ft_data_ready_event = skin_test_module.ft_data_ready_event

# =============================================================================
# SHEAR FORCE COLLECTION FUNCTIONS
# =============================================================================

def press_to_fz(r, ft_thread, target_fz, initial_z=None, max_iterations=500):
    """Press down until Fz reaches -target_fz (negative value, e.g., -2N for target_fz=2N).
    Respects MAX_INDENTATION limit (5mm).
    """
    target_fz_negative = -abs(target_fz)  # Target is negative (e.g., -2N for target_fz=2N)
    
    # Get initial Z position if not provided
    if initial_z is None:
        current_state = r.getState()
        initial_z = current_state.T[2, 3]
    
    iteration = 0
    for _ in range(max_iterations):
        iteration += 1
        current_ft = ft_thread.get_ft()
        current_fz = current_ft[2]  # Use raw value (negative when pressing)
        current_fx = current_ft[0]
        current_fy = current_ft[1]
        
        # Check indentation limit (5mm)
        current_state = r.getState()
        current_pos = current_state.T[:3, 3]
        current_indentation = initial_z - current_pos[2]  # Positive = indentation
        
        if current_indentation >= MAX_INDENTATION:
            print(f"    ⚠️  Maximum indentation reached ({current_indentation*1000:.2f}mm >= {MAX_INDENTATION*1000:.2f}mm)")
            return True
        
        # Print forces every 10 iterations
        if iteration % 10 == 0:
            print(f"    Fz: {current_fz:.3f}N (target: {target_fz_negative:.3f}N), Fx: {current_fx:.3f}N, Fy: {current_fy:.3f}N, Indentation: {current_indentation*1000:.2f}mm")
        
        # Check if we've reached or exceeded the target (more negative = more force)
        if current_fz <= target_fz_negative:
            print(f"    ✅ Target Fz reached: {current_fz:.3f}N (target: {target_fz_negative:.3f}N), Fx: {current_fx:.3f}N, Fy: {current_fy:.3f}N")
            return True
        
        # Move down slightly (same as franka_skin_test.py: move_relative(r, 0, 0, -0.0001, ...))
        target_pos = current_pos.copy()
        target_pos[2] -= 0.0001  # Move down 0.1mm (Z negative = down)
        target_pose = current_state.T.copy()
        target_pose[:3, 3] = target_pos
        
        r.move("absolute", target_pose, 0.1)
        time.sleep(0.01)
    
    print(f"    ⚠️  Maximum iterations reached. Final Fz: {current_fz:.3f}N, Fx: {current_fx:.3f}N, Fy: {current_fy:.3f}N")
    return False

def move_to_shear_force(r, ft_thread, direction, target_force, maintain_fz, start_position, max_iterations=500):
    """Move robot in X or Y direction until target shear force is reached, while maintaining Fz.
    Returns (success, final_position, total_displacement).
    
    Note: Robot movement direction is inverted relative to FT sensor readings:
    - Moving in x+ produces negative Fx
    - Moving in x- produces positive Fx
    - Moving in y+ produces negative Fy
    - Moving in y- produces positive Fy
    """
    iteration = 0
    for _ in range(max_iterations):
        iteration += 1
        current_ft = ft_thread.get_ft()
        current_fx = current_ft[0]
        current_fy = current_ft[1]
        current_fz = current_ft[2]
        
        if direction == 'x+' or direction == 'x-':
            current_force = current_ft[0]  # Fx
            axis_idx = 0
            # Normal step: x+ moves in +X, x- moves in -X
            step = 0.0001 if direction == 'x+' else -0.0001
        elif direction == 'y+' or direction == 'y-':
            current_force = current_ft[1]  # Fy
            axis_idx = 1
            # Normal step: y+ moves in +Y, y- moves in -Y
            step = 0.0001 if direction == 'y+' else -0.0001
        else:
            return False, None, 0.0
        
        # Check displacement limit (5mm)
        current_state = r.getState()
        current_pos = current_state.T[:3, 3]
        displacement = abs(current_pos[axis_idx] - start_position[axis_idx])
        
        if displacement >= MAX_SHEAR_DISPLACEMENT:
            print(f"    ⚠️  Maximum displacement reached ({displacement*1000:.2f}mm >= {MAX_SHEAR_DISPLACEMENT*1000:.2f}mm)")
            return True, current_pos.copy(), displacement
        
        # Print forces every 10 iterations
        if iteration % 10 == 0:
            print(f"    {direction}: {current_force:.3f}N (target: {target_force:.3f}N), Fz: {current_fz:.3f}N, Displacement: {displacement*1000:.2f}mm")
        
        # Check if target force reached
        # For positive targets: check if current_force >= target_force
        # For negative targets: check if current_force <= target_force
        if target_force >= 0:
            # Positive target (e.g., +1N)
            if current_force >= target_force:
                print(f"    ✅ Target {direction} reached: {current_force:.3f}N, Displacement: {displacement*1000:.2f}mm")
                return True, current_pos.copy(), displacement
        else:
            # Negative target (e.g., -1N)
            if current_force <= target_force:
                print(f"    ✅ Target {direction} reached: {current_force:.3f}N, Displacement: {displacement*1000:.2f}mm")
                return True, current_pos.copy(), displacement
        
        # Move in the direction
        target_pos = current_pos.copy()
        target_pos[axis_idx] += step
        
        # Try to maintain Fz by adjusting Z slightly (using negative values)
        maintain_fz_negative = -abs(maintain_fz)  # Target is negative (e.g., -2N for maintain_fz=2N)
        
        # If Fz is less negative than target (closer to zero), move down
        if current_fz > maintain_fz_negative * 0.9:  # current_fz is less negative (e.g., -1.0N vs -2N)
            target_pos[2] -= 0.00005  # Move down slightly (Z negative = down, increases negative Fz)
        # If Fz is more negative than target (too much force), move up
        elif current_fz < maintain_fz_negative * 1.1:  # current_fz is more negative (e.g., -2.5N vs -2N)
            target_pos[2] += 0.00005  # Move up slightly (Z positive = up, decreases negative Fz)
        
        target_pose = current_state.T.copy()
        target_pose[:3, 3] = target_pos
        
        r.move("absolute", target_pose, 0.1)
        time.sleep(0.01)
    
    print(f"    ⚠️  Maximum iterations reached. Displacement: {displacement*1000:.2f}mm")
    return False, current_pos.copy(), displacement

def return_to_center(r, target_pos):
    """Return robot to center position (target_pos)."""
    current_state = r.getState()
    current_pose = current_state.T.copy()
    current_pose[:3, 3] = target_pos
    r.move("absolute", current_pose, config.ABSOLUTE_MOVEMENT_DURATION)
    time.sleep(0.5)

def return_by_displacement(r, direction, displacement):
    """Return robot by exact displacement in the opposite direction.
    displacement is always positive (absolute value).
    """
    current_state = r.getState()
    current_pos = current_state.T[:3, 3]
    target_pos = current_pos.copy()
    
    if direction == 'x+' or direction == 'x-':
        axis_idx = 0
        # Return in opposite direction: if moved x+, return x- (subtract displacement)
        if direction == 'x+':
            target_pos[axis_idx] -= displacement
        else:  # x-
            target_pos[axis_idx] += displacement
    elif direction == 'y+' or direction == 'y-':
        axis_idx = 1
        # Return in opposite direction: if moved y+, return y- (subtract displacement)
        if direction == 'y+':
            target_pos[axis_idx] -= displacement
        else:  # y-
            target_pos[axis_idx] += displacement
    else:
        return
    
    target_pose = current_state.T.copy()
    target_pose[:3, 3] = target_pos
    r.move("absolute", target_pose, config.ABSOLUTE_MOVEMENT_DURATION)
    time.sleep(0.5)

def identify_shear_outliers(movements, z_threshold=3.0):
    """Identify outlier movements using statistical methods (MAD).
    Returns indices of outliers to remove (first + 2 worst).
    """
    if len(movements) < 3:
        return []
    
    # Extract features for each movement
    durations = []
    max_fz = []
    num_samples = []
    
    for mov in movements:
        if 'shear_data' in mov and len(mov['shear_data']['forces']) > 0:
            forces = np.array(mov['shear_data']['forces'])
            fz_values = forces[:, 2]
            durations.append(mov['shear_data']['timestamps'][-1] - mov['shear_data']['timestamps'][0] if len(mov['shear_data']['timestamps']) > 1 else 0.0)
            max_fz.append(float(np.max(np.abs(fz_values))))
            num_samples.append(len(forces))
        else:
            durations.append(0.0)
            max_fz.append(0.0)
            num_samples.append(0)
    
    durations = np.array(durations)
    max_fz = np.array(max_fz)
    num_samples = np.array(num_samples)
    
    outlier_indices = set()
    
    # Check each feature with MAD
    for feature_name, feature_values in [
        ('duration', durations),
        ('max_fz', max_fz),
        ('num_samples', num_samples),
    ]:
        median = np.median(feature_values)
        mad = np.median(np.abs(feature_values - median))
        
        if mad > 0:
            threshold = z_threshold if feature_name != 'duration' else 2.0
            modified_z_scores = 0.6745 * (feature_values - median) / mad
            outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
            outlier_indices.update(outliers)
    
    # Calculate outlier scores for all movements
    movements_with_scores = []
    fz_median = np.median(max_fz)
    fz_mad = np.median(np.abs(max_fz - fz_median)) if len(max_fz) > 0 else 1.0
    duration_median = np.median(durations)
    duration_mad = np.median(np.abs(durations - duration_median)) if len(durations) > 0 else 1.0
    num_samples_median = np.median(num_samples)
    num_samples_mad = np.median(np.abs(num_samples - num_samples_median)) if len(num_samples) > 0 else 1.0
    
    for idx, mov in enumerate(movements):
        score = 0
        if fz_mad > 0:
            score += abs((max_fz[idx] - fz_median) / fz_mad)
        if duration_mad > 0:
            duration_score = abs((durations[idx] - duration_median) / duration_mad)
            if durations[idx] > duration_median * 2.0:
                duration_score *= 2.0
            score += duration_score * 1.5
        if num_samples_mad > 0:
            score += abs((num_samples[idx] - num_samples_median) / num_samples_mad)
        
        is_identified_outlier = idx in outlier_indices
        if is_identified_outlier:
            score *= 2.0
        
        movements_with_scores.append((idx, score, is_identified_outlier))
    
    # Sort by outlier status and score
    movements_with_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    # Remove first movement (index 0) + 2 worst
    indices_to_remove = {0}  # Always remove first
    num_to_remove = min(2, len(movements_with_scores) - 1)
    for idx, _, _ in movements_with_scores[:num_to_remove]:
        if idx != 0:  # Don't add index 0 twice
            indices_to_remove.add(idx)
    
    return sorted(list(indices_to_remove))

def lift_robot(r, target_pos, lift_height=0.005):
    """Lift robot by lift_height (default 5mm)."""
    current_state = r.getState()
    current_pos = current_state.T[:3, 3].copy()
    lift_pos = current_pos.copy()
    lift_pos[2] += lift_height
    lift_pose = current_state.T.copy()
    lift_pose[:3, 3] = lift_pos
    r.move("absolute", lift_pose, config.ABSOLUTE_MOVEMENT_DURATION)
    time.sleep(0.5)

def collect_shear_data_for_point(r, ft_thread, stretchmagtec_reader, ft_calibration, stretchmagtec_calibration, 
                                  target_pos, point_name, stretch_value, stretch_label):
    """Collect shear force data for a single point.
    
    For each direction, performs 33 movements:
    - Press to Fz = -2N
    - Move in direction until target force (max 5mm displacement)
    - Collect data
    - Return by exact displacement
    - Repeat 33 times
    - Remove first movement + 2 worst outliers
    - Keep 30 final movements
    """
    all_sequences = []
    
    # Move to point
    rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = target_pos
    r.move("absolute", pose, config.ABSOLUTE_MOVEMENT_DURATION)
    time.sleep(1.0)  # Wait for stabilization
    
    # Get initial Z position for indentation limit
    initial_state = r.getState()
    initial_z = initial_state.T[2, 3]
    
    for seq_idx, (direction, target_force) in enumerate(SHEAR_SEQUENCES):
        print(f"\n{'='*70}")
        print(f"  Direction {seq_idx + 1}/{len(SHEAR_SEQUENCES)}: {direction} to {target_force:.1f}N")
        print(f"{'='*70}")
        
        # Press once to Fz target (before all movements in this direction)
        print(f"  Pressing to {FZ_TARGET}N on Fz (max indentation: {MAX_INDENTATION*1000:.1f}mm)...")
        if not press_to_fz(r, ft_thread, FZ_TARGET, initial_z=initial_z):
            print(f"  ⚠️  Failed to reach {FZ_TARGET}N on Fz, skipping direction {direction}")
            continue
        
        # Collect 33 movements for this direction
        movements = []
        for mov_idx in range(MOVEMENTS_PER_DIRECTION):
            print(f"\n  Movement {mov_idx + 1}/{MOVEMENTS_PER_DIRECTION} in {direction}")
            
            # Get start position for displacement calculation
            start_state = r.getState()
            start_position = start_state.T[:3, 3].copy()
            
            # Move to shear force (with displacement limit)
            print(f"    Moving in {direction} to {target_force:.1f}N (max displacement: {MAX_SHEAR_DISPLACEMENT*1000:.1f}mm)...")
            success, final_position, displacement = move_to_shear_force(r, ft_thread, direction, target_force, FZ_TARGET, start_position)
            
            if not success:
                print(f"    ⚠️  Failed to reach target force, skipping this movement")
                # Return to start position anyway
                return_by_displacement(r, direction, displacement)
                continue
            
            # Collect data during shear
            shear_data = {'forces': [], 'positions': [], 'timestamps': [], 'stretchmagtec': []}
            start_time = time.time()
            while time.time() - start_time < DATA_COLLECTION_DURATION:
                # Get FT data and apply calibration
                raw_ft = ft_thread.get_ft()
                compensated_ft = ft_calibration.compensate_force(raw_ft) if ft_calibration else raw_ft
                current_state = r.getState()
                current_pos = current_state.T[:3, 3]
                
                shear_data['forces'].append(compensated_ft.copy())
                shear_data['positions'].append(current_pos.copy())
                shear_data['timestamps'].append(time.time())
                
                # Get StretchMagTec data and apply calibration
                raw_stretch_data = read_stretchmagtec_data()
                if raw_stretch_data is not None:
                    compensated_stretch = stretchmagtec_calibration.compensate_sensors(raw_stretch_data) if stretchmagtec_calibration else raw_stretch_data
                else:
                    compensated_stretch = np.zeros((config.STRETCHMAGTEC_SENSORS, config.STRETCHMAGTEC_CHANNELS))
                shear_data['stretchmagtec'].append(compensated_stretch.copy())
                
                time.sleep(0.01)  # 100 Hz
            
            # Return by exact displacement
            print(f"    Returning by {displacement*1000:.2f}mm...")
            return_by_displacement(r, direction, displacement)
            time.sleep(0.2)  # Brief pause before next movement
            
            movements.append({
                'direction': direction,
                'target_force': target_force,
                'displacement': displacement,
                'shear_data': shear_data,
            })
        
        # Remove first movement + 2 worst outliers
        print(f"\n  Removing outliers from {len(movements)} movements...")
        outlier_indices = identify_shear_outliers(movements)
        print(f"  Removing {len(outlier_indices)} movements (indices: {outlier_indices})")
        
        cleaned_movements = [mov for idx, mov in enumerate(movements) if idx not in outlier_indices]
        print(f"  Kept {len(cleaned_movements)} movements after outlier removal")
        
        # Add cleaned movements to all_sequences
        all_sequences.extend(cleaned_movements)
        
        # Lift after completing all movements for this direction
        print(f"\n  Lifting after completing {direction}...")
        lift_robot(r, target_pos)
        time.sleep(0.5)
    
    return all_sequences

def save_shear_data_to_h5(output_file, sequences_by_point, stretch_value, stretch_label):
    """Save collected shear data to HDF5 file (compatible with multiple_points structure)."""
    import h5py
    
    print(f"\nSaving data to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Top-level attributes (MUST match franka_skin_test.py structure)
        f.attrs['sensor_name'] = config.SENSOR_NAME
        f.attrs['robot_ip'] = config.ROBOT_IP
        f.attrs['reference_position'] = np.array(TARGET_POSITION_COORDS, dtype=np.float64)
        f.attrs['grid_offsets'] = str(TARGET_OFFSETS)
        f.attrs['number_of_presses'] = len(TARGET_OFFSETS) * len(SHEAR_SEQUENCES)
        f.attrs['steps_per_press'] = 1
        f.attrs['dz_press'] = 0.0
        f.attrs['dz_lift'] = 0.0
        f.attrs['ft_calibration_enabled'] = np.bool_(True)
        f.attrs['stretchmagtec_calibration_enabled'] = np.bool_(True)
        f.attrs['target_freq'] = config.TARGET_FREQ
        f.attrs['stretch_level'] = float(stretch_value)
        f.attrs['stretch_label'] = str(stretch_label)
        f.attrs['press_profile'] = 'shear_force_3d'
        f.attrs['description'] = f"Multi-point shear force data collection - {len(TARGET_OFFSETS)} points, {len(SHEAR_SEQUENCES)} sequences per point"
        
        # Create top-level concatenated datasets
        all_forces = []
        all_positions = []
        all_timestamps = []
        all_labels = []
        all_stretchmagtec = []
        
        # Create presses group (MANDATORY - same as franka_skin_test.py)
        presses_group = f.create_group("presses")
        
        press_idx = 0
        start_idx = 0
        
        for point_name, sequences in sequences_by_point.items():
            for seq_idx, seq_data in enumerate(sequences):
                # Combine press and shear data into one sequence
                combined_forces = seq_data['press_data']['forces'] + seq_data['shear_data']['forces']
                combined_positions = seq_data['press_data']['positions'] + seq_data['shear_data']['positions']
                combined_timestamps = seq_data['press_data']['timestamps'] + seq_data['shear_data']['timestamps']
                combined_stretchmagtec = seq_data['press_data']['stretchmagtec'] + seq_data['shear_data']['stretchmagtec']
                
                # Add to concatenated arrays
                end_idx = start_idx + len(combined_forces)
                all_forces.extend(combined_forces)
                all_positions.extend(combined_positions)
                all_timestamps.extend(combined_timestamps)
                all_stretchmagtec.extend(combined_stretchmagtec)
                
                # Create label (MUST match format from franka_skin_test.py)
                label = f"pos_{point_name}_{point_name}_press_{seq_idx:02d}_shear_{seq_data['direction']}_{seq_data['target_force']:.1f}N"
                all_labels.extend([label.encode('utf-8')] * len(combined_forces))
                
                # Create press group inside presses (MANDATORY - same as franka_skin_test.py)
                press_group = presses_group.create_group(f"press_{press_idx:03d}")
                
                # Core attributes (MANDATORY - same as original)
                press_group.attrs['label'] = label
                press_group.attrs['start_idx'] = int(start_idx)
                press_group.attrs['end_idx'] = int(end_idx)
                press_group.attrs['num_samples'] = int(end_idx - start_idx)
                
                # Offset attribute (MANDATORY - same as original)
                offset_value = point_name
                if isinstance(offset_value, str):
                    press_group.attrs['offset'] = offset_value.encode('utf-8') if not isinstance(offset_value, bytes) else offset_value
                else:
                    press_group.attrs['offset'] = b"unknown"
                
                # Stretch attributes (MANDATORY - same as original)
                press_group.attrs['stretch_level'] = float(stretch_value)
                press_group.attrs['stretch_label'] = str(stretch_label)
                
                # Additional metadata (for shear force data)
                press_group.attrs['sequence_idx'] = int(seq_idx)
                press_group.attrs['direction'] = str(seq_data['direction'])
                press_group.attrs['target_force'] = float(seq_data['target_force'])
                
                # Calculate indentation (Z-axis movement) for compatibility
                if combined_positions:
                    positions_array = np.array(combined_positions)
                    initial_z = float(positions_array[0, 2])
                    indentation = positions_array[:, 2] - initial_z
                    press_group.attrs['initial_z'] = float(initial_z)
                    press_group.attrs['max_indentation'] = float(np.min(indentation))
                    press_group.create_dataset('indentation', data=indentation)
                else:
                    press_group.attrs['initial_z'] = float('nan')
                    press_group.attrs['max_indentation'] = float('nan')
                
                # Core datasets (MANDATORY - same as original)
                press_group.create_dataset('forces', data=np.array(combined_forces))
                press_group.create_dataset('positions', data=np.array(combined_positions))
                press_group.create_dataset('stretchmagtec', data=np.array(combined_stretchmagtec))
                press_group.create_dataset('labels', data=np.array([label.encode('utf-8')] * len(combined_forces), dtype='S64'))
                
                # Timestamps (relative to first timestamp for compatibility)
                if combined_timestamps:
                    timestamps_array = np.array(combined_timestamps)
                    relative_times = timestamps_array - timestamps_array[0]
                    press_group.create_dataset('timestamps', data=relative_times)
                else:
                    press_group.create_dataset('timestamps', data=np.array([], dtype=float))
                
                start_idx = end_idx
                press_idx += 1
        
        # Top-level concatenated datasets (MANDATORY - same as original)
        if all_forces:
            if all_timestamps:
                timestamps_array = np.array(all_timestamps)
                relative_timestamps = timestamps_array - timestamps_array[0]
            else:
                relative_timestamps = np.array([])
            
            f.create_dataset('forces', data=np.array(all_forces))
            f.create_dataset('positions', data=np.array(all_positions))
            f.create_dataset('timestamps', data=relative_timestamps)
            f.create_dataset('labels', data=np.array(all_labels, dtype='S64'))
            f.create_dataset('stretchmagtec', data=np.array(all_stretchmagtec))
        
        print(f"  Total presses: {press_idx}")
        print(f"  Total samples: {len(all_forces)}")
    
    print(f"✅ Data saved: {output_file}")

def run_data_collection(collection_done_event: threading.Event):
    """
    Execute shear force data collection directly (instead of calling franka_skin_test).
    Runs inside a background thread so the GUI can remain responsive on the main thread.
    """
    global ft_thread, stretchmagtec_reader, r, ft_calibration, stretchmagtec_calibration
    
    # Make these available to GUI adapter
    import __main__
    __main__.ft_thread = None
    __main__.stretchmagtec_reader = None
    __main__.ft_calibration = None
    __main__.stretchmagtec_calibration = None
    __main__.read_stretchmagtec_data = read_stretchmagtec_data
    
    try:
        # Initialize sensors (same as franka_skin_test.py)
        stretchmagtec_ready_event.clear()
        ft_data_ready_event.clear()
        
        print("Starting sensor threads...")
        # Use baud rate and port from config (same as franka_skin_test.py)
        stretchmagtec_reader = StretchMagTecSerialReader(port=config.STRETCHMAGTEC_PORT, baud=config.STRETCHMAGTEC_BAUD)
        print(f"  StretchMagTec sensor: port={config.STRETCHMAGTEC_PORT}, baud={config.STRETCHMAGTEC_BAUD}")
        stretchmagtec_reader.daemon = True
        stretchmagtec_reader.start()
        
        ft_thread = FTSensorThread()
        ft_thread.daemon = True
        ft_thread.start()
        time.sleep(2)
        
        __main__.stretchmagtec_reader = stretchmagtec_reader
        __main__.ft_thread = ft_thread
        
        # Connect to robot
        r = None
        print(f"Connecting to robot at {config.ROBOT_IP}...")
        max_connection_retries = 5
        for attempt in range(max_connection_retries):
            try:
                r = franka.Robot_(config.ROBOT_IP, False, hand_franka=False, auto_init=True, speed_factor=config.ROBOT_SPEED_FACTOR)
                print("✅ Robot connected")
                break
            except Exception as e:
                if attempt < max_connection_retries - 1:
                    print(f"Retrying connection ({attempt + 1}/{max_connection_retries})...")
                    time.sleep(2)
                else:
                    print(f"❌ Failed to connect to robot: {e}")
                    raise
        
        # Move to initial joint positions
        if INITIAL_JOINT_POSITIONS is not None:
            print(f"\n📍 Moving to initial joint positions...")
            current_state = r.getState()
            current_joints = np.array(current_state.q)
            joint_diffs = np.array(INITIAL_JOINT_POSITIONS) - current_joints
            max_joint_diff = np.max(np.abs(joint_diffs))
            num_steps = max(10, int(max_joint_diff * 50))
            speed_factor = 0.05
            for step in range(num_steps):
                alpha = (step + 1) / num_steps
                intermediate_joints = current_joints + alpha * joint_diffs
                r.move_joints(intermediate_joints.tolist(), speed_factor)
                time.sleep(0.2)
            time.sleep(1.0)
            print("✅ Initial joint positions reached")
        
        # Calibration
        print("\n" + "="*70)
        print("CALIBRATION")
        print("="*70)
        
        ft_calibration = DynamicFTCalibration(enabled=True)
        stretchmagtec_calibration = StretchMagTecCalibration(enabled=True)
        
        __main__.ft_calibration = ft_calibration
        __main__.stretchmagtec_calibration = stretchmagtec_calibration
        
        # Wait for StretchMagTec stream (same logic as franka_skin_test.py)
        print("\n" + "="*70)
        print("CHECKING MAGNETIC SENSOR CONNECTION")
        print("="*70)
        
        print("Waiting for StretchMagTec stream to stabilize...")
        STRETCHMAGTEC_STREAM_TIMEOUT = 60.0
        STRETCHMAGTEC_STREAM_STABILIZATION = 2.0
        if not stretchmagtec_ready_event.wait(timeout=STRETCHMAGTEC_STREAM_TIMEOUT):
            print("\n" + "="*70)
            print("❌ CRITICAL: StretchMagTec sensor did not start streaming in time.")
            print("="*70)
            print("Possible causes:")
            print("  1. Port does not exist (check with: ls -l /dev/tty*)")
            print("  2. Cable not connected or loose")
            print("  3. Device not powered on")
            print("  4. Wrong port configured in config.py")
            print("\nPlease:")
            print("  1. Check the magnetic sensor cable connection")
            print("  2. Ensure the device is powered on")
            print("  3. Check available ports: ls -l /dev/tty*")
            print("  4. Update STRETCHMAGTEC_PORT in config.py if needed")
            print("  5. Restart the script after fixing the connection")
            raise RuntimeError("StretchMagTec sensor did not start streaming in time. Check hardware connection.")
        
        print(f"StretchMagTec stream detected. Waiting {STRETCHMAGTEC_STREAM_STABILIZATION:.1f} seconds for stabilization...")
        time.sleep(STRETCHMAGTEC_STREAM_STABILIZATION)
        
        # Check if sensor is reading valid data (not all zeros) - same as franka_skin_test.py
        max_check_attempts = 10
        check_interval = 1.0  # seconds
        sensor_working = False
        
        for check_attempt in range(max_check_attempts):
            sensor_data = read_stretchmagtec_data()
            # Check if any sensor has non-zero readings
            if sensor_data is not None and np.any(np.abs(sensor_data) > 1.0):  # At least 1 unit of magnetic field
                sensor_working = True
                print(f"✅ Magnetic sensor is reading valid data (non-zero values detected)")
                break
            else:
                print(f"⚠️  Check {check_attempt + 1}/{max_check_attempts}: Magnetic sensor readings are zero or invalid")
                if check_attempt < max_check_attempts - 1:
                    print(f"   Waiting {check_interval} seconds before retry...")
                    time.sleep(check_interval)
        
        if not sensor_working:
            print("\n" + "="*70)
            print("❌ MAGNETIC SENSOR NOT WORKING")
            print("="*70)
            print("The magnetic sensor is not reading valid data (all zeros or no data).")
            print("Please:")
            print("  1. Disconnect the magnetic sensor cable")
            print("  2. Reconnect the cable securely")
            print("  3. Press Enter to retry connection...")
            input()
            
            # Restart the sensor thread
            print("Restarting magnetic sensor connection...")
            stretchmagtec_reader.running = False
            stretchmagtec_reader.join(timeout=2.0)
            stretchmagtec_ready_event.clear()
            stretchmagtec_reader = StretchMagTecSerialReader(port=config.STRETCHMAGTEC_PORT, baud=config.STRETCHMAGTEC_BAUD)
            print(f"  StretchMagTec sensor: port={config.STRETCHMAGTEC_PORT}, baud={config.STRETCHMAGTEC_BAUD}")
            stretchmagtec_reader.daemon = True
            stretchmagtec_reader.start()
            __main__.stretchmagtec_reader = stretchmagtec_reader
            time.sleep(2)
            
            # Retry the check
            if not stretchmagtec_ready_event.wait(timeout=STRETCHMAGTEC_STREAM_TIMEOUT):
                raise RuntimeError("StretchMagTec sensor did not start streaming after reconnection.")
            
            time.sleep(STRETCHMAGTEC_STREAM_STABILIZATION)
            sensor_data = read_stretchmagtec_data()
            if sensor_data is None or not np.any(np.abs(sensor_data) > 1.0):
                raise RuntimeError("Magnetic sensor still not working after reconnection. Please check hardware.")
            
            print("✅ Magnetic sensor is now working after reconnection")
        
        print("="*70 + "\n")
        
        # Initial calibrations (same logic as franka_skin_test.py)
        print("\n" + "="*70)
        print("INITIAL SENSOR CALIBRATION")
        print("="*70)
        
        # Temporarily enable calibration objects for initial calibration
        STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED = getattr(config, 'STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED', True)
        STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED = getattr(config, 'STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED', False)
        FT_INITIAL_CALIBRATION_ENABLED = getattr(config, 'FT_INITIAL_CALIBRATION_ENABLED', True)
        FT_PER_POSITION_CALIBRATION_ENABLED = getattr(config, 'FT_PER_POSITION_CALIBRATION_ENABLED', False)
        FT_STREAM_TIMEOUT = 30.0
        FT_STREAM_STABILIZATION = 1.0
        
        if STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED:
            stretchmagtec_calibration.enabled = True
            print("Starting initial StretchMagTec calibration...")
            stretchmagtec_calibration.measure_offsets(stretchmagtec_reader, "initial StretchMagTec calibration")
            # Set back to per-position setting
            stretchmagtec_calibration.enabled = STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED
        else:
            print("⚠️  StretchMagTec initial calibration DISABLED (not recommended)")
        
        if FT_INITIAL_CALIBRATION_ENABLED:
            ft_calibration.enabled = True
            print("\nWaiting for FT sensor stream to stabilize...")
            if not ft_data_ready_event.wait(timeout=FT_STREAM_TIMEOUT):
                raise RuntimeError("FT sensor did not start streaming in time for calibration.")
            time.sleep(FT_STREAM_STABILIZATION)
            print("Starting initial FT sensor calibration...")
            ft_calibration.measure_offset(ft_thread, "initial FT calibration")
            # Set back to per-position setting
            ft_calibration.enabled = FT_PER_POSITION_CALIBRATION_ENABLED
        else:
            print("⚠️  FT initial calibration DISABLED (not recommended)")
        
        print("="*70 + "\n")
        
        # Wait for initial calibration to complete (same as franka_skin_test.py)
        wait_for_initial_calibration_complete = skin_test_module.wait_for_initial_calibration_complete
        try:
            wait_for_initial_calibration_complete(ft_calibration, stretchmagtec_calibration)
            print("Initial calibrations complete.")
        except RuntimeError as exc:
            print(str(exc))
            raise
        
        # Get current stretch configuration from config
        stretch_value = getattr(config, 'CURRENT_STRETCH_VALUE', 0.0)
        stretch_label = getattr(config, 'CURRENT_STRETCH_LABEL', '000pct')
        run_root = Path(config.DATA_DIR)
        run_name = getattr(config, 'CURRENT_RUN_LABEL', 'test')
        
        # Build offsets for current stretch
        updated_offsets = build_offsets_for_stretch(stretch_value)
        config.GRID_OFFSETS.update(updated_offsets)
        
        # Collect data for all points
        sequences_by_point = {}
        base_position = config.MAIN_GRID_POSITIONS[TARGET_POSITION_ID]
        rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
        
        for point_idx, point_name in enumerate(TARGET_OFFSETS):
            print(f"\n{'='*70}")
            print(f" POINT {point_name} ({point_idx + 1}/{len(TARGET_OFFSETS)})")
            print(f"{'='*70}")
            
            # Calculate target position
            target_pos = config.get_position_with_offset(base_position, point_name)
            print(f"Target position: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
            
            # Collect shear sequences
            sequences = collect_shear_data_for_point(
                r, ft_thread, stretchmagtec_reader, ft_calibration, stretchmagtec_calibration,
                target_pos, point_name, stretch_value, stretch_label
            )
            sequences_by_point[point_name] = sequences
            
            # Ask user before next point
            if point_idx < len(TARGET_OFFSETS) - 1:
                print(f"\nPress Enter to continue to next point...")
                input()
        
        # Save data
        output_file = run_root / f"{run_name}_shear_{stretch_label}.h5"
        save_shear_data_to_h5(output_file, sequences_by_point, stretch_value, stretch_label)
        
        # Store output file in config for main() to find
        config.LAST_OUTPUT_FILE = str(output_file)
        
        # Return to initial joint positions
        if INITIAL_JOINT_POSITIONS is not None:
            print(f"\n📍 Returning to initial joint positions...")
            current_state = r.getState()
            current_joints = np.array(current_state.q)
            joint_diffs = np.array(INITIAL_JOINT_POSITIONS) - current_joints
            num_steps = max(10, int(np.max(np.abs(joint_diffs)) * 50))
            speed_factor = 0.05
            for step in range(num_steps):
                alpha = (step + 1) / num_steps
                intermediate_joints = current_joints + alpha * joint_diffs
                r.move_joints(intermediate_joints.tolist(), speed_factor)
                time.sleep(0.2)
            time.sleep(1.0)
            print("✅ Robot at initial joint positions")
        
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if ft_thread:
            ft_thread.running = False
            ft_thread.join(timeout=2.0)
        if stretchmagtec_reader:
            stretchmagtec_reader.running = False
            stretchmagtec_reader.join(timeout=2.0)
        if r:
            try:
                r.stop()
            except:
                pass
        collection_done_event.set()


def execute_collection():
    collection_done = threading.Event()
    collection_thread = threading.Thread(
        target=run_data_collection,
        args=(collection_done,),
        daemon=True,
        name="franka_skin_test_runner",
    )
    collection_thread.start()

    if ENABLE_GUI:
        launch_predictor_gui(collection_done)
    else:
        print("GUI disabled; collecting data without live visualization...")
        while not collection_done.wait(timeout=0.5):
            pass

    collection_thread.join()


def main():
    original_state = capture_original_config()

    try:
        run_label, run_root = generate_run_name()
        run_root.mkdir(parents=True, exist_ok=True)

        print(f"\nSelected run label: {run_label}")
        print(f"Data for this session will be stored under: {run_root}\n")

        explore_offsets()

        collected_files: List[Path] = []

        for idx, stretch_value in enumerate(STRETCH_LEVELS):
            stretch_label = format_stretch_label(stretch_value)
            horizontal_offset_mm = BASE_EW_OFFSET * (1.0 + stretch_value) * 1000.0

            print("\n" + "=" * 70)
            print(f"Stretch Run {idx + 1}/{len(STRETCH_LEVELS)} → {int(round(stretch_value * 100))}% elongation")
            print("=" * 70)
            print(f"Output directory: {run_root}")
            print(f"Shear sequences per point: {len(SHEAR_SEQUENCES)}")
            print(f"Fz target: {FZ_TARGET}N, Fx/Fy target: ±{FX_TARGET}N")
            print(f"Data collection duration: {DATA_COLLECTION_DURATION}s per sequence")

            if PROMPT_FOR_STRETCH:
                input(f"\nSet the skin to approximately {int(round(stretch_value * 100))}% stretch and press Enter to continue...")

            configure_for_stretch(stretch_value, stretch_label, run_root, run_label, idx + 1)
            execute_collection()

            last_file = getattr(config, "LAST_OUTPUT_FILE", None)
            if last_file:
                path = Path(last_file)
                collected_files.append(path)
                print(f"\nCompleted stretch level {stretch_label}. Data stored in {path}")
            else:
                print(f"\nCompleted stretch level {stretch_label}. (No output file reported)")
            
            # Return to initial joint positions between stretch levels (except after the last one)
            # Note: The robot should already be at initial joint positions after each press,
            # but we ensure it here as well for consistency
            if idx < len(STRETCH_LEVELS) - 1 and INITIAL_JOINT_POSITIONS is not None:
                print(f"\n📍 Ensuring robot is at initial joint positions before next stretch level...")
                try:
                    robot = franka.Robot_(config.ROBOT_IP, False, hand_franka=False,
                                         auto_init=True, speed_factor=config.ROBOT_SPEED_FACTOR)
                    # Move incrementally in smaller steps to ensure slow, safe movement
                    import numpy as np
                    current_state = robot.getState()
                    current_joints = np.array(current_state.q)
                    joint_diffs = np.array(INITIAL_JOINT_POSITIONS) - current_joints
                    max_joint_diff = np.max(np.abs(joint_diffs))
                    
                    num_steps = max(10, int(max_joint_diff * 50))  # At least 10 steps
                    speed_factor = 0.05  # 5% speed for very slow, safe movement
                    print(f"   Moving in {num_steps} steps with {speed_factor*100}% speed...")
                    
                    for step in range(num_steps):
                        alpha = (step + 1) / num_steps
                        intermediate_joints = current_joints + alpha * joint_diffs
                        # Use very low speed_factor (0.05 = 5% speed) for safe, slow movement
                        robot.move_joints(intermediate_joints.tolist(), speed_factor)  # speed_factor, not duration!
                        time.sleep(0.2)  # Pause between steps to ensure completion
                    time.sleep(1.0)  # Wait for stabilization
                    print("✅ Robot at initial joint positions")
                    robot.stop()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to return to initial joint positions: {e}")
                    print("   Continuing to next stretch level...")
            
            time.sleep(2.0)

        if collected_files:
            print(f"\n✅ Data collection complete. {len(collected_files)} file(s) collected:")
            for f in collected_files:
                print(f"   - {f}")
            print("\n⚠️  Training pipeline disabled. Run training manually if needed.")
            # run_training_pipeline(run_root, run_label)  # Disabled to avoid reconnection issues
        else:
            print("\nNo data files were collected; skipping training pipeline.")

    finally:
        restore_original_config(original_state)
        print(f"\nAll configured stretch levels processed. Results saved in {run_root}")
        print("Configuration restored.")


if __name__ == "__main__":
    main()

