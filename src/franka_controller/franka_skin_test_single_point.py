#!/usr/bin/env python3
"""
Single-point variant of the Franka skin data collection script.

This wrapper sets the configuration to collect data only on one main grid
position (default: 32) and a selected set of offsets. It also spins up the
same visualization GUI used in `validation_tests/real_time_predictor.py`
so you can monitor StretchMagTec readings in real time while data collection
is running.

Update `TARGET_POSITION_ID` or `TARGET_OFFSETS` below as needed.
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
TARGET_OFFSETS = ['center', 'nw', 'ne', 'se', 'sw']
TARGET_POSITION_COORDS = [0.493776, 0.440500, 0.031311]

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

# Stretch levels to test (percentages expressed as decimal fractions)
STRETCH_LEVELS = [0.00, 0.10, 0.20]
PROMPT_FOR_STRETCH = True  # Prompt operator before each stretch run
PROMPT_FOR_RUN_LABEL = False  # Run label generated automatically

# Pressing profile configuration (stepwise: 0.5 mm × 5 steps = 2.5 mm)
PRESS_DEPTH_MM = 2.5            # Total press depth (mm)
PRESS_STEP_MM = PRESS_DEPTH_MM  # Single indentation (no intermediate steps)
STEPWISE_MODE = False           # Continuous press, single movement
PRESS_HOLD_S = 1.0              # Hold at maximum indentation before lift
DWELL_AFTER_LIFT_S = 0.5        # Pause after lift (seconds)
PRESSES_PER_POINT = 50          # Number of press cycles per offset

# GUI flag (set False to disable visualization)
ENABLE_GUI = True

# Base references for restoring configuration after the test
BASE_DATA_DIR = Path(config.DATA_DIR)

# Apply single-point selection defaults
config.SELECTED_POSITIONS = [TARGET_POSITION_ID]
config.SELECTED_OFFSETS = TARGET_OFFSETS
config.MAIN_GRID_POSITIONS[TARGET_POSITION_ID] = TARGET_POSITION_COORDS
config.GRID_OFFSETS.update(BASE_OFFSETS)

print("=" * 70)
print(" SINGLE-POINT SKIN TEST (MULTI-STRETCH) ")
print("=" * 70)
print(f"Target position ID: {TARGET_POSITION_ID}")
print(f"Offsets under test: {', '.join(TARGET_OFFSETS)}")
print(f"Base coordinates: {TARGET_POSITION_COORDS}")
print(f"Baseline offset distances → NS: {BASE_NS_OFFSET*1000:.1f} mm, EW: {BASE_EW_OFFSET*1000:.1f} mm")
print(f"Stretch levels configured: {', '.join(f'{int(s*100)}%' for s in STRETCH_LEVELS)}")
print(f"Press strategy: single indentation {PRESS_DEPTH_MM:.1f} mm (no intermediate steps)")
print(f"Hold before lift: {PRESS_HOLD_S:.1f}s  |  Dwell after lift: {DWELL_AFTER_LIFT_S:.1f}s")
print(f"Presses per point: {PRESSES_PER_POINT}")
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
    return offsets


def generate_run_name() -> Tuple[str, Path]:
    indent_tag = f"{PRESS_DEPTH_MM:.1f}mm_single"
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
    config.NUMBER_OF_PRESSES = PRESSES_PER_POINT
    config.STEPS_PER_PRESS = 1
    config.DZ_PRESS = -(PRESS_DEPTH_MM / 1000.0)
    config.DZ_LIFT = PRESS_DEPTH_MM / 1000.0
    config.PRESS_DELAY = PRESS_HOLD_S
    config.LIFT_DELAY = DWELL_AFTER_LIFT_S
    press_profile = "single_press"

    config.DATA_DIR = run_root

    config.CURRENT_RUN_LABEL = run_name
    config.CURRENT_STRETCH_VALUE = stretch_value
    config.CURRENT_STRETCH_LABEL = stretch_label
    config.CURRENT_PRESS_PROFILE = press_profile
    config.CURRENT_PRESS_SETTINGS = {
        "press_depth_mm": PRESS_DEPTH_MM,
        "press_step_mm": PRESS_STEP_MM,
        "stepwise": bool(STEPWISE_MODE),
        "step_hold_s": PRESS_HOLD_S if STEPWISE_MODE else PRESS_HOLD_S,
        "presses_per_point": PRESSES_PER_POINT,
    }
    config.CURRENT_STRETCH_INDEX = stretch_idx
    config.CURRENT_OUTPUT_PREFIX = f"{run_name}_{stretch_label}"
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


def run_data_collection(collection_done_event: threading.Event):
    """
    Execute the standard `franka_skin_test` script as if it were launched
    directly. Runs inside a background thread so the GUI can remain responsive
    on the main thread.
    """
    try:
        runpy.run_module('franka_controller.franka_skin_test', run_name='__main__', alter_sys=True)
    except SystemExit:
        pass
    finally:
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

        collected_files: List[Path] = []

        for idx, stretch_value in enumerate(STRETCH_LEVELS):
            stretch_label = format_stretch_label(stretch_value)
            horizontal_offset_mm = BASE_EW_OFFSET * (1.0 + stretch_value) * 1000.0
            if STEPWISE_MODE:
                steps = max(1, int(round(PRESS_DEPTH_MM / PRESS_STEP_MM)))
                profile_desc = f"Stepwise {PRESS_STEP_MM:.2f} mm × {steps} steps (hold {PRESS_HOLD_S:.1f}s)"
            else:
                profile_desc = f"Single press {PRESS_DEPTH_MM:.1f} mm (hold {PRESS_HOLD_S:.1f}s)"

            print("\n" + "=" * 70)
            print(f"Stretch Run {idx + 1}/{len(STRETCH_LEVELS)} → {int(round(stretch_value * 100))}% elongation")
            print("=" * 70)
            print(f"Output directory: {run_root}")
            print(f"Horizontal offset distance: {horizontal_offset_mm:.2f} mm")
            print(f"Press profile: {profile_desc}")
            print(f"Presses per offset: {PRESSES_PER_POINT}")

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
            time.sleep(2.0)

        if collected_files:
            run_training_pipeline(run_root, run_label)
        else:
            print("\nNo data files were collected; skipping training pipeline.")

    finally:
        restore_original_config(original_state)
        print(f"\nAll configured stretch levels processed. Results saved in {run_root}")
        print("Configuration restored.")


if __name__ == "__main__":
    main()

