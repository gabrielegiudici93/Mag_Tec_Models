#!/usr/bin/env python3
import numpy as np
import pyfranka_interface as franka
from scipy.spatial.transform import Rotation as R
import sys
import tty
import termios
import os
import threading
import time
import tkinter as tk

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *
import franka_controller.franka_skin_test as skin_module
import validation_tests.real_time_predictor as predictor

# =============================================================================
# TELEOP CONFIGURATION
# =============================================================================
# Select which position to start teleoperation from
SELECTED_POSITION_ID = 23  # Default to position 11 (row 1, col 1)
SELECTED_OFFSET = 'center'  # Default to center offset
ENABLE_GUI = True  # Set to False to disable real-time visualization

if SELECTED_POSITION_ID not in MAIN_GRID_POSITIONS:
    raise ValueError(f"Invalid position ID: {SELECTED_POSITION_ID}. Available IDs: {list(MAIN_GRID_POSITIONS.keys())}")

# Get desired position with offset
base_position = MAIN_GRID_POSITIONS[SELECTED_POSITION_ID]
desired_position = get_position_with_offset(base_position, SELECTED_OFFSET)

# Set rotation matrix
rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()

# Create desired pose
des_pos_fingertip_setup = np.eye(4)
des_pos_fingertip_setup[:3, :3] = rotation_matrix
des_pos_fingertip_setup[:3, 3] = desired_position

STEP_SIZE = 0.001  # Initial step size


def move_relative(r, dx, dy, dz):
    delta_transform = np.eye(4)
    delta_transform[:3, 3] = [dx, dy, dz]
    r.move("relative", delta_transform)


def get_key():
    """Read one keypress as a single character."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_controls():
    print("\n" + "=" * 60)
    print("ROBOT TELEOPERATION CONTROLS")
    print("=" * 60)
    print("Movement Keys:")
    print("  8 : Move X -step_size (Up)")
    print("  2 : Move X +step_size (Down)")
    print("  6 : Move Y +step_size (Right)")
    print("  4 : Move Y -step_size (Left)")
    print("Depth Keys:")
    print("  + : Move Z +step_size")
    print("  - : Move Z -step_size")
    print("Step size control:")
    print("  * : Increase step size by 0.001")
    print("  / : Decrease step size by 0.001 (min 0.001)")
    print("Other Keys:")
    print("  q or c: Quit teleoperation")
    print("  p : Print current position")
    print("  0 : Clear accumulated displacement")
    print("=" * 60)
    print(f"Current step size: {STEP_SIZE:.3f} m")
    print("Press any key to start...")
    get_key()


def print_position_info(current_pos, accumulated_displacement):
    print(
        f"\nCurrent Position (X, Y, Z): ({current_pos[0]:.6f}, "
        f"{current_pos[1]:.6f}, {current_pos[2]:.6f})"
    )
    print(
        f"Accumulated Displacement (X, Y, Z): "
        f"({accumulated_displacement[0]:.6f}, "
        f"{accumulated_displacement[1]:.6f}, "
        f"{accumulated_displacement[2]:.6f})"
    )


class TeleopSensorAdapter:
    """
    Adapts the StretchMagTec + FT sensor streams (managed by franka_skin_test module)
    to the interface expected by RealTimePredictorGUI.
    """

    def __init__(self, ft_thread, max_buffer_size=1000):
        self.ft_thread = ft_thread
        self.running = False
        self.max_buffer_size = max_buffer_size

        self.ft_buffer = []
        self.stretch_buffer = []
        self.time_buffer = []

        self.sensor_hz_values = [0.0] * skin_module.STRETCHMAGTEC_SENSORS
        self._sensor_hz_counts = [0] * skin_module.STRETCHMAGTEC_SENSORS
        self._last_hz_time = time.time()

        self.individual_sensor_buffers = {
            sensor_id: {'X': [], 'Y': [], 'Z': [], 'time': []}
            for sensor_id in range(skin_module.STRETCHMAGTEC_SENSORS)
        }

        self._latest_data = np.zeros((skin_module.STRETCHMAGTEC_SENSORS, skin_module.STRETCHMAGTEC_CHANNELS))
        self._latest_ft = np.zeros(6)
        self._lock = threading.Lock()
        self._poll_thread = None
        self._poll_interval = getattr(skin_module, 'PERIOD', 0.01)

    def start_sensors(self):
        if self.running:
            return
        self.running = True
        self.ft_buffer.clear()
        self.stretch_buffer.clear()
        self.time_buffer.clear()
        self._sensor_hz_counts = [0] * skin_module.STRETCHMAGTEC_SENSORS
        self._last_hz_time = time.time()
        for buffer in self.individual_sensor_buffers.values():
            for key in buffer:
                buffer[key].clear()

        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="teleop_gui_poll")
            self._poll_thread.start()

    def stop_sensors(self):
        self.running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self):
        while self.running:
            stretch_event = getattr(skin_module, 'stretchmagtec_ready_event', None)
            stretch_stream_ready = stretch_event.is_set() if stretch_event else False

            data = skin_module.read_stretchmagtec_data()
            if data is not None and data.size:
                if stretch_stream_ready and skin_module.stretchmagtec_calibration.is_calibrated:
                    try:
                        data = skin_module.stretchmagtec_calibration.compensate_sensors(data)
                    except Exception:
                        pass

                ft_reading = np.zeros(6)
                if self.ft_thread is not None and hasattr(self.ft_thread, 'get_ft'):
                    try:
                        ft_reading = np.array(self.ft_thread.get_ft(), dtype=float)
                    except Exception:
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

                    for sensor_id in range(skin_module.STRETCHMAGTEC_SENSORS):
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
                        for sensor_id in range(skin_module.STRETCHMAGTEC_SENSORS):
                            self.sensor_hz_values[sensor_id] = self._sensor_hz_counts[sensor_id] / elapsed
                            self._sensor_hz_counts[sensor_id] = 0
                        self._last_hz_time = current_time

            time.sleep(max(0.001, self._poll_interval))

    def get_ft_data(self):
        with self._lock:
            return self._latest_ft.copy()

    def get_stretchmagtec_data(self):
        with self._lock:
            return self._latest_data.copy()

    def get_plot_data(self):
        with self._lock:
            if not self.time_buffer:
                return [], [], []
            t0 = self.time_buffer[0]
            relative_time = [t - t0 for t in self.time_buffer]
            return self.ft_buffer.copy(), self.stretch_buffer.copy(), relative_time


class VisualizationModelStub:
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


def launch_teleop_gui(session_done_event, sensor_adapter):
    gui = predictor.RealTimePredictorGUI(model_dir=MODELS_DIR)

    gui.sensor_reader = sensor_adapter
    gui.model_predictor = VisualizationModelStub()

    def load_models_stub():
        gui.status_label.config(text="Status: Visualization only (no models)", foreground="purple")
        return False

    gui.load_models = load_models_stub
    gui.load_models_button.config(state=tk.DISABLED)
    gui.grid_viz_button.config(state=tk.DISABLED)
    gui.start_button.config(state=tk.DISABLED)
    gui.stop_button.config(state=tk.DISABLED)

    gui.contact_label.config(text="Contact: visualization only", foreground="purple")
    gui.confidence_label.config(text="Confidence: --", foreground="purple")
    for lbl in gui.force_pred_labels:
        lbl.config(text="Fx/Fy/Fz: --", foreground="gray")

    sensor_adapter.start_sensors()
    gui.update_running = True

    def gui_update_loop():
        if not gui.update_running:
            return

        ft_data = gui.sensor_reader.get_ft_data()
        stretch_data = gui.sensor_reader.get_stretchmagtec_data()

        ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
        for i, value in enumerate(ft_data):
            color = "red" if abs(value) > 1.0 else "black"
            gui.ft_labels[i].config(text=f"{ft_names[i]}: {value:7.3f}", foreground=color)

        for sensor_id in range(skin_module.STRETCHMAGTEC_SENSORS):
            for channel_id in range(skin_module.STRETCHMAGTEC_CHANNELS):
                channel_name = ['X', 'Y', 'Z'][channel_id]
                value = stretch_data[sensor_id, channel_id]
                color = "red" if abs(value) > skin_module.STRETCHMAGTEC_THRESHOLD else "black"
                gui.stretchmagtec_labels[sensor_id][channel_id].config(
                    text=f"{channel_name}: {value:6.0f}", foreground=color
                )

        hz_values = gui.sensor_reader.sensor_hz_values
        for sensor_id in range(skin_module.STRETCHMAGTEC_SENSORS):
            gui.stretchmagtec_labels[sensor_id][3].config(text=f"Hz: {hz_values[sensor_id]:.1f}")

        ft_plot, stretch_plot, time_plot = gui.sensor_reader.get_plot_data()
        if time_plot and gui.selected_sensors:
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

            if ft_plot:
                fx = [row[0] for row in ft_plot]
                fy = [row[1] for row in ft_plot]
                fz = [row[2] for row in ft_plot]
                gui.ax1.plot(time_plot, fx, label="Fx")
                gui.ax1.plot(time_plot, fy, label="Fy")
                gui.ax1.plot(time_plot, fz, label="Fz")
                gui.ax1.legend(loc="upper right")

            selected = sorted(gui.selected_sensors)
            for sensor_id in selected:
                if sensor_id < len(stretch_plot):
                    series = [row[sensor_id, :] for row in stretch_plot]
                    x_series = [row[0] for row in series]
                    y_series = [row[1] for row in series]
                    z_series = [row[2] for row in series]
                    gui.ax2.plot(time_plot, x_series, label=f"S{sensor_id+1}")
                    gui.ax3.plot(time_plot, y_series, label=f"S{sensor_id+1}")
                    gui.ax4.plot(time_plot, z_series, label=f"S{sensor_id+1}")

            if selected:
                labels = [f"S{s+1}" for s in selected]
                gui.ax2.set_title(f"StretchMagTec X-Axis: {labels}")
                gui.ax3.set_title(f"StretchMagTec Y-Axis: {labels}")
                gui.ax4.set_title(f"StretchMagTec Z-Axis: {labels}")
                gui.ax2.legend(loc="upper right", fontsize=8)
                gui.ax3.legend(loc="upper right", fontsize=8)
                gui.ax4.legend(loc="upper right", fontsize=8)

            gui.fig.tight_layout()
            gui.canvas.draw_idle()

        gui.root.after(gui.update_interval, gui_update_loop)

    gui.root.after(gui.update_interval, gui_update_loop)

    def refresh_status():
        ft_enabled = getattr(skin_module, 'FT_INITIAL_CALIBRATION_ENABLED', True)
        stretch_enabled = getattr(skin_module, 'STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED', True)

        ft_ready = True
        stretch_ready = True

        if ft_enabled and getattr(skin_module, 'ft_calibration', None) is not None:
            ft_ready = getattr(skin_module.ft_calibration, 'is_calibrated', False)

        if stretch_enabled and getattr(skin_module, 'stretchmagtec_calibration', None) is not None:
            stretch_ready = getattr(skin_module.stretchmagtec_calibration, 'is_calibrated', False)

        if ft_ready and stretch_ready:
            gui.status_label.config(text="Status: Receiving data (calibrated)", foreground="green")
        else:
            gui.status_label.config(text="Status: Reading data (uncalibrated)", foreground="orange")

        gui.root.after(500, refresh_status)

    gui.root.after(500, refresh_status)

    def check_session():
        if session_done_event.is_set():
            gui.update_running = False
            gui.root.after(250, gui.root.quit)
        else:
            gui.root.after(500, check_session)

    gui.root.after(500, check_session)

    try:
        gui.root.mainloop()
    finally:
        sensor_adapter.stop_sensors()

def teleop_loop(stop_event, session_done_event):
    global STEP_SIZE
    print("Connecting to robot...")
    robot = franka.Robot_(ROBOT_IP, False, hand_franka=False, speed_factor=ROBOT_SPEED_FACTOR)

    print(
        f"Moving to position {SELECTED_POSITION_ID} "
        f"(matrix position {SELECTED_POSITION_ID // 10},{SELECTED_POSITION_ID % 10})"
    )
    robot.move("absolute", des_pos_fingertip_setup)
    print("Robot moved to desired position")

    accumulated_displacement = [0.0, 0.0, 0.0]

    print_controls()

    print(f"\nStarting teleoperation at position {SELECTED_POSITION_ID}")
    print("Press 'q' to quit, 'p' to print position, 'c' to clear displacement")

    current_pos = robot.getState().T[:3, 3]

    try:
        while not stop_event.is_set():
            cur_state = robot.getState()
            current_pos = cur_state.T[:3, 3]

            key = get_key()
            print(f"Key pressed: {repr(key)}")

            if key in ('q', 'c'):
                print("\nQuitting teleoperation...")
                stop_event.set()
                break            

            if key == 'p':
                print_position_info(current_pos, accumulated_displacement)
            elif key == '0':
                accumulated_displacement = [0.0, 0.0, 0.0]
                print(
                    f"\nAccumulated displacement cleared. Current: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '*':
                if STEP_SIZE < 0.001:
                    STEP_SIZE += 0.0001
                else:
                    STEP_SIZE += 0.001
                print(f"Step size increased to {STEP_SIZE:.4f} m")
            elif key == '/':
                if STEP_SIZE > 0.001:
                    STEP_SIZE -= 0.001
                else:
                    STEP_SIZE = max(0.0005, STEP_SIZE - 0.0001)
                    print(f"Step size decreased to {STEP_SIZE:.4f} m")
            elif key == '8':
                print(f"Moving X -{STEP_SIZE:.3f}m...")
                move_relative(robot, -STEP_SIZE, 0, 0)
                accumulated_displacement[0] -= STEP_SIZE
                print(
                    f"↑ Moved X -{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '2':
                print(f"Moving X +{STEP_SIZE:.3f}m...")
                move_relative(robot, STEP_SIZE, 0, 0)
                accumulated_displacement[0] += STEP_SIZE
                print(
                    f"↓ Moved X +{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '6':
                print(f"Moving Y +{STEP_SIZE:.3f}m...")
                move_relative(robot, 0, STEP_SIZE, 0)
                accumulated_displacement[1] += STEP_SIZE
                print(
                    f"→ Moved Y +{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '4':
                print(f"Moving Y -{STEP_SIZE:.3f}m...")
                move_relative(robot, 0, -STEP_SIZE, 0)
                accumulated_displacement[1] -= STEP_SIZE
                print(
                    f"← Moved Y -{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '+':
                print(f"Moving Z +{STEP_SIZE:.3f}m...")
                move_relative(robot, 0, 0, STEP_SIZE)
                accumulated_displacement[2] += STEP_SIZE
                print(
                    f"+ Moved Z +{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            elif key == '-':
                print(f"Moving Z -{STEP_SIZE:.3f}m...")
                move_relative(robot, 0, 0, -STEP_SIZE)
                accumulated_displacement[2] -= STEP_SIZE
                print(
                    f"- Moved Z -{STEP_SIZE:.3f}m | Accumulated: "
                    f"({accumulated_displacement[0]:.6f}, "
                    f"{accumulated_displacement[1]:.6f}, "
                    f"{accumulated_displacement[2]:.6f})"
                )
            else:
                print(f"Unknown key: {repr(key)}")

    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user")
        stop_event.set()

    finally:
        print_position_info(current_pos, accumulated_displacement)
        print("Teleoperation session ended.")
        session_done_event.set()


def perform_initial_calibrations(stretch_reader, ft_thread):
    print("\n" + "=" * 70)
    print("INITIAL SENSOR CALIBRATION")
    print("=" * 70)

    if skin_module.STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED:
        skin_module.stretchmagtec_calibration.enabled = True
        print("Waiting for StretchMagTec stream to stabilize...")
        if not skin_module.stretchmagtec_ready_event.wait(timeout=skin_module.STRETCHMAGTEC_STREAM_TIMEOUT):
            raise RuntimeError("StretchMagTec sensor did not start streaming in time for calibration.")
        print(f"StretchMagTec stream detected. Waiting {skin_module.STRETCHMAGTEC_STREAM_STABILIZATION:.1f} seconds before calibration...")
        time.sleep(skin_module.STRETCHMAGTEC_STREAM_STABILIZATION)
        skin_module.stretchmagtec_calibration.measure_offsets(stretch_reader, "initial StretchMagTec calibration")
        skin_module.stretchmagtec_calibration.enabled = skin_module.STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED
    else:
        print("⚠️  StretchMagTec initial calibration DISABLED (not recommended)")

    if skin_module.FT_INITIAL_CALIBRATION_ENABLED:
        skin_module.ft_calibration.enabled = True
        print("\nWaiting for FT sensor stream to stabilize...")
        if not skin_module.ft_data_ready_event.wait(timeout=skin_module.FT_STREAM_TIMEOUT):
            raise RuntimeError("FT sensor did not start streaming in time for calibration.")
        time.sleep(skin_module.FT_STREAM_STABILIZATION)
        skin_module.ft_calibration.measure_offset(ft_thread, "initial FT calibration")
        skin_module.ft_calibration.enabled = skin_module.FT_PER_POSITION_CALIBRATION_ENABLED
    else:
        print("⚠️  FT initial calibration DISABLED (not recommended)")

    try:
        skin_module.wait_for_initial_calibration_complete(
            skin_module.ft_calibration,
            skin_module.stretchmagtec_calibration
        )
    except RuntimeError:
        pass

    print("=" * 70 + "\n")


def stop_sensor_threads(stretch_reader, ft_thread):
    if stretch_reader:
        stretch_reader.running = False
        stretch_reader.join(timeout=2.0)
    if ft_thread:
        ft_thread.running = False
        ft_thread.join(timeout=2.0)


def main():
    if hasattr(skin_module, 'stretchmagtec_ready_event'):
        skin_module.stretchmagtec_ready_event.clear()
    if hasattr(skin_module, 'ft_data_ready_event'):
        skin_module.ft_data_ready_event.clear()

    stretch_reader = skin_module.StretchMagTecSerialReader()
    stretch_reader.daemon = True
    stretch_reader.start()

    ft_thread = skin_module.FTSensorThread()
    ft_thread.daemon = True
    ft_thread.start()

    stop_event = threading.Event()
    session_done_event = threading.Event()

    try:
        perform_initial_calibrations(stretch_reader, ft_thread)
    except Exception as exc:
        print(f"❌ Initial calibration failed: {exc}")
        stop_sensor_threads(stretch_reader, ft_thread)
        return

    teleop_thread = threading.Thread(target=teleop_loop, args=(stop_event, session_done_event), daemon=True)
    teleop_thread.start()

    try:
        if ENABLE_GUI:
            adapter = TeleopSensorAdapter(ft_thread)
            try:
                launch_teleop_gui(session_done_event, adapter)
            finally:
                stop_event.set()
        else:
            teleop_thread.join()
    except KeyboardInterrupt:
        print("\nTeleoperation interrupted by user")
        stop_event.set()
    finally:
        stop_event.set()
        teleop_thread.join(timeout=2.0)
        stop_sensor_threads(stretch_reader, ft_thread)


if __name__ == '__main__':
    main()
