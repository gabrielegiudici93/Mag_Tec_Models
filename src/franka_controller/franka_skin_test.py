#!/usr/bin/env python3
"""
Franka Skin Test - Data Collection Script

This script performs automated data collection for the MagTecK_PM tactile sensing system.
It visits each grid position with 9-point offsets and collects:
- FT sensor data
- StretchMagTec 3x5 sensor data  
- Robot end-effector position

Features:
- Configurable grid with 9-point offset system (center, nw, n, ne, w, e, sw, s, se)
- Optional FT sensor and StretchMagTec calibration
- Continuous data logging at 100Hz
- HDF5 data storage with comprehensive metadata

Author: Gabriele Giudici
Date: 2025
"""

import numpy as np
import time
import serial
import threading
from datetime import datetime
import h5py
import libscrc
import minimalmodbus as mm
import pyfranka_interface as franka
from scipy.spatial.transform import Rotation as R
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *
import franka_controller.config as config_module

# =============================================================================
# STREAM STABILIZATION CONSTANTS
# =============================================================================
STRETCHMAGTEC_STREAM_TIMEOUT = 60.0  # Maximum wait for first StretchMagTec frame
STRETCHMAGTEC_STREAM_STABILIZATION = 15.0  # Additional wait after first frame (seconds)
FT_STREAM_TIMEOUT = 30.0  # Maximum wait for first FT frame
FT_STREAM_STABILIZATION = 3.0  # Additional wait after first frame (seconds)

# =============================================================================
# FT SENSOR DYNAMIC CALIBRATION
# =============================================================================
class DynamicFTCalibration:
    """
    Dynamic FT sensor calibration system.
    
    This system measures the actual force offset during stable periods (before/after press cycles)
    and applies compensation to zero out the sensor readings.
    """
    def __init__(self, enabled=FT_CALIBRATION_ENABLED):
        self.enabled = enabled
        self.current_offset = [0, 0, 0, 0, 0, 0]  # [fx, fy, fz, tx, ty, tz]
        self.is_calibrated = False
        self.calibration_history = []  # Store all calibration results
    
    def measure_offset(self, ft_sensor, description="calibration"):
        """Measure the force offset during calibration period."""
        if not self.enabled:
            print(f"FT calibration disabled - skipping {description}")
            return self.current_offset
            
        print(f"Starting {description} measurement ({FT_CALIBRATION_DURATION} seconds)...")
        
        # Collect samples
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < FT_CALIBRATION_DURATION:
            force_reading = ft_sensor.get_raw_ft()  # Use raw reading for calibration
            samples.append(force_reading)
            time.sleep(0.01)  # 100 Hz sampling
        
        # Calculate the mean offset
        if samples:
            samples_array = np.array(samples)
            mean_offset = np.mean(samples_array, axis=0)
            std_offset = np.std(samples_array, axis=0)
            
            self.current_offset = mean_offset.tolist()
            self.is_calibrated = True
            
            # Store calibration result in history
            calibration_result = {
                'description': description,
                'mean_offset': mean_offset.tolist(),
                'std_offset': std_offset.tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            }
            self.calibration_history.append(calibration_result)
            
            print(f"{description.capitalize()} complete:")
            print(f"  Mean offset: {[round(x, 3) for x in self.current_offset]}")
            print(f"  Std deviation: {[round(x, 3) for x in std_offset]}")
            
            return self.current_offset
        else:
            print(f"Warning: No samples collected during {description}")
            return [0, 0, 0, 0, 0, 0]
    
    def compensate_force(self, force_reading):
        """Apply the current offset compensation to a force reading."""
        if not self.is_calibrated:
            return force_reading
        
        compensated_force = []
        for i in range(6):
            compensated_force.append(force_reading[i] - self.current_offset[i])
        return compensated_force
    
    def reset_calibration(self):
        """Reset the calibration status."""
        self.is_calibrated = False
        self.current_offset = [0, 0, 0, 0, 0, 0]
    
    def print_calibration_summary(self):
        """Print a summary of all calibration results."""
        if not self.calibration_history:
            print("No FT calibration data available.")
            return
        
        print("\n" + "="*60)
        print("FT SENSOR CALIBRATION SUMMARY")
        print("="*60)
        
        for i, cal in enumerate(self.calibration_history, 1):
            print(f"\n{i}. {cal['description'].upper()}")
            print(f"   Timestamp: {cal['timestamp']}")
            print(f"   Mean offset: [fx={cal['mean_offset'][0]:.3f}, fy={cal['mean_offset'][1]:.3f}, fz={cal['mean_offset'][2]:.3f}, tx={cal['mean_offset'][3]:.3f}, ty={cal['mean_offset'][4]:.3f}, tz={cal['mean_offset'][5]:.3f}]")
            print(f"   Std deviation: [fx={cal['std_offset'][0]:.3f}, fy={cal['std_offset'][1]:.3f}, fz={cal['std_offset'][2]:.3f}, tx={cal['std_offset'][3]:.3f}, ty={cal['std_offset'][4]:.3f}, tz={cal['std_offset'][5]:.3f}]")
        
        print("\n" + "="*60)

# Global calibration object
ft_calibration = DynamicFTCalibration()

# =============================================================================
# STRETCHMAGTEC CALIBRATION
# =============================================================================
class StretchMagTecCalibration:
    """
    StretchMagTec 3x5 sensor calibration system.
    
    This system measures the offset for each of the 15 magnetic sensors independently during
    calibration window at the beginning of the experiment.
    """
    def __init__(self, enabled=STRETCHMAGTEC_CALIBRATION_ENABLED, num_sensors=STRETCHMAGTEC_SENSORS, num_channels=STRETCHMAGTEC_CHANNELS):
        self.enabled = enabled
        self.num_sensors = num_sensors
        self.num_channels = num_channels
        self.offsets = np.zeros((num_sensors, num_channels))  # [sensor_id, channel]
        self.is_calibrated = False
        self.calibration_history = []  # Store all calibration results
        
    def measure_offsets(self, sensor_reader, description="StretchMagTec calibration"):
        """
        Measure the offset for each sensor during calibration period.
        This should be called when the sensors are in a stable position (no contact).
        """
        if not self.enabled:
            print(f"StretchMagTec calibration disabled - skipping {description}")
            return self.offsets
            
        print(f"Starting {description} ({STRETCHMAGTEC_CALIBRATION_DURATION} seconds)...")
        
        # Collect samples
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < STRETCHMAGTEC_CALIBRATION_DURATION:
            sensor_data = read_stretchmagtec_data()
            if sensor_data is not None and sensor_data.shape == (self.num_sensors, self.num_channels):
                samples.append(sensor_data.copy())
            time.sleep(0.01)  # 100 Hz sampling
        
        # Calculate the mean offset for each sensor and channel
        if samples:
            samples_array = np.array(samples)  # Shape: [time_samples, num_sensors, num_channels]
            mean_offsets = np.mean(samples_array, axis=0)  # Shape: [num_sensors, num_channels]
            std_offsets = np.std(samples_array, axis=0)
            
            self.offsets = mean_offsets
            self.is_calibrated = True
            
            # Store calibration result in history
            calibration_result = {
                'description': description,
                'mean_offsets': mean_offsets.tolist(),
                'std_offsets': std_offsets.tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            }
            self.calibration_history.append(calibration_result)
            
            print(f"{description.capitalize()} complete:")
            for sensor_id in range(self.num_sensors):
                print(f"  Sensor {sensor_id+1}: offset = {[round(x, 4) for x in self.offsets[sensor_id]]}, std = {[round(x, 4) for x in std_offsets[sensor_id]]}")
            
            return self.offsets
        else:
            print(f"Warning: No samples collected during {description}")
            return np.zeros((self.num_sensors, self.num_channels))
    
    def compensate_sensors(self, sensor_data):
        """Apply offset compensation to sensor data."""
        if not self.is_calibrated or sensor_data is None:
            return sensor_data
        
        # Subtract the offsets from the sensor data
        compensated_data = sensor_data - self.offsets
        return compensated_data
    
    def reset_calibration(self):
        """Reset the calibration offsets."""
        self.offsets = np.zeros((self.num_sensors, self.num_channels))
        self.is_calibrated = False
    
    def print_calibration_summary(self):
        """Print a summary of all sensor calibration results."""
        if not self.calibration_history:
            print("No StretchMagTec calibration data available.")
            return
        
        print("\n" + "="*60)
        print("STRETCHMAGTEC 3x5 CALIBRATION SUMMARY")
        print("="*60)
        
        for i, cal in enumerate(self.calibration_history, 1):
            print(f"\n{i}. {cal['description'].upper()}")
            print(f"   Timestamp: {cal['timestamp']}")
            for sensor_id in range(self.num_sensors):
                mean_offsets = cal['mean_offsets'][sensor_id]
                std_offsets = cal['std_offsets'][sensor_id]
                print(f"   Sensor {sensor_id+1}: offset = [x={mean_offsets[0]:.4f}, y={mean_offsets[1]:.4f}, z={mean_offsets[2]:.4f}], std = [x={std_offsets[0]:.4f}, y={std_offsets[1]:.4f}, z={std_offsets[2]:.4f}]")
        
        print("\n" + "="*60)

# Global StretchMagTec calibration object
stretchmagtec_calibration = StretchMagTecCalibration()

# =============================================================================
# FT SENSOR THREAD
# =============================================================================
def forceFromSerialMessage(serialMessage, zeroRef=[0,0,0,0,0,0]):
    forceTorque = [0,0,0,0,0,0]
    forceTorque[0] = int.from_bytes(serialMessage[2:4], byteorder='little', signed=True)/100 - zeroRef[0]
    forceTorque[1] = int.from_bytes(serialMessage[4:6], byteorder='little', signed=True)/100 - zeroRef[1]
    forceTorque[2] = int.from_bytes(serialMessage[6:8], byteorder='little', signed=True)/100 - zeroRef[2]
    forceTorque[3] = int.from_bytes(serialMessage[8:10], byteorder='little', signed=True)/1000 - zeroRef[3]
    forceTorque[4] = int.from_bytes(serialMessage[10:12], byteorder='little', signed=True)/1000 - zeroRef[4]
    forceTorque[5] = int.from_bytes(serialMessage[12:14], byteorder='little', signed=True)/1000 - zeroRef[5]
    return [round(val, 3) for val in forceTorque]

def crcCheck(serialMessage):
    crc = int.from_bytes(serialMessage[14:16], byteorder='little', signed=False)
    crcCalc = libscrc.modbus(serialMessage[0:14])
    return crc == crcCalc

class FTSensorThread(threading.Thread):
    def __init__(self, port=FT_PORT, baudrate=FT_BAUDRATE):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        self.force_reading = [0]*6
        self.raw_force_reading = [0]*6  # Raw reading without compensation
        self.lock = threading.Lock()

    def run(self):
        try:
            ser_tmp = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=8, parity='N', stopbits=1, timeout=1)
            ser_tmp.write(bytearray([0xff]*50))
            ser_tmp.close()
            mm.BAUDRATE = self.baudrate
            mm.BYTESIZE = 8
            mm.PARITY = 'N'
            mm.STOPBITS = 1
            mm.TIMEOUT = 1
            ft300 = mm.Instrument(self.port, slaveaddress=9)
            ft300.close_port_after_each_call = True
            ft300.write_register(410, 0x0200)
            del ft300
            ser = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=8, parity='N', stopbits=1, timeout=1)
            STARTBYTES = bytes([0x20, 0x4e])
            ser.read_until(STARTBYTES)
            data = ser.read_until(STARTBYTES)
            dataArray = bytearray(data)
            dataArray = STARTBYTES + dataArray[:-2]
            if not crcCheck(dataArray):
                print("CRC ERROR on ZeroRef")
                return
            zeroRef = forceFromSerialMessage(dataArray)
            while self.running:
                data = ser.read_until(STARTBYTES)
                dataArray = bytearray(data)
                dataArray = STARTBYTES + dataArray[:-2]
                if not crcCheck(dataArray):
                    continue
                raw_force = forceFromSerialMessage(dataArray, zeroRef)
                ft_cleaned = [0 if abs(val) < FT_NOISE_THRESHOLD else val for val in raw_force]
                
                # Store both raw and compensated readings
                with self.lock:
                    self.raw_force_reading = ft_cleaned.copy()
                    self.force_reading = ft_calibration.compensate_force(ft_cleaned)
                ft_data_ready_event.set()
            ser.close()
        except Exception as e:
            print(f"FT Sensor thread error: {e}")

    def get_ft(self):
        """Return compensated force reading"""
        with self.lock:
            return self.force_reading.copy()
    
    def get_raw_ft(self):
        """Return raw force reading (without compensation)"""
        with self.lock:
            return self.raw_force_reading.copy()

# =============================================================================
# STRETCHMAGTEC SENSOR THREAD
# =============================================================================
stretchmagtec_data_lock = threading.Lock()
stretchmagtec_data = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
stretchmagtec_ready_event = threading.Event()
ft_data_ready_event = threading.Event()

# Per-press summary buffers
press_summary_sensors = []
press_summary_forces = []
press_summary_metadata = []

def wait_for_initial_calibration_complete(ft_calib, stretch_calib, timeout=30.0, poll_interval=0.25):
    """
    Block until the initial FT and StretchMagTec calibrations are completed.
    Raises RuntimeError if calibration does not finish within the timeout.
    """
    start_time = time.time()
    while True:
        ft_ready = True if not FT_INITIAL_CALIBRATION_ENABLED else getattr(ft_calib, 'is_calibrated', False)
        stretch_ready = True if not STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED else getattr(stretch_calib, 'is_calibrated', False)
        
        if ft_ready and stretch_ready:
            return
        
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            missing = []
            if not ft_ready:
                missing.append("FT sensor")
            if not stretch_ready:
                missing.append("StretchMagTec sensor")
            raise RuntimeError(f"Initial calibration timeout: {', '.join(missing)} not calibrated after {timeout:.1f}s.")
        
        time.sleep(poll_interval)

def parse_stretchmagtec_line(line):
    """
    Parse StretchMagTec sensor line data - supports multiple formats.
    Formats supported:
    1. Optimized: "DATA:1:x,y,z|2:x,y,z|..."
    2. Original: "S1: X=1234 Y=5678 Z=9012 | S2: X=2345 Y=6789 Z=0123 | ..."
    3. Simple: space-separated values
    """
    try:
        sensor_values = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        # Skip batch separators
        if line.strip() == "=== BATCH END ===":
            return None
        
        # Try new optimized format: "DATA:1:x,y,z|2:x,y,z|..."
        if line.startswith("DATA:"):
            data_part = line[5:]  # Remove "DATA:" prefix
            sensor_entries = data_part.split('|')
            
            for entry in sensor_entries:
                if ':' in entry:
                    sensor_id_str, values_str = entry.split(':', 1)
                    try:
                        sensor_id = int(sensor_id_str) - 1  # Convert to 0-based index
                        if 0 <= sensor_id < STRETCHMAGTEC_SENSORS:
                            # Remove status indicators like (OK) or (ERR)
                            if '(' in values_str:
                                values_str = values_str.split('(')[0]
                            
                            values = values_str.split(',')
                            if len(values) == 3:
                                sensor_values[sensor_id, 0] = float(values[0])  # X
                                sensor_values[sensor_id, 1] = float(values[1])  # Y
                                sensor_values[sensor_id, 2] = float(values[2])  # Z
                    except (ValueError, IndexError):
                        continue
            
            # Check if we got any valid data
            if np.any(sensor_values != 0):
                # Apply threshold filter
                sensor_values[(sensor_values >= -STRETCHMAGTEC_THRESHOLD) & (sensor_values <= STRETCHMAGTEC_THRESHOLD)] = 0
                return sensor_values
            else:
                return None
        
        # Try original format: "S1: X=1234 Y=5678 Z=9012 | S2: X=2345 Y=6789 Z=0123 | ..."
        elif ' | ' in line:
            sensor_parts = line.split(' | ')
            if len(sensor_parts) >= STRETCHMAGTEC_SENSORS:
                for i, sensor_part in enumerate(sensor_parts[:STRETCHMAGTEC_SENSORS]):
                    sensor_part = sensor_part.strip()
                    if ':' not in sensor_part:
                        continue
                        
                    values_part = sensor_part.split(':', 1)[1].strip()
                    
                    coords = {'X': 0, 'Y': 0, 'Z': 0}
                    for coord_pair in values_part.split():
                        if '=' in coord_pair:
                            coord, value = coord_pair.split('=', 1)
                            if coord in coords:
                                try:
                                    coords[coord] = float(value)
                                except ValueError:
                                    coords[coord] = 0
                    
                    sensor_values[i, 0] = coords['X']
                    sensor_values[i, 1] = coords['Y'] 
                    sensor_values[i, 2] = coords['Z']
        
        # Apply threshold filter
        sensor_values[(sensor_values >= -STRETCHMAGTEC_THRESHOLD) & (sensor_values <= STRETCHMAGTEC_THRESHOLD)] = 0
        return sensor_values
        
    except Exception as e:
        print(f"Error parsing StretchMagTec data: {e}")
        return None

class StretchMagTecSerialReader(threading.Thread):
    def __init__(self, port=STRETCHMAGTEC_PORT, baud=STRETCHMAGTEC_BAUD):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True
        self.ser = None

    def run(self):
        global stretchmagtec_data
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            while self.running:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    sensor_values = parse_stretchmagtec_line(line)
                    if sensor_values is not None:
                        with stretchmagtec_data_lock:
                            stretchmagtec_data[:, :] = sensor_values
                        stretchmagtec_ready_event.set()
        except Exception as e:
            print(f"StretchMagTec serial error: {e}")
        finally:
            if self.ser:
                self.ser.close()

def read_stretchmagtec_data():
    with stretchmagtec_data_lock:
        return stretchmagtec_data.copy()

# =============================================================================
# CONTINUOUS LOGGER THREAD
# =============================================================================
class ContinuousLoggerThread(threading.Thread):
    def __init__(self, robot, ft_sensor):
        super().__init__()
        self.robot = robot
        self.ft_sensor = ft_sensor
        self.running = True
        self.timestamps = []
        self.positions = []
        self.forces = []
        self.stretchmagtec = []
        self.labels = []
        self._current_label = "idle"
        self.lock = threading.Lock()

    def set_label(self, val):
        with self.lock:
            self._current_label = val

    def run(self):
        while self.running:
            loop_start = time.time()
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            cur_state = self.robot.getState()
            current_pos = cur_state.T[:3, 3]
            ft = self.ft_sensor.get_ft()
            sensors = read_stretchmagtec_data()
            
            # Apply StretchMagTec calibration compensation
            if sensors is not None:
                sensors = stretchmagtec_calibration.compensate_sensors(sensors)
            
            with self.lock:
                label = self._current_label
            self.timestamps.append(timestamp)
            self.positions.append(current_pos.tolist())
            self.forces.append(ft)
            self.stretchmagtec.append(sensors)
            self.labels.append(label)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, PERIOD - elapsed))

    def stop(self):
        self.running = False

# =============================================================================
# ROBOT MOVEMENT FUNCTIONS
# =============================================================================
def move_relative(r, dx, dy, dz, duration=MOVEMENT_DURATION):
    """Move robot relative to current position"""
    delta_transform = np.eye(4)
    delta_transform[:3, 3] = [dx, dy, dz]
    r.move("relative", delta_transform, duration)

# =============================================================================
# MAIN DATA COLLECTION
# =============================================================================
if __name__ == '__main__':
    stretchmagtec_ready_event.clear()
    ft_data_ready_event.clear()

    # Start sensor threads
    stretchmagtec_reader = StretchMagTecSerialReader()
    stretchmagtec_reader.daemon = True
    stretchmagtec_reader.start()
    
    ft_thread = FTSensorThread()
    ft_thread.daemon = True
    ft_thread.start()
    time.sleep(2)

    # Initialize robot
    print(f"Connecting to robot at {ROBOT_IP}...")
    r = franka.Robot_(ROBOT_IP, False, hand_franka=False, auto_init=True, speed_factor=ROBOT_SPEED_FACTOR)
    print("Robot connected successfully")
    
    # Initial calibrations (ALWAYS done at start)
    print("\n" + "="*70)
    print("INITIAL SENSOR CALIBRATION")
    print("="*70)
    
    # Temporarily enable calibration objects for initial calibration
    if STRETCHMAGTEC_INITIAL_CALIBRATION_ENABLED:
        stretchmagtec_calibration.enabled = True
        print("Waiting for StretchMagTec stream to stabilize...")
        if not stretchmagtec_ready_event.wait(timeout=STRETCHMAGTEC_STREAM_TIMEOUT):
            raise RuntimeError("StretchMagTec sensor did not start streaming in time for calibration.")
        print(f"StretchMagTec stream detected. Waiting {STRETCHMAGTEC_STREAM_STABILIZATION:.1f} seconds before calibration...")
        time.sleep(STRETCHMAGTEC_STREAM_STABILIZATION)
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

    try:
        wait_for_initial_calibration_complete(ft_calibration, stretchmagtec_calibration)
        print("Initial calibrations complete. Proceeding with data collection.\n")
    except RuntimeError as exc:
        print(str(exc))
        raise
    
    # Start continuous logger
    logger = ContinuousLoggerThread(r, ft_thread)
    logger.daemon = True
    logger.start()

    # Set "Z-down" orientation
    rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()

    try:
        # Determine which positions to test from config
        position_ids_to_test = get_positions_to_test()
        offsets_to_test = get_offsets_to_test()
        
        all_position_ids = sorted(MAIN_GRID_POSITIONS.keys())
        
        if len(position_ids_to_test) < len(all_position_ids) or len(offsets_to_test) < len(GRID_OFFSETS):
            print(f"\n{'⚠️ '*20}")
            print(f"SELECTIVE TESTING MODE")
            print(f"{'⚠️ '*20}")
            print(f"Testing {len(position_ids_to_test)} positions (of {len(all_position_ids)}): {position_ids_to_test}")
            print(f"Testing {len(offsets_to_test)} offsets (of {len(GRID_OFFSETS)}): {offsets_to_test}")
            print(f"Total test points: {len(position_ids_to_test)} × {len(offsets_to_test)} = {len(position_ids_to_test) * len(offsets_to_test)}")
            print(f"⚠️  Set DEBUG_MODE = False in config.py for full collection")
            print(f"{'⚠️ '*20}\n")
        else:
            position_ids_to_test = all_position_ids
            offsets_to_test = list(GRID_OFFSETS.keys())
            print(f"\n🚀 FULL COLLECTION MODE")
            print(f"   Testing {len(position_ids_to_test)} positions x {len(offsets_to_test)} offsets = {len(position_ids_to_test) * len(offsets_to_test)} test points\n")
        
        # Iterate through selected positions
        total_positions = len(position_ids_to_test)
        position_count = 0
        
        for position_id in position_ids_to_test:
            position_count += 1
            row = position_id // 10
            col = position_id % 10
            base_position = MAIN_GRID_POSITIONS[position_id]
            
            print(f"\n{'='*70}")
            print(f"MAIN POSITION {position_count}/{total_positions}: Position {position_id} (Row {row}, Col {col})")
            print(f"{'='*70}")
            
            # Iterate through selected offsets
            offset_count = 0
            total_offsets = len(offsets_to_test)
            
            for offset_key in offsets_to_test:
                offset_count += 1
                print(f"\n--- Offset {offset_count}/{total_offsets}: {offset_key.upper()} ---")
                print(f"\n{'='*70}")
                print(f"POSITION {position_id} ({row},{col}) - OFFSET: {offset_key}")
                print(f"{'='*70}")
                
                # Get position coordinates with offset
                desired_position = get_position_with_offset(base_position, offset_key)
                
                # Create desired pose
                des_pos_fingertip_setup = np.eye(4)
                des_pos_fingertip_setup[:3, :3] = rotation_matrix
                des_pos_fingertip_setup[:3, 3] = desired_position
                
                # Move to position
                r.move("absolute", des_pos_fingertip_setup, ABSOLUTE_MOVEMENT_DURATION)
                print(f"Moved to position {position_id} ({row},{col}) - {offset_key}")
                print(f"Coordinates: [{desired_position[0]:.6f}, {desired_position[1]:.6f}, {desired_position[2]:.6f}]")
                
                # Wait for stabilization
                time.sleep(1)
                
                # Perform press cycles at this position
                for press_num in range(NUMBER_OF_PRESSES):
                    press_id = PRESS_IDS[press_num]
                    logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_start")
                    
                    print(f"Starting press cycle {press_num + 1}/{NUMBER_OF_PRESSES} (Press ID: {press_id})")
                    
                    # Calibrate before press (optional, per-position calibration)
                    if FT_PER_POSITION_CALIBRATION_ENABLED:
                        ft_calibration.measure_offset(ft_thread, f"pos_{position_id}_{offset_key}_pre-press_{press_id}")
                    if STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED:
                        stretchmagtec_calibration.measure_offsets(stretchmagtec_reader, f"pos_{position_id}_{offset_key}_pre-press_{press_id}")
                    
                    # Perform press steps
                    for step_num in range(STEPS_PER_PRESS):
                        logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_step_{step_num+1}")
                        move_relative(r, 0, 0, DZ_PRESS)
                        print(f"Press {press_id}, Step {step_num + 1}/{STEPS_PER_PRESS} - Moving down by {abs(DZ_PRESS)}m")
                        time.sleep(PRESS_DELAY)

                    # Capture press snapshot before lifting
                    with stretchmagtec_data_lock:
                        sensor_snapshot = stretchmagtec_data.copy()
                    ft_snapshot = ft_thread.get_ft()
                    summary_meta = {
                        "position_id": int(position_id),
                        "offset_key": offset_key,
                        "press_id": press_id,
                        "press_index": press_num,
                        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
                        "stretch_level": float(getattr(config_module, "CURRENT_STRETCH_VALUE", 0.0)),
                        "stretch_label": str(getattr(config_module, "CURRENT_STRETCH_LABEL", "")),
                        "press_profile": str(getattr(config_module, "CURRENT_PRESS_PROFILE", "")),
                        "press_depth_m": float(abs(DZ_PRESS) * STEPS_PER_PRESS),
                        "steps_per_press": int(STEPS_PER_PRESS),
                    }
                    press_summary_sensors.append(sensor_snapshot)
                    press_summary_forces.append(np.array(ft_snapshot, dtype=float))
                    press_summary_metadata.append(summary_meta)
                    
                    # Lift back up
                    logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_lift")
                    lift_distance = DZ_LIFT * STEPS_PER_PRESS
                    move_relative(r, 0, 0, lift_distance, MOVEMENT_DURATION * STEPS_PER_PRESS)
                    print(f"Press {press_id} complete - Lifting up by {lift_distance}m")
                    time.sleep(LIFT_DELAY)
                    
                    # Calibrate after press (optional, per-position calibration)
                    if FT_PER_POSITION_CALIBRATION_ENABLED:
                        ft_calibration.measure_offset(ft_thread, f"pos_{position_id}_{offset_key}_post-press_{press_id}")
                    if STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED:
                        stretchmagtec_calibration.measure_offsets(stretchmagtec_reader, f"pos_{position_id}_{offset_key}_post-press_{press_id}")
                
                print(f"Completed position {position_id} ({row},{col}) - {offset_key}")
                time.sleep(1)

        logger.set_label("final_position")
        print(f"\n{'='*70}")
        print("ALL POSITIONS COMPLETE")
        print(f"{'='*70}")
        time.sleep(1)

        center_position = None
        if position_ids_to_test and 'center' in GRID_OFFSETS:
            center_id = position_ids_to_test[0]
            center_position = get_position_with_offset(MAIN_GRID_POSITIONS[center_id], 'center')
        elif position_ids_to_test:
            center_position = MAIN_GRID_POSITIONS.get(position_ids_to_test[0], None)
        elif MAIN_GRID_POSITIONS:
            first_key = sorted(MAIN_GRID_POSITIONS.keys())[0]
            center_position = MAIN_GRID_POSITIONS[first_key]

        if center_position is not None:
            center_pose = np.eye(4)
            center_pose[:3, :3] = rotation_matrix
            center_pose[:3, 3] = center_position
            print("\nReturning robot to central position before completing the stretch run...")
            r.move("absolute", center_pose, ABSOLUTE_MOVEMENT_DURATION)
            time.sleep(1)

    finally:
        logger.stop()
        logger.join()
        ft_thread.running = False
        stretchmagtec_reader.running = False
        ft_thread.join()
        stretchmagtec_reader.join()
        
        # Generate filename with sensor name and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_suffix = "_debug" if DEBUG_MODE else ""
        output_prefix = getattr(config_module, "CURRENT_OUTPUT_PREFIX", None)
        if output_prefix:
            filename = get_data_path(f"{output_prefix}.h5")
        else:
            filename = get_data_path(f"{SENSOR_NAME}_data_{GRID_ROWS}x{GRID_COLS}_9pt{debug_suffix}_{timestamp}.h5")
        print(f"Saving data to: {filename}")
        
        forces_array = np.array(logger.forces)
        stretchmagtec_array = np.array(logger.stretchmagtec)
        positions_arr = np.array(logger.positions)
        timestamps_arr = np.array(logger.timestamps, dtype='S26')
        labels_arr = np.array(logger.labels, dtype='S64')
        
        with h5py.File(filename, "w") as f:
            # Save continuous data
            f.create_dataset("forces", data=forces_array)
            f.create_dataset("stretchmagtec", data=stretchmagtec_array)
            f.create_dataset("positions", data=positions_arr)
            f.create_dataset("timestamps", data=timestamps_arr)
            f.create_dataset("labels", data=labels_arr)
            
            # Save file attributes
            f.attrs["sensor_name"] = SENSOR_NAME
            f.attrs["robot_ip"] = ROBOT_IP
            f.attrs["grid_rows"] = GRID_ROWS
            f.attrs["grid_cols"] = GRID_COLS
            f.attrs["grid_dx"] = GRID_DX
            f.attrs["grid_dy"] = GRID_DY
            f.attrs["reference_position"] = REFERENCE_POSITION
            f.attrs["grid_offsets"] = str(GRID_OFFSETS)
            f.attrs["number_of_presses"] = NUMBER_OF_PRESSES
            f.attrs["steps_per_press"] = STEPS_PER_PRESS
            f.attrs["dz_press"] = DZ_PRESS
            f.attrs["dz_lift"] = DZ_LIFT
            f.attrs["ft_calibration_enabled"] = FT_CALIBRATION_ENABLED
            f.attrs["stretchmagtec_calibration_enabled"] = STRETCHMAGTEC_CALIBRATION_ENABLED
            f.attrs["target_freq"] = TARGET_FREQ
            if hasattr(config_module, "CURRENT_STRETCH_VALUE"):
                f.attrs["stretch_level"] = float(getattr(config_module, "CURRENT_STRETCH_VALUE"))
            if hasattr(config_module, "CURRENT_STRETCH_LABEL"):
                f.attrs["stretch_label"] = str(getattr(config_module, "CURRENT_STRETCH_LABEL"))
            if hasattr(config_module, "CURRENT_PRESS_PROFILE"):
                f.attrs["press_profile"] = str(getattr(config_module, "CURRENT_PRESS_PROFILE"))
            if hasattr(config_module, "CURRENT_PRESS_SETTINGS"):
                for key, value in getattr(config_module, "CURRENT_PRESS_SETTINGS").items():
                    f.attrs[f"press_{key}"] = value
            
            if press_summary_sensors:
                f.create_dataset(
                    "press_summaries/sensors",
                    data=np.array(press_summary_sensors, dtype=float)
                )
                f.create_dataset(
                    "press_summaries/forces",
                    data=np.array(press_summary_forces, dtype=float)
                )
                summary_strings = [json.dumps(meta) for meta in press_summary_metadata]
                str_dtype = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(
                    "press_summaries/metadata",
                    data=np.array(summary_strings, dtype=str_dtype)
                )

            f.attrs["description"] = f"MagTecK_PM skin test - {GRID_ROWS}x{GRID_COLS} grid with 9-point offsets"
            f.flush()
        
        # Print calibration summary
        ft_calibration.print_calibration_summary()
        stretchmagtec_calibration.print_calibration_summary()
        
        print(f"\nData collection complete. Log written to {filename}.")
        
        press_summary_sensors.clear()
        press_summary_forces.clear()
        press_summary_metadata.clear()
        config_module.LAST_OUTPUT_FILE = str(filename)

