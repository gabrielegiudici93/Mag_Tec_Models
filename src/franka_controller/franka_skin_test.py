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
import re
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
            time.sleep(0.1)  # Brief pause after reset
            
            mm.BAUDRATE = self.baudrate
            mm.BYTESIZE = 8
            mm.PARITY = 'N'
            mm.STOPBITS = 1
            mm.TIMEOUT = 1
            
            # Retry logic for minimalmodbus initialization (checksum errors can occur)
            max_retries = 3
            retry_delay = 0.5
            ft300 = None
            for attempt in range(max_retries):
                try:
                    ft300 = mm.Instrument(self.port, slaveaddress=9)
                    ft300.close_port_after_each_call = True
                    ft300.write_register(410, 0x0200)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[FT Thread] MinimalModbus initialization attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"[FT Thread] Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        if ft300:
                            try:
                                del ft300
                            except:
                                pass
                    else:
                        # Last attempt failed - log but continue (sensor might still work)
                        print(f"[FT Thread] ⚠️  MinimalModbus initialization failed after {max_retries} attempts: {e}")
                        print(f"[FT Thread] Continuing anyway - sensor may still work in streaming mode")
            if ft300:
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
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self.running:
                try:
                    data = ser.read_until(STARTBYTES)
                    dataArray = bytearray(data)
                    dataArray = STARTBYTES + dataArray[:-2]
                    if not crcCheck(dataArray):
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"[FT Thread] ⚠️  Too many CRC errors ({consecutive_errors}), sensor may be disconnected")
                        continue
                    
                    # Reset error counter on successful read
                    consecutive_errors = 0
                    
                    raw_force = forceFromSerialMessage(dataArray, zeroRef)
                    
                    # Store raw values (not filtered) for GUI display
                    # Filtering is only for noise reduction in plots, but GUI should show actual values
                    with self.lock:
                        self.raw_force_reading = raw_force.copy()  # Store raw (unfiltered) for GUI
                        # Apply noise threshold for compensated reading (used in logging)
                        ft_cleaned = [0 if abs(val) < FT_NOISE_THRESHOLD else val for val in raw_force]
                        self.force_reading = ft_calibration.compensate_force(ft_cleaned)
                    ft_data_ready_event.set()
                    
                except serial.SerialTimeoutException:
                    # Timeout is normal if no data available, just continue
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors % 10 == 0:  # Print every 10th error to avoid spam
                        print(f"[FT Thread] ⚠️  Error reading FT data (error #{consecutive_errors}): {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[FT Thread] ⚠️  Too many consecutive errors ({consecutive_errors}), stopping FT thread")
                        break
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                    
            ser.close()
        except Exception as e:
            print(f"[FT Thread] Fatal error in FT sensor thread: {e}")
            import traceback
            traceback.print_exc()
            # Mark as error so GUI can show it
            with self.lock:
                self.raw_force_reading = [float('nan')] * 6
                self.force_reading = [float('nan')] * 6

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
# Global variables to expose to GUI adapter
ft_thread = None
stretchmagtec_reader = None

def main():
    global ft_thread, stretchmagtec_reader
    
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
    
    # Determine target position for calibration (first position to test)
    position_ids_to_test = get_positions_to_test()
    if position_ids_to_test:
        first_position_id = position_ids_to_test[0]
        base_position = MAIN_GRID_POSITIONS[first_position_id]
        print(f"\nBase position (from MAIN_GRID_POSITIONS[{first_position_id}]): [{base_position[0]:.6f}, {base_position[1]:.6f}, {base_position[2]:.6f}]")
        # Get center offset position - center has offset [0,0,0], so base_position is already the center
        offsets_to_test = get_offsets_to_test()
        center_offset = 'center' if 'center' in offsets_to_test else offsets_to_test[0] if offsets_to_test else 'center'
        # For center, target_position should be the same as base_position (offset is [0,0,0])
        target_position = get_position_with_offset(base_position, center_offset)
        print(f"Target position (center offset applied): [{target_position[0]:.6f}, {target_position[1]:.6f}, {target_position[2]:.6f}]")
        
        # Set "Z-down" orientation
        rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
        
        # Move to calibration position (5mm above target)
        calibration_position = np.array(target_position).copy()
        calibration_position[2] += 0.005  # Lift 5mm above target
        
        calibration_pose = np.eye(4)
        calibration_pose[:3, :3] = rotation_matrix
        calibration_pose[:3, 3] = calibration_position
        
        print(f"\nMoving to calibration position: 5mm above target position {first_position_id} ({center_offset})")
        print(f"Calibration coordinates: [{calibration_position[0]:.6f}, {calibration_position[1]:.6f}, {calibration_position[2]:.6f}]")
        r.move("absolute", calibration_pose, ABSOLUTE_MOVEMENT_DURATION)
        time.sleep(1.0)  # Wait for stabilization
    else:
        # Fallback: use current position
        print("⚠️  No positions to test - calibration will be done at current position")
        rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
    
    # Initial calibrations (ALWAYS done at start, at 5mm above target)
    print("\n" + "="*70)
    print("INITIAL SENSOR CALIBRATION (at 5mm above target position)")
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
        print("Initial calibrations complete.")
    except RuntimeError as exc:
        print(str(exc))
        raise
    
    # Move to center position after calibration
    # Use base_position directly (it's already the center position, no offset needed)
    if position_ids_to_test:
        center_position = base_position  # Base position is already the center (no offset applied)
        center_pose = np.eye(4)
        center_pose[:3, :3] = rotation_matrix
        center_pose[:3, 3] = center_position
        
        print(f"\nMoving to center position: [{center_position[0]:.6f}, {center_position[1]:.6f}, {center_position[2]:.6f}]")
        r.move("absolute", center_pose, ABSOLUTE_MOVEMENT_DURATION)
        time.sleep(1.0)  # Wait for stabilization
        print("Proceeding with data collection.\n")
    else:
        print("Proceeding with data collection.\n")
    
    # Start continuous logger
    logger = ContinuousLoggerThread(r, ft_thread)
    logger.daemon = True
    logger.start()

    try:
        # Determine which positions to test from config (already done above, but get offsets)
        if not position_ids_to_test:
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
                
                # Before moving to new position: lift 5mm to avoid sliding on surface
                # (Only lift if not the first offset, i.e., when moving from one offset to another)
                if offset_count > 1:  # Not the first offset (center)
                    print(f"Lifting 5mm before moving to {offset_key}...")
                    move_relative(r, 0, 0, 0.005, duration=ABSOLUTE_MOVEMENT_DURATION)
                    time.sleep(0.5)
                
                # Create desired pose
                des_pos_fingertip_setup = np.eye(4)
                des_pos_fingertip_setup[:3, :3] = rotation_matrix
                des_pos_fingertip_setup[:3, 3] = desired_position
                
                # If we lifted, move to position at lifted height (add 5mm to Z)
                if offset_count > 1:
                    des_pos_fingertip_setup[2, 3] += 0.005
                
                # Move to position
                r.move("absolute", des_pos_fingertip_setup, ABSOLUTE_MOVEMENT_DURATION)
                print(f"Moved to position {position_id} ({row},{col}) - {offset_key}")
                
                # Lower 5mm to target position (if we lifted)
                if offset_count > 1:
                    print(f"Lowering 5mm to target position...")
                    move_relative(r, 0, 0, -0.005, duration=ABSOLUTE_MOVEMENT_DURATION)
                    time.sleep(0.5)
                
                print(f"Final coordinates: [{desired_position[0]:.6f}, {desired_position[1]:.6f}, {desired_position[2]:.6f}]")
                
                # Wait for stabilization
                time.sleep(1)
                
                # For force-controlled pressing: get the ORIGINAL starting Z position ONCE before all presses
                # This ensures all presses start from and return to the same position
                # INCREASE by 5mm to avoid starting with force already at 0.8N
                reference_initial_z = None
                if getattr(config_module, "FORCE_CONTROLLED_PRESS", False):
                    reference_state = r.getState()
                    reference_initial_z = reference_state.T[2, 3] + 0.005  # Add 5mm (0.005m) to starting position
                    print(f"Reference Z position for all presses: {reference_initial_z:.6f}m (increased by 5mm from {reference_state.T[2, 3]:.6f}m)")
                    # Move to the increased position
                    current_state = r.getState()
                    current_z = current_state.T[2, 3]
                    z_diff = reference_initial_z - current_z
                    if abs(z_diff) > 0.0001:  # More than 0.1mm difference
                        print(f"  Moving up 5mm to starting position...")
                        move_relative(r, 0, 0, z_diff, MOVEMENT_DURATION)
                        time.sleep(0.5)
                
                # Perform press cycles at this position
                for press_num in range(NUMBER_OF_PRESSES):
                    press_id = PRESS_IDS[press_num]
                    logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_start")
                    
                    # Skip first press (discard it)
                    skip_first_press = (press_num == 0)
                    if skip_first_press:
                        print(f"Starting press cycle {press_num + 1}/{NUMBER_OF_PRESSES} (Press ID: {press_id}) - DISCARDING (first press)")
                    else:
                        print(f"Starting press cycle {press_num + 1}/{NUMBER_OF_PRESSES} (Press ID: {press_id})")
                    
                    # Calibrate before press (optional, per-position calibration)
                    if FT_PER_POSITION_CALIBRATION_ENABLED:
                        ft_calibration.measure_offset(ft_thread, f"pos_{position_id}_{offset_key}_pre-press_{press_id}")
                    if STRETCHMAGTEC_PER_POSITION_CALIBRATION_ENABLED:
                        stretchmagtec_calibration.measure_offsets(stretchmagtec_reader, f"pos_{position_id}_{offset_key}_pre-press_{press_id}")
                    
                    # Perform force-controlled press: stop at 1.0N to 3.0N in 0.1N steps for 1s each
                    # OR position-controlled press (if FORCE_CONTROLLED_PRESS is False)
                    if getattr(config_module, "FORCE_CONTROLLED_PRESS", False):
                        # Force-controlled pressing: use config parameters
                        force_min = getattr(config_module, "FORCE_MIN", 1.0)
                        force_max = getattr(config_module, "FORCE_MAX", 3.0)
                        force_step = getattr(config_module, "FORCE_STEP_SIZE", 0.1)
                        target_forces = np.arange(force_min, force_max + force_step, force_step).tolist()
                        force_tolerance = getattr(config_module, "FORCE_TOLERANCE", 0.01)
                        data_collection_duration = getattr(config_module, "FORCE_STEP_DELAY", 1.0)  # Stay at each force level
                        
                        # CRITICAL: Return to reference position BEFORE starting this press
                        # This ensures all presses start from the same position
                        # NO LIFTING - just return to reference position
                        if reference_initial_z is not None:
                            current_state = r.getState()
                            current_z = current_state.T[2, 3]
                            z_diff = current_z - reference_initial_z
                            if abs(z_diff) > 0.0001:  # More than 0.1mm difference
                                print(f"  Returning to reference position (moving {z_diff*1000:.2f}mm)...")
                                move_relative(r, 0, 0, -z_diff, MOVEMENT_DURATION)
                                time.sleep(0.3)
                        
                        # Wait a moment for stabilization
                        time.sleep(0.2)
                        
                        # Get current Z position for tracking indentation (should be at reference_initial_z)
                        start_state = r.getState()
                        start_z = start_state.T[2, 3]
                        max_indentation = 0.010  # 10mm safety limit (5mm initial lift + 5mm safety margin)
                        
                        print(f"Press {press_id} - Force-controlled pressing: {target_forces}N")
                        print(f"  Starting from Z position: {start_z:.6f}m")
                        
                        # Flag to track if this is the first movement (for sequence_start)
                        first_movement = True
                        
                        for force_step, target_force in enumerate(target_forces):
                            logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_force_{target_force:.1f}N")
                            print(f"  Target force: {target_force:.1f} N (step {force_step + 1}/{len(target_forces)})")
                            
                            # Control loop: adjust position to reach target force
                            max_iterations = 500  # Safety limit
                            iteration = 0
                            
                            while iteration < max_iterations:
                                # Check current position for safety
                                current_state = r.getState()
                                current_z = current_state.T[2, 3]
                                current_indentation = abs(start_z - current_z)
                                
                                # Safety check: stop if maximum indentation exceeded
                                if current_indentation >= max_indentation:
                                    print(f"    ⚠️  Safety stop: Maximum indentation ({max_indentation*1000:.1f}mm) reached")
                                    break
                                
                                # Read force (single reading for speed - sensor thread updates continuously)
                                current_ft = ft_thread.get_ft()
                                current_fz_abs = abs(current_ft[2])
                                
                                # Determine movement direction based on current vs target force (using abs values)
                                # IMPORTANT: If force exceeds target, don't go back - just stop and proceed
                                if current_fz_abs >= target_force - force_tolerance:
                                    # Force is at or above target - stop here and proceed (don't go back)
                                    print(f"    Target reached (or exceeded): {current_fz_abs:.3f} N (indentation: {current_indentation*1000:.2f}mm)")
                                    break
                                elif current_fz_abs < target_force - force_tolerance:
                                    # CRITICAL: Mark the start of data collection IMMEDIATELY before the FIRST movement
                                    # This ensures the sequence starts exactly when the robot begins pressing
                                    if first_movement:
                                        # Print FT sensor values at the beginning of sequence
                                        current_ft = ft_thread.get_ft()
                                        print(f"  📊 FT sensor values at sequence start: Fx={current_ft[0]:.4f}N, Fy={current_ft[1]:.4f}N, Fz={current_ft[2]:.4f}N, "
                                              f"Tx={current_ft[3]:.4f}Nm, Ty={current_ft[4]:.4f}Nm, Tz={current_ft[5]:.4f}Nm")
                                        logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_sequence_start")
                                        first_movement = False
                                    
                                    # Force is too low - press down to increase force
                                    # This is the actual movement - sequence_start is set just before this
                                    move_relative(r, 0, 0, -0.0001, duration=0.05)  # Move down, faster
                                    time.sleep(0.05)  # Reduced wait time
                                
                                iteration += 1
                            
                            if iteration >= max_iterations:
                                print(f"    ⚠️  Warning: Max iterations reached for {target_force:.1f}N target")
                            
                            # Stay at this force level for data collection (1 second)
                            print(f"    Collecting data at {target_force:.1f}N for {data_collection_duration:.1f}s...")
                            time.sleep(data_collection_duration)
                            
                            # CRITICAL: Mark sequence_end IMMEDIATELY after the wait at the LAST target force
                            # This ensures the sequence ends exactly when data collection is complete, before any return movement
                            if force_step == len(target_forces) - 1:  # Last target force
                                logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_sequence_end")
                                print(f"  Sequence complete - reached {target_forces[-1]:.1f}N")
                                # CRITICAL: Wait a tiny bit to ensure the logger captures at least one sample with sequence_end label
                                time.sleep(0.01)  # 10ms to ensure one sample is logged with sequence_end
                                # Then immediately change label to prevent logging more samples with sequence_end
                                logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_return")
                        
                        print(f"  Force-controlled press complete - reached {target_forces[-1]:.1f}N")
                        
                        # For force-controlled pressing: return to REFERENCE initial Z position
                        # This ensures the next press starts from the same reference position
                        if reference_initial_z is not None:
                            current_state = r.getState()
                            current_z = current_state.T[2, 3]
                            z_diff = current_z - reference_initial_z  # How much we need to move to return to reference
                            if abs(z_diff) > 0.0001:  # More than 0.1mm difference
                                logger.set_label(f"pos_{position_id}_{offset_key}_press_{press_id}_lift")
                                print(f"Press {press_id} complete - Returning to reference position (moving {z_diff*1000:.2f}mm)")
                                move_relative(r, 0, 0, -z_diff, MOVEMENT_DURATION)  # Move back to reference_initial_z
                                time.sleep(LIFT_DELAY)
                            else:
                                print(f"Press {press_id} complete - Already at reference position")
                                time.sleep(0.2)
                    else:
                        # Original position-controlled pressing
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
        
        # Extract individual press sequences using sequence_start and sequence_end labels
        # Each press is saved as a separate entry in the HDF5 file
        # IMPORTANT: We need to find the FIRST occurrence of _sequence_start for each press,
        # not all occurrences (since the label persists until changed)
        press_sequences = []
        current_sequence_start = None
        last_sequence_start_label = None  # Track the label to detect new sequences
        
        for idx, label in enumerate(labels_arr):
            label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
            
            if '_sequence_start' in label_str:
                # Check if this is a NEW sequence (different press_id) or continuation of same label
                if current_sequence_start is None or label_str != last_sequence_start_label:
                    # This is a new sequence start
                    if current_sequence_start is not None:
                        # Save previous sequence if it exists (shouldn't happen normally, but handle it)
                        press_sequences.append({
                            'start_idx': current_sequence_start,
                            'end_idx': idx - 1,
                            'label': labels_arr[current_sequence_start].decode('utf-8') if isinstance(labels_arr[current_sequence_start], bytes) else str(labels_arr[current_sequence_start])
                        })
                    current_sequence_start = idx
                    last_sequence_start_label = label_str
            elif '_sequence_end' in label_str and current_sequence_start is not None:
                # End of current press sequence
                # CRITICAL: end_idx is the index of the sample WITH the sequence_end label
                # We include this sample (inclusive), so we use end_idx: end_idx+1 in slicing
                press_sequences.append({
                    'start_idx': current_sequence_start,
                    'end_idx': idx,  # This is the index of the sample with sequence_end label
                    'label': labels_arr[current_sequence_start].decode('utf-8') if isinstance(labels_arr[current_sequence_start], bytes) else str(labels_arr[current_sequence_start])
                })
                current_sequence_start = None
                last_sequence_start_label = None
        
        # If there's a sequence that didn't end, include it
        if current_sequence_start is not None:
            press_sequences.append({
                'start_idx': current_sequence_start,
                'end_idx': len(labels_arr) - 1,
                'label': labels_arr[current_sequence_start].decode('utf-8') if isinstance(labels_arr[current_sequence_start], bytes) else str(labels_arr[current_sequence_start])
            })
        
        # Filter out first press (discard it)
        # First press has press_id = PRESS_IDS[0] (usually 'A')
        filtered_sequences = []
        for seq in press_sequences:
            label_str = seq['label']
            # Extract press_id from label (e.g., "pos_32_center_press_A_sequence_start" -> "A")
            match = re.search(r'press_([A-Za-z0-9_]+)_sequence', label_str)
            if match:
                press_id = match.group(1)
                # Skip first press (PRESS_IDS[0])
                if press_id != PRESS_IDS[0]:
                    filtered_sequences.append(seq)
            else:
                # If we can't identify the press, keep it (shouldn't happen)
                filtered_sequences.append(seq)
        
        print(f"Found {len(press_sequences)} press sequences, keeping {len(filtered_sequences)} (discarded first press)")
        
        with h5py.File(filename, "w") as f:
            # Save continuous data (for backward compatibility)
            f.create_dataset("forces", data=forces_array)
            f.create_dataset("stretchmagtec", data=stretchmagtec_array)
            f.create_dataset("positions", data=positions_arr)
            f.create_dataset("timestamps", data=timestamps_arr)
            f.create_dataset("labels", data=labels_arr)
            
            # Save each press as a separate entry in presses group
            if filtered_sequences:
                presses_group = f.create_group("presses")
                for seq_idx, seq in enumerate(filtered_sequences):
                    start_idx = seq['start_idx']
                    # end_idx is the index of the sample WITH the sequence_end label
                    # We want to include this sample, so we use end_idx + 1 for slicing (exclusive end)
                    end_idx = seq['end_idx'] + 1  # +1 because Python slicing is exclusive of end
                    
                    press_group = presses_group.create_group(f"press_{seq_idx:03d}")
                    press_group.create_dataset("forces", data=forces_array[start_idx:end_idx])
                    press_group.create_dataset("stretchmagtec", data=stretchmagtec_array[start_idx:end_idx])
                    press_group.create_dataset("positions", data=positions_arr[start_idx:end_idx])
                    
                    # CRITICAL: Normalize timestamps to start from 0 for each sequence
                    # Convert timestamps to relative time (seconds since sequence start)
                    seq_timestamps = timestamps_arr[start_idx:end_idx]
                    if len(seq_timestamps) > 0:
                        # Convert first timestamp to reference
                        first_ts_str = seq_timestamps[0].decode('utf-8') if isinstance(seq_timestamps[0], bytes) else str(seq_timestamps[0])
                        try:
                            first_ts = datetime.fromisoformat(first_ts_str.replace('Z', '+00:00')) if 'T' in first_ts_str else datetime.fromisoformat(first_ts_str)
                            # Create relative timestamps (seconds since sequence start)
                            relative_times = []
                            for ts_bytes in seq_timestamps:
                                ts_str = ts_bytes.decode('utf-8') if isinstance(ts_bytes, bytes) else str(ts_bytes)
                                try:
                                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if 'T' in ts_str else datetime.fromisoformat(ts_str)
                                    relative_seconds = (ts - first_ts).total_seconds()
                                    relative_times.append(relative_seconds)
                                except:
                                    # Fallback: use index-based time
                                    relative_times.append(len(relative_times) / 100.0)
                            
                            # Save as numeric array (seconds, starting from 0)
                            press_group.create_dataset("timestamps", data=np.array(relative_times, dtype=float))
                        except:
                            # Fallback: use index-based time (100 Hz sampling)
                            relative_times = np.arange(len(seq_timestamps)) / 100.0
                            press_group.create_dataset("timestamps", data=relative_times)
                    else:
                        press_group.create_dataset("timestamps", data=np.array([], dtype=float))
                    
                    press_group.create_dataset("labels", data=labels_arr[start_idx:end_idx])
                    
                    # Calculate and save indentation (relative Z change from initial position)
                    seq_positions = positions_arr[start_idx:end_idx]
                    if len(seq_positions) > 0:
                        initial_z = seq_positions[0][2]  # Z coordinate of first sample
                        indentation = seq_positions[:, 2] - initial_z  # Negative values = indentation (pressing down)
                        press_group.create_dataset("indentation", data=indentation)
                        press_group.attrs["initial_z"] = float(initial_z)
                        press_group.attrs["max_indentation"] = float(np.min(indentation))  # Most negative = deepest press
                    
                    # Store metadata
                    press_group.attrs["label"] = seq['label']
                    press_group.attrs["start_idx"] = start_idx
                    press_group.attrs["end_idx"] = end_idx
                    press_group.attrs["num_samples"] = end_idx - start_idx
                    
                    # Store stretch level (from file attributes or config)
                    if hasattr(config_module, "CURRENT_STRETCH_VALUE"):
                        press_group.attrs["stretch_level"] = float(getattr(config_module, "CURRENT_STRETCH_VALUE"))
                    elif "stretch_level" in f.attrs:
                        press_group.attrs["stretch_level"] = float(f.attrs["stretch_level"])
                    else:
                        press_group.attrs["stretch_level"] = np.nan
                    
                    if hasattr(config_module, "CURRENT_STRETCH_LABEL"):
                        press_group.attrs["stretch_label"] = str(getattr(config_module, "CURRENT_STRETCH_LABEL"))
                    elif "stretch_label" in f.attrs:
                        press_group.attrs["stretch_label"] = str(f.attrs["stretch_label"])
                    else:
                        press_group.attrs["stretch_label"] = "unknown"
            
            # Save file attributes
            f.attrs["sensor_name"] = SENSOR_NAME
            f.attrs["robot_ip"] = ROBOT_IP
            f.attrs["grid_rows"] = GRID_ROWS
            f.attrs["grid_cols"] = GRID_COLS
            f.attrs["grid_dx"] = GRID_DX
            f.attrs["grid_dy"] = GRID_DY
            # Ensure reference_position is saved as numpy array (matching stable format)
            f.attrs["reference_position"] = np.array(REFERENCE_POSITION, dtype=np.float64)
            f.attrs["grid_offsets"] = str(GRID_OFFSETS)
            f.attrs["number_of_presses"] = NUMBER_OF_PRESSES
            f.attrs["steps_per_press"] = STEPS_PER_PRESS
            f.attrs["dz_press"] = DZ_PRESS
            f.attrs["dz_lift"] = DZ_LIFT
            # Save boolean attributes as numpy bool_ (matching stable format)
            f.attrs["ft_calibration_enabled"] = np.bool_(FT_CALIBRATION_ENABLED)
            f.attrs["stretchmagtec_calibration_enabled"] = np.bool_(STRETCHMAGTEC_CALIBRATION_ENABLED)
            f.attrs["target_freq"] = TARGET_FREQ
            if hasattr(config_module, "CURRENT_STRETCH_VALUE"):
                f.attrs["stretch_level"] = float(getattr(config_module, "CURRENT_STRETCH_VALUE"))
            if hasattr(config_module, "CURRENT_STRETCH_LABEL"):
                f.attrs["stretch_label"] = str(getattr(config_module, "CURRENT_STRETCH_LABEL"))
            if hasattr(config_module, "CURRENT_PRESS_PROFILE"):
                f.attrs["press_profile"] = str(getattr(config_module, "CURRENT_PRESS_PROFILE"))
            if hasattr(config_module, "CURRENT_PRESS_SETTINGS"):
                for key, value in getattr(config_module, "CURRENT_PRESS_SETTINGS").items():
                    # Convert value to HDF5-compatible type
                    if value is None:
                        # Skip None values or convert to empty string
                        continue
                    elif isinstance(value, (list, dict)):
                        # Convert complex types to string
                        f.attrs[f"press_{key}"] = str(value)
                    elif isinstance(value, bool):
                        # Convert bool to int (HDF5 doesn't support bool natively)
                        f.attrs[f"press_{key}"] = int(value)
                    elif isinstance(value, (int, float, str)):
                        f.attrs[f"press_{key}"] = value
                    else:
                        # Fallback: convert to string
                        f.attrs[f"press_{key}"] = str(value)
            
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


if __name__ == '__main__':
    main()
