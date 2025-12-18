#!/usr/bin/env python3
"""
Real-Time 10-Point Multi-Point Predictor with GUI

This script provides real-time prediction for 10-point multi-point configuration:
1. Contact location (10 locations: '1' through '10')
2. Force values in Newtons from StretchMagTec sensors

The GUI displays:
- Real-time sensor readings (15 sensors in 3x5 grid)
- Predicted contact location with top 3 probabilities
- FT sensor readings (ground truth)
- StretchMagTec sensor raw values
- Predicted forces from StretchMagTec sensors
- Real-time grid visualization with 10 taxels
- Model switching (combined, 000pct, 010pct, 020pct)

Usage:
    python3 10_points_real_time_predictor.py [--model-dir PATH]

Author: Gabriele Giudici
Date: 2025
"""

import os
import sys
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import joblib
from pathlib import Path
import serial
import minimalmodbus as mm

import libscrc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

# Default model directory (will be set based on data directory)
DEFAULT_MODEL_DIR = None

# 10-point location names + no_touch
# Physical layout: 2 rows × 5 columns
# Row 1 (X=-0.01): 2, 3, 6, 7, 10 (ordered by Y: 0, 0.01, 0.02, 0.03, 0.04)
# Row 2 (X=0): 1, 4, 5, 8, 9 (ordered by Y: 0, 0.01, 0.02, 0.03, 0.04)
# Class 10: 'no_touch' (no contact)
MULTIPOINT_OFFSET_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'no_touch']

# Grid layout mapping: [row, col] -> location_id
# Row 0: locations with X=-0.01 (2, 3, 6, 7, 10)
# Row 1: locations with X=0 (1, 4, 5, 8, 9)
GRID_LAYOUT = [
    ['2', '3', '6', '7', '10'],  # Row 0: X=-0.01, ordered by Y
    ['1', '4', '5', '8', '9']    # Row 1: X=0, ordered by Y
]

# Mapping from location ID to grid position [row, col]
LOCATION_TO_GRID = {}
for row_idx, row in enumerate(GRID_LAYOUT):
    for col_idx, loc_id in enumerate(row):
        LOCATION_TO_GRID[loc_id] = (row_idx, col_idx)

# Mapping from location ID to magnetic sensor ID (1-15)
# NOTE: This mapping needs to be verified based on physical sensor layout
# The 15 magnetic sensors are arranged in a 3×5 grid:
#   Row 1: S1  S2  S3  S4  S5
#   Row 2: S6  S7  S8  S9  S10
#   Row 3: S11 S12 S13 S14 S15
# The 10 contact points are arranged in 2 rows × 5 columns
# TODO: Verify the physical mapping between contact points and magnetic sensors
LOCATION_TO_SENSOR = {
    '1': 1,   # Center - needs verification
    '2': 2,   # X=-0.01, Y=0 - needs verification
    '3': 3,   # X=-0.01, Y=0.01 - needs verification
    '4': 4,   # X=0, Y=0.01 - needs verification
    '5': 5,   # X=0, Y=0.02 - needs verification
    '6': 6,   # X=-0.01, Y=0.02 - needs verification
    '7': 7,   # X=-0.01, Y=0.03 - needs verification
    '8': 8,   # X=0, Y=0.03 - needs verification
    '9': 9,   # X=0, Y=0.04 - needs verification
    '10': 10  # X=-0.01, Y=0.04 - needs verification
}

class SensorReader:
    """Handles real-time reading from FT sensor and StretchMagTec 3x5 sensors."""
    
    def __init__(self, enable_ft_sensor=False):
        self.enable_ft_sensor = enable_ft_sensor
        self.ft_data = np.zeros(6)
        self.stretchmagtec_data = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.running = False
        
        # Hz tracking for each sensor
        self.last_hz_time = time.time()
        self.sensor_hz_counts = [0] * STRETCHMAGTEC_SENSORS
        self.sensor_hz_values = [0.0] * STRETCHMAGTEC_SENSORS
        
        # FT sensor calibration
        self.ft_offset = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        self.ft_calibration_samples = []
        self.ft_calibration_start_time = None
        self.ft_calibration_duration = 2.0  # seconds (from config)
        self.ft_is_calibrated = False
        
        # StretchMagTec sensor calibration
        self.stretchmagtec_offsets = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.stretchmagtec_calibration_samples = []
        self.stretchmagtec_calibration_start_time = None
        self.stretchmagtec_calibration_duration = 5.0  # seconds (from config)
        self.stretchmagtec_is_calibrated = False
        
        # FT sensor setup
        self.ft_thread = None
        self.ft_ser = None
        
        # StretchMagTec sensor setup
        self.stretchmagtec_thread = None
        self.stretchmagtec_ser = None
        
        # Data buffers for real-time plotting
        self.ft_buffer = []
        self.stretchmagtec_buffer = []
        self.time_buffer = []
        self.max_buffer_size = 1000
        
        # Locks for thread safety
        self.ft_lock = threading.Lock()
        self.stretchmagtec_lock = threading.Lock()
    
    def start_sensors(self):
        """Start sensor reading threads."""
        if self.running:
            return
        
        # Reset calibration state
        self.ft_is_calibrated = False
        self.ft_calibration_samples = []
        self.ft_calibration_start_time = None
        self.ft_offset = np.zeros(6)
        
        self.stretchmagtec_is_calibrated = False
        self.stretchmagtec_calibration_samples = []
        self.stretchmagtec_calibration_start_time = None
        self.stretchmagtec_offsets = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        self.running = True
        
        # Start FT sensor thread (only if enabled)
        if self.enable_ft_sensor:
            self.ft_thread = threading.Thread(target=self._ft_sensor_loop, daemon=True)
            self.ft_thread.start()
        else:
            print("[SensorReader] FT sensor disabled - skipping initialization")
            self.ft_thread = None
        
        # Start StretchMagTec sensor thread
        self.stretchmagtec_thread = threading.Thread(target=self._stretchmagtec_sensor_loop, daemon=True)
        self.stretchmagtec_thread.start()
        
        print("Sensors started successfully - Calibration will begin automatically")
    
    def stop_sensors(self):
        """Stop sensor reading threads."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        if self.ft_thread and self.ft_thread.is_alive():
            self.ft_thread.join(timeout=2)
        if self.stretchmagtec_thread and self.stretchmagtec_thread.is_alive():
            self.stretchmagtec_thread.join(timeout=2)
        
        # Close serial connections
        if self.ft_ser:
            try:
                self.ft_ser.close()
            except:
                pass
        if self.stretchmagtec_ser:
            try:
                self.stretchmagtec_ser.close()
            except:
                pass
        
        print("Sensors stopped successfully")
    
    def _ft_sensor_loop(self):
        """FT sensor reading loop."""
        if not self.enable_ft_sensor:
            print("[FT Thread] FT sensor disabled - exiting loop")
            return
        
        try:
            print(f"[FT Thread] Starting FT sensor initialization on {FT_PORT}...")
            # Initialize FT sensor
            ser_tmp = serial.Serial(port=FT_PORT, baudrate=FT_BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1)
            ser_tmp.write(bytearray([0xff]*50))
            ser_tmp.close()
            
            mm.BAUDRATE = FT_BAUDRATE
            mm.BYTESIZE = 8
            mm.PARITY = 'N'
            mm.STOPBITS = 1
            mm.TIMEOUT = 1
            ft300 = mm.Instrument(FT_PORT, slaveaddress=9)
            ft300.close_port_after_each_call = True
            ft300.write_register(410, 0x0200)
            del ft300
            
            self.ft_ser = serial.Serial(port=FT_PORT, baudrate=FT_BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1)
            STARTBYTES = bytes([0x20, 0x4e])
            print(f"[FT Thread] Reading initial data for zero reference...")
            self.ft_ser.read_until(STARTBYTES)
            data = self.ft_ser.read_until(STARTBYTES)
            dataArray = bytearray(data)
            dataArray = STARTBYTES + dataArray[:-2]
            
            if not self._crc_check(dataArray):
                print("[FT Thread] CRC ERROR on ZeroRef")
                with self.ft_lock:
                    self.ft_data[:] = [float('nan')] * 6
                return
            
            zeroRef = self._force_from_serial_message(dataArray)
            print(f"[FT Thread] Zero reference set: {zeroRef}")
            print(f"[FT Thread] Starting real-time reading loop...")
            
            read_count = 0
            while self.running:
                data = self.ft_ser.read_until(STARTBYTES)
                dataArray = bytearray(data)
                dataArray = STARTBYTES + dataArray[:-2]
                
                if not self._crc_check(dataArray):
                    continue
                
                raw_force = self._force_from_serial_message(dataArray, zeroRef)
                
                read_count += 1
                if read_count <= 5:
                    print(f"[FT Thread] Reading #{read_count}: Fx={raw_force[0]:.4f}N, Fy={raw_force[1]:.4f}N, Fz={raw_force[2]:.4f}N")
                
                # Handle FT calibration
                current_time = time.time()
                if not self.ft_is_calibrated:
                    if self.ft_calibration_start_time is None:
                        self.ft_calibration_start_time = current_time
                        print(f"[FT Thread] Starting calibration ({self.ft_calibration_duration} seconds)...")
                    
                    # Collect calibration samples
                    self.ft_calibration_samples.append(raw_force.copy())
                    
                    # Check if calibration is complete
                    if current_time - self.ft_calibration_start_time >= self.ft_calibration_duration:
                        if self.ft_calibration_samples:
                            samples_array = np.array(self.ft_calibration_samples)
                            self.ft_offset = np.mean(samples_array, axis=0)
                            self.ft_is_calibrated = True
                            print(f"[FT Thread] ✅ Calibration complete! Offset: {self.ft_offset}")
                            self.ft_calibration_samples = []  # Clear to save memory
                
                # Apply offset compensation
                compensated_force = raw_force - self.ft_offset
                
                with self.ft_lock:
                    self.ft_data[:] = compensated_force
                
                # Use compensated force for buffer (already compensated above)
                ft_cleaned = [0 if abs(val) < FT_NOISE_THRESHOLD else val for val in compensated_force]
                
                current_time = time.time()
                if len(self.time_buffer) >= self.max_buffer_size:
                    self.ft_buffer.pop(0)
                    self.time_buffer.pop(0)
                
                self.ft_buffer.append(ft_cleaned.copy())
                self.time_buffer.append(current_time)
                
        except Exception as e:
            print(f"FT Sensor error: {e}")
            import traceback
            traceback.print_exc()
            with self.ft_lock:
                self.ft_data[:] = [float('nan')] * 6
        finally:
            if self.ft_ser:
                try:
                    self.ft_ser.close()
                except:
                    pass
    
    def _stretchmagtec_sensor_loop(self):
        """StretchMagTec sensor reading loop."""
        try:
            # Try multiple ports
            ports_to_try = [STRETCHMAGTEC_PORT] + [f"/dev/ttyACM{i}" for i in range(10)]
            self.stretchmagtec_ser = None
            
            print(f"[StretchMagTec Thread] Trying to connect with baud rate: {STRETCHMAGTEC_BAUD}")
            
            for port in ports_to_try:
                try:
                    if os.path.exists(port):
                        print(f"[StretchMagTec Thread] Trying port {port}...")
                        self.stretchmagtec_ser = serial.Serial(port, STRETCHMAGTEC_BAUD, timeout=1)
                        print(f"[StretchMagTec Thread] ✅ Connected to {port} at {STRETCHMAGTEC_BAUD} baud")
                        break
                except Exception as e:
                    print(f"[StretchMagTec Thread] Failed to connect to {port}: {e}")
                    continue
            
            if self.stretchmagtec_ser is None:
                print(f"[StretchMagTec Thread] ❌ Could not connect to any port")
                return
            
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Flush any existing data
            self.stretchmagtec_ser.reset_input_buffer()
            
            line_count = 0
            parse_fail_count = 0
            no_data_count = 0
            
            while self.running:
                if self.stretchmagtec_ser.in_waiting > 0:
                    line = self.stretchmagtec_ser.readline().decode('utf-8', errors='ignore').strip()
                    line_count += 1
                    
                    if line:
                        sensor_values = self._parse_stretchmagtec_line(line)
                        
                        if sensor_values is not None:
                            # Handle StretchMagTec calibration
                            current_time = time.time()
                            if not self.stretchmagtec_is_calibrated:
                                if self.stretchmagtec_calibration_start_time is None:
                                    self.stretchmagtec_calibration_start_time = current_time
                                    print(f"[StretchMagTec Thread] Starting calibration ({self.stretchmagtec_calibration_duration} seconds)...")
                                
                                # Collect calibration samples
                                self.stretchmagtec_calibration_samples.append(sensor_values.copy())
                                
                                # Check if calibration is complete
                                if current_time - self.stretchmagtec_calibration_start_time >= self.stretchmagtec_calibration_duration:
                                    if self.stretchmagtec_calibration_samples:
                                        samples_array = np.array(self.stretchmagtec_calibration_samples)
                                        self.stretchmagtec_offsets = np.mean(samples_array, axis=0)
                                        self.stretchmagtec_is_calibrated = True
                                        print(f"[StretchMagTec Thread] ✅ Calibration complete!")
                                        for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                            print(f"  S{sensor_id+1}: offset = [{self.stretchmagtec_offsets[sensor_id, 0]:.2f}, {self.stretchmagtec_offsets[sensor_id, 1]:.2f}, {self.stretchmagtec_offsets[sensor_id, 2]:.2f}]")
                                        self.stretchmagtec_calibration_samples = []  # Clear to save memory
                            
                            # Apply offset compensation
                            compensated_values = sensor_values - self.stretchmagtec_offsets
                            
                            with self.stretchmagtec_lock:
                                self.stretchmagtec_data[:] = compensated_values
                            
                            # Update Hz tracking
                            current_time = time.time()
                            if current_time - self.last_hz_time >= 1.0:
                                for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                    self.sensor_hz_values[sensor_id] = self.sensor_hz_counts[sensor_id]
                                    self.sensor_hz_counts[sensor_id] = 0
                                self.last_hz_time = current_time
                            
                            # Count Hz for each sensor (if any channel changed)
                            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                if np.any(np.abs(compensated_values[sensor_id, :]) > 0):
                                    self.sensor_hz_counts[sensor_id] += 1
                            
                            # Add to buffer (use compensated values)
                            current_time = time.time()
                            if len(self.stretchmagtec_buffer) >= self.max_buffer_size:
                                self.stretchmagtec_buffer.pop(0)
                            
                            self.stretchmagtec_buffer.append(compensated_values.copy())
                            
                            # Debug: print first few successful reads
                            if line_count <= 5:
                                print(f"[StretchMagTec Thread] Reading #{line_count}: {len(line)} chars, parsed OK")
                        else:
                            parse_fail_count += 1
                            if parse_fail_count <= 5:
                                print(f"[StretchMagTec Thread] Parse failed (line #{line_count}): '{line[:50]}...' (expected {STRETCHMAGTEC_SENSORS * STRETCHMAGTEC_CHANNELS} values)")
                else:
                    no_data_count += 1
                    if no_data_count == 100:  # Print every 100 iterations when no data
                        print(f"[StretchMagTec Thread] ⚠️ No data available (waiting...)")
                        no_data_count = 0
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                            
        except Exception as e:
            print(f"StretchMagTec Sensor error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.stretchmagtec_ser:
                try:
                    self.stretchmagtec_ser.close()
                except:
                    pass
    
    def _parse_stretchmagtec_line(self, line):
        """Parse StretchMagTec sensor line - handles multiple formats."""
        try:
            sensor_values = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
            line = line.strip()
            
            if not line:
                return None
            
            # Try format: comma-separated values (45 values: 15 sensors × 3 channels)
            if ',' in line and ' | ' not in line and not line.startswith('S'):
                parts = line.split(',')
                expected_values = STRETCHMAGTEC_SENSORS * STRETCHMAGTEC_CHANNELS
                if len(parts) == expected_values:
                    values = np.array([float(x) for x in parts])
                    sensor_data = values.reshape(STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS)
                    return sensor_data
            
            # Try format: "S1: X=1234 Y=5678 Z=9012 | S2: X=2345 Y=6789 Z=0123 | ..."
            if ' | ' in line:
                sensor_parts = line.split(' | ')
                if len(sensor_parts) >= STRETCHMAGTEC_SENSORS:
                    for i, sensor_part in enumerate(sensor_parts[:STRETCHMAGTEC_SENSORS]):
                        sensor_part = sensor_part.strip()
                        if ':' not in sensor_part:
                            continue
                        
                        # Extract the values part after ':'
                        values_part = sensor_part.split(':', 1)[1].strip()
                        
                        # Parse X=, Y=, Z= values
                        coords = {'X': 0, 'Y': 0, 'Z': 0}
                        for coord_pair in values_part.split():
                            if '=' in coord_pair:
                                coord, value = coord_pair.split('=', 1)
                                if coord in coords:
                                    try:
                                        coords[coord] = float(value)
                                    except ValueError:
                                        coords[coord] = 0
                        
                        # Store in array [sensor_id, channel] where channels are [X, Y, Z]
                        sensor_values[i, 0] = coords['X']
                        sensor_values[i, 1] = coords['Y']
                        sensor_values[i, 2] = coords['Z']
                    
                    return sensor_values
            
            # Try format with regex: "S1: X=1234 Y=5678 Z=9012" (without | separator)
            import re
            pattern = r'S(\d+):\s*X=([-\d.]+)\s*Y=([-\d.]+)\s*Z=([-\d.]+)'
            matches = re.findall(pattern, line)
            
            if matches:
                for match in matches:
                    sensor_id = int(match[0]) - 1  # Convert to 0-based index
                    if 0 <= sensor_id < STRETCHMAGTEC_SENSORS:
                        try:
                            sensor_values[sensor_id, 0] = float(match[1])  # X
                            sensor_values[sensor_id, 1] = float(match[2])  # Y
                            sensor_values[sensor_id, 2] = float(match[3])  # Z
                        except ValueError:
                            continue
                
                if np.any(sensor_values != 0):
                    return sensor_values
            
            # Try format: "DATA:1:x,y,z|2:x,y,z|..."
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
                    return sensor_values
            
            return None
            
        except Exception as e:
            print(f"[StretchMagTec Thread] Parse error: {e}")
            return None
    
    def _crc_check(self, dataArray):
        """Check CRC for FT sensor data."""
        try:
            crc = libscrc.modbus(dataArray[:-2])
            return crc == int.from_bytes(dataArray[-2:], byteorder='little')
        except:
            return False
    
    def _force_from_serial_message(self, dataArray, zeroRef=None):
        """Extract force from FT sensor serial message."""
        try:
            fx = int.from_bytes(dataArray[2:4], byteorder='little', signed=True) / 100.0
            fy = int.from_bytes(dataArray[4:6], byteorder='little', signed=True) / 100.0
            fz = int.from_bytes(dataArray[6:8], byteorder='little', signed=True) / 100.0
            tx = int.from_bytes(dataArray[8:10], byteorder='little', signed=True) / 1000.0
            ty = int.from_bytes(dataArray[10:12], byteorder='little', signed=True) / 1000.0
            tz = int.from_bytes(dataArray[12:14], byteorder='little', signed=True) / 1000.0
            
            force = np.array([fx, fy, fz, tx, ty, tz])
            
            if zeroRef is not None:
                force = force - zeroRef
            
            return force
        except:
            return np.zeros(6)
    
    def get_ft_data(self):
        """Get current FT sensor data."""
        with self.ft_lock:
            return self.ft_data.copy()
    
    def get_stretchmagtec_data(self):
        """Get current StretchMagTec sensor data."""
        with self.stretchmagtec_lock:
            return self.stretchmagtec_data.copy()
    
    def get_sensor_hz(self, sensor_id):
        """Get Hz for a specific sensor."""
        return self.sensor_hz_values[sensor_id] if sensor_id < len(self.sensor_hz_values) else 0.0


class ModelPredictor:
    """Handles model loading and predictions for 10-point multi-point configuration."""
    
    def __init__(self, model_dir=None, sensor_reader=None):
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR
        if model_dir is None:
            # Try to find models in common locations
            possible_dirs = [
                Path("data/Multiple_Points/2.5mm_single_test24/cleaned/raw/models"),
                Path("models"),
                MODELS_DIR
            ]
            for pd in possible_dirs:
                if pd.exists() and (pd / "force_regressor_combined.joblib").exists():
                    model_dir = pd
                    break
        
        self.model_dir = Path(model_dir) if model_dir else None
        self.sensor_reader = sensor_reader
        self.current_model_type = "combined"  # Default: combined
        
        # Models for each stretch level
        self.force_models = {}
        self.location_models = {}
        self.stretch_models = {}  # Stretch classifier (only for combined)
        self.scalers = {}
        
        self.models_loaded = False
    
    def load_models(self, model_type="combined"):
        """Load trained models from disk."""
        if self.model_dir is None or not self.model_dir.exists():
            print(f"❌ Model directory not found: {self.model_dir}")
            return False
        
        try:
            print(f"Looking for models in: {self.model_dir}")
            
            if model_type == "combined":
                # Load combined model
                force_path = self.model_dir / "force_regressor_combined.joblib"
                location_path = self.model_dir / "location_classifier_combined.joblib"
                scaler_path = self.model_dir / "scaler_combined.joblib"
                stretch_path = self.model_dir / "stretch_classifier_combined.joblib"
                
                if force_path.exists() and location_path.exists() and scaler_path.exists():
                    self.force_models["combined"] = joblib.load(force_path)
                    self.location_models["combined"] = joblib.load(location_path)
                    self.scalers["combined"] = joblib.load(scaler_path)
                    # Load stretch classifier if available (optional)
                    if stretch_path.exists():
                        self.stretch_models["combined"] = joblib.load(stretch_path)
                        print("✅ Combined model loaded (with stretch classifier)")
                    else:
                        self.stretch_models["combined"] = None
                        print("✅ Combined model loaded (stretch classifier not available)")
                    self.current_model_type = "combined"
                    self.models_loaded = True
                    return True
                else:
                    print(f"⚠️ Combined model files not found")
                    return False
            
            else:
                # Load specific stretch model
                stretch = model_type  # e.g., "000pct"
                force_path = self.model_dir / f"force_regressor_{stretch}.joblib"
                location_path = self.model_dir / f"location_classifier_{stretch}.joblib"
                scaler_path = self.model_dir / f"scaler_{stretch}.joblib"
                
                if force_path.exists() and location_path.exists() and scaler_path.exists():
                    self.force_models[stretch] = joblib.load(force_path)
                    self.location_models[stretch] = joblib.load(location_path)
                    self.scalers[stretch] = joblib.load(scaler_path)
                    self.current_model_type = stretch
                    print(f"✅ {stretch} model loaded")
                    self.models_loaded = True
                    return True
                else:
                    print(f"⚠️ {stretch} model files not found")
                    return False
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def switch_model(self, model_type):
        """Switch to a different model type."""
        if model_type not in ["combined", "000pct", "010pct", "020pct"]:
            print(f"⚠️ Invalid model type: {model_type}")
            return False
        
        if model_type == self.current_model_type and model_type in self.force_models:
            print(f"Model {model_type} already loaded")
            return True
        
        return self.load_models(model_type)
    
    def extract_features(self, stretchmagtec_data):
        """
        Extract 45 raw features from StretchMagTec data (same as training).
        
        Args:
            stretchmagtec_data: Sensor data (15, 3) or (n_samples, 15, 3)
        
        Returns:
            features: Feature vector (45,) or (n_samples, 45)
        """
        # Make a copy to avoid modifying original data
        data = stretchmagtec_data.copy()
        
        # Invert x and y of sensor 8 (index 7, 0-based) - magnet may be flipped
        # Sensor 8 is at index 7 (0-based: sensors 0-14)
        SENSOR_8_INDEX = 7
        if data.ndim == 2:
            # Single sample: (15, 3)
            data[SENSOR_8_INDEX, 0] = -data[SENSOR_8_INDEX, 0]  # Invert x
            data[SENSOR_8_INDEX, 1] = -data[SENSOR_8_INDEX, 1]  # Invert y
            # z (index 2) is not inverted
        else:
            # Multiple samples: (n_samples, 15, 3)
            data[:, SENSOR_8_INDEX, 0] = -data[:, SENSOR_8_INDEX, 0]  # Invert x
            data[:, SENSOR_8_INDEX, 1] = -data[:, SENSOR_8_INDEX, 1]  # Invert y
            # z (index 2) is not inverted
        
        # If single sample, reshape
        if data.ndim == 2:
            # Single sample: (15, 3) -> (1, 45)
            features = data.flatten()
            return features.reshape(1, -1)
        else:
            # Multiple samples: (n_samples, 15, 3) -> (n_samples, 45)
            n_samples = data.shape[0]
            features = data.reshape(n_samples, -1)
            return features
    
    def predict_location(self, stretchmagtec_data):
        """Predict contact location from StretchMagTec data."""
        if not self.models_loaded or self.current_model_type not in self.location_models:
            return None, 0.0, {}
        
        try:
            # Extract features
            features = self.extract_features(stretchmagtec_data)
            
            # Scale features
            scaler = self.scalers[self.current_model_type]
            features_scaled = scaler.transform(features)
            
            # Predict
            location_model = self.location_models[self.current_model_type]
            prediction = location_model.predict(features_scaled)[0]
            probabilities = location_model.predict_proba(features_scaled)[0]
            
            # Ensure prediction is within valid range (0-10 for 10 points + no_touch)
            if prediction < 0 or prediction >= len(MULTIPOINT_OFFSET_NAMES):
                print(f"⚠️  Warning: Invalid prediction {prediction}, clamping to valid range [0, {len(MULTIPOINT_OFFSET_NAMES)-1}]")
                prediction = max(0, min(prediction, len(MULTIPOINT_OFFSET_NAMES) - 1))
            
            # Map prediction to location name (0-10 -> '1'-'10', 'no_touch')
            location_name = MULTIPOINT_OFFSET_NAMES[prediction]
            confidence = probabilities[prediction] if prediction < len(probabilities) else 0.0
            
            # Get top 3 predictions
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3_predictions = {
                MULTIPOINT_OFFSET_NAMES[idx]: float(probabilities[idx]) 
                for idx in top3_indices if idx < len(MULTIPOINT_OFFSET_NAMES)
            }
            
            return location_name, confidence, top3_predictions
            
        except Exception as e:
            print(f"Location prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, {}
    
    def predict_force(self, stretchmagtec_data):
        """Predict force from StretchMagTec data."""
        if not self.models_loaded or self.current_model_type not in self.force_models:
            return {"fx": 0.0, "fy": 0.0, "fz": 0.0}
        
        try:
            # Extract features
            features = self.extract_features(stretchmagtec_data)
            
            # Scale features
            scaler = self.scalers[self.current_model_type]
            features_scaled = scaler.transform(features)
            
            # Predict (only Fz for now, as per training)
            force_model = self.force_models[self.current_model_type]
            fz_pred = force_model.predict(features_scaled)[0]
            
            return {"fx": 0.0, "fy": 0.0, "fz": float(fz_pred)}
            
        except Exception as e:
            print(f"Force prediction error: {e}")
            return {"fx": 0.0, "fy": 0.0, "fz": 0.0}
    
    def predict_stretch(self, stretchmagtec_data):
        """Predict stretch level from StretchMagTec data (only for combined model)."""
        if (not self.models_loaded or 
            self.current_model_type != "combined" or 
            "combined" not in self.stretch_models or 
            self.stretch_models["combined"] is None):
            return None, 0.0, {}
        
        try:
            # Extract features
            features = self.extract_features(stretchmagtec_data)
            
            # Scale features
            scaler = self.scalers[self.current_model_type]
            features_scaled = scaler.transform(features)
            
            # Predict
            stretch_model = self.stretch_models["combined"]
            prediction = stretch_model.predict(features_scaled)[0]
            probabilities = stretch_model.predict_proba(features_scaled)[0]
            
            # Map prediction to stretch name (3 classes: 000pct, 010pct, 020pct)
            # Note: no_touch is NOT a stretch class, but no_touch sequences are classified
            # by their physical stretch level (000pct/010pct/020pct)
            stretch_names = ['000pct', '010pct', '020pct']
            stretch_name = stretch_names[prediction] if prediction < len(stretch_names) else 'unknown'
            confidence = probabilities[prediction]
            
            # Get all probabilities
            all_predictions = {
                stretch_names[i]: float(prob) 
                for i, prob in enumerate(probabilities) 
                if i < len(stretch_names)
            }
            
            return stretch_name, confidence, all_predictions
            
        except Exception as e:
            print(f"Stretch prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, {}


class GridVisualizationWindow:
    """Popup window showing the 10-point contact grid with real-time predictions."""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("10-Point Contact Grid Visualization")
        self.window.geometry("1000x400")
        
        # 10 locations
        self.locations = MULTIPOINT_OFFSET_NAMES  # ['1', '2', ..., '10']
        
        # Contact probabilities (will be updated in real-time)
        self.location_probabilities = {loc: 0.0 for loc in self.locations}
        
        # Create matplotlib figure
        self.create_visualization()
        
        # Update timer
        self.update_interval = 100  # ms
        self.running = True
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_visualization(self):
        """Create the grid visualization."""
        # Create figure
        self.fig = Figure(figsize=(12, 4), dpi=90)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info label
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, 
                                     text="10-Point Contact Grid | Locations: 1-10",
                                     font=("Arial", 10, "bold"))
        self.info_label.pack()
        
        self.prediction_label = ttk.Label(info_frame, 
                                          text="Current prediction: None", 
                                          font=("Arial", 10), 
                                          foreground="blue")
        self.prediction_label.pack()
        
        # Draw initial grid
        self.draw_grid()
    
    def draw_grid(self):
        """Draw the 10-point contact grid with current probabilities."""
        self.ax.clear()
        
        # Grid layout: 2 rows × 5 columns
        cell_size = 1.0
        cell_spacing = 0.2
        
        # Draw each location using the correct grid layout
        for row_idx, row_locations in enumerate(GRID_LAYOUT):
            for col_idx, location in enumerate(row_locations):
                # Calculate cell position
                x = col_idx * (cell_size + cell_spacing)
                y = (1 - row_idx) * (cell_size + cell_spacing)  # Flip Y axis
                
                # Get probability for this location
                prob = self.location_probabilities.get(location, 0.0)
                
                # Color based on probability (white = 0, red = 1)
                # Use a colormap that goes from white to red
                color = plt.cm.Reds(prob)
                
                # Draw cell
                rect = Rectangle((x, y), cell_size, cell_size,
                              fill=True, facecolor=color, edgecolor='black', linewidth=2)
                self.ax.add_patch(rect)
                
                # Add location label
                self.ax.text(x + cell_size/2, y + cell_size/2,
                            location, ha='center', va='center', 
                            fontsize=14, fontweight='bold',
                            color='white' if prob > 0.5 else 'black')
                
                # Add probability label
                if prob > 0.01:
                    self.ax.text(x + cell_size/2, y + cell_size/2 - 0.3,
                                f"{prob*100:.1f}%", ha='center', va='center',
                                fontsize=10, color='white' if prob > 0.5 else 'black')
        
        # Set axis properties
        x_max = 5 * (cell_size + cell_spacing)
        y_max = 2 * (cell_size + cell_spacing)
        
        self.ax.set_xlim(-0.1, x_max)
        self.ax.set_ylim(-0.1, y_max)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        
        # Add colorbar
        if not hasattr(self, 'colorbar'):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            self.colorbar = self.fig.colorbar(sm, ax=self.ax, orientation='horizontal', 
                                              pad=0.05, fraction=0.046, label='Contact Probability')
        
        self.canvas.draw_idle()
    
    def update_predictions(self, location, confidence, top3_predictions):
        """
        Update contact probabilities based on prediction.
        
        Args:
            location: Predicted location ('1'-'10')
            confidence: Confidence value (0-1)
            top3_predictions: Dictionary of {location: probability} for top 3
        """
        # Reset all probabilities
        for loc in self.locations:
            self.location_probabilities[loc] = 0.0
        
        # Update with top 3 predictions
        for loc, prob in top3_predictions.items():
            self.location_probabilities[loc] = prob
        
        # Update prediction label
        if location:
            self.prediction_label.config(
                text=f"Predicted: Location {location} ({confidence*100:.1f}%)",
                foreground="green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            )
        
        # Redraw grid
        self.draw_grid()
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        self.window.destroy()


class RealTimePredictorGUI:
    """Main GUI application for real-time 10-point multi-point prediction."""
    
    def __init__(self, model_dir=None, enable_ft_sensor=False):
        self.root = tk.Tk()
        self.root.title("10-Point Multi-Point Real-Time Predictor")
        self.root.geometry("1400x900")
        
        # Grid visualization window
        self.grid_viz_window = None
        
        # Initialize components
        self.sensor_reader = SensorReader(enable_ft_sensor=enable_ft_sensor)
        self.model_predictor = ModelPredictor(model_dir, self.sensor_reader)
        
        # GUI update control
        self.update_running = False
        self.update_interval = 50  # ms
        
        # Create GUI elements
        self.create_widgets()
        
        # Load default model (combined)
        self.load_models()
    
    def create_widgets(self):
        """Create and layout GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Sensors", command=self.start_sensors)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Sensors", command=self.stop_sensors, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.load_models_button = ttk.Button(control_frame, text="Reload Models", command=self.load_models)
        self.load_models_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.grid_viz_button = ttk.Button(control_frame, text="Show Grid Visualization", command=self.toggle_grid_viz)
        self.grid_viz_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="combined")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                   values=["combined", "000pct", "010pct", "020pct"],
                                   state="readonly", width=10)
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready", foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Data display frame
        data_frame = ttk.Frame(main_frame)
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Sensor data
        left_frame = ttk.LabelFrame(data_frame, text="Sensor Data")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # FT sensor data
        ft_frame = ttk.LabelFrame(left_frame, text="FT Sensor (Ground Truth)")
        ft_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.ft_labels = []
        ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
        for i, name in enumerate(ft_names):
            label = ttk.Label(ft_frame, text=f"{name}: 0.000", font=("Courier", 10))
            label.pack(anchor=tk.W, padx=5)
            self.ft_labels.append(label)
        
        # StretchMagTec sensor data
        stretchmagtec_frame = ttk.LabelFrame(left_frame, text="StretchMagTec 3x5 Sensors")
        stretchmagtec_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for sensor data
        canvas = tk.Canvas(stretchmagtec_frame)
        scrollbar = ttk.Scrollbar(stretchmagtec_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create labels for all 15 sensors
        self.stretchmagtec_labels = []
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            sensor_frame = ttk.LabelFrame(scrollable_frame, text=f"Sensor {sensor_id + 1}")
            sensor_frame.pack(fill=tk.X, padx=2, pady=2)
            
            sensor_labels = []
            for channel, name in enumerate(['X', 'Y', 'Z']):
                label = ttk.Label(sensor_frame, text=f"{name}: 0", font=("Courier", 9))
                label.pack(anchor=tk.W, padx=2)
                sensor_labels.append(label)
            
            # Add Hz label
            hz_label = ttk.Label(sensor_frame, text="Hz: 0.0", font=("Courier", 9, "bold"), foreground="blue")
            hz_label.pack(anchor=tk.W, padx=2)
            sensor_labels.append(hz_label)
            
            self.stretchmagtec_labels.append(sensor_labels)
        
        # Right column - Predictions
        right_frame = ttk.LabelFrame(data_frame, text="Predictions")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Current model label
        model_info_frame = ttk.LabelFrame(right_frame, text="Current Model")
        model_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_info_label = ttk.Label(model_info_frame, text="Model: combined", font=("Arial", 10, "bold"))
        self.model_info_label.pack(pady=5)
        
        # Location prediction
        location_frame = ttk.LabelFrame(right_frame, text="Location Prediction")
        location_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.location_label = ttk.Label(location_frame, text="Location: Unknown", font=("Arial", 14, "bold"))
        self.location_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(location_frame, text="Confidence: 0.0%", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Top 3 predictions
        top3_frame = ttk.LabelFrame(location_frame, text="Top 3 Predictions")
        top3_frame.pack(fill=tk.X, pady=5)
        
        self.top3_labels = []
        for i in range(3):
            label = ttk.Label(top3_frame, text=f"{i+1}. Location X: 0.0%", font=("Courier", 10))
            label.pack(anchor=tk.W, padx=5, pady=2)
            self.top3_labels.append(label)
        
        # Stretch prediction (only for combined model)
        stretch_frame = ttk.LabelFrame(right_frame, text="Stretch Classification (Combined Model)")
        stretch_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stretch_label = ttk.Label(stretch_frame, text="Stretch: N/A", font=("Arial", 12, "bold"))
        self.stretch_label.pack(pady=5)
        
        self.stretch_confidence_label = ttk.Label(stretch_frame, text="Confidence: 0.0%", font=("Arial", 10))
        self.stretch_confidence_label.pack(pady=2)
        
        # Stretch probabilities
        stretch_probs_frame = ttk.Frame(stretch_frame)
        stretch_probs_frame.pack(fill=tk.X, pady=5)
        
        self.stretch_prob_labels = {}
        for stretch in ['000pct', '010pct', '020pct']:
            label = ttk.Label(stretch_probs_frame, text=f"{stretch}: 0.0%", font=("Courier", 9))
            label.pack(anchor=tk.W, padx=5)
            self.stretch_prob_labels[stretch] = label
        
        # Force prediction
        force_pred_frame = ttk.LabelFrame(right_frame, text="Force Prediction (from StretchMagTec)")
        force_pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.force_pred_labels = []
        force_names = ["Fx (N)", "Fy (N)", "Fz (N)"]
        for name in force_names:
            label = ttk.Label(force_pred_frame, text=f"{name}: 0.000", font=("Courier", 12))
            label.pack(anchor=tk.W, padx=5)
            self.force_pred_labels.append(label)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(right_frame, text="Real-Time Plots")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sensor selection frame
        selection_frame = ttk.Frame(plot_frame)
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Select sensors to plot:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Selected sensors tracking
        self.selected_sensors = set()
        
        # Predefined colors for each sensor (15 distinct colors)
        self.sensor_colors = [
            '#FF0000',  # Red
            '#0000FF',  # Blue  
            '#00FF00',  # Green
            '#FF8000',  # Orange
            '#8000FF',  # Purple
            '#00FFFF',  # Cyan
            '#FF0080',  # Magenta
            '#FFFF00',  # Yellow
            '#FF4000',  # Red-Orange
            '#4000FF',  # Blue-Purple
            '#00FF80',  # Green-Cyan
            '#FF0040',  # Red-Pink
            '#8000FF',  # Purple-Blue
            '#40FF00',  # Yellow-Green
            '#FF8080'   # Light Red
        ]
        
        # Create sensor selection buttons
        self.sensor_buttons = []
        buttons_frame = ttk.Frame(selection_frame)
        buttons_frame.pack(side=tk.LEFT, padx=10)
        
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            btn = tk.Button(buttons_frame, text=f"S{sensor_id+1}", width=3,
                           command=lambda s_id=sensor_id: self.toggle_sensor_selection(s_id),
                           bg=self.sensor_colors[sensor_id], fg='white', font=('Arial', 8, 'bold'))
            btn.pack(side=tk.LEFT, padx=1)
            self.sensor_buttons.append(btn)
        
        # Clear selection button
        clear_btn = ttk.Button(selection_frame, text="Clear All", 
                              command=self.clear_sensor_selection)
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 12), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots - 4 subplots vertically: FT, X, Y, Z
        self.ax1 = self.fig.add_subplot(411)  # FT sensor
        self.ax2 = self.fig.add_subplot(412)   # X-axis
        self.ax3 = self.fig.add_subplot(413)   # Y-axis
        self.ax4 = self.fig.add_subplot(414)   # Z-axis
        
        self.ax1.set_title("FT Sensor Data")
        self.ax1.set_ylabel("Force/Torque")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("StretchMagTec X-Axis")
        self.ax2.set_ylabel("Magnetic Field")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title("StretchMagTec Y-Axis")
        self.ax3.set_ylabel("Magnetic Field")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title("StretchMagTec Z-Axis")
        self.ax4.set_ylabel("Magnetic Field")
        self.ax4.set_xlabel("Time (s)")
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def on_model_change(self, event=None):
        """Handle model selection change."""
        new_model = self.model_var.get()
        if self.model_predictor.switch_model(new_model):
            self.model_info_label.config(text=f"Model: {new_model}")
            self.status_label.config(text=f"Status: Model switched to {new_model}", foreground="green")
        else:
            self.status_label.config(text=f"Status: Failed to load {new_model}", foreground="red")
    
    def load_models(self):
        """Load prediction models."""
        if self.model_predictor.load_models("combined"):
            self.status_label.config(text="Status: Models loaded", foreground="green")
        else:
            self.status_label.config(text="Status: Model loading failed", foreground="red")
    
    def start_sensors(self):
        """Start sensor reading and GUI updates."""
        try:
            self.sensor_reader.start_sensors()
            self.update_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Sensors running", foreground="green")
            self.update_gui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start sensors: {e}")
    
    def stop_sensors(self):
        """Stop sensor reading and GUI updates."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Sensors stopped", foreground="orange")
    
    def toggle_grid_viz(self):
        """Toggle grid visualization window."""
        if self.grid_viz_window is None or not self.grid_viz_window.running:
            self.grid_viz_window = GridVisualizationWindow(self.root)
            self.grid_viz_button.config(text="Hide Grid Visualization")
        else:
            self.grid_viz_window.window.destroy()
            self.grid_viz_window = None
            self.grid_viz_button.config(text="Show Grid Visualization")
    
    def toggle_sensor_selection(self, sensor_id):
        """Toggle sensor selection for plotting."""
        if sensor_id in self.selected_sensors:
            self.selected_sensors.remove(sensor_id)
            self.sensor_buttons[sensor_id].configure(bg=self.sensor_colors[sensor_id], relief='raised')
        else:
            self.selected_sensors.add(sensor_id)
            self.sensor_buttons[sensor_id].configure(bg=self.sensor_colors[sensor_id], relief='sunken')
        
        # Update plot immediately
        self.update_plots()
    
    def clear_sensor_selection(self):
        """Clear all sensor selections."""
        self.selected_sensors.clear()
        for i, btn in enumerate(self.sensor_buttons):
            btn.configure(bg=self.sensor_colors[i], relief='raised')
        self.update_plots()
    
    def update_plots(self):
        """Update real-time plots with selected sensors."""
        if not self.update_running:
            return
        
        try:
            # Get buffer data (with locks to ensure thread safety)
            with self.sensor_reader.ft_lock:
                ft_buffer = self.sensor_reader.ft_buffer.copy()
            with self.sensor_reader.stretchmagtec_lock:
                stretchmagtec_buffer = self.sensor_reader.stretchmagtec_buffer.copy()
            with self.sensor_reader.ft_lock:  # time_buffer is updated with ft_buffer
                time_buffer = self.sensor_reader.time_buffer.copy()
            
            if not time_buffer:
                return
            
            # Find minimum length to ensure all arrays have same size
            min_length = len(time_buffer)
            if ft_buffer:
                min_length = min(min_length, len(ft_buffer))
            if stretchmagtec_buffer:
                min_length = min(min_length, len(stretchmagtec_buffer))
            
            if min_length == 0:
                return
            
            # Trim all buffers to same length
            time_buffer = time_buffer[:min_length]
            if ft_buffer:
                ft_buffer = ft_buffer[:min_length]
            if stretchmagtec_buffer:
                stretchmagtec_buffer = stretchmagtec_buffer[:min_length]
            
            # Calculate relative time
            start_time = time_buffer[0]
            relative_time = np.array([(t - start_time) for t in time_buffer])
            
            # Clear plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot FT sensor data
            if ft_buffer and len(ft_buffer) == len(relative_time):
                ft_array = np.array(ft_buffer)
                self.ax1.plot(relative_time, ft_array[:, 0], 'r-', label='Fx', linewidth=1, alpha=0.7)
                self.ax1.plot(relative_time, ft_array[:, 1], 'g-', label='Fy', linewidth=1, alpha=0.7)
                self.ax1.plot(relative_time, ft_array[:, 2], 'b-', label='Fz', linewidth=1, alpha=0.7)
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force (N)")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.grid(True, alpha=0.3)
            
            # Plot StretchMagTec data for selected sensors
            if stretchmagtec_buffer and self.selected_sensors and len(stretchmagtec_buffer) == len(relative_time):
                stretch_array = np.array(stretchmagtec_buffer)
                
                # X-axis
                for sensor_id in self.selected_sensors:
                    if sensor_id < STRETCHMAGTEC_SENSORS:
                        color = self.sensor_colors[sensor_id]
                        self.ax2.plot(relative_time, stretch_array[:, sensor_id, 0], 
                                     color=color, label=f'S{sensor_id+1}', linewidth=1.5, alpha=0.8)
                
                # Y-axis
                for sensor_id in self.selected_sensors:
                    if sensor_id < STRETCHMAGTEC_SENSORS:
                        color = self.sensor_colors[sensor_id]
                        self.ax3.plot(relative_time, stretch_array[:, sensor_id, 1], 
                                     color=color, label=f'S{sensor_id+1}', linewidth=1.5, alpha=0.8)
                
                # Z-axis
                for sensor_id in self.selected_sensors:
                    if sensor_id < STRETCHMAGTEC_SENSORS:
                        color = self.sensor_colors[sensor_id]
                        self.ax4.plot(relative_time, stretch_array[:, sensor_id, 2], 
                                     color=color, label=f'S{sensor_id+1}', linewidth=1.5, alpha=0.8)
                
                # Add legends
                if self.selected_sensors:
                    self.ax2.legend(loc='upper right', fontsize=8)
                    self.ax3.legend(loc='upper right', fontsize=8)
                    self.ax4.legend(loc='upper right', fontsize=8)
            
            # Set titles and labels
            self.ax2.set_title("StretchMagTec X-Axis")
            self.ax2.set_ylabel("Magnetic Field")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.grid(True, alpha=0.3)
            
            self.ax3.set_title("StretchMagTec Y-Axis")
            self.ax3.set_ylabel("Magnetic Field")
            self.ax3.set_xlabel("Time (s)")
            self.ax3.grid(True, alpha=0.3)
            
            self.ax4.set_title("StretchMagTec Z-Axis")
            self.ax4.set_ylabel("Magnetic Field")
            self.ax4.set_xlabel("Time (s)")
            self.ax4.grid(True, alpha=0.3)
            
            # Use subplots_adjust instead of tight_layout to avoid warnings
            self.fig.subplots_adjust(hspace=0.4)
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_gui(self):
        """Update GUI with latest sensor data and predictions."""
        if not self.update_running:
            return
        
        try:
            # Get sensor data
            ft_data = self.sensor_reader.get_ft_data()
            stretchmagtec_data = self.sensor_reader.get_stretchmagtec_data()
            
            # Update FT sensor display
            ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
            for i, (name, value) in enumerate(zip(ft_names, ft_data)):
                color = "red" if abs(value) > 1.0 else "black"
                self.ft_labels[i].config(text=f"{name}: {value:7.3f}", foreground=color)
            
            # Update StretchMagTec sensor display
            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                for channel_id in range(STRETCHMAGTEC_CHANNELS):
                    value = stretchmagtec_data[sensor_id, channel_id]
                    channel_name = ['X', 'Y', 'Z'][channel_id]
                    color = "red" if abs(value) > 1000 else "black"
                    self.stretchmagtec_labels[sensor_id][channel_id].config(
                        text=f"{channel_name}: {value:6.0f}", foreground=color
                    )
                
                # Update Hz
                hz = self.sensor_reader.get_sensor_hz(sensor_id)
                self.stretchmagtec_labels[sensor_id][3].config(text=f"Hz: {hz:.1f}")
            
            # Make predictions
            location, confidence, top3_predictions = self.model_predictor.predict_location(stretchmagtec_data)
            predicted_forces = self.model_predictor.predict_force(stretchmagtec_data)
            
            # Predict stretch (only for combined model)
            stretch_pred, stretch_conf, stretch_probs = self.model_predictor.predict_stretch(stretchmagtec_data)
            
            # Update location prediction display
            if location:
                location_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                self.location_label.config(text=f"Location: {location}", foreground=location_color)
                self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%", foreground=location_color)
            else:
                self.location_label.config(text="Location: Unknown", foreground="gray")
                self.confidence_label.config(text="Confidence: 0.0%", foreground="gray")
            
            # Update top 3 predictions
            sorted_top3 = sorted(top3_predictions.items(), key=lambda x: x[1], reverse=True)
            for i, (loc, prob) in enumerate(sorted_top3[:3]):
                if i < len(self.top3_labels):
                    self.top3_labels[i].config(text=f"{i+1}. Location {loc}: {prob*100:.1f}%")
            
            # Update stretch prediction display (only for combined model)
            if self.model_predictor.current_model_type == "combined" and stretch_pred:
                stretch_color = "green" if stretch_conf > 0.7 else "orange" if stretch_conf > 0.5 else "red"
                self.stretch_label.config(text=f"Stretch: {stretch_pred}", foreground=stretch_color)
                self.stretch_confidence_label.config(text=f"Confidence: {stretch_conf*100:.1f}%", foreground=stretch_color)
                
                # Update stretch probabilities
                for stretch, prob in stretch_probs.items():
                    if stretch in self.stretch_prob_labels:
                        self.stretch_prob_labels[stretch].config(text=f"{stretch}: {prob*100:.1f}%")
            else:
                self.stretch_label.config(text="Stretch: N/A (not combined model)", foreground="gray")
                self.stretch_confidence_label.config(text="Confidence: 0.0%", foreground="gray")
                for stretch in ['000pct', '010pct', '020pct']:
                    if stretch in self.stretch_prob_labels:
                        self.stretch_prob_labels[stretch].config(text=f"{stretch}: 0.0%")
            
            # Update force predictions
            force_names = ["Fx (N)", "Fy (N)", "Fz (N)"]
            for i, (name, value) in enumerate(zip(force_names, [predicted_forces['fx'], predicted_forces['fy'], predicted_forces['fz']])):
                self.force_pred_labels[i].config(text=f"{name}: {value:7.3f}")
            
            # Update grid visualization if open
            if self.grid_viz_window and self.grid_viz_window.running:
                self.grid_viz_window.update_predictions(location, confidence, top3_predictions)
            
            # Update plots
            self.update_plots()
            
        except Exception as e:
            print(f"GUI update error: {e}")
            import traceback
            traceback.print_exc()
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_gui)
    
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="10-Point Multi-Point Real-Time Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=None,
        help='Directory containing trained models (default: auto-detect)'
    )
    parser.add_argument(
        '--enable-ft-sensor',
        action='store_true',
        help='Enable FT sensor connection (disabled by default to avoid conflicts with other scripts)'
    )
    
    args = parser.parse_args()
    
    # Set default model directory if provided
    global DEFAULT_MODEL_DIR
    if args.model_dir:
        DEFAULT_MODEL_DIR = args.model_dir
    
    # Create and run GUI
    enable_ft_sensor = args.enable_ft_sensor
    if enable_ft_sensor:
        print("✅ FT sensor enabled via command-line flag")
    else:
        print("⚠️  FT sensor disabled by default (use --enable-ft-sensor to enable)")
    app = RealTimePredictorGUI(model_dir=args.model_dir, enable_ft_sensor=enable_ft_sensor)
    app.run()


if __name__ == "__main__":
    main()

