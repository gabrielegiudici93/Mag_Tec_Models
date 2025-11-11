#!/usr/bin/env python3
"""
Real-Time StretchMagTec 3x5 Sensor Predictor with GUI - MagTec_KPM

This script provides real-time prediction of:
1. Contact point location (9-point grid: center, nw, n, ne, w, e, sw, s, se)
2. Force values in Newtons from StretchMagTec 3x5 sensors

The GUI displays:
- Real-time sensor readings (15 sensors in 3x5 grid)
- Predicted contact location with confidence
- FT sensor readings (ground truth)
- StretchMagTec sensor raw values
- Predicted forces from StretchMagTec sensors
- Real-time plots of sensor data

Usage:
    python3 real_time_predictor.py

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

# Model directory is from config
MODEL_DIR = MODELS_DIR

class SensorReader:
    """Handles real-time reading from FT sensor and StretchMagTec 3x5 sensors."""
    
    def __init__(self):
        self.ft_data = np.zeros(6)
        self.stretchmagtec_data = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.running = False
        
        # Hz tracking for each sensor
        self.last_hz_time = time.time()
        self.sensor_hz_counts = [0] * STRETCHMAGTEC_SENSORS
        self.sensor_hz_values = [0.0] * STRETCHMAGTEC_SENSORS
        
        # Offset calibration
        self.offsets = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))  # 15x3 = 45 offsets
        self.calibration_samples = []
        self.calibration_start_time = None
        self.calibration_duration = 10.0  # 10 seconds calibration
        self.is_calibrated = False
        
        # Live time axis
        self.session_start_time = None
        
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
        
        # Individual sensor data buffers for detailed plotting
        self.individual_sensor_buffers = {}
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            self.individual_sensor_buffers[sensor_id] = {
                'X': [], 'Y': [], 'Z': [], 'time': []
            }
        
        # Locks for thread safety
        self.ft_lock = threading.Lock()
        self.stretchmagtec_lock = threading.Lock()
    
    def start_sensors(self):
        """Start sensor reading threads."""
        if self.running:
            return
        
        # Reset calibration state for fresh start
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_start_time = None
        self.offsets = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        # Reset session start time for live time axis
        self.session_start_time = time.time()
        
        self.running = True
        
        # Start FT sensor thread
        self.ft_thread = threading.Thread(target=self._ft_sensor_loop, daemon=True)
        self.ft_thread.start()
        
        # Start StretchMagTec sensor thread
        self.stretchmagtec_thread = threading.Thread(target=self._stretchmagtec_sensor_loop, daemon=True)
        self.stretchmagtec_thread.start()
        
        print("Sensors started successfully - Fresh calibration will begin")
    
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
        try:
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
            self.ft_ser.read_until(STARTBYTES)
            data = self.ft_ser.read_until(STARTBYTES)
            dataArray = bytearray(data)
            dataArray = STARTBYTES + dataArray[:-2]
            
            if not self._crc_check(dataArray):
                print("CRC ERROR on ZeroRef")
                return
            
            zeroRef = self._force_from_serial_message(dataArray)
            
            while self.running:
                data = self.ft_ser.read_until(STARTBYTES)
                dataArray = bytearray(data)
                dataArray = STARTBYTES + dataArray[:-2]
                
                if not self._crc_check(dataArray):
                    continue
                
                raw_force = self._force_from_serial_message(dataArray, zeroRef)
                ft_cleaned = [0 if abs(val) < FT_NOISE_THRESHOLD else val for val in raw_force]
                
                with self.ft_lock:
                    self.ft_data[:] = ft_cleaned
                    
                # Add to buffer for plotting
                current_time = time.time()
                if len(self.time_buffer) >= self.max_buffer_size:
                    self.ft_buffer.pop(0)
                    self.time_buffer.pop(0)
                
                self.ft_buffer.append(ft_cleaned.copy())
                self.time_buffer.append(current_time)
                
        except Exception as e:
            print(f"FT Sensor error: {e}")
        finally:
            if self.ft_ser:
                try:
                    self.ft_ser.close()
                except:
                    pass
    
    def _stretchmagtec_sensor_loop(self):
        """StretchMagTec sensor reading loop."""
        try:
            self.stretchmagtec_ser = serial.Serial(STRETCHMAGTEC_PORT, STRETCHMAGTEC_BAUD, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            while self.running:
                if self.stretchmagtec_ser.in_waiting > 0:
                    line = self.stretchmagtec_ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:  # Only process non-empty lines
                        sensor_values = self._parse_stretchmagtec_line(line)
                        
                        if sensor_values is not None:
                            # Apply offset correction if calibrated
                            if self.is_calibrated:
                                sensor_values = sensor_values - self.offsets
                            
                            # Handle calibration
                            current_time = time.time()
                            if not self.is_calibrated:
                                if self.calibration_start_time is None:
                                    self.calibration_start_time = current_time
                                    print("Starting 10-second calibration...")
                                
                                # Collect calibration samples
                                self.calibration_samples.append(sensor_values.copy())
                                
                                # Check if calibration is complete
                                if current_time - self.calibration_start_time >= self.calibration_duration:
                                    self.calculate_offsets()
                                    self.is_calibrated = True
                                    print("Calibration complete! Offsets calculated.")
                            
                            with self.stretchmagtec_lock:
                                self.stretchmagtec_data[:, :] = sensor_values
                                
                            # Add to buffer for plotting
                            current_time = time.time()
                            if len(self.stretchmagtec_buffer) >= self.max_buffer_size:
                                self.stretchmagtec_buffer.pop(0)
                                self.time_buffer.pop(0)
                                
                            self.stretchmagtec_buffer.append(sensor_values.copy())
                            self.time_buffer.append(current_time)
                            
                            # Calculate Hz for each sensor
                            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                if sensor_values[sensor_id, 0] != 0 or sensor_values[sensor_id, 1] != 0 or sensor_values[sensor_id, 2] != 0:
                                    self.sensor_hz_counts[sensor_id] += 1
                            
                            if current_time - self.last_hz_time >= 1.0:  # Update every second
                                for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                    if self.sensor_hz_counts[sensor_id] > 0:
                                        self.sensor_hz_values[sensor_id] = self.sensor_hz_counts[sensor_id] / (current_time - self.last_hz_time)
                                    else:
                                        self.sensor_hz_values[sensor_id] = 0.0
                                    self.sensor_hz_counts[sensor_id] = 0
                                self.last_hz_time = current_time
                            
                            # Store individual sensor data
                            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                sensor_buffer = self.individual_sensor_buffers[sensor_id]
                                
                                # Add new data
                                sensor_buffer['X'].append(sensor_values[sensor_id, 0])
                                sensor_buffer['Y'].append(sensor_values[sensor_id, 1])
                                sensor_buffer['Z'].append(sensor_values[sensor_id, 2])
                                sensor_buffer['time'].append(current_time)
                                
                                # Keep buffer size manageable
                                if len(sensor_buffer['time']) > self.max_buffer_size:
                                    sensor_buffer['X'].pop(0)
                                    sensor_buffer['Y'].pop(0)
                                    sensor_buffer['Z'].pop(0)
                                    sensor_buffer['time'].pop(0)
                        
        except Exception as e:
            print(f"StretchMagTec Sensor error: {e}")
        finally:
            if self.stretchmagtec_ser:
                try:
                    self.stretchmagtec_ser.close()
                except:
                    pass
    
    def calculate_offsets(self):
        """Calculate offsets from calibration samples."""
        if not self.calibration_samples:
            return
        
        # Convert to numpy array
        calibration_data = np.array(self.calibration_samples)
        
        # Calculate mean offset for each sensor and axis
        self.offsets = np.mean(calibration_data, axis=0)
        
        print(f"Calculated offsets for {len(self.calibration_samples)} samples:")
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            print(f"S{sensor_id+1}: X={self.offsets[sensor_id, 0]:.2f}, Y={self.offsets[sensor_id, 1]:.2f}, Z={self.offsets[sensor_id, 2]:.2f}")
        
        # Clear calibration samples to save memory
        self.calibration_samples = []

    def _parse_stretchmagtec_line(self, line):
        """Parse StretchMagTec sensor line data."""
        try:
            # Handle different possible formats
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
                    return sensor_values
                else:
                    return None
            
            # Try original format: "S1: X=1234 Y=5678 Z=9012 | S2: X=2345 Y=6789 Z=0123 | ..."
            elif ' | ' in line:
                sensor_parts = line.split(' | ')
                if len(sensor_parts) >= STRETCHMAGTEC_SENSORS:
                    for i, sensor_part in enumerate(sensor_parts[:STRETCHMAGTEC_SENSORS]):
                        # Parse format like "S1: X=1234 Y=5678 Z=9012"
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

                    if np.any(sensor_values != 0):
                        return sensor_values
                    else:
                        return None
            
            # Try format: individual sensor data without separators
            elif 'S' in line and ':' in line:
                # Look for patterns like "S1: X=1234 Y=5678 Z=9012"
                import re
                pattern = r'S(\d+):\s*X=([-\d.]+)\s*Y=([-\d.]+)\s*Z=([-\d.]+)'
                matches = re.findall(pattern, line)
                
                for match in matches:
                    sensor_id = int(match[0]) - 1  # Convert to 0-based index
                    if 0 <= sensor_id < STRETCHMAGTEC_SENSORS:
                        try:
                            sensor_values[sensor_id, 0] = float(match[1])  # X
                            sensor_values[sensor_id, 1] = float(match[2])  # Y
                            sensor_values[sensor_id, 2] = float(match[3])  # Z
                        except ValueError:
                            continue
            
            # Try format: space-separated values (fallback)
            else:
                # Try to parse as space-separated numbers
                try:
                    values = line.split()
                    if len(values) >= STRETCHMAGTEC_SENSORS * STRETCHMAGTEC_CHANNELS:
                        for i in range(STRETCHMAGTEC_SENSORS):
                            for j in range(STRETCHMAGTEC_CHANNELS):
                                idx = i * STRETCHMAGTEC_CHANNELS + j
                                if idx < len(values):
                                    try:
                                        sensor_values[i, j] = float(values[idx])
                                    except ValueError:
                                        sensor_values[i, j] = 0
                except Exception:
                    pass
            
            # Apply threshold filter
            sensor_values[(sensor_values >= -STRETCHMAGTEC_THRESHOLD) & (sensor_values <= STRETCHMAGTEC_THRESHOLD)] = 0
            
            # Check if we got any valid data
            if np.any(sensor_values != 0):
                return sensor_values
            else:
                return None
            
        except Exception as e:
            print(f"Error parsing StretchMagTec data: {e}")
            return None
    
    def _force_from_serial_message(self, serialMessage, zeroRef=[0,0,0,0,0,0]):
        forceTorque = [0,0,0,0,0,0]
        forceTorque[0] = int.from_bytes(serialMessage[2:4], byteorder='little', signed=True)/100 - zeroRef[0]
        forceTorque[1] = int.from_bytes(serialMessage[4:6], byteorder='little', signed=True)/100 - zeroRef[1]
        forceTorque[2] = int.from_bytes(serialMessage[6:8], byteorder='little', signed=True)/100 - zeroRef[2]
        forceTorque[3] = int.from_bytes(serialMessage[8:10], byteorder='little', signed=True)/1000 - zeroRef[3]
        forceTorque[4] = int.from_bytes(serialMessage[10:12], byteorder='little', signed=True)/1000 - zeroRef[4]
        forceTorque[5] = int.from_bytes(serialMessage[12:14], byteorder='little', signed=True)/1000 - zeroRef[5]
        return [round(val, 3) for val in forceTorque]

    def _crc_check(self, serialMessage):
        crc = int.from_bytes(serialMessage[14:16], byteorder='little', signed=False)
        crcCalc = libscrc.modbus(serialMessage[0:14])
        return crc == crcCalc
    
    def get_ft_data(self):
        """Get current FT sensor data."""
        with self.ft_lock:
            return self.ft_data.copy()
    
    def get_stretchmagtec_data(self):
        """Get current StretchMagTec sensor data."""
        with self.stretchmagtec_lock:
            return self.stretchmagtec_data.copy()
    
    def get_plot_data(self):
        """Get data for plotting."""
        ft_data = self.ft_buffer.copy() if self.ft_buffer else []
        stretchmagtec_data = self.stretchmagtec_buffer.copy() if self.stretchmagtec_buffer else []
        time_data = self.time_buffer.copy() if self.time_buffer else []
        
        
        return ft_data, stretchmagtec_data, time_data

class ModelPredictor:
    """Handles model loading and predictions."""
    
    def __init__(self, model_dir=None, sensor_reader=None):
        if model_dir is None:
            model_dir = MODELS_DIR
        self.model_dir = Path(model_dir)
        self.sensor_reader = sensor_reader  # Reference to get buffer for feature extraction
        self.contact_classifier = None
        self.contact_scaler = None
        self.ft_mappers = {}
        self.ft_scalers = {}
        self.ft_output_scalers = {}
        self.models_loaded = False
        self.use_spatial_features = False  # Flag for which feature type to use
        self.use_normalized_features = False  # Flag for normalized features
        self.baseline = None  # Baseline for spatial features
        self.contact_threshold = 100  # Threshold for normalization
        
    def load_models(self):
        """Load trained models from disk."""
        try:
            print(f"Looking for models in: {self.model_dir}")
            
            # Priority 1: Try normalized spatial classifier (best - force-independent)
            normalized_model_path = self.model_dir / "skin_1_normalized_spatial_classifier.joblib"
            normalized_scaler_path = self.model_dir / "skin_1_normalized_spatial_scaler.joblib"
            
            if normalized_model_path.exists() and normalized_scaler_path.exists():
                self.contact_classifier = joblib.load(normalized_model_path)
                self.contact_scaler = joblib.load(normalized_scaler_path)
                self.use_spatial_features = True
                self.use_normalized_features = True
                print("✅ Normalized spatial classifier loaded (force-independent, ~85-90% accuracy)")
            
            # Priority 2: Try regular spatial classifier
            elif (self.model_dir / "skin_1_spatial_classifier.joblib").exists():
                spatial_model_path = self.model_dir / "skin_1_spatial_classifier.joblib"
                spatial_scaler_path = self.model_dir / "skin_1_spatial_scaler.joblib"
                
                self.contact_classifier = joblib.load(spatial_model_path)
                self.contact_scaler = joblib.load(spatial_scaler_path)
                self.use_spatial_features = True
                self.use_normalized_features = False
                print("✅ Spatial contact classifier loaded (80% accuracy)")
            
            # Priority 3: Fallback to original statistical classifier
            else:
                contact_model_path = self.model_dir / CONTACT_CLASSIFIER_MODEL
                contact_scaler_path = self.model_dir / CONTACT_SCALER_MODEL
                
                if contact_model_path.exists() and contact_scaler_path.exists():
                    self.contact_classifier = joblib.load(contact_model_path)
                    self.contact_scaler = joblib.load(contact_scaler_path)
                    self.use_spatial_features = False
                    self.use_normalized_features = False
                    print("✅ Statistical contact classifier loaded")
                else:
                    print(f"⚠️ No contact classifier models found")
                    return False
            
            # Load FT mapping models (using filenames from config)
            for component in ['fx', 'fy', 'fz']:
                model_name = f"{SENSOR_NAME}_ft_mapping_{component}.joblib"
                scaler_name = f"{SENSOR_NAME}_ft_mapping_scaler_{component}.joblib"
                output_scaler_name = f"{SENSOR_NAME}_ft_mapping_output_scaler_{component}.joblib"
                
                mapper_path = self.model_dir / model_name
                scaler_path = self.model_dir / scaler_name
                output_scaler_path = self.model_dir / output_scaler_name
                
                if mapper_path.exists() and scaler_path.exists() and output_scaler_path.exists():
                    self.ft_mappers[component] = joblib.load(mapper_path)
                    self.ft_scalers[component] = joblib.load(scaler_path)
                    self.ft_output_scalers[component] = joblib.load(output_scaler_path)
                    print(f"✅ FT mapping {component} loaded successfully")
                else:
                    print(f"⚠️ FT mapping {component} models not found")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def extract_spatial_features(self, sensor_window, baseline=None):
        """
        Extract spatial features from sensor data (same as training).
        
        Args:
            sensor_window: Sensor data (n_samples, 15, 3)
            baseline: Baseline to subtract (15, 3) or None
        
        Returns:
            features: Spatial feature vector (56,)
        """
        # Get mean activation over time window
        mean_activation = np.mean(sensor_window, axis=0)  # (15, 3)
        
        # Subtract baseline if provided
        if baseline is not None:
            delta_activation = mean_activation - baseline
        else:
            delta_activation = mean_activation
        
        # Flatten raw delta values
        raw_features = delta_activation.flatten()  # 45 features
        
        # Compute spatial features (sensor grid is 3 rows × 5 cols)
        sensor_grid = delta_activation.reshape(3, 5, 3)
        
        # Calculate center of pressure for each channel
        cop_features = []
        for ch in range(3):
            channel_grid = np.abs(sensor_grid[:, :, ch])
            total_activation = np.sum(channel_grid)
            
            if total_activation > 0:
                row_indices, col_indices = np.meshgrid(range(3), range(5), indexing='ij')
                cop_row = np.sum(channel_grid * row_indices) / total_activation
                cop_col = np.sum(channel_grid * col_indices) / total_activation
                cop_features.extend([cop_row, cop_col])
            else:
                cop_features.extend([0, 0])
        
        # Add activation concentration (variance)
        for ch in range(3):
            channel_grid = np.abs(sensor_grid[:, :, ch])
            cop_features.append(np.std(channel_grid))
        
        # Add maximum activation location (argmax for Z channel)
        z_grid = np.abs(sensor_grid[:, :, 2])
        max_idx = np.unravel_index(np.argmax(z_grid), z_grid.shape)
        cop_features.extend([max_idx[0], max_idx[1]])
        
        # Combine all features
        all_features = np.concatenate([raw_features, cop_features])
        
        return all_features
    
    def extract_normalized_spatial_features(self, sensor_window, baseline=None):
        """
        Extract NORMALIZED spatial features (force-independent).
        
        Args:
            sensor_window: Sensor data (n_samples, 15, 3)
            baseline: Baseline to subtract (15, 3) or None
        
        Returns:
            features: Normalized spatial feature vector (56,)
        """
        # Get mean activation over time window
        mean_activation = np.mean(sensor_window, axis=0)  # (15, 3)
        
        # Subtract baseline if provided
        if baseline is not None:
            delta_activation = mean_activation - baseline
        else:
            delta_activation = mean_activation
        
        # Calculate total activation magnitude
        total_magnitude = np.linalg.norm(delta_activation)
        
        if total_magnitude > self.contact_threshold:
            # NORMALIZE by magnitude (makes it force-independent)
            normalized_activation = delta_activation / total_magnitude
        else:
            # No contact - return zeros
            normalized_activation = np.zeros_like(delta_activation)
        
        # Flatten normalized values
        raw_features = normalized_activation.flatten()  # 45 features
        
        # Compute spatial features on NORMALIZED data
        sensor_grid = normalized_activation.reshape(3, 5, 3)
        
        # Calculate center of pressure for each channel
        cop_features = []
        for ch in range(3):
            channel_grid = np.abs(sensor_grid[:, :, ch])
            total_activation = np.sum(channel_grid)
            
            if total_activation > 0:
                row_indices, col_indices = np.meshgrid(range(3), range(5), indexing='ij')
                cop_row = np.sum(channel_grid * row_indices) / total_activation
                cop_col = np.sum(channel_grid * col_indices) / total_activation
                cop_features.extend([cop_row, cop_col])
            else:
                cop_features.extend([0, 0])
        
        # Add activation concentration (variance)
        for ch in range(3):
            channel_grid = np.abs(sensor_grid[:, :, ch])
            cop_features.append(np.std(channel_grid))
        
        # Add maximum activation location (argmax for Z channel)
        z_grid = np.abs(sensor_grid[:, :, 2])
        max_idx = np.unravel_index(np.argmax(z_grid), z_grid.shape)
        cop_features.extend([max_idx[0], max_idx[1]])
        
        # Combine all features
        all_features = np.concatenate([raw_features, cop_features])
        
        return all_features
    
    def extract_statistical_features(self, sensor_window):
        """
        Extract statistical features from sensor window (original method).
        
        Args:
            sensor_window: Sensor data array, shape (n_samples, 15, 3) or (15, 3)
        
        Returns:
            features: 1D feature vector (315,)
        """
        # If single sample, get buffer
        if sensor_window.ndim == 2:
            stretchmagtec_buffer = getattr(self.sensor_reader, 'stretchmagtec_buffer', [])
            if len(stretchmagtec_buffer) >= 10:
                sensor_window = np.array(stretchmagtec_buffer[-10:])
            else:
                return np.zeros(315)
        
        features = []
        
        # Extract 7 statistical features for each of 15 sensors × 3 channels
        for sensor_id in range(15):
            for channel in range(3):
                channel_data = sensor_window[:, sensor_id, channel]
                
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.ptp(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                ])
        
        return np.array(features)
    
    def extract_contact_features(self, sensor_window):
        """
        Extract features based on which classifier is loaded.
        
        Args:
            sensor_window: Sensor data array, shape (n_samples, 15, 3) or (15, 3)
        
        Returns:
            features: Feature vector
        """
        # If single sample, get buffer
        if sensor_window.ndim == 2:
            stretchmagtec_buffer = getattr(self.sensor_reader, 'stretchmagtec_buffer', [])
            if len(stretchmagtec_buffer) >= 10:
                sensor_window = np.array(stretchmagtec_buffer[-10:])
            else:
                # Return zeros of appropriate size
                return np.zeros(56 if self.use_spatial_features else 315)
        
        # Calculate baseline if not set (use first 50 samples as baseline)
        if self.use_spatial_features and self.baseline is None:
            stretchmagtec_buffer = getattr(self.sensor_reader, 'stretchmagtec_buffer', [])
            if len(stretchmagtec_buffer) >= 50:
                self.baseline = np.mean(stretchmagtec_buffer[:50], axis=0)
        
        # Extract features based on classifier type
        if self.use_normalized_features:
            return self.extract_normalized_spatial_features(sensor_window, self.baseline)
        elif self.use_spatial_features:
            return self.extract_spatial_features(sensor_window, self.baseline)
        else:
            return self.extract_statistical_features(sensor_window)
    
    def predict_contact_point(self, stretchmagtec_data):
        """Predict contact point from StretchMagTec data."""
        if not self.models_loaded or self.contact_classifier is None:
            return "No Model", 0.0
        
        try:
            # Extract statistical features (same as training)
            features = self.extract_contact_features(stretchmagtec_data)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.contact_scaler.transform(features)
            
            # Predict
            prediction = self.contact_classifier.predict(features_scaled)[0]
            probabilities = self.contact_classifier.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            # Print detailed error info for debugging
            error_msg = f"Contact prediction error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Also print buffer status
            if self.sensor_reader:
                buffer_len = len(self.sensor_reader.stretchmagtec_buffer)
                print(f"Debug: Buffer has {buffer_len} samples, need at least 10")
            else:
                print(f"Debug: No sensor_reader available")
            
            return "Error", 0.0
    
    def get_contact_probabilities(self, stretchmagtec_data):
        """Get probability distribution for all contact classes."""
        if not self.models_loaded or self.contact_classifier is None:
            return {}
        
        try:
            # Extract statistical features (same as training)
            features = self.extract_contact_features(stretchmagtec_data)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.contact_scaler.transform(features)
            
            # Get probabilities for all classes
            probabilities = self.contact_classifier.predict_proba(features_scaled)[0]
            classes = self.contact_classifier.classes_
            
            # Create dictionary of class: probability
            prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            return prob_dict
            
        except Exception as e:
            print(f"Probability extraction error: {e}")
            return {}
    
    def predict_ft_forces(self, stretchmagtec_data):
        """Predict FT forces from StretchMagTec data."""
        if not self.models_loaded:
            return {"fx": 0.0, "fy": 0.0, "fz": 0.0}
        
        predicted_forces = {}
        
        try:
            # FT mapping uses STATISTICAL features (315), not spatial features (56)
            # This is because FT mapping models were trained with statistical features
            features = self.extract_statistical_features(stretchmagtec_data)
            features = features.reshape(1, -1)
            
            for component in ['fx', 'fy', 'fz']:
                if component in self.ft_mappers:
                    # Scale features
                    features_scaled = self.ft_scalers[component].transform(features)
                    
                    # Predict
                    prediction_scaled = self.ft_mappers[component].predict(features_scaled)
                    
                    # Inverse transform output
                    prediction = self.ft_output_scalers[component].inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
                    predicted_forces[component] = prediction
                else:
                    predicted_forces[component] = 0.0
            
            return predicted_forces
            
        except Exception as e:
            print(f"FT prediction error: {e}")
            return {"fx": 0.0, "fy": 0.0, "fz": 0.0}

class GridVisualizationWindow:
    """Popup window showing the contact grid with real-time predictions."""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Contact Grid Visualization")
        self.window.geometry("900x700")
        
        # Grid parameters from config
        self.grid_rows = GRID_ROWS
        self.grid_cols = GRID_COLS
        self.offsets = list(GRID_OFFSETS.keys())  # 9 offset positions
        self.total_cells = self.grid_rows * self.grid_cols * len(self.offsets)
        
        # Contact probabilities (will be updated in real-time)
        self.contact_probabilities = {}  # {(row, col, offset): probability}
        self.init_probabilities()
        
        # Create matplotlib figure
        self.create_visualization()
        
        # Update timer
        self.update_interval = 100  # ms
        self.running = True
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def init_probabilities(self):
        """Initialize all contact probabilities to zero."""
        for row in range(1, self.grid_rows + 1):
            for col in range(1, self.grid_cols + 1):
                for offset in self.offsets:
                    self.contact_probabilities[(row, col, offset)] = 0.0
    
    def create_visualization(self):
        """Create the grid visualization."""
        # Create figure
        self.fig = Figure(figsize=(10, 7), dpi=90)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info label
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, 
                                     text=f"Grid: {self.grid_rows}x{self.grid_cols} | Total cells: {self.total_cells} | 9-point offsets per cell",
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
        """Draw the contact grid with current probabilities."""
        self.ax.clear()
        
        # Calculate cell sizes
        offset_spacing = 0.3  # Space between offset points
        main_cell_size = 3.0  # Size of main grid cell
        offset_cell_size = (main_cell_size - 2 * offset_spacing) / 3  # Size of each 3x3 offset grid
        
        # Offset positions in 3x3 grid (nw, n, ne, w, center, e, sw, s, se)
        offset_positions_3x3 = {
            'nw': (0, 2), 'n': (1, 2), 'ne': (2, 2),
            'w':  (0, 1), 'center': (1, 1), 'e':  (2, 1),
            'sw': (0, 0), 's': (1, 0), 'se': (2, 0)
        }
        
        # Draw each main grid cell
        for row in range(1, self.grid_rows + 1):
            for col in range(1, self.grid_cols + 1):
                # Calculate main cell position
                main_x = (col - 1) * (main_cell_size + 0.5)
                main_y = (self.grid_rows - row) * (main_cell_size + 0.5)
                
                # Draw main cell border
                main_rect = Rectangle((main_x, main_y), main_cell_size, main_cell_size,
                                      fill=False, edgecolor='black', linewidth=2)
                self.ax.add_patch(main_rect)
                
                # Add main cell label
                self.ax.text(main_x + main_cell_size/2, main_y + main_cell_size + 0.15,
                           f"({row},{col})", ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Draw 9-point offset grid inside main cell
                for offset, (ox, oy) in offset_positions_3x3.items():
                    # Calculate offset cell position
                    offset_x = main_x + offset_spacing + ox * offset_cell_size
                    offset_y = main_y + offset_spacing + oy * offset_cell_size
                    
                    # Get probability for this cell
                    prob = self.contact_probabilities.get((row, col, offset), 0.0)
                    
                    # Color based on probability (white = 0, red = 1)
                    color = plt.cm.Reds(prob)
                    
                    # Draw offset cell
                    offset_rect = Rectangle((offset_x, offset_y), offset_cell_size, offset_cell_size,
                                           fill=True, facecolor=color, edgecolor='gray', linewidth=0.5)
                    self.ax.add_patch(offset_rect)
                    
                    # Add offset label if probability is high
                    if prob > 0.1:
                        label_text = offset if offset != 'center' else 'C'
                        self.ax.text(offset_x + offset_cell_size/2, offset_y + offset_cell_size/2,
                                   label_text, ha='center', va='center', fontsize=6,
                                   color='white' if prob > 0.5 else 'black')
        
        # Set axis properties (FIXED - don't recalculate)
        x_max = self.grid_cols * (main_cell_size + 0.5)
        y_max = self.grid_rows * (main_cell_size + 0.5) + 0.5
        
        self.ax.set_xlim(-0.5, x_max)
        self.ax.set_ylim(-0.5, y_max)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        
        # Add colorbar (only once)
        if not hasattr(self, 'colorbar'):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            self.colorbar = self.fig.colorbar(sm, ax=self.ax, orientation='horizontal', 
                                              pad=0.05, fraction=0.046, label='Contact Probability')
        
        self.canvas.draw_idle()  # Use draw_idle() instead of draw() for better performance
    
    def update_predictions(self, position_id, offset, probabilities):
        """
        Update contact probabilities based on prediction.
        
        Args:
            position_id: Main grid position (e.g., 11, 12, etc.)
            offset: Predicted offset ('center', 'nw', etc.)
            probabilities: Dictionary of {full_position: probability} for all 135 classes
                          Format: {"11_center": 0.85, "11_n": 0.10, "12_center": 0.03, ...}
        """
        # Reset all probabilities to zero (show only current prediction)
        self.init_probabilities()
        
        # Parse probabilities which now include full position info
        if probabilities:
            for full_class, prob in probabilities.items():
                # Parse "11_center" → row=1, col=1, offset="center"
                if '_' in full_class:
                    parts = full_class.split('_', 1)
                    try:
                        pos_id = int(parts[0])
                        offset_key = parts[1]
                        row = pos_id // 10
                        col = pos_id % 10
                        
                        # Update this specific cell
                        self.contact_probabilities[(row, col, offset_key)] = prob
                    except (ValueError, IndexError):
                        continue
        else:
            # Fallback: just highlight the predicted position
            row = position_id // 10
            col = position_id % 10
            self.contact_probabilities[(row, col, offset)] = 1.0
        
        # Update prediction label
        row = position_id // 10
        col = position_id % 10
        full_class = f"{position_id}_{offset}"
        conf_value = probabilities.get(full_class, 0.0) if probabilities else 0.0
        self.prediction_label.config(text=f"Prediction: {full_class} - Pos({row},{col})-{offset.upper()} (conf: {conf_value:.2f})")
        
        # Redraw grid
        self.draw_grid()
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        self.window.destroy()

class RealTimePredictorGUI:
    """Main GUI application for real-time StretchMagTec prediction."""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = MODELS_DIR
        
        self.root = tk.Tk()
        self.root.title("StretchMagTec 3x5 Real-Time Predictor")
        self.root.geometry("1400x900")
        
        # Grid visualization window
        self.grid_viz_window = None
        
        # Initialize components
        self.sensor_reader = SensorReader()
        self.model_predictor = ModelPredictor(model_dir, self.sensor_reader)
        
        # GUI update control
        self.update_running = False
        self.update_interval = 50  # ms
        
        # Create GUI elements
        self.create_widgets()
        
        # Load models
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
        
        # Create labels for all 15 sensors with clickable functionality
        self.stretchmagtec_labels = []
        self.sensor_frames = []
        
        # Add instruction label
        instruction_label = ttk.Label(scrollable_frame, text="🖱️ Click sensors to select for plotting", 
                                    font=("Arial", 10, "italic"), foreground="blue")
        instruction_label.pack(pady=5)
        
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            sensor_frame = ttk.LabelFrame(scrollable_frame, text=f"Sensor {sensor_id + 1} (Click to plot)")
            sensor_frame.pack(fill=tk.X, padx=2, pady=2)
            
            # Make sensor frame clickable with visual feedback
            sensor_frame.bind("<Button-1>", lambda e, s_id=sensor_id: self.toggle_sensor_selection(s_id))
            sensor_frame.bind("<Enter>", lambda e, frame=sensor_frame: frame.configure(relief="raised"))
            sensor_frame.bind("<Leave>", lambda e, frame=sensor_frame: frame.configure(relief="flat"))
            
            sensor_labels = []
            for channel, name in enumerate(['X', 'Y', 'Z']):
                label = ttk.Label(sensor_frame, text=f"{name}: 0", font=("Courier", 9))
                label.pack(anchor=tk.W, padx=2)
                # Make labels clickable too
                label.bind("<Button-1>", lambda e, s_id=sensor_id: self.toggle_sensor_selection(s_id))
                sensor_labels.append(label)
            
            # Add Hz label for this sensor
            hz_label = ttk.Label(sensor_frame, text="Hz: 0.0", font=("Courier", 9, "bold"), foreground="blue")
            hz_label.pack(anchor=tk.W, padx=2)
            # Make Hz label clickable too
            hz_label.bind("<Button-1>", lambda e, s_id=sensor_id: self.toggle_sensor_selection(s_id))
            sensor_labels.append(hz_label)  # Add Hz label to the sensor_labels list
            
            self.stretchmagtec_labels.append(sensor_labels)
            self.sensor_frames.append(sensor_frame)
        
        # Right column - Predictions
        right_frame = ttk.LabelFrame(data_frame, text="Predictions")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Contact point prediction
        contact_frame = ttk.LabelFrame(right_frame, text="Contact Point Prediction")
        contact_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.contact_label = ttk.Label(contact_frame, text="Contact: Unknown", font=("Arial", 14, "bold"))
        self.contact_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(contact_frame, text="Confidence: 0.0%", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
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
    
    def toggle_sensor_selection(self, sensor_id):
        """Toggle sensor selection for plotting."""
        print(f"Toggling sensor {sensor_id}, current selection: {self.selected_sensors}")
        
        if sensor_id in self.selected_sensors:
            self.selected_sensors.remove(sensor_id)
            # Reset button to original color
            self.sensor_buttons[sensor_id].configure(bg=self.sensor_colors[sensor_id], relief='raised')
            print(f"Removed sensor {sensor_id}")
        else:
            self.selected_sensors.add(sensor_id)
            # Make button appear pressed/selected
            self.sensor_buttons[sensor_id].configure(bg=self.sensor_colors[sensor_id], relief='sunken')
            print(f"Added sensor {sensor_id}")
        
        print(f"New selection: {self.selected_sensors}")
        
        # Update plot immediately
        self.update_plots()
    
    def clear_sensor_selection(self):
        """Clear all sensor selections."""
        self.selected_sensors.clear()
        for i, btn in enumerate(self.sensor_buttons):
            btn.configure(bg=self.sensor_colors[i], relief='raised')
        self.update_plots()
    
    def load_models(self):
        """Load prediction models."""
        if self.model_predictor.load_models():
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
            
            # Start GUI update loop
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start sensors: {e}")
            self.status_label.config(text="Status: Sensor start failed", foreground="red")
    
    def stop_sensors(self):
        """Stop sensor reading and GUI updates."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Sensors stopped", foreground="orange")
    
    def toggle_grid_viz(self):
        """Toggle the grid visualization window."""
        if self.grid_viz_window and self.grid_viz_window.running:
            # Close existing window
            self.grid_viz_window.on_closing()
            self.grid_viz_window = None
            self.grid_viz_button.config(text="Show Grid Visualization")
        else:
            # Open new window
            self.grid_viz_window = GridVisualizationWindow(self.root)
            self.grid_viz_button.config(text="Hide Grid Visualization")
    
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
                    color = "red" if abs(value) > STRETCHMAGTEC_THRESHOLD else "black"
                    self.stretchmagtec_labels[sensor_id][channel_id].config(
                        text=f"{channel_name}: {value:6.0f}", foreground=color
                    )
            
            # Update Hz display
            self.update_hz_display()
            
            # Make predictions
            contact_point, confidence = self.model_predictor.predict_contact_point(stretchmagtec_data)
            predicted_forces = self.model_predictor.predict_ft_forces(stretchmagtec_data)
            
            # Get class probabilities for grid visualization
            contact_probabilities = self.model_predictor.get_contact_probabilities(stretchmagtec_data)
            
            # Update prediction display
            contact_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            self.contact_label.config(text=f"Contact: {contact_point}", foreground=contact_color)
            self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%", foreground=contact_color)
            
            # Update grid visualization if open
            if self.grid_viz_window and self.grid_viz_window.running:
                # Parse contact_point to extract position_id and offset
                # Format: "11_center", "23_nw", etc.
                if '_' in contact_point:
                    parts = contact_point.split('_', 1)
                    try:
                        position_id = int(parts[0])
                        offset = parts[1]
                        self.grid_viz_window.update_predictions(position_id, offset, contact_probabilities)
                    except (ValueError, IndexError):
                        pass  # Invalid format, skip grid update
            
            # Update force predictions
            force_names = ["fx", "fy", "fz"]
            display_names = ["Fx (N)", "Fy (N)", "Fz (N)"]
            for i, (force_name, display_name) in enumerate(zip(force_names, display_names)):
                pred_value = predicted_forces.get(force_name, 0.0)
                color = "red" if abs(pred_value) > 1.0 else "black"
                self.force_pred_labels[i].config(text=f"{display_name}: {pred_value:7.3f}", foreground=color)
            
            # Update plots (if implemented)
            if hasattr(self, 'update_plots'):
                self.update_plots()
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_gui)
    
    def show_sensor_plot(self, sensor_id):
        """Show individual sensor plot in a new window."""
        try:
            # Get sensor data
            sensor_buffer = self.sensor_reader.individual_sensor_buffers[sensor_id]
            
            if not sensor_buffer['time']:
                messagebox.showinfo("No Data", f"No data available for Sensor {sensor_id + 1}")
                return
            
            # Create new window
            sensor_window = tk.Toplevel(self.root)
            sensor_window.title(f"Sensor {sensor_id + 1} - Detailed Plot")
            sensor_window.geometry("800x600")
            
            # Create matplotlib figure
            fig = Figure(figsize=(10, 8), dpi=80)
            canvas = FigureCanvasTkAgg(fig, sensor_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create subplots
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            
            # Convert absolute time to relative time
            start_time = sensor_buffer['time'][0]
            relative_time = [(t - start_time) for t in sensor_buffer['time']]
            
            # Plot X channel
            ax1.plot(relative_time, sensor_buffer['X'], 'r-', linewidth=1.5, alpha=0.8)
            ax1.set_title(f"Sensor {sensor_id + 1} - X Channel")
            ax1.set_ylabel("Magnetic Field X")
            ax1.grid(True, alpha=0.3)
            
            # Plot Y channel
            ax2.plot(relative_time, sensor_buffer['Y'], 'g-', linewidth=1.5, alpha=0.8)
            ax2.set_title(f"Sensor {sensor_id + 1} - Y Channel")
            ax2.set_ylabel("Magnetic Field Y")
            ax2.grid(True, alpha=0.3)
            
            # Plot Z channel
            ax3.plot(relative_time, sensor_buffer['Z'], 'b-', linewidth=1.5, alpha=0.8)
            ax3.set_title(f"Sensor {sensor_id + 1} - Z Channel")
            ax3.set_ylabel("Magnetic Field Z")
            ax3.set_xlabel("Time (s)")
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Samples: {len(sensor_buffer['time'])}\n"
            stats_text += f"X: min={min(sensor_buffer['X']):.1f}, max={max(sensor_buffer['X']):.1f}, mean={np.mean(sensor_buffer['X']):.1f}\n"
            stats_text += f"Y: min={min(sensor_buffer['Y']):.1f}, max={max(sensor_buffer['Y']):.1f}, mean={np.mean(sensor_buffer['Y']):.1f}\n"
            stats_text += f"Z: min={min(sensor_buffer['Z']):.1f}, max={max(sensor_buffer['Z']):.1f}, mean={np.mean(sensor_buffer['Z']):.1f}"
            
            fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig.tight_layout()
            canvas.draw()
            
            # Add update button
            update_button = ttk.Button(sensor_window, text="Update Plot", 
                                     command=lambda: self.update_sensor_plot(sensor_window, sensor_id))
            update_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sensor plot: {e}")
    
    def update_sensor_plot(self, window, sensor_id):
        """Update the sensor plot window."""
        try:
            # Get current sensor data
            sensor_buffer = self.sensor_reader.individual_sensor_buffers[sensor_id]
            
            if not sensor_buffer['time']:
                return
            
            # Find the canvas and update the plot
            for widget in window.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    canvas = widget
                    break
            
            fig = canvas.figure
            fig.clear()
            
            # Recreate plots with updated data
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            
            # Convert absolute time to relative time
            start_time = sensor_buffer['time'][0]
            relative_time = [(t - start_time) for t in sensor_buffer['time']]
            
            # Plot updated data
            ax1.plot(relative_time, sensor_buffer['X'], 'r-', linewidth=1.5, alpha=0.8)
            ax1.set_title(f"Sensor {sensor_id + 1} - X Channel")
            ax1.set_ylabel("Magnetic Field X")
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(relative_time, sensor_buffer['Y'], 'g-', linewidth=1.5, alpha=0.8)
            ax2.set_title(f"Sensor {sensor_id + 1} - Y Channel")
            ax2.set_ylabel("Magnetic Field Y")
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(relative_time, sensor_buffer['Z'], 'b-', linewidth=1.5, alpha=0.8)
            ax3.set_title(f"Sensor {sensor_id + 1} - Z Channel")
            ax3.set_ylabel("Magnetic Field Z")
            ax3.set_xlabel("Time (s)")
            ax3.grid(True, alpha=0.3)
            
            # Add updated statistics
            stats_text = f"Samples: {len(sensor_buffer['time'])}\n"
            stats_text += f"X: min={min(sensor_buffer['X']):.1f}, max={max(sensor_buffer['X']):.1f}, mean={np.mean(sensor_buffer['X']):.1f}\n"
            stats_text += f"Y: min={min(sensor_buffer['Y']):.1f}, max={max(sensor_buffer['Y']):.1f}, mean={np.mean(sensor_buffer['Y']):.1f}\n"
            stats_text += f"Z: min={min(sensor_buffer['Z']):.1f}, max={max(sensor_buffer['Z']):.1f}, mean={np.mean(sensor_buffer['Z']):.1f}"
            
            fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            print(f"Error updating sensor plot: {e}")

    def update_hz_display(self):
        """Update Hz display labels for each sensor."""
        try:
            # Get Hz values from sensor reader
            hz_values = self.sensor_reader.sensor_hz_values
            
            # Update each Hz label (Hz label is the 4th element in sensor_labels: X, Y, Z, Hz)
            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                hz_value = hz_values[sensor_id]
                self.stretchmagtec_labels[sensor_id][3].config(text=f"Hz: {hz_value:.1f}")
                
        except Exception as e:
            print(f"Error updating Hz display: {e}")

    def update_plots(self):
        """Update real-time plots with 4 separate plots."""
        try:
            ft_data, stretchmagtec_data, time_data = self.sensor_reader.get_plot_data()
            
            if not time_data:
                return
            
            # Clear all plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Convert absolute time to live relative time (starts from 0 when sensors start)
            if time_data and hasattr(self.sensor_reader, 'session_start_time') and self.sensor_reader.session_start_time:
                relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
            else:
                relative_time = []
            
            # Plot FT data (top-left)
            if ft_data and relative_time:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                # Ensure arrays have same length
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                for i in range(6):
                    self.ax1.plot(relative_time_trimmed, ft_array_trimmed[:, i], label=labels[i], color=colors[i], alpha=0.7)
                
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force/Torque")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.relim()  # Recalculate limits
                self.ax1.autoscale_view()  # Auto-scale the view
            
            # Plot X-axis data (top-right)
            if stretchmagtec_data and relative_time and self.selected_sensors:
                stretchmagtec_array = np.array(stretchmagtec_data)
                
                # Ensure arrays have same length
                min_len = min(len(relative_time), len(stretchmagtec_array))
                relative_time_trimmed = relative_time[:min_len]
                stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                
                for sensor_id in sorted(self.selected_sensors):
                    if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                        sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                        color = self.sensor_colors[sensor_id]
                        
                        # Scale for visualization
                        self.ax2.plot(relative_time_trimmed, sensor_data[:, 0] / PLOT_SCALE_FACTOR, 
                                    label=f'S{sensor_id+1}', color=color, alpha=0.8, 
                                    linestyle='-', linewidth=2.0)
                
                self.ax2.set_title(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                self.ax2.set_ylabel(f"Magnetic Field (/{PLOT_SCALE_FACTOR})")
                self.ax2.set_xlabel("Time (s)")
                self.ax2.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
                self.ax2.legend(loc='upper right', fontsize=8)
                self.ax2.grid(True, alpha=0.3)
            else:
                self.ax2.set_title("X-Axis: Select sensors to plot")
                self.ax2.set_ylabel("Magnetic Field")
                self.ax2.set_xlabel("Time (s)")
                self.ax2.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                            ha='center', va='center', transform=self.ax2.transAxes, fontsize=10)
                self.ax2.grid(True, alpha=0.3)
            
            # Plot Y-axis data (bottom-left)
            if stretchmagtec_data and relative_time and self.selected_sensors:
                stretchmagtec_array = np.array(stretchmagtec_data)
                
                # Ensure arrays have same length
                min_len = min(len(relative_time), len(stretchmagtec_array))
                relative_time_trimmed = relative_time[:min_len]
                stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                
                for sensor_id in sorted(self.selected_sensors):
                    if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                        sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                        color = self.sensor_colors[sensor_id]
                        
                        # Scale for visualization
                        self.ax3.plot(relative_time_trimmed, sensor_data[:, 1] / PLOT_SCALE_FACTOR, 
                                    label=f'S{sensor_id+1}', color=color, alpha=0.8, 
                                    linestyle='-', linewidth=2.0)
                
                self.ax3.set_title(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                self.ax3.set_ylabel(f"Magnetic Field (/{PLOT_SCALE_FACTOR})")
                self.ax3.set_xlabel("Time (s)")
                self.ax3.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
                self.ax3.legend(loc='upper right', fontsize=8)
                self.ax3.grid(True, alpha=0.3)
            else:
                self.ax3.set_title("Y-Axis: Select sensors to plot")
                self.ax3.set_ylabel("Magnetic Field")
                self.ax3.set_xlabel("Time (s)")
                self.ax3.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                            ha='center', va='center', transform=self.ax3.transAxes, fontsize=10)
                self.ax3.grid(True, alpha=0.3)
            
            # Plot Z-axis data (bottom-right)
            if stretchmagtec_data and relative_time and self.selected_sensors:
                stretchmagtec_array = np.array(stretchmagtec_data)
                
                # Ensure arrays have same length
                min_len = min(len(relative_time), len(stretchmagtec_array))
                relative_time_trimmed = relative_time[:min_len]
                stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                
                for sensor_id in sorted(self.selected_sensors):
                    if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                        sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                        color = self.sensor_colors[sensor_id]
                        
                        # Scale for visualization
                        self.ax4.plot(relative_time_trimmed, sensor_data[:, 2] / PLOT_SCALE_FACTOR, 
                                    label=f'S{sensor_id+1}', color=color, alpha=0.8, 
                                    linestyle='-', linewidth=2.0)
                
                self.ax4.set_title(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                self.ax4.set_ylabel(f"Magnetic Field (/{PLOT_SCALE_FACTOR})")
                self.ax4.set_xlabel("Time (s)")
                self.ax4.set_ylim(-PLOT_Y_LIMIT, PLOT_Y_LIMIT)
                self.ax4.legend(loc='upper right', fontsize=8)
                self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.set_title("Z-Axis: Select sensors to plot")
                self.ax4.set_ylabel("Magnetic Field")
                self.ax4.set_xlabel("Time (s)")
                self.ax4.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                            ha='center', va='center', transform=self.ax4.transAxes, fontsize=10)
                self.ax4.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Handle application closing."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        self.root.quit()
        self.root.destroy()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="StretchMagTec 3x5 Real-Time Predictor")
    parser.add_argument("--model_dir", default=str(MODELS_DIR), help=f"Directory containing trained models (default: {MODELS_DIR})")
    
    args = parser.parse_args()
    
    print("="*60)
    print("STRETCHMAGTEC 3x5 REAL-TIME PREDICTOR")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("="*60)
    
    # Create and run GUI
    app = RealTimePredictorGUI(args.model_dir)
    app.run()

if __name__ == "__main__":
    main()
