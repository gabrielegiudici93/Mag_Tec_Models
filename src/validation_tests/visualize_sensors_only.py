#!/usr/bin/env python3
"""
Real-Time Sensor Visualization - FT Sensor and StretchMagTec Sensors Only

This script provides real-time visualization of:
1. FT sensor readings (Fx, Fy, Fz, Tx, Ty, Tz)
2. StretchMagTec 3x5 sensor readings (15 sensors × 3 channels)

No model predictions, just pure sensor data visualization.

Usage:
    python3 visualize_sensors_only.py

Author: Gabriele Giudici
Date: 2025
"""

import os
import sys
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import serial
import minimalmodbus as mm
import libscrc
import glob
import ast
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *


def auto_detect_stretchmagtec_port():
    """
    Auto-detect StretchMagTec port by scanning available /dev/ttyACM* ports.
    Returns the first port that successfully opens and can read data.
    """
    # Get all available ACM ports
    acm_ports = sorted(glob.glob('/dev/ttyACM*'))
    
    if not acm_ports:
        print("⚠️  No /dev/ttyACM* ports found. Using default from config.")
        return STRETCHMAGTEC_PORT
    
    print(f"🔍 Found {len(acm_ports)} ACM port(s): {acm_ports}")
    print("   Attempting to detect StretchMagTec sensor...")
    
    for port in acm_ports:
        try:
            print(f"   Trying {port}...", end=" ")
            ser = serial.Serial(port, STRETCHMAGTEC_BAUD, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Try to read a line to see if it's responding
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and ('DATA:' in line or 'S' in line or 'X=' in line):
                    ser.close()
                    print(f"✅ SUCCESS - StretchMagTec detected on {port}")
                    return port
            else:
                # Wait a bit and try again
                time.sleep(0.5)
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ('DATA:' in line or 'S' in line or 'X=' in line):
                        ser.close()
                        print(f"✅ SUCCESS - StretchMagTec detected on {port}")
                        return port
            
            # If port opens successfully but no data yet, still use it (might be starting up)
            # Close and return - the main loop will handle reconnection
            ser.close()
            print("⚠️  Port opens but no data yet - will use it anyway")
            return port
        except (serial.SerialException, OSError) as e:
            print(f"❌ Error: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue
    
    print(f"⚠️  Could not auto-detect StretchMagTec. Using default: {STRETCHMAGTEC_PORT}")
    return STRETCHMAGTEC_PORT

class SensorReader:
    """Handles real-time reading from FT sensor and StretchMagTec 3x5 sensors."""
    
    def __init__(self):
        self.ft_data = np.zeros(6)
        self.stretchmagtec_data = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.running = False
        
        # FT sensor setup
        self.ft_thread = None
        self.ft_ser = None
        
        # StretchMagTec sensor setup
        self.stretchmagtec_thread = None
        self.stretchmagtec_ser = None
        self._stretchmagtec_port = None  # Will be auto-detected on first use
        
        # Data buffers for real-time plotting
        self.ft_buffer = []
        self.stretchmagtec_buffer = []
        self.time_buffer = []
        self.max_buffer_size = 1000
        
        # Last valid sensor values (to use if buffer is empty or data is corrupted)
        self.last_valid_stretchmagtec = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        # Median filter for outlier rejection (per sensor/channel)
        # Keep last N values for each sensor/channel to compute median
        self.median_filter_size = 5  # Use median of last 5 values
        self.median_filter_buffer = {}  # Dict: (sensor_id, channel_id) -> deque of values
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            for channel_id in range(STRETCHMAGTEC_CHANNELS):
                self.median_filter_buffer[(sensor_id, channel_id)] = deque(maxlen=self.median_filter_size)
        
        # Hz tracking for StretchMagTec sensors
        self.last_hz_time = time.time()
        self.sensor_hz_counts = [0] * STRETCHMAGTEC_SENSORS
        self.sensor_hz_values = [0.0] * STRETCHMAGTEC_SENSORS
        
        # Locks for thread safety
        self.ft_lock = threading.Lock()
        self.stretchmagtec_lock = threading.Lock()
        
        # Session start time for relative time axis
        self.session_start_time = None
    
    def start_sensors(self):
        """Start sensor reading threads."""
        if self.running:
            return
        
        self.session_start_time = time.time()
        self.running = True
        
        # Start FT sensor thread
        self.ft_thread = threading.Thread(target=self._ft_sensor_loop, daemon=True)
        self.ft_thread.start()
        
        # Start StretchMagTec sensor thread
        self.stretchmagtec_thread = threading.Thread(target=self._stretchmagtec_sensor_loop, daemon=True)
        self.stretchmagtec_thread.start()
        
        print("Sensors started successfully")
    
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
            
            while self.running:
                data = self.ft_ser.read_until(STARTBYTES)
                dataArray = bytearray(data)
                dataArray = STARTBYTES + dataArray[:-2]
                
                if not self._crc_check(dataArray):
                    continue
                
                raw_force = self._force_from_serial_message(dataArray, zeroRef)
                
                # Store raw values
                with self.ft_lock:
                    self.ft_data[:] = raw_force
                
                # Add to buffer for plotting
                current_time = time.time()
                if len(self.time_buffer) >= self.max_buffer_size:
                    self.ft_buffer.pop(0)
                    self.time_buffer.pop(0)
                
                self.ft_buffer.append(raw_force.copy())
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
            # Auto-detect port if not explicitly set
            port = getattr(self, '_stretchmagtec_port', None)
            if port is None:
                port = auto_detect_stretchmagtec_port()
                self._stretchmagtec_port = port
            
            print(f"[StretchMagTec Thread] Starting on {port}...")
            self.stretchmagtec_ser = serial.Serial(port, STRETCHMAGTEC_BAUD, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"[StretchMagTec Thread] Serial connection established")
            
            while self.running:
                if self.stretchmagtec_ser.in_waiting > 0:
                    line = self.stretchmagtec_ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:  # Only process non-empty lines
                        sensor_values = self._parse_stretchmagtec_line(line)
                        
                        if sensor_values is not None:
                            # Apply median filter to remove transient spikes
                            # This handles individual sensor I2C glitches
                            filtered_values = np.zeros_like(sensor_values)
                            
                            with self.stretchmagtec_lock:
                                last_valid = self.last_valid_stretchmagtec.copy()
                            
                            # Filter each sensor/channel independently
                            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                for channel_id in range(STRETCHMAGTEC_CHANNELS):
                                    raw_value = sensor_values[sensor_id, channel_id]
                                    key = (sensor_id, channel_id)
                                    
                                    # Add new value to median filter buffer
                                    self.median_filter_buffer[key].append(raw_value)
                                    
                                    # Use median of buffer if we have enough samples, otherwise use raw value
                                    if len(self.median_filter_buffer[key]) >= 3:
                                        # Use median of last N values
                                        median_value = np.median(list(self.median_filter_buffer[key]))
                                        filtered_values[sensor_id, channel_id] = median_value
                                    else:
                                        # Not enough samples yet, use raw value
                                        filtered_values[sensor_id, channel_id] = raw_value
                            
                            # Additional check: reject if ALL sensors spike simultaneously (parsing error)
                            is_outlier = False
                            if np.any(last_valid != 0):
                                diff = np.abs(filtered_values - last_valid)
                                OUTLIER_THRESHOLD = 50000  # Very large threshold for simultaneous spikes
                                spiked_sensors = 0
                                
                                for i in range(STRETCHMAGTEC_SENSORS):
                                    # Count sensors where ALL 3 channels spiked dramatically
                                    if (diff[i, 0] > OUTLIER_THRESHOLD and 
                                        diff[i, 1] > OUTLIER_THRESHOLD and 
                                        diff[i, 2] > OUTLIER_THRESHOLD):
                                        spiked_sensors += 1
                                
                                # Only reject if ALL 15 sensors spiked simultaneously (definite parsing error)
                                if spiked_sensors >= 15:
                                    is_outlier = True
                            
                            # Use filtered data if it's not a parsing error
                            if not is_outlier:
                                current_time = time.time()
                                
                                # Update data with filtered values
                                with self.stretchmagtec_lock:
                                    self.stretchmagtec_data[:, :] = filtered_values
                                    self.last_valid_stretchmagtec[:, :] = filtered_values
                                
                                # Calculate Hz for each sensor - count every reading
                                with self.stretchmagtec_lock:
                                    for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                        # Count every reading (sensor is active if we got data)
                                        self.sensor_hz_counts[sensor_id] += 1
                                    
                                    # Update Hz values every second
                                    elapsed = current_time - self.last_hz_time
                                    if elapsed >= 1.0:
                                        for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                            if elapsed > 0:
                                                self.sensor_hz_values[sensor_id] = self.sensor_hz_counts[sensor_id] / elapsed
                                            else:
                                                self.sensor_hz_values[sensor_id] = 0.0
                                            self.sensor_hz_counts[sensor_id] = 0
                                        self.last_hz_time = current_time
                                
                                # Add to buffer for plotting
                                if len(self.stretchmagtec_buffer) >= self.max_buffer_size:
                                    self.stretchmagtec_buffer.pop(0)
                                    self.time_buffer.pop(0)
                                
                                self.stretchmagtec_buffer.append(filtered_values.copy())
                                self.time_buffer.append(current_time)
                
                # Small sleep to avoid CPU spinning
                time.sleep(0.001)  # 1ms sleep
                            
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
        """
        Parse StretchMagTec sensor line data.
        Supports multiple formats:
        1. DATA format: DATA:1:x,y,z|2:x,y,z|...
        2. Python list format: [timestamp, [S1_X,S1_Y,S1_Z], [S2_X,S2_Y,S2_Z], ...]
        3. Original format: S1: X=1234 Y=5678 Z=9012 | S2: ...
        """
        try:
            sensor_values = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
            
            # Skip empty lines
            line = line.strip()
            if not line:
                return None
            
            # Try DATA format first (most common)
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
            
            # Try Python list format: [timestamp, [S1_X,S1_Y,S1_Z], ...]
            if line.startswith('['):
                # Parse Python list format using ast.literal_eval (safe eval)
                try:
                    data = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    return None
                
                # Expected format: [timestamp, [S1_X,S1_Y,S1_Z], [S2_X,S2_Y,S2_Z], ...]
                if not isinstance(data, list) or len(data) < 2:
                    return None
                
                # First element is timestamp, rest are sensor data
                sensor_data_list = data[1:]  # Skip timestamp
                
                # Parse each sensor's [X, Y, Z] array
                for i, sensor_data in enumerate(sensor_data_list):
                    if i >= STRETCHMAGTEC_SENSORS:
                        break
                    
                    if not isinstance(sensor_data, list) or len(sensor_data) != 3:
                        continue
                    
                    try:
                        x_val = int(sensor_data[0])
                        y_val = int(sensor_data[1])
                        z_val = int(sensor_data[2])
                        
                        sensor_values[i, 0] = x_val
                        sensor_values[i, 1] = y_val
                        sensor_values[i, 2] = z_val
                    except (ValueError, IndexError, TypeError):
                        continue
                
                return sensor_values
            
            # Try original format: "S1: X=1234 Y=5678 Z=9012 | S2: ..."
            if ' | ' in line:
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
                
                return sensor_values
            
            # No recognized format
            return None
            
        except Exception as e:
            # Return None on any parsing error
            if hasattr(self, '_debug_line_count') and self._debug_line_count <= 10:
                print(f"[DEBUG StretchMagTec] Parse error: {e}")
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
        """Get data for plotting. Uses last valid data if buffer is empty."""
        ft_data = self.ft_buffer.copy() if self.ft_buffer else []
        
        # For StretchMagTec, use buffer if available, otherwise use last valid data
        with self.stretchmagtec_lock:
            if self.stretchmagtec_buffer:
                stretchmagtec_data = self.stretchmagtec_buffer.copy()
            else:
                # Buffer is empty - create a single-entry buffer with last valid data
                if np.any(self.last_valid_stretchmagtec != 0):
                    stretchmagtec_data = [self.last_valid_stretchmagtec.copy()]
                else:
                    stretchmagtec_data = []
            
            # Sync time buffer - if stretchmagtec has data but time doesn't, create time entry
            if stretchmagtec_data and not self.time_buffer:
                current_time = time.time()
                time_data = [current_time]
            else:
                time_data = self.time_buffer.copy() if self.time_buffer else []
        
        return ft_data, stretchmagtec_data, time_data


class SensorVisualizationGUI:
    """GUI for real-time sensor visualization."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sensor Visualization - FT & StretchMagTec")
        self.root.geometry("1400x900")
        
        # Initialize sensor reader
        self.sensor_reader = SensorReader()
        
        # GUI update control
        self.update_running = False
        self.update_interval = 50  # ms
        self.plot_max_points = 500  # Limit plotting to last N points for performance
        self.tight_layout_counter = 0
        self.tight_layout_frequency = 20  # Update layout every 20 frames (1 second)
        
        # Selected sensors for plotting
        self.selected_sensors = set()
        
        # Predefined colors for each sensor
        self.sensor_colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF',
            '#00FFFF', '#FF0080', '#FFFF00', '#FF4000', '#4000FF',
            '#00FF80', '#FF0040', '#8000FF', '#40FF00', '#FF8080'
        ]
        
        # Create GUI elements
        self.create_widgets()
    
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
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready", foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Data display frame
        data_frame = ttk.Frame(main_frame)
        data_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Sensor data
        left_frame = ttk.LabelFrame(data_frame, text="Sensor Data")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        # FT sensor data
        ft_frame = ttk.LabelFrame(left_frame, text="FT Sensor")
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
        canvas = tk.Canvas(stretchmagtec_frame, width=200)
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
        self.sensor_frames = []
        
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
            self.sensor_frames.append(sensor_frame)
        
        # Right column - Plots
        right_frame = ttk.LabelFrame(data_frame, text="Real-Time Plots")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Sensor selection frame
        selection_frame = ttk.Frame(right_frame)
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Select sensors to plot:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
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
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
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
        if sensor_id in self.selected_sensors:
            self.selected_sensors.remove(sensor_id)
            self.sensor_buttons[sensor_id].configure(relief='raised')
        else:
            self.selected_sensors.add(sensor_id)
            self.sensor_buttons[sensor_id].configure(relief='sunken')
        
        self.update_plots()
    
    def clear_sensor_selection(self):
        """Clear all sensor selections."""
        self.selected_sensors.clear()
        for i, btn in enumerate(self.sensor_buttons):
            btn.configure(relief='raised')
        self.update_plots()
    
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
            self.status_label.config(text=f"Status: Error - {e}", foreground="red")
    
    def stop_sensors(self):
        """Stop sensor reading and GUI updates."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Sensors stopped", foreground="orange")
    
    def update_gui(self):
        """Update GUI with latest sensor data."""
        if not self.update_running:
            return
        
        try:
            # Get sensor data
            ft_data = self.sensor_reader.get_ft_data()
            stretchmagtec_data = self.sensor_reader.get_stretchmagtec_data()
            
            # Update FT sensor display
            ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
            for i, (name, value) in enumerate(zip(ft_names, ft_data)):
                if np.isnan(value):
                    self.ft_labels[i].config(text=f"{name}: ERROR", foreground="red")
                else:
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
                
                # Update Hz display (Hz label is the 4th element, index 3)
                if len(self.stretchmagtec_labels[sensor_id]) > 3:
                    hz_value = self.sensor_reader.sensor_hz_values[sensor_id]
                    self.stretchmagtec_labels[sensor_id][3].config(text=f"Hz: {hz_value:.1f}")
            
            # Update plots
            self.update_plots()
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Adaptive update interval based on number of selected sensors
        # More sensors = longer interval to prevent blocking
        num_sensors = len(self.selected_sensors) if self.selected_sensors else 0
        adaptive_interval = self.update_interval + (num_sensors * 5)  # Add 5ms per sensor
        
        # Schedule next update
        self.root.after(adaptive_interval, self.update_gui)
    
    def update_plots(self):
        """Update real-time plots - original approach with minimal optimizations."""
        try:
            ft_data, stretchmagtec_data, time_data = self.sensor_reader.get_plot_data()
            
            if not time_data:
                return
            
            # Limit data to last N points for performance (only optimization)
            plot_points = min(len(time_data), self.plot_max_points)
            if plot_points > 0:
                time_data = time_data[-plot_points:]
                if ft_data:
                    ft_data = ft_data[-plot_points:]
                if stretchmagtec_data:
                    stretchmagtec_data = stretchmagtec_data[-plot_points:]
            
            # Convert absolute time to relative time
            if time_data and self.sensor_reader.session_start_time:
                relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
            else:
                relative_time = []
            
            # Clear all plots (original approach - reliable)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot FT data
            if ft_data and relative_time:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                for i in range(6):
                    self.ax1.plot(relative_time_trimmed, ft_array_trimmed[:, i], 
                                label=labels[i], color=colors[i], alpha=0.7)
                
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force/Torque")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.relim()
                self.ax1.autoscale_view()
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing
                    if not relative_time and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax2.plot(relative_time_trimmed, sensor_data[:, 0], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax2.set_title(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax2.set_ylabel("Magnetic Field")
                        self.ax2.set_xlabel("Time (s)")
                        self.ax2.legend(loc='upper right', fontsize=8)
                        self.ax2.grid(True, alpha=0.3)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                    else:
                        self.ax2.set_title("X-Axis: Waiting for data...")
                        self.ax2.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax2.set_title("X-Axis: Error plotting data")
                    self.ax2.grid(True, alpha=0.3)
            else:
                self.ax2.set_title("X-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax2.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax2.transAxes, fontsize=10)
                self.ax2.grid(True, alpha=0.3)
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax3.plot(relative_time_trimmed, sensor_data[:, 1], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax3.set_title(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax3.set_ylabel("Magnetic Field")
                        self.ax3.set_xlabel("Time (s)")
                        self.ax3.legend(loc='upper right', fontsize=8)
                        self.ax3.grid(True, alpha=0.3)
                        self.ax3.relim()
                        self.ax3.autoscale_view()
                    else:
                        self.ax3.set_title("Y-Axis: Waiting for data...")
                        self.ax3.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax3.set_title("Y-Axis: Error plotting data")
                    self.ax3.grid(True, alpha=0.3)
            else:
                self.ax3.set_title("Y-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax3.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax3.transAxes, fontsize=10)
                self.ax3.grid(True, alpha=0.3)
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax4.plot(relative_time_trimmed, sensor_data[:, 2], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax4.set_title(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax4.set_ylabel("Magnetic Field")
                        self.ax4.set_xlabel("Time (s)")
                        self.ax4.legend(loc='upper right', fontsize=8)
                        self.ax4.grid(True, alpha=0.3)
                        self.ax4.relim()
                        self.ax4.autoscale_view()
                    else:
                        self.ax4.set_title("Z-Axis: Waiting for data...")
                        self.ax4.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax4.set_title("Z-Axis: Error plotting data")
                    self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.set_title("Z-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax4.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax4.transAxes, fontsize=10)
                self.ax4.grid(True, alpha=0.3)
            
            # Always update layout and draw
            self.fig.tight_layout()
            self.canvas.draw()  # Use draw() to ensure immediate update
            
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
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("="*60)
    
    # Create and run GUI
    app = SensorVisualizationGUI()
    app.run()


if __name__ == "__main__":
    main()


            
            # Clear all plots (original approach - reliable)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot FT data
            if ft_data and relative_time:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                for i in range(6):
                    self.ax1.plot(relative_time_trimmed, ft_array_trimmed[:, i], 
                                label=labels[i], color=colors[i], alpha=0.7)
                
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force/Torque")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.relim()
                self.ax1.autoscale_view()
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing
                    if not relative_time and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax2.plot(relative_time_trimmed, sensor_data[:, 0], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax2.set_title(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax2.set_ylabel("Magnetic Field")
                        self.ax2.set_xlabel("Time (s)")
                        self.ax2.legend(loc='upper right', fontsize=8)
                        self.ax2.grid(True, alpha=0.3)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                    else:
                        self.ax2.set_title("X-Axis: Waiting for data...")
                        self.ax2.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax2.set_title("X-Axis: Error plotting data")
                    self.ax2.grid(True, alpha=0.3)
            else:
                self.ax2.set_title("X-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax2.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax2.transAxes, fontsize=10)
                self.ax2.grid(True, alpha=0.3)
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax3.plot(relative_time_trimmed, sensor_data[:, 1], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax3.set_title(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax3.set_ylabel("Magnetic Field")
                        self.ax3.set_xlabel("Time (s)")
                        self.ax3.legend(loc='upper right', fontsize=8)
                        self.ax3.grid(True, alpha=0.3)
                        self.ax3.relim()
                        self.ax3.autoscale_view()
                    else:
                        self.ax3.set_title("Y-Axis: Waiting for data...")
                        self.ax3.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax3.set_title("Y-Axis: Error plotting data")
                    self.ax3.grid(True, alpha=0.3)
            else:
                self.ax3.set_title("Y-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax3.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax3.transAxes, fontsize=10)
                self.ax3.grid(True, alpha=0.3)
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax4.plot(relative_time_trimmed, sensor_data[:, 2], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax4.set_title(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax4.set_ylabel("Magnetic Field")
                        self.ax4.set_xlabel("Time (s)")
                        self.ax4.legend(loc='upper right', fontsize=8)
                        self.ax4.grid(True, alpha=0.3)
                        self.ax4.relim()
                        self.ax4.autoscale_view()
                    else:
                        self.ax4.set_title("Z-Axis: Waiting for data...")
                        self.ax4.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax4.set_title("Z-Axis: Error plotting data")
                    self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.set_title("Z-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax4.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax4.transAxes, fontsize=10)
                self.ax4.grid(True, alpha=0.3)
            
            # Always update layout and draw
            self.fig.tight_layout()
            self.canvas.draw()  # Use draw() to ensure immediate update
            
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
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("="*60)
    
    # Create and run GUI
    app = SensorVisualizationGUI()
    app.run()


if __name__ == "__main__":
    main()


            
            # Clear all plots (original approach - reliable)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot FT data
            if ft_data and relative_time:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                for i in range(6):
                    self.ax1.plot(relative_time_trimmed, ft_array_trimmed[:, i], 
                                label=labels[i], color=colors[i], alpha=0.7)
                
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force/Torque")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.relim()
                self.ax1.autoscale_view()
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing
                    if not relative_time and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax2.plot(relative_time_trimmed, sensor_data[:, 0], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax2.set_title(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax2.set_ylabel("Magnetic Field")
                        self.ax2.set_xlabel("Time (s)")
                        self.ax2.legend(loc='upper right', fontsize=8)
                        self.ax2.grid(True, alpha=0.3)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                    else:
                        self.ax2.set_title("X-Axis: Waiting for data...")
                        self.ax2.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax2.set_title("X-Axis: Error plotting data")
                    self.ax2.grid(True, alpha=0.3)
            else:
                self.ax2.set_title("X-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax2.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax2.transAxes, fontsize=10)
                self.ax2.grid(True, alpha=0.3)
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax3.plot(relative_time_trimmed, sensor_data[:, 1], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax3.set_title(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax3.set_ylabel("Magnetic Field")
                        self.ax3.set_xlabel("Time (s)")
                        self.ax3.legend(loc='upper right', fontsize=8)
                        self.ax3.grid(True, alpha=0.3)
                        self.ax3.relim()
                        self.ax3.autoscale_view()
                    else:
                        self.ax3.set_title("Y-Axis: Waiting for data...")
                        self.ax3.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax3.set_title("Y-Axis: Error plotting data")
                    self.ax3.grid(True, alpha=0.3)
            else:
                self.ax3.set_title("Y-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax3.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax3.transAxes, fontsize=10)
                self.ax3.grid(True, alpha=0.3)
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax4.plot(relative_time_trimmed, sensor_data[:, 2], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax4.set_title(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax4.set_ylabel("Magnetic Field")
                        self.ax4.set_xlabel("Time (s)")
                        self.ax4.legend(loc='upper right', fontsize=8)
                        self.ax4.grid(True, alpha=0.3)
                        self.ax4.relim()
                        self.ax4.autoscale_view()
                    else:
                        self.ax4.set_title("Z-Axis: Waiting for data...")
                        self.ax4.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax4.set_title("Z-Axis: Error plotting data")
                    self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.set_title("Z-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax4.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax4.transAxes, fontsize=10)
                self.ax4.grid(True, alpha=0.3)
            
            # Always update layout and draw
            self.fig.tight_layout()
            self.canvas.draw()  # Use draw() to ensure immediate update
            
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
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("="*60)
    
    # Create and run GUI
    app = SensorVisualizationGUI()
    app.run()


if __name__ == "__main__":
    main()

            
            # Clear all plots (original approach - reliable)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot FT data
            if ft_data and relative_time:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                for i in range(6):
                    self.ax1.plot(relative_time_trimmed, ft_array_trimmed[:, i], 
                                label=labels[i], color=colors[i], alpha=0.7)
                
                self.ax1.set_title("FT Sensor Data")
                self.ax1.set_ylabel("Force/Torque")
                self.ax1.set_xlabel("Time (s)")
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.relim()
                self.ax1.autoscale_view()
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing
                    if not relative_time and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax2.plot(relative_time_trimmed, sensor_data[:, 0], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax2.set_title(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax2.set_ylabel("Magnetic Field")
                        self.ax2.set_xlabel("Time (s)")
                        self.ax2.legend(loc='upper right', fontsize=8)
                        self.ax2.grid(True, alpha=0.3)
                        self.ax2.relim()
                        self.ax2.autoscale_view()
                    else:
                        self.ax2.set_title("X-Axis: Waiting for data...")
                        self.ax2.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax2.set_title("X-Axis: Error plotting data")
                    self.ax2.grid(True, alpha=0.3)
            else:
                self.ax2.set_title("X-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax2.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax2.transAxes, fontsize=10)
                self.ax2.grid(True, alpha=0.3)
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax3.plot(relative_time_trimmed, sensor_data[:, 1], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax3.set_title(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax3.set_ylabel("Magnetic Field")
                        self.ax3.set_xlabel("Time (s)")
                        self.ax3.legend(loc='upper right', fontsize=8)
                        self.ax3.grid(True, alpha=0.3)
                        self.ax3.relim()
                        self.ax3.autoscale_view()
                    else:
                        self.ax3.set_title("Y-Axis: Waiting for data...")
                        self.ax3.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax3.set_title("Y-Axis: Error plotting data")
                    self.ax3.grid(True, alpha=0.3)
            else:
                self.ax3.set_title("Y-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax3.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax3.transAxes, fontsize=10)
                self.ax3.grid(True, alpha=0.3)
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                self.ax4.plot(relative_time_trimmed, sensor_data[:, 2], 
                                            label=f'S{sensor_id+1}', color=color, alpha=0.8, linewidth=2.0)
                        
                        self.ax4.set_title(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        self.ax4.set_ylabel("Magnetic Field")
                        self.ax4.set_xlabel("Time (s)")
                        self.ax4.legend(loc='upper right', fontsize=8)
                        self.ax4.grid(True, alpha=0.3)
                        self.ax4.relim()
                        self.ax4.autoscale_view()
                    else:
                        self.ax4.set_title("Z-Axis: Waiting for data...")
                        self.ax4.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ax4.set_title("Z-Axis: Error plotting data")
                    self.ax4.grid(True, alpha=0.3)
            else:
                self.ax4.set_title("Z-Axis: Select sensors to plot")
                if not self.selected_sensors:
                    self.ax4.text(0.5, 0.5, 'No sensors selected\nClick sensor buttons', 
                                ha='center', va='center', transform=self.ax4.transAxes, fontsize=10)
                self.ax4.grid(True, alpha=0.3)
            
            # Always update layout and draw
            self.fig.tight_layout()
            self.canvas.draw()  # Use draw() to ensure immediate update
            
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
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("="*60)
    
    # Create and run GUI
    app = SensorVisualizationGUI()
    app.run()


if __name__ == "__main__":
    main()
