#!/usr/bin/env python3
"""
Real-Time Sensor Visualization - GPU Accelerated Version
Uses PyQtGraph for high-performance GPU-accelerated plotting

This script provides real-time visualization of:
1. FT sensor readings (Fx, Fy, Fz, Tx, Ty, Tz)
2. StretchMagTec 3x5 sensor readings (15 sensors × 3 channels)

Usage:
    python3 visualize_sensors_only_gpu.py

Author: Gabriele Giudici
Date: 2025
"""

import os
import sys
import time
import threading
import numpy as np
import serial
import minimalmodbus as mm
import libscrc
import glob
import ast
from collections import deque

# PyQtGraph for GPU-accelerated plotting
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    # Enable OpenGL for GPU acceleration
    pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
except ImportError:
    print("ERROR: PyQtGraph not installed. Install with: pip install pyqtgraph pyqt5")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *


def auto_detect_stretchmagtec_port():
    """
    Auto-detect StretchMagTec port by scanning available /dev/ttyACM* ports.
    Returns the first port that successfully opens and can read data.
    """
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
            time.sleep(2)
            
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and ('DATA:' in line or 'S' in line or 'X=' in line or line.startswith('[')):
                    ser.close()
                    print(f"✅ SUCCESS - StretchMagTec detected on {port}")
                    return port
            else:
                time.sleep(0.5)
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ('DATA:' in line or 'S' in line or 'X=' in line or line.startswith('[')):
                        ser.close()
                        print(f"✅ SUCCESS - StretchMagTec detected on {port}")
                        return port
            
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
        self._stretchmagtec_port = None
        
        # Data buffers for real-time plotting
        self.ft_buffer = []
        self.stretchmagtec_buffer = []
        self.time_buffer = []
        self.max_buffer_size = 1000
        
        # Last valid sensor values
        self.last_valid_stretchmagtec = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        # Median filter for outlier rejection (per sensor/channel)
        self.median_filter_size = 5
        self.median_filter_buffer = {}
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
        
        self.ft_thread = threading.Thread(target=self._ft_sensor_loop, daemon=True)
        self.ft_thread.start()
        
        self.stretchmagtec_thread = threading.Thread(target=self._stretchmagtec_sensor_loop, daemon=True)
        self.stretchmagtec_thread.start()
        
        print("Sensors started successfully")
    
    def stop_sensors(self):
        """Stop sensor reading threads."""
        if not self.running:
            return
            
        self.running = False
        
        if self.ft_thread and self.ft_thread.is_alive():
            self.ft_thread.join(timeout=2)
        if self.stretchmagtec_thread and self.stretchmagtec_thread.is_alive():
            self.stretchmagtec_thread.join(timeout=2)
        
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
                
                with self.ft_lock:
                    self.ft_data[:] = raw_force
                
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
            port = getattr(self, '_stretchmagtec_port', None)
            if port is None:
                port = auto_detect_stretchmagtec_port()
                self._stretchmagtec_port = port
            
            print(f"[StretchMagTec Thread] Starting on {port}...")
            self.stretchmagtec_ser = serial.Serial(port, STRETCHMAGTEC_BAUD, timeout=1)
            time.sleep(2)
            print(f"[StretchMagTec Thread] Serial connection established")
            
            while self.running:
                if self.stretchmagtec_ser.in_waiting > 0:
                    line = self.stretchmagtec_ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        sensor_values = self._parse_stretchmagtec_line(line)
                        
                        if sensor_values is not None:
                            filtered_values = np.zeros_like(sensor_values)
                            
                            with self.stretchmagtec_lock:
                                last_valid = self.last_valid_stretchmagtec.copy()
                            
                            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                                for channel_id in range(STRETCHMAGTEC_CHANNELS):
                                    raw_value = sensor_values[sensor_id, channel_id]
                                    key = (sensor_id, channel_id)
                                    
                                    self.median_filter_buffer[key].append(raw_value)
                                    
                                    if len(self.median_filter_buffer[key]) >= 3:
                                        median_value = np.median(list(self.median_filter_buffer[key]))
                                        filtered_values[sensor_id, channel_id] = median_value
                                    else:
                                        filtered_values[sensor_id, channel_id] = raw_value
                            
                            is_outlier = False
                            if np.any(last_valid != 0):
                                diff = np.abs(filtered_values - last_valid)
                                OUTLIER_THRESHOLD = 50000
                                spiked_sensors = 0
                                
                                for i in range(STRETCHMAGTEC_SENSORS):
                                    if (diff[i, 0] > OUTLIER_THRESHOLD and 
                                        diff[i, 1] > OUTLIER_THRESHOLD and 
                                        diff[i, 2] > OUTLIER_THRESHOLD):
                                        spiked_sensors += 1
                                
                                if spiked_sensors >= 15:
                                    is_outlier = True
                            
                            if not is_outlier:
                                current_time = time.time()
                                
                                with self.stretchmagtec_lock:
                                    self.stretchmagtec_data[:, :] = filtered_values
                                    self.last_valid_stretchmagtec[:, :] = filtered_values
                                
                                if len(self.stretchmagtec_buffer) >= self.max_buffer_size:
                                    self.stretchmagtec_buffer.pop(0)
                                    self.time_buffer.pop(0)
                                
                                self.stretchmagtec_buffer.append(filtered_values.copy())
                                self.time_buffer.append(current_time)
                                
                                # Calculate Hz for each sensor - count every reading (not just non-zero)
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
                
                time.sleep(0.001)
                            
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
        """Parse StretchMagTec sensor line data."""
        try:
            sensor_values = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
            
            line = line.strip()
            if not line:
                return None
            
            if not line.startswith('['):
                return None
            
            try:
                data = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                return None
            
            if not isinstance(data, list) or len(data) < 2:
                return None
            
            sensor_data_list = data[1:]
            
            for i, sensor_data in enumerate(sensor_data_list):
                if i >= STRETCHMAGTEC_SENSORS:
                    break
                
                if not isinstance(sensor_data, list) or len(sensor_data) != 3:
                    continue
                
                try:
                    sensor_values[i, 0] = int(sensor_data[0])
                    sensor_values[i, 1] = int(sensor_data[1])
                    sensor_values[i, 2] = int(sensor_data[2])
                except (ValueError, IndexError, TypeError):
                    continue
            
            return sensor_values
            
        except Exception as e:
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
        
        with self.stretchmagtec_lock:
            if self.stretchmagtec_buffer:
                stretchmagtec_data = self.stretchmagtec_buffer.copy()
            else:
                if np.any(self.last_valid_stretchmagtec != 0):
                    stretchmagtec_data = [self.last_valid_stretchmagtec.copy()]
                else:
                    stretchmagtec_data = []
            
            if stretchmagtec_data and not self.time_buffer:
                current_time = time.time()
                time_data = [current_time]
            else:
                time_data = self.time_buffer.copy() if self.time_buffer else []
        
        return ft_data, stretchmagtec_data, time_data


class SensorVisualizationGUI(QtWidgets.QMainWindow):
    """GUI for real-time sensor visualization."""
    
    def __init__(self):
        super().__init__()
        self.sensor_reader = SensorReader()
        
        # GUI update control
        self.update_running = False
        self.update_interval = 50  # ms
        self.plot_max_points = 500
        self.tight_layout_counter = 0
        self.tight_layout_frequency = 20
        
        # Selected sensors for plotting
        self.selected_sensors = set()
        
        # Predefined colors for each sensor
        self.sensor_colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF',
            '#00FFFF', '#FF0080', '#FFFF00', '#FF4000', '#4000FF',
            '#00FF80', '#FF0040', '#8000FF', '#40FF00', '#FF8080'
        ]
        
        # Store plot items for updating instead of recreating
        self.ft_plot_items = {}
        self.x_plot_items = {}
        self.y_plot_items = {}
        self.z_plot_items = {}
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Sensor Visualization - FT & StretchMagTec (GPU Accelerated)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set white background for the entire window
        self.setStyleSheet("background-color: white;")
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        central_widget.setStyleSheet("background-color: white;")
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control frame
        control_frame = QtWidgets.QGroupBox("Control Panel")
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        
        self.start_button = QtWidgets.QPushButton("Start Sensors")
        self.start_button.clicked.connect(self.start_sensors)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QtWidgets.QPushButton("Stop Sensors")
        self.stop_button.clicked.connect(self.stop_sensors)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: blue;")
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_frame)
        
        # Data display frame
        data_frame = QtWidgets.QWidget()
        data_layout = QtWidgets.QHBoxLayout(data_frame)
        
        # Left column - Sensor data
        left_frame = QtWidgets.QGroupBox("Sensor Data")
        left_layout = QtWidgets.QVBoxLayout(left_frame)
        left_frame.setMaximumWidth(250)
        
        # FT sensor data
        ft_frame = QtWidgets.QGroupBox("FT Sensor")
        ft_layout = QtWidgets.QVBoxLayout(ft_frame)
        self.ft_labels = []
        ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
        for name in ft_names:
            label = QtWidgets.QLabel(f"{name}: 0.000")
            label.setFont(QtGui.QFont("Courier", 10))
            ft_layout.addWidget(label)
            self.ft_labels.append(label)
        left_layout.addWidget(ft_frame)
        
        # StretchMagTec sensor data
        sm_frame = QtWidgets.QGroupBox("StretchMagTec 3x5 Sensors")
        sm_layout = QtWidgets.QVBoxLayout(sm_frame)
        scroll = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        self.stretchmagtec_labels = []
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            sensor_frame = QtWidgets.QGroupBox(f"Sensor {sensor_id + 1}")
            sensor_layout = QtWidgets.QVBoxLayout(sensor_frame)
            sensor_labels = []
            for channel, name in enumerate(['X', 'Y', 'Z']):
                label = QtWidgets.QLabel(f"{name}: 0")
                label.setFont(QtGui.QFont("Courier", 9))
                sensor_layout.addWidget(label)
                sensor_labels.append(label)
            # Add Hz label
            hz_label = QtWidgets.QLabel("Hz: 0.0")
            hz_label.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
            hz_label.setStyleSheet("color: blue;")
            sensor_layout.addWidget(hz_label)
            sensor_labels.append(hz_label)
            self.stretchmagtec_labels.append(sensor_labels)
            scroll_layout.addWidget(sensor_frame)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        sm_layout.addWidget(scroll)
        left_layout.addWidget(sm_frame)
        
        data_layout.addWidget(left_frame)
        
        # Right column - Plots
        right_frame = QtWidgets.QGroupBox("Real-Time Plots")
        right_layout = QtWidgets.QVBoxLayout(right_frame)
        
        # Sensor selection frame
        selection_frame = QtWidgets.QWidget()
        selection_layout = QtWidgets.QHBoxLayout(selection_frame)
        
        selection_label = QtWidgets.QLabel("Select sensors to plot:")
        selection_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        selection_layout.addWidget(selection_label)
        
        # Create sensor selection buttons
        self.sensor_buttons = []
        buttons_frame = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        for sensor_id in range(STRETCHMAGTEC_SENSORS):
            btn = QtWidgets.QPushButton(f"S{sensor_id+1}")
            btn.setCheckable(True)
            btn.setStyleSheet(f"background-color: {self.sensor_colors[sensor_id]}; color: white; font-weight: bold;")
            btn.clicked.connect(lambda checked, s_id=sensor_id: self.toggle_sensor_selection(s_id))
            buttons_layout.addWidget(btn)
            self.sensor_buttons.append(btn)
        
        selection_layout.addWidget(buttons_frame)
        
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_sensor_selection)
        selection_layout.addWidget(clear_btn)
        
        right_layout.addWidget(selection_frame)
        
        # Create PyQtGraph plot widget with 4 subplots
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setBackground('w')  # White background for the entire plot widget
        
        # FT sensor plot
        self.ft_plot = plot_widget.addPlot(title="FT Sensor Data", row=0, col=0)
        self.ft_plot.setLabel('left', 'Force/Torque')
        self.ft_plot.setLabel('bottom', 'Time (s)')
        self.ft_plot.addLegend()
        self.ft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ft_plot.getViewBox().setBackgroundColor('w')
        # Create plot items once for Fx, Fy, Fz with initial test data to ensure they work
        color_map = {'r': (255, 0, 0), 'g': (0, 200, 0), 'b': (0, 0, 255)}
        labels = ["Fx", "Fy", "Fz"]
        colors = ['r', 'g', 'b']
        for i, label in enumerate(labels):
            rgb = color_map.get(colors[i], (0, 0, 0))
            # Create with empty data first
            plot_item = self.ft_plot.plot([], [], pen=pg.mkPen(rgb, width=3), name=label)
            plot_item.setVisible(True)
            self.ft_plot_items[label] = plot_item
        
        # X-axis plot
        self.x_plot = plot_widget.addPlot(title="StretchMagTec X-Axis", row=1, col=0)
        self.x_plot.setLabel('left', 'Magnetic Field')
        self.x_plot.setLabel('bottom', 'Time (s)')
        self.x_plot.addLegend()
        self.x_plot.showGrid(x=True, y=True, alpha=0.3)
        self.x_plot.getViewBox().setBackgroundColor('w')
        
        # Y-axis plot
        self.y_plot = plot_widget.addPlot(title="StretchMagTec Y-Axis", row=2, col=0)
        self.y_plot.setLabel('left', 'Magnetic Field')
        self.y_plot.setLabel('bottom', 'Time (s)')
        self.y_plot.addLegend()
        self.y_plot.showGrid(x=True, y=True, alpha=0.3)
        self.y_plot.getViewBox().setBackgroundColor('w')
        
        # Z-axis plot
        self.z_plot = plot_widget.addPlot(title="StretchMagTec Z-Axis", row=3, col=0)
        self.z_plot.setLabel('left', 'Magnetic Field')
        self.z_plot.setLabel('bottom', 'Time (s)')
        self.z_plot.addLegend()
        self.z_plot.showGrid(x=True, y=True, alpha=0.3)
        self.z_plot.getViewBox().setBackgroundColor('w')
        
        right_layout.addWidget(plot_widget)
        data_layout.addWidget(right_frame)
        
        main_layout.addWidget(data_frame)
    
    def toggle_sensor_selection(self, sensor_id):
        """Toggle sensor selection for plotting."""
        if sensor_id in self.selected_sensors:
            self.selected_sensors.remove(sensor_id)
            self.sensor_buttons[sensor_id].setChecked(False)
        else:
            self.selected_sensors.add(sensor_id)
            self.sensor_buttons[sensor_id].setChecked(True)
        
        self.update_plots()
    
    def clear_sensor_selection(self):
        """Clear all sensor selections."""
        self.selected_sensors.clear()
        for btn in self.sensor_buttons:
            btn.setChecked(False)
        self.update_plots()
    
    def start_sensors(self):
        """Start sensor reading and GUI updates."""
        try:
            self.sensor_reader.start_sensors()
            self.update_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Status: Sensors running")
            self.status_label.setStyleSheet("color: green;")
            
            # Start GUI update loop
            self.update_gui()
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {e}")
            self.status_label.setStyleSheet("color: red;")
    
    def stop_sensors(self):
        """Stop sensor reading and GUI updates."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Sensors stopped")
        self.status_label.setStyleSheet("color: orange;")
    
    def update_gui(self):
        """Update GUI with latest sensor data."""
        if not self.update_running:
            return
        
        try:
            ft_data = self.sensor_reader.get_ft_data()
            stretchmagtec_data = self.sensor_reader.get_stretchmagtec_data()
            
            # Update FT sensor display
            ft_names = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
            for i, (name, value) in enumerate(zip(ft_names, ft_data)):
                if np.isnan(value):
                    self.ft_labels[i].setText(f"{name}: ERROR")
                    self.ft_labels[i].setStyleSheet("color: red;")
                else:
                    color = "red" if abs(value) > 1.0 else "black"
                    self.ft_labels[i].setText(f"{name}: {value:7.3f}")
                    self.ft_labels[i].setStyleSheet(f"color: {color};")
            
            # Update StretchMagTec sensor display
            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                for channel_id in range(STRETCHMAGTEC_CHANNELS):
                    value = stretchmagtec_data[sensor_id, channel_id]
                    channel_name = ['X', 'Y', 'Z'][channel_id]
                    color = "red" if abs(value) > STRETCHMAGTEC_THRESHOLD else "black"
                    self.stretchmagtec_labels[sensor_id][channel_id].setText(f"{channel_name}: {value:6.0f}")
                    self.stretchmagtec_labels[sensor_id][channel_id].setStyleSheet(f"color: {color};")
                
                # Update Hz display (add Hz label if not exists)
                if len(self.stretchmagtec_labels[sensor_id]) == 3:
                    # Add Hz label
                    hz_label = QtWidgets.QLabel("Hz: 0.0")
                    hz_label.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
                    hz_label.setStyleSheet("color: blue;")
                    # Find the parent widget and add Hz label
                    parent_widget = self.stretchmagtec_labels[sensor_id][0].parent()
                    if parent_widget:
                        layout = parent_widget.layout()
                        if layout:
                            hz_label = QtWidgets.QLabel("Hz: 0.0")
                            hz_label.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
                            hz_label.setStyleSheet("color: blue;")
                            layout.addWidget(hz_label)
                            self.stretchmagtec_labels[sensor_id].append(hz_label)
                
                # Update Hz value
                if len(self.stretchmagtec_labels[sensor_id]) > 3:
                    hz_value = self.sensor_reader.sensor_hz_values[sensor_id]
                    self.stretchmagtec_labels[sensor_id][3].setText(f"Hz: {hz_value:.1f}")
            
            # Update plots
            self.update_plots()
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Adaptive update interval
        num_sensors = len(self.selected_sensors) if self.selected_sensors else 0
        adaptive_interval = self.update_interval + (num_sensors * 5)
        
        # Schedule next update
        QtCore.QTimer.singleShot(adaptive_interval, self.update_gui)
    
    def update_plots(self):
        """Update real-time plots - SIMPLIFIED WITH DEBUG."""
        try:
            ft_data, stretchmagtec_data, time_data = self.sensor_reader.get_plot_data()
            
            # DEBUG: Print data status every 20 updates (once per second)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            if self._debug_counter % 20 == 0:
                print(f"[DEBUG] FT data: {len(ft_data) if ft_data else 0} points, "
                      f"StretchMagTec: {len(stretchmagtec_data) if stretchmagtec_data else 0} points, "
                      f"Time: {len(time_data) if time_data else 0} points")
                if ft_data:
                    print(f"[DEBUG] FT data sample: {ft_data[0] if len(ft_data) > 0 else 'empty'}")
                if time_data:
                    print(f"[DEBUG] Time data sample: {time_data[0] if len(time_data) > 0 else 'empty'}")
            
            if not time_data:
                if self._debug_counter % 20 == 0:
                    print("[DEBUG] No time data, skipping plot update")
                return
            
            # Ensure we have at least some data before plotting
            if len(time_data) < 2:
                if self._debug_counter % 20 == 0:
                    print(f"[DEBUG] Not enough time data ({len(time_data)} points), skipping plot update")
                return
            
            # Limit data to last N points
            plot_points = min(len(time_data), self.plot_max_points)
            if plot_points > 0:
                time_data = time_data[-plot_points:]
                if ft_data:
                    ft_data = ft_data[-plot_points:]
                if stretchmagtec_data:
                    stretchmagtec_data = stretchmagtec_data[-plot_points:]
            
            # Convert absolute time to relative time - EXACT SAME LOGIC AS NON-GPU VERSION
            if time_data and self.sensor_reader.session_start_time:
                # Simple calculation: subtract session_start_time from each timestamp
                relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
            else:
                relative_time = []
            
            # Don't clear plots - update existing items or create new ones
            # This avoids the ItemHasNoContents issue
            
            # Plot FT data - SAME LOGIC AS NON-GPU VERSION
            if ft_data and len(relative_time) > 0:
                ft_array = np.array(ft_data)
                labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
                colors = ['r', 'g', 'b', 'c', 'm', 'y']
                
                min_len = min(len(relative_time), len(ft_array))
                relative_time_trimmed = relative_time[:min_len]
                ft_array_trimmed = ft_array[:min_len]
                
                # Ensure time starts from 0 (or positive)
                if len(relative_time_trimmed) > 0 and relative_time_trimmed[0] < 0:
                    time_offset = relative_time_trimmed[0]
                    relative_time_trimmed = [t - time_offset for t in relative_time_trimmed]
                
                # Convert to numpy arrays for PyQtGraph - ensure they are contiguous
                time_array = np.ascontiguousarray(relative_time_trimmed, dtype=np.float64)
                
                # Plot only Fx, Fy, Fz (forces only, no torques)
                for i in range(3):  # Only plot first 3 components: Fx, Fy, Fz
                    # x=time, y=force values - ensure arrays are 1D and contiguous
                    force_data = np.ascontiguousarray(ft_array_trimmed[:, i], dtype=np.float64)
                    if len(time_array) == len(force_data) and len(force_data) > 0:
                        # Debug: print first and last values more frequently
                        if self._debug_counter % 20 == 0:
                            print(f"[DEBUG FT {labels[i]}] Points: {len(force_data)}, "
                                  f"Range: [{force_data.min():.3f}, {force_data.max():.3f}], "
                                  f"Time: [{time_array[0]:.2f}, {time_array[-1]:.2f}], "
                                  f"First 3 values: {force_data[:3]}")
                        # Update existing plot item - ensure data is valid
                        if np.any(np.isfinite(force_data)) and np.any(np.isfinite(time_array)):
                            if labels[i] in self.ft_plot_items:
                                # Convert to lists to avoid numpy issues with PyQtGraph
                                time_list = time_array.tolist() if isinstance(time_array, np.ndarray) else list(time_array)
                                force_list = force_data.tolist() if isinstance(force_data, np.ndarray) else list(force_data)
                                # Update existing item with new data
                                self.ft_plot_items[labels[i]].setData(time_list, force_list, autoDownsample=True)
                                # Force the item to be visible
                                self.ft_plot_items[labels[i]].setVisible(True)
                                if self._debug_counter % 20 == 0:
                                    item = self.ft_plot_items[labels[i]]
                                    print(f"[DEBUG] Updated plot item for {labels[i]}, "
                                          f"visible: {item.isVisible()}, "
                                          f"hasData: {hasattr(item, 'hasData') and item.hasData()}, "
                                          f"data length: {len(time_list)}")
                        else:
                            if self._debug_counter % 20 == 0:
                                print(f"[DEBUG] Skipping {labels[i]} - invalid data (NaN/Inf)")
                
                self.ft_plot.setTitle("FT Sensor Data (Fx, Fy, Fz)")
                # Set fixed Y range for FT plot: -15 to 15 N
                self.ft_plot.setYRange(-15, 15, padding=0)
                # Auto-range X axis - ensure it starts from 0 or positive
                if len(time_array) > 0:
                    x_min = max(0, time_array[0] - 0.1)
                    x_max = time_array[-1] + 0.1
                    self.ft_plot.setXRange(x_min, x_max, padding=0)
            else:
                if self._debug_counter % 20 == 0:
                    print(f"[DEBUG] FT plot skipped: ft_data={ft_data is not None}, "
                          f"len(relative_time)={len(relative_time) if relative_time else 0}")
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing - SAME LOGIC AS NON-GPU
                    if len(relative_time) == 0 and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = np.array([(t - self.sensor_reader.session_start_time) for t in time_data])
                        else:
                            relative_time = np.array(list(range(len(stretchmagtec_array)))) * 0.01  # Assume 100Hz
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # x=time, y=magnetic field values - ensure arrays are 1D
                                mag_data = sensor_data[:, 0]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.x_plot_items:
                                        self.x_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.x_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.x_plot_items[key] = plot_item
                        
                        self.x_plot.setTitle(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.x_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.x_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.x_plot.setTitle("X-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.x_plot.setTitle("X-Axis: Error plotting data")
            else:
                self.x_plot.setTitle("X-Axis: Select sensors to plot")
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 1]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.y_plot_items:
                                        self.y_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.y_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.y_plot_items[key] = plot_item
                        
                        self.y_plot.setTitle(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.y_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.y_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.y_plot.setTitle("Y-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.y_plot.setTitle("Y-Axis: Error plotting data")
            else:
                self.y_plot.setTitle("Y-Axis: Select sensors to plot")
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 2]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    # Convert to lists to avoid numpy issues
                                    time_list = time_array.tolist() if isinstance(time_array, np.ndarray) else list(time_array)
                                    mag_list = mag_data.tolist() if isinstance(mag_data, np.ndarray) else list(mag_data)
                                    if key in self.z_plot_items:
                                        self.z_plot_items[key].setData(time_list, mag_list, autoDownsample=True)
                                        self.z_plot_items[key].setVisible(True)
                                    else:
                                        plot_item = self.z_plot.plot(time_list, mag_list, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.z_plot_items[key] = plot_item
                                        plot_item.setVisible(True)
                        
                        self.z_plot.setTitle(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.z_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.z_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.z_plot.setTitle("Z-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.z_plot.setTitle("Z-Axis: Error plotting data")
            else:
                self.z_plot.setTitle("Z-Axis: Select sensors to plot")
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window closing."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        event.accept()


def main():
    """Main function."""
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC (GPU ACCELERATED)")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("Using PyQtGraph with OpenGL acceleration")
    print("="*60)
    
    app = QtWidgets.QApplication(sys.argv)
    window = SensorVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

                                print(f"[DEBUG] Skipping {labels[i]} - invalid data (NaN/Inf)")
                
                self.ft_plot.setTitle("FT Sensor Data (Fx, Fy, Fz)")
                # Set fixed Y range for FT plot: -15 to 15 N
                self.ft_plot.setYRange(-15, 15, padding=0)
                # Auto-range X axis - ensure it starts from 0 or positive
                if len(time_array) > 0:
                    x_min = max(0, time_array[0] - 0.1)
                    x_max = time_array[-1] + 0.1
                    self.ft_plot.setXRange(x_min, x_max, padding=0)
            else:
                if self._debug_counter % 20 == 0:
                    print(f"[DEBUG] FT plot skipped: ft_data={ft_data is not None}, "
                          f"len(relative_time)={len(relative_time) if relative_time else 0}")
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing - SAME LOGIC AS NON-GPU
                    if len(relative_time) == 0 and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = np.array([(t - self.sensor_reader.session_start_time) for t in time_data])
                        else:
                            relative_time = np.array(list(range(len(stretchmagtec_array)))) * 0.01  # Assume 100Hz
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # x=time, y=magnetic field values - ensure arrays are 1D
                                mag_data = sensor_data[:, 0]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.x_plot_items:
                                        self.x_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.x_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.x_plot_items[key] = plot_item
                        
                        self.x_plot.setTitle(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.x_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.x_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.x_plot.setTitle("X-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.x_plot.setTitle("X-Axis: Error plotting data")
            else:
                self.x_plot.setTitle("X-Axis: Select sensors to plot")
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 1]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.y_plot_items:
                                        self.y_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.y_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.y_plot_items[key] = plot_item
                        
                        self.y_plot.setTitle(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.y_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.y_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.y_plot.setTitle("Y-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.y_plot.setTitle("Y-Axis: Error plotting data")
            else:
                self.y_plot.setTitle("Y-Axis: Select sensors to plot")
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 2]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    # Convert to lists to avoid numpy issues
                                    time_list = time_array.tolist() if isinstance(time_array, np.ndarray) else list(time_array)
                                    mag_list = mag_data.tolist() if isinstance(mag_data, np.ndarray) else list(mag_data)
                                    if key in self.z_plot_items:
                                        self.z_plot_items[key].setData(time_list, mag_list, autoDownsample=True)
                                        self.z_plot_items[key].setVisible(True)
                                    else:
                                        plot_item = self.z_plot.plot(time_list, mag_list, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.z_plot_items[key] = plot_item
                                        plot_item.setVisible(True)
                        
                        self.z_plot.setTitle(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.z_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.z_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.z_plot.setTitle("Z-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.z_plot.setTitle("Z-Axis: Error plotting data")
            else:
                self.z_plot.setTitle("Z-Axis: Select sensors to plot")
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window closing."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        event.accept()


def main():
    """Main function."""
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC (GPU ACCELERATED)")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("Using PyQtGraph with OpenGL acceleration")
    print("="*60)
    
    app = QtWidgets.QApplication(sys.argv)
    window = SensorVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

                                print(f"[DEBUG] Skipping {labels[i]} - invalid data (NaN/Inf)")
                
                self.ft_plot.setTitle("FT Sensor Data (Fx, Fy, Fz)")
                # Set fixed Y range for FT plot: -15 to 15 N
                self.ft_plot.setYRange(-15, 15, padding=0)
                # Auto-range X axis - ensure it starts from 0 or positive
                if len(time_array) > 0:
                    x_min = max(0, time_array[0] - 0.1)
                    x_max = time_array[-1] + 0.1
                    self.ft_plot.setXRange(x_min, x_max, padding=0)
            else:
                if self._debug_counter % 20 == 0:
                    print(f"[DEBUG] FT plot skipped: ft_data={ft_data is not None}, "
                          f"len(relative_time)={len(relative_time) if relative_time else 0}")
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing - SAME LOGIC AS NON-GPU
                    if len(relative_time) == 0 and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = np.array([(t - self.sensor_reader.session_start_time) for t in time_data])
                        else:
                            relative_time = np.array(list(range(len(stretchmagtec_array)))) * 0.01  # Assume 100Hz
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # x=time, y=magnetic field values - ensure arrays are 1D
                                mag_data = sensor_data[:, 0]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.x_plot_items:
                                        self.x_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.x_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.x_plot_items[key] = plot_item
                        
                        self.x_plot.setTitle(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.x_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.x_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.x_plot.setTitle("X-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.x_plot.setTitle("X-Axis: Error plotting data")
            else:
                self.x_plot.setTitle("X-Axis: Select sensors to plot")
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 1]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.y_plot_items:
                                        self.y_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.y_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.y_plot_items[key] = plot_item
                        
                        self.y_plot.setTitle(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.y_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.y_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.y_plot.setTitle("Y-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.y_plot.setTitle("Y-Axis: Error plotting data")
            else:
                self.y_plot.setTitle("Y-Axis: Select sensors to plot")
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 2]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    # Convert to lists to avoid numpy issues
                                    time_list = time_array.tolist() if isinstance(time_array, np.ndarray) else list(time_array)
                                    mag_list = mag_data.tolist() if isinstance(mag_data, np.ndarray) else list(mag_data)
                                    if key in self.z_plot_items:
                                        self.z_plot_items[key].setData(time_list, mag_list, autoDownsample=True)
                                        self.z_plot_items[key].setVisible(True)
                                    else:
                                        plot_item = self.z_plot.plot(time_list, mag_list, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.z_plot_items[key] = plot_item
                                        plot_item.setVisible(True)
                        
                        self.z_plot.setTitle(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.z_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.z_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.z_plot.setTitle("Z-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.z_plot.setTitle("Z-Axis: Error plotting data")
            else:
                self.z_plot.setTitle("Z-Axis: Select sensors to plot")
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window closing."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        event.accept()


def main():
    """Main function."""
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC (GPU ACCELERATED)")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("Using PyQtGraph with OpenGL acceleration")
    print("="*60)
    
    app = QtWidgets.QApplication(sys.argv)
    window = SensorVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
                                print(f"[DEBUG] Skipping {labels[i]} - invalid data (NaN/Inf)")
                
                self.ft_plot.setTitle("FT Sensor Data (Fx, Fy, Fz)")
                # Set fixed Y range for FT plot: -15 to 15 N
                self.ft_plot.setYRange(-15, 15, padding=0)
                # Auto-range X axis - ensure it starts from 0 or positive
                if len(time_array) > 0:
                    x_min = max(0, time_array[0] - 0.1)
                    x_max = time_array[-1] + 0.1
                    self.ft_plot.setXRange(x_min, x_max, padding=0)
            else:
                if self._debug_counter % 20 == 0:
                    print(f"[DEBUG] FT plot skipped: ft_data={ft_data is not None}, "
                          f"len(relative_time)={len(relative_time) if relative_time else 0}")
            
            # Plot X-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined even if FT data is missing - SAME LOGIC AS NON-GPU
                    if len(relative_time) == 0 and len(stretchmagtec_array) > 0:
                        # Create relative time from stretchmagtec data
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = np.array([(t - self.sensor_reader.session_start_time) for t in time_data])
                        else:
                            relative_time = np.array(list(range(len(stretchmagtec_array)))) * 0.01  # Assume 100Hz
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # x=time, y=magnetic field values - ensure arrays are 1D
                                mag_data = sensor_data[:, 0]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.x_plot_items:
                                        self.x_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.x_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.x_plot_items[key] = plot_item
                        
                        self.x_plot.setTitle(f"X-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.x_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.x_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.x_plot.setTitle("X-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting X-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.x_plot.setTitle("X-Axis: Error plotting data")
            else:
                self.x_plot.setTitle("X-Axis: Select sensors to plot")
            
            # Plot Y-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 1]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    if key in self.y_plot_items:
                                        self.y_plot_items[key].setData(time_array, mag_data)
                                    else:
                                        plot_item = self.y_plot.plot(time_array, mag_data, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.y_plot_items[key] = plot_item
                        
                        self.y_plot.setTitle(f"Y-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.y_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.y_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.y_plot.setTitle("Y-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Y-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.y_plot.setTitle("Y-Axis: Error plotting data")
            else:
                self.y_plot.setTitle("Y-Axis: Select sensors to plot")
            
            # Plot Z-axis data
            if stretchmagtec_data and self.selected_sensors:
                try:
                    stretchmagtec_array = np.array(stretchmagtec_data)
                    
                    # Ensure relative_time is defined - SAME LOGIC AS NON-GPU
                    if not relative_time and len(stretchmagtec_array) > 0:
                        if self.sensor_reader.session_start_time and time_data:
                            relative_time = [(t - self.sensor_reader.session_start_time) for t in time_data]
                        else:
                            relative_time = list(range(len(stretchmagtec_array)))
                    
                    if len(relative_time) > 0 and len(stretchmagtec_array) > 0:
                        min_len = min(len(relative_time), len(stretchmagtec_array))
                        relative_time_trimmed = relative_time[:min_len]
                        stretchmagtec_array_trimmed = stretchmagtec_array[:min_len]
                        
                        # Use 100Hz frequency for time axis (0.01s per sample)
                        relative_time_trimmed = [i * 0.01 for i in range(len(relative_time_trimmed))]
                        time_array = np.array(relative_time_trimmed)
                        
                        for sensor_id in sorted(self.selected_sensors):
                            if sensor_id < stretchmagtec_array_trimmed.shape[1]:
                                sensor_data = stretchmagtec_array_trimmed[:, sensor_id, :]
                                color = self.sensor_colors[sensor_id]
                                # Ensure arrays are 1D
                                mag_data = sensor_data[:, 2]
                                if len(time_array) == len(mag_data) and len(mag_data) > 0:
                                    # Convert hex color to RGB tuple for PyQtGraph
                                    if color.startswith('#'):
                                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                    else:
                                        rgb = color
                                    # Update existing plot item or create new one
                                    key = f'S{sensor_id+1}'
                                    # Convert to lists to avoid numpy issues
                                    time_list = time_array.tolist() if isinstance(time_array, np.ndarray) else list(time_array)
                                    mag_list = mag_data.tolist() if isinstance(mag_data, np.ndarray) else list(mag_data)
                                    if key in self.z_plot_items:
                                        self.z_plot_items[key].setData(time_list, mag_list, autoDownsample=True)
                                        self.z_plot_items[key].setVisible(True)
                                    else:
                                        plot_item = self.z_plot.plot(time_list, mag_list, 
                                                                    pen=pg.mkPen(rgb, width=3), name=key)
                                        self.z_plot_items[key] = plot_item
                                        plot_item.setVisible(True)
                        
                        self.z_plot.setTitle(f"Z-Axis: {[f'S{s+1}' for s in sorted(self.selected_sensors)]}")
                        # Set fixed Y range for magnetic plots: -30000 to 30000
                        self.z_plot.setYRange(-30000, 30000, padding=0)
                        # Auto-range X axis - ensure it starts from 0
                        if len(relative_time_trimmed) > 0:
                            x_min = max(0, relative_time_trimmed[0] - 0.1)
                            x_max = relative_time_trimmed[-1] + 0.1
                            self.z_plot.setXRange(x_min, x_max, padding=0)
                    else:
                        self.z_plot.setTitle("Z-Axis: Waiting for data...")
                except Exception as e:
                    print(f"Error plotting Z-axis: {e}")
                    import traceback
                    traceback.print_exc()
                    self.z_plot.setTitle("Z-Axis: Error plotting data")
            else:
                self.z_plot.setTitle("Z-Axis: Select sensors to plot")
            
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window closing."""
        self.update_running = False
        self.sensor_reader.stop_sensors()
        event.accept()


def main():
    """Main function."""
    print("="*60)
    print("SENSOR VISUALIZATION - FT & STRETCHMAGTEC (GPU ACCELERATED)")
    print("="*60)
    print(f"FT sensor port: {FT_PORT}")
    print(f"StretchMagTec port: {STRETCHMAGTEC_PORT}")
    print(f"Sensor configuration: {STRETCHMAGTEC_SENSORS} sensors ({STRETCHMAGTEC_ROWS}x{STRETCHMAGTEC_COLS}) with {STRETCHMAGTEC_CHANNELS} channels each")
    print("Using PyQtGraph with OpenGL acceleration")
    print("="*60)
    
    app = QtWidgets.QApplication(sys.argv)
    window = SensorVisualizationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()