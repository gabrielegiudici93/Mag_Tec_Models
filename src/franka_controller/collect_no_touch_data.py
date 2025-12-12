#!/usr/bin/env python3
"""
No-Touch Data Collection Script

This script collects sensor data without any contact for training a "no touch" classifier.
It connects only to FT and StretchMagTec sensors (no robot) and records sequences.

Features:
- Records 33 sequences of 5 seconds each
- No robot movement required
- Saves data in HDF5 format compatible with training pipeline
- Includes calibration for both sensors
- Supports multiple stretch levels (000pct, 010pct, 020pct)

Usage:
    python3 src/franka_controller/collect_no_touch_data.py --stretch 000pct
    python3 src/franka_controller/collect_no_touch_data.py --stretch 010pct
    python3 src/franka_controller/collect_no_touch_data.py --stretch 020pct

Author: Gabriele Giudici
Date: 2025
"""

import numpy as np
import time
import serial
import threading
import re
from datetime import datetime
from collections import deque
import h5py
import libscrc
import minimalmodbus as mm
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *
import franka_controller.config as config_module

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_SEQUENCES = 33  # Number of sequences to record
SEQUENCE_DURATION = 5.0  # Duration of each sequence in seconds
SAMPLING_RATE = 100  # Hz
SAMPLING_PERIOD = 1.0 / SAMPLING_RATE

# Output directory (will be set based on --data-dir and --run-label)

# =============================================================================
# FT SENSOR CALIBRATION
# =============================================================================
class DynamicFTCalibration:
    """FT sensor calibration system."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.current_offset = [0, 0, 0, 0, 0, 0]
        self.is_calibrated = False
    
    def measure_offset(self, ft_ser, description="calibration"):
        """Measure the force offset during calibration period."""
        if not self.enabled:
            print(f"FT calibration disabled - skipping {description}")
            return self.current_offset
            
        print(f"Starting {description} measurement ({FT_CALIBRATION_DURATION} seconds)...")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < FT_CALIBRATION_DURATION:
            try:
                force_reading = self._read_ft(ft_ser)
                if force_reading is not None:
                    samples.append(force_reading)
            except:
                pass
            time.sleep(0.01)  # 100 Hz sampling
        
        if samples:
            samples_array = np.array(samples)
            self.current_offset = np.mean(samples_array, axis=0).tolist()
            self.is_calibrated = True
            print(f"{description.capitalize()} complete:")
            print(f"  Mean offset: {[round(x, 3) for x in self.current_offset]}")
            return self.current_offset
        else:
            print(f"Warning: No samples collected during {description}")
            return [0, 0, 0, 0, 0, 0]
    
    def _read_ft(self, ft_ser):
        """Read FT sensor data."""
        try:
            STARTBYTES = bytes([0x20, 0x4e])
            ft_ser.write(STARTBYTES)
            data = ft_ser.read(25)
            
            if len(data) == 25 and data[0] == 0x20 and data[1] == 0x4e:
                dataArray = list(data[2:24])
                crc = libscrc.modbus(dataArray)
                crcLow = crc & 0xFF
                crcHigh = (crc >> 8) & 0xFF
                
                if crcLow == data[24] and crcHigh == data[23]:
                    forceTorque = []
                    for i in range(6):
                        value = (dataArray[i * 2 + 1] << 8) | dataArray[i * 2]
                        if value > 32767:
                            value = value - 65536
                        forceTorque.append(value / 1000.0)
                    return forceTorque
        except:
            pass
        return None
    
    def compensate_force(self, force_reading):
        """Apply offset compensation."""
        if not self.is_calibrated:
            return force_reading
        return [f - o for f, o in zip(force_reading, self.current_offset)]

# =============================================================================
# STRETCHMAGTEC CALIBRATION
# =============================================================================
class StretchMagTecCalibration:
    """StretchMagTec sensor calibration system."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.offsets = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.is_calibrated = False
    
    def measure_offsets(self, sensor_reader, description="StretchMagTec calibration"):
        """Measure offsets for all sensors."""
        if not self.enabled:
            print(f"StretchMagTec calibration disabled - skipping {description}")
            return self.offsets
            
        print(f"Starting {description} ({STRETCHMAGTEC_CALIBRATION_DURATION} seconds)...")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < STRETCHMAGTEC_CALIBRATION_DURATION:
            sensor_data = sensor_reader.get_latest_data()
            if sensor_data is not None and sensor_data.shape == (STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS):
                samples.append(sensor_data.copy())
            time.sleep(0.01)  # 100 Hz sampling
        
        if samples:
            samples_array = np.array(samples)
            self.offsets = np.mean(samples_array, axis=0)
            self.is_calibrated = True
            print(f"{description.capitalize()} complete!")
            for sensor_id in range(STRETCHMAGTEC_SENSORS):
                print(f"  Sensor {sensor_id+1}: offset = {[round(x, 2) for x in self.offsets[sensor_id]]}")
            return self.offsets
        else:
            print(f"Warning: No samples collected during {description}")
            return np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
    
    def compensate_sensors(self, sensor_data):
        """Apply offset compensation."""
        if not self.is_calibrated or sensor_data is None:
            return sensor_data
        return sensor_data - self.offsets

# =============================================================================
# STRETCHMAGTEC SENSOR READER
# =============================================================================
class StretchMagTecReader:
    """Reads data from StretchMagTec sensors."""
    
    def __init__(self):
        self.ser = None
        self.running = False
        self.latest_data = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        self.lock = threading.Lock()
        self.thread = None
    
    def start(self):
        """Start reading thread."""
        if self.running:
            return
        
        # Try to find available port
        port_found = False
        for port_num in range(10):
            port = f"/dev/ttyACM{port_num}"
            try:
                self.ser = serial.Serial(port, STRETCHMAGTEC_BAUD, timeout=1)
                print(f"✅ Connected to {port} at {STRETCHMAGTEC_BAUD} baud")
                port_found = True
                break
            except (FileNotFoundError, serial.SerialException):
                continue
        
        if not port_found:
            raise RuntimeError(f"Could not find StretchMagTec sensor on /dev/ttyACM0-9")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
        # Wait for first data
        time.sleep(2.0)
    
    def stop(self):
        """Stop reading thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.ser:
            self.ser.close()
    
    def get_latest_data(self):
        """Get latest sensor data."""
        with self.lock:
            return self.latest_data.copy()
    
    def _read_loop(self):
        """Main reading loop."""
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                sensor_values = self._parse_line(line)
                if sensor_values is not None:
                    with self.lock:
                        self.latest_data[:] = sensor_values
            except:
                pass
            time.sleep(0.001)
    
    def _parse_line(self, line):
        """Parse sensor data line."""
        sensor_values = np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS))
        
        # Try format: "S1: X=... Y=... Z=... | S2: ..."
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
        
        # Try regex format
        pattern = r'S(\d+):\s*X=([-\d.]+)\s*Y=([-\d.]+)\s*Z=([-\d.]+)'
        matches = re.findall(pattern, line)
        
        if matches:
            for match in matches:
                sensor_id = int(match[0]) - 1
                if 0 <= sensor_id < STRETCHMAGTEC_SENSORS:
                    try:
                        sensor_values[sensor_id, 0] = float(match[1])
                        sensor_values[sensor_id, 1] = float(match[2])
                        sensor_values[sensor_id, 2] = float(match[3])
                    except ValueError:
                        continue
            
            if np.any(sensor_values != 0):
                return sensor_values
        
        return None

# =============================================================================
# MAIN DATA COLLECTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect no-touch data for training")
    parser.add_argument("--stretch", type=str, required=True, 
                       choices=['000pct', '010pct', '020pct'],
                       help="Stretch level (000pct, 010pct, or 020pct)")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Data directory (e.g., data/Multiple_Points/2.5mm_single_test24)")
    parser.add_argument("--run-label", type=str, required=True,
                       help="Run label (e.g., test24)")
    parser.add_argument("--run-training", action="store_true",
                       help="Run training automatically after data collection")
    args = parser.parse_args()
    
    stretch_level = args.stretch
    data_dir = Path(args.data_dir)
    run_label = args.run_label
    
    # Create output directory
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NO-TOUCH DATA COLLECTION")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Run label: {run_label}")
    print(f"Stretch level: {stretch_level}")
    print(f"Sequences: {NUM_SEQUENCES}")
    print(f"Duration per sequence: {SEQUENCE_DURATION} seconds")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print("=" * 70 + "\n")
    
    # Initialize FT sensor (optional for no-touch data collection)
    print("Initializing FT sensor...")
    ft_ser = None
    ft_available = False
    try:
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
        
        ft_ser = serial.Serial(port=FT_PORT, baudrate=FT_BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1)
        ft_available = True
        print(f"✅ FT sensor connected on {FT_PORT}")
    except Exception as e:
        print(f"⚠️  FT sensor not available: {e}")
        print("   Continuing without FT sensor (only StretchMagTec data will be recorded)")
        ft_available = False
    
    # Initialize StretchMagTec sensor
    print("Initializing StretchMagTec sensor...")
    stretch_reader = StretchMagTecReader()
    try:
        stretch_reader.start()
        print("✅ StretchMagTec sensor connected")
    except Exception as e:
        print(f"❌ Failed to connect to StretchMagTec sensor: {e}")
        if ft_ser:
            ft_ser.close()
        return
    
    # Calibration
    print("\n" + "=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    ft_cal = DynamicFTCalibration(enabled=True)
    stretch_cal = StretchMagTecCalibration(enabled=True)
    
    # FT calibration (only if FT sensor is available)
    if ft_available:
        zero_ref = ft_cal.measure_offset(ft_ser, "initial FT calibration")
    else:
        print("⚠️  Skipping FT calibration (sensor not available)")
        ft_cal.is_calibrated = False
    
    # StretchMagTec calibration
    time.sleep(2.0)  # Wait for stream to stabilize
    stretch_cal.measure_offsets(stretch_reader, "initial StretchMagTec calibration")
    
    print("=" * 70 + "\n")
    
    # Data collection
    print("Starting data collection...")
    print("Press Enter to start recording sequences...")
    input()
    
    all_forces = []
    all_stretchmagtec = []
    all_timestamps = []
    all_labels = []
    all_positions = []  # Dummy positions (no robot)
    
    sequence_starts = []
    sequence_ends = []
    
    for seq_num in range(NUM_SEQUENCES):
        print(f"\nSequence {seq_num + 1}/{NUM_SEQUENCES}...")
        
        seq_forces = []
        seq_stretchmagtec = []
        seq_timestamps = []
        seq_positions = []
        
        start_time = time.time()
        last_sample_time = start_time
        
        # Record sequence
        while time.time() - start_time < SEQUENCE_DURATION:
            current_time = time.time()
            
            # Read FT sensor (if available)
            if ft_available:
                try:
                    force_reading = ft_cal._read_ft(ft_ser)
                    if force_reading is not None:
                        compensated_force = ft_cal.compensate_force(force_reading)
                        seq_forces.append(compensated_force)
                    else:
                        seq_forces.append([0.0] * 6)
                except:
                    seq_forces.append([0.0] * 6)
            else:
                seq_forces.append([0.0] * 6)
            
            # Read StretchMagTec sensor
            stretch_data = stretch_reader.get_latest_data()
            if stretch_data is not None:
                compensated_stretch = stretch_cal.compensate_sensors(stretch_data)
                seq_stretchmagtec.append(compensated_stretch.copy())
            else:
                seq_stretchmagtec.append(np.zeros((STRETCHMAGTEC_SENSORS, STRETCHMAGTEC_CHANNELS)))
            
            # Timestamp
            seq_timestamps.append(current_time)
            
            # Dummy position (no robot)
            seq_positions.append([0.0, 0.0, 0.0])
            
            # Maintain sampling rate
            elapsed = time.time() - last_sample_time
            sleep_time = max(0, SAMPLING_PERIOD - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_sample_time = time.time()
        
        # Mark sequence boundaries
        if seq_num == 0:
            sequence_starts.append(0)
        else:
            sequence_starts.append(len(all_forces))
        
        # Append to main arrays
        all_forces.extend(seq_forces)
        all_stretchmagtec.extend(seq_stretchmagtec)
        all_timestamps.extend(seq_timestamps)
        all_positions.extend(seq_positions)
        
        # Labels: "sequence_start", "no_touch", "sequence_end"
        if seq_num == 0:
            all_labels.append(f"sequence_start_no_touch_{seq_num:03d}".encode('utf-8'))
        else:
            all_labels.append(f"sequence_start_no_touch_{seq_num:03d}".encode('utf-8'))
        
        for _ in range(len(seq_forces) - 2):
            all_labels.append(f"no_touch_{seq_num:03d}".encode('utf-8'))
        
        all_labels.append(f"sequence_end_no_touch_{seq_num:03d}".encode('utf-8'))
        sequence_ends.append(len(all_forces) - 1)
        
        print(f"  Recorded {len(seq_forces)} samples")
    
    # Convert to numpy arrays
    forces_array = np.array(all_forces)
    stretchmagtec_array = np.array(all_stretchmagtec)
    timestamps_array = np.array(all_timestamps)
    positions_array = np.array(all_positions)
    labels_array = np.array(all_labels, dtype='|S64')
    
    # Save to HDF5 (same naming convention as regular data collection)
    filename = output_dir / f"no_touch_{run_label}_stretch_{stretch_level}.h5"
    
    print(f"\nSaving data to {filename}...")
    
    with h5py.File(filename, "w") as f:
        # Top-level datasets
        f.create_dataset("forces", data=forces_array)
        f.create_dataset("stretchmagtec", data=stretchmagtec_array)
        f.create_dataset("positions", data=positions_array)
        f.create_dataset("timestamps", data=timestamps_array)
        f.create_dataset("labels", data=labels_array)
        
        # Attributes
        f.attrs["run_label"] = "no_touch"
        f.attrs["num_sequences"] = NUM_SEQUENCES
        f.attrs["sequence_duration"] = SEQUENCE_DURATION
        f.attrs["sampling_rate"] = SAMPLING_RATE
        f.attrs["timestamp"] = timestamp_str
        f.attrs["stretch"] = stretch_level
        
        # Press groups (one per sequence)
        presses_group = f.create_group("presses")
        cumulative_idx = 0
        
        for seq_idx in range(NUM_SEQUENCES):
            start_idx = sequence_starts[seq_idx]
            end_idx = sequence_ends[seq_idx] + 1  # +1 for exclusive end
            
            press_group = presses_group.create_group(f"press_{seq_idx:03d}")
            press_group.create_dataset("forces", data=forces_array[start_idx:end_idx])
            press_group.create_dataset("stretchmagtec", data=stretchmagtec_array[start_idx:end_idx])
            press_group.create_dataset("positions", data=positions_array[start_idx:end_idx])
            
            # Normalize timestamps
            seq_timestamps = timestamps_array[start_idx:end_idx]
            if len(seq_timestamps) > 0:
                seq_start_time = seq_timestamps[0]
                relative_timestamps = seq_timestamps - seq_start_time
                press_group.create_dataset("timestamps", data=relative_timestamps)
            else:
                press_group.create_dataset("timestamps", data=np.array([]))
            
            # Attributes
            press_group.attrs["offset"] = "no_touch"
            press_group.attrs["stretch"] = stretch_level
            press_group.attrs["label"] = f"no_touch_{stretch_level}_{seq_idx:03d}"
            press_group.attrs["start_idx"] = cumulative_idx
            press_group.attrs["end_idx"] = cumulative_idx + (end_idx - start_idx) - 1
            
            cumulative_idx += (end_idx - start_idx)
    
    print(f"✅ Data saved: {filename}")
    print(f"   Total samples: {len(all_forces)}")
    print(f"   Sequences: {NUM_SEQUENCES}")
    
    # Cleanup
    stretch_reader.stop()
    if ft_ser:
        ft_ser.close()
    
    print("\n✅ Data collection complete!")
    print(f"   File saved: {filename}")
    
    # Run training if requested
    if args.run_training:
        print("\n" + "=" * 70)
        print("RUNNING TRAINING")
        print("=" * 70)
        
        import subprocess
        train_script = Path(__file__).parent.parent / "training" / "train_multipoint.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--data-dir", str(data_dir),
            "--run-label", run_label,
            "--feature-method", "raw"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent.parent)
            print("\n✅ Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            sys.exit(1)

if __name__ == "__main__":
    main()

