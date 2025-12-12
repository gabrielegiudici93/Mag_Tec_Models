#!/usr/bin/env python3
"""
Franka 10 Random Points - Random Order Pressing Script

This script moves the Franka robot to 10 predefined points in random order,
performing force-controlled presses up to 3N at each location.
- No magnetic sensor connection (only FT sensor for force control)
- Randomizes point order (no repeats until all 10 are visited)
- Asks user to press Enter before moving to next point
- Returns to initial joint positions after each complete cycle
- Loops indefinitely until interrupted

Author: Gabriele Giudici
Date: 2025
"""

import numpy as np
import time
import serial
import threading
import random
import sys
import os
import libscrc
import minimalmodbus as mm
import pyfranka_interface as franka
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base position coordinates (point 1 center)
TARGET_POSITION_COORDS = [0.500781, 0.419620, 0.034311]

# Initial joint positions (same as multiple points data collection)
INITIAL_JOINT_POSITIONS = [-1.460883997177473, -1.4397968588005559, 1.8498105422813298, 
                           -1.680352194797862, 1.4646542101436189, 1.8593807739681665, 
                           0.8594902150722012]

# Multi-point offsets relative to center (point 1)
MULTI_POINT_OFFSETS = {
    '1': [0.0, 0.0, 0.0],           # Center
    '2': [-0.01, 0.0, 0.0],        # X-0.01
    '3': [-0.01, 0.01, 0.0],       # X-0.01, Y+0.01
    '4': [0.0, 0.01, 0.0],         # Y+0.01
    '5': [0.0, 0.02, 0.0],         # Y+0.02
    '6': [-0.01, 0.02, 0.0],       # X-0.01, Y+0.02
    '7': [-0.01, 0.03, 0.0],       # X-0.01, Y+0.03
    '8': [0.0, 0.03, 0.0],         # Y+0.03
    '9': [0.0, 0.04, 0.0],         # Y+0.04
    '10': [-0.01, 0.04, 0.0],      # X-0.01, Y+0.04
}

# Force control parameters (same as data collection)
FORCE_MIN = 0.0
FORCE_MAX = 3.0
FORCE_STEP_SIZE = 0.1
FORCE_STEP_DELAY = 0.2  # Reduced from 1.0s to 0.2s
FORCE_TOLERANCE = 0.01
MAX_INDENTATION = 0.010  # 10mm safety limit

# Movement parameters
JOINT_MOVEMENT_SPEED_FACTOR = 0.05
JOINT_MOVEMENT_STEPS = 10
JOINT_MOVEMENT_PAUSE = 0.2
ABSOLUTE_MOVEMENT_DURATION = 5.0

# =============================================================================
# FT SENSOR CLASSES
# =============================================================================

def crcCheck(serialMessage):
    """Check CRC of serial message."""
    if len(serialMessage) < 16:
        return False
    crc = (serialMessage[15] << 8) | serialMessage[14]
    crcCalc = libscrc.modbus(serialMessage[0:14])
    return crc == crcCalc

def forceFromSerialMessage(dataArray, zeroRef=None):
    """Extract force values from serial message."""
    forceTorque = []
    for i in range(6):
        value = (dataArray[i * 2 + 1] << 8) | dataArray[i * 2]
        if value > 32767:
            value = value - 65536
        forceTorque.append(value / 1000.0)
    
    if zeroRef is not None:
        forceTorque = [f - z for f, z in zip(forceTorque, zeroRef)]
    
    return forceTorque

class FTSensorThread(threading.Thread):
    """FT sensor reading thread."""
    def __init__(self, port=FT_PORT, baudrate=FT_BAUDRATE):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        self.force_reading = [0.0] * 6
        self.lock = threading.Lock()
        self.zero_ref = None

    def run(self):
        try:
            # Reset sensor
            ser_tmp = serial.Serial(port=self.port, baudrate=self.baudrate, 
                                   bytesize=8, parity='N', stopbits=1, timeout=1)
            ser_tmp.write(bytearray([0xff] * 50))
            ser_tmp.close()
            time.sleep(0.1)
            
            # Initialize minimalmodbus
            mm.BAUDRATE = self.baudrate
            mm.BYTESIZE = 8
            mm.PARITY = 'N'
            mm.STOPBITS = 1
            mm.TIMEOUT = 1
            
            ft300 = None
            for attempt in range(3):
                try:
                    ft300 = mm.Instrument(self.port, slaveaddress=9)
                    ft300.close_port_after_each_call = True
                    ft300.write_register(410, 0x0200)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        print(f"[FT Thread] ⚠️  Failed to initialize FT sensor after 3 attempts: {e}")
            
            if ft300:
                del ft300
            
            # Open serial connection
            ser = serial.Serial(port=self.port, baudrate=self.baudrate, 
                              bytesize=8, parity='N', stopbits=1, timeout=1)
            
            # Read zero reference
            STARTBYTES = bytes([0x20, 0x4e])
            ser.read_until(STARTBYTES)
            data = ser.read_until(STARTBYTES)
            dataArray = bytearray(data)
            dataArray = STARTBYTES + dataArray[:-2]
            if crcCheck(dataArray):
                self.zero_ref = forceFromSerialMessage(dataArray)
                print(f"[FT Thread] ✅ Zero reference measured: {self.zero_ref}")
            else:
                print("[FT Thread] ⚠️  CRC error on zero reference")
                self.zero_ref = [0.0] * 6
            
            # Main reading loop
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
                            print(f"[FT Thread] ⚠️  Too many CRC errors ({consecutive_errors})")
                        continue
                    
                    consecutive_errors = 0
                    raw_force = forceFromSerialMessage(dataArray, self.zero_ref)
                    
                    with self.lock:
                        self.force_reading = raw_force.copy()
                    
                except serial.SerialTimeoutException:
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors % 10 == 0:
                        print(f"[FT Thread] ⚠️  Error reading FT data: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[FT Thread] ⚠️  Too many errors, stopping")
                        break
                    time.sleep(0.01)
                    continue
            
            ser.close()
        except Exception as e:
            print(f"[FT Thread] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            with self.lock:
                self.force_reading = [float('nan')] * 6

    def get_ft(self):
        """Return current force reading."""
        with self.lock:
            return self.force_reading.copy()

    def stop(self):
        """Stop the thread."""
        self.running = False

# =============================================================================
# ROBOT MOVEMENT FUNCTIONS
# =============================================================================

def move_joints_safe(r, target_joints, speed_factor=JOINT_MOVEMENT_SPEED_FACTOR):
    """Move robot to target joint positions safely in steps."""
    current_joints = r.getState().q
    joint_diffs = np.array(target_joints) - np.array(current_joints)
    num_steps = max(JOINT_MOVEMENT_STEPS, int(np.max(np.abs(joint_diffs)) / 0.1))
    
    for step in range(num_steps):
        alpha = (step + 1) / num_steps
        intermediate_joints = current_joints + alpha * joint_diffs
        r.move_joints(intermediate_joints, speed_factor)
        time.sleep(JOINT_MOVEMENT_PAUSE)
    
    # Verify final position
    final_joints = r.getState().q
    joint_error = np.max(np.abs(np.array(final_joints) - np.array(target_joints)))
    if joint_error > 0.05:
        print(f"⚠️  Warning: Joint position error {joint_error:.4f} rad after movement")

def safe_robot_move(r, move_type, target, duration=None, max_retries=3):
    """
    Safely move robot with retry logic for reflex errors.
    Same as in franka_skin_test.py
    """
    for attempt in range(max_retries):
        try:
            if duration is not None:
                r.move(move_type, target, duration)
            else:
                r.move(move_type, target)
            return True
        except Exception as e:
            error_str = str(e)
            print(f"   ❌ Movement attempt {attempt + 1}/{max_retries} failed: {error_str}")
            
            # Check if it's a reflex mode error
            if "Reflex" in error_str or "reflex" in error_str.lower():
                print("   🛑 Robot in reflex mode (safety stop).")
                print("   🔓 Please unlock the safety button on the robot and reset it, then press Enter to continue...")
                try:
                    input()
                    print("   ✅ Robot reset acknowledged, retrying movement...")
                    time.sleep(3)  # Give robot time to fully reset
                except KeyboardInterrupt:
                    print("   ⚠️  User interrupted during reflex recovery.")
                    return False
            elif attempt < max_retries - 1:
                print(f"   🔄 Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"   ❌ Failed to complete movement after {max_retries} attempts")
                raise
    return False

def move_to_position(r, target_pos, orientation=None):
    """Move robot to target Cartesian position."""
    if orientation is None:
        # Use default orientation (same as data collection)
        orientation = R.from_euler('xyz', [180, 0, 0], degrees=True)
    
    target_pose = np.eye(4)
    target_pose[:3, :3] = orientation.as_matrix()
    target_pose[:3, 3] = target_pos
    
    safe_robot_move(r, "absolute", target_pose, duration=ABSOLUTE_MOVEMENT_DURATION)

def move_relative(r, dx, dy, dz, duration=0.05):
    """Move robot relative to current position."""
    delta_transform = np.eye(4)
    delta_transform[:3, 3] = [dx, dy, dz]
    safe_robot_move(r, "relative", delta_transform, duration=duration)

def perform_force_controlled_press(r, ft_thread, target_forces):
    """Perform force-controlled press up to target forces."""
    print(f"  Starting force-controlled press: {target_forces}N")
    
    # Get starting position
    start_state = r.getState()
    start_z = start_state.T[2, 3]
    print(f"  Starting Z position: {start_z:.6f}m")
    
    # Wait for stabilization
    time.sleep(0.2)
    
    for force_step, target_force in enumerate(target_forces):
        print(f"  Target force: {target_force:.1f}N (step {force_step + 1}/{len(target_forces)})")
        
        max_iterations = 500
        iteration = 0
        
        while iteration < max_iterations:
            # Check current position
            current_state = r.getState()
            current_z = current_state.T[2, 3]
            current_indentation = abs(start_z - current_z)
            
            # Safety check
            if current_indentation >= MAX_INDENTATION:
                print(f"    ⚠️  Safety stop: Maximum indentation ({MAX_INDENTATION*1000:.1f}mm) reached")
                break
            
            # Read force
            current_ft = ft_thread.get_ft()
            current_fz_abs = abs(current_ft[2])
            
            # Check if target reached
            if current_fz_abs >= target_force - FORCE_TOLERANCE:
                print(f"    Target reached: {current_fz_abs:.3f}N (indentation: {current_indentation*1000:.2f}mm)")
                break
            elif current_fz_abs < target_force - FORCE_TOLERANCE:
                # Press down
                move_relative(r, 0, 0, -0.0001, duration=0.05)
                time.sleep(0.05)
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"    ⚠️  Warning: Max iterations reached for {target_force:.1f}N target")
        
        # Stay at this force level
        print(f"    Holding at {target_force:.1f}N for {FORCE_STEP_DELAY:.1f}s...")
        time.sleep(FORCE_STEP_DELAY)
    
    # Return to starting Z position
    current_state = r.getState()
    current_z = current_state.T[2, 3]
    z_diff = start_z - current_z
    if abs(z_diff) > 0.001:
        print(f"  Returning to starting Z position (difference: {z_diff*1000:.2f}mm)...")
        move_relative(r, 0, 0, z_diff, duration=0.5)
        time.sleep(0.5)

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 70)
    print(" FRANKA 10 RANDOM POINTS - RANDOM ORDER PRESSING")
    print("=" * 70)
    print(f"Base position: {TARGET_POSITION_COORDS}")
    print(f"Points: {list(MULTI_POINT_OFFSETS.keys())}")
    print(f"Force range: {FORCE_MIN}N to {FORCE_MAX}N (step: {FORCE_STEP_SIZE}N)")
    print("=" * 70 + "\n")
    
    # Connect to robot
    print(f"Connecting to robot at {ROBOT_IP}...")
    r = None
    max_connection_retries = 5
    for attempt in range(max_connection_retries):
        try:
            r = franka.Robot_(ROBOT_IP, False, hand_franka=False, auto_init=True, speed_factor=ROBOT_SPEED_FACTOR)
            print("✅ Robot connected successfully")
            break  # Success, exit retry loop
        except KeyboardInterrupt:
            raise  # Re-raise to be caught by outer handler
        except Exception as e:
            error_str = str(e)
            print(f"Error connecting to robot (attempt {attempt + 1}/{max_connection_retries}): {error_str}")
            
            # Check if it's a reflex mode error
            if "Reflex" in error_str or "reflex" in error_str.lower():
                print("🛑 Robot is in reflex mode (safety stop).")
                print("   Please unlock the robot and press Enter to retry...")
                try:
                    input()
                except KeyboardInterrupt:
                    raise
            
            if attempt < max_connection_retries - 1:
                time.sleep(2.0)  # Wait before retry
            else:
                print(f"❌ Failed to connect to robot after {max_connection_retries} attempts")
                return
    
    # Connect to FT sensor
    print("Connecting to FT sensor...")
    ft_thread = None
    try:
        ft_thread = FTSensorThread()
        ft_thread.daemon = True
        ft_thread.start()
        time.sleep(2.0)  # Wait for sensor to initialize
        print("✅ FT sensor connected")
    except Exception as e:
        print(f"❌ Failed to connect to FT sensor: {e}")
        print("   Force control requires FT sensor. Exiting.")
        r.stop()
        return
    
    # Move to initial joint positions
    print("\nMoving to initial joint positions...")
    try:
        move_joints_safe(r, INITIAL_JOINT_POSITIONS)
        print("✅ Initial joint positions reached")
    except Exception as e:
        print(f"⚠️  Error moving to initial joints: {e}")
    
    # Main loop
    cycle_count = 0
    point_names = list(MULTI_POINT_OFFSETS.keys())
    
    try:
        while True:
            cycle_count += 1
            print("\n" + "=" * 70)
            print(f" CYCLE {cycle_count}")
            print("=" * 70)
            
            # Randomize point order (no repeats until all visited)
            random.shuffle(point_names)
            print(f"Random order: {', '.join(point_names)}")
            
            # Visit each point
            for point_idx, point_name in enumerate(point_names):
                print(f"\n--- Point {point_name} ({point_idx + 1}/10) ---")
                
                # Calculate target position
                offset = MULTI_POINT_OFFSETS[point_name]
                target_pos = np.array(TARGET_POSITION_COORDS) + np.array(offset)
                print(f"Target position: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
                
                # Move to position
                print("Moving to position...")
                try:
                    move_to_position(r, target_pos)
                    time.sleep(0.5)  # Stabilization
                    print("✅ Position reached")
                except Exception as e:
                    print(f"⚠️  Error moving to position: {e}")
                    continue
                
                # Perform force-controlled press
                target_forces = np.arange(FORCE_MIN, FORCE_MAX + FORCE_STEP_SIZE, FORCE_STEP_SIZE).tolist()
                try:
                    perform_force_controlled_press(r, ft_thread, target_forces)
                    print("✅ Press completed")
                except Exception as e:
                    print(f"⚠️  Error during press: {e}")
                
                # Wait for user input before next point
                if point_idx < len(point_names) - 1:
                    print("\nPress Enter to move to next point...")
                    try:
                        input()
                    except KeyboardInterrupt:
                        print("\n⚠️  Interrupted by user")
                        raise
            
            # Return to initial joint positions after cycle
            print("\n--- Cycle complete ---")
            print("Returning to initial joint positions...")
            try:
                move_joints_safe(r, INITIAL_JOINT_POSITIONS)
                print("✅ Initial joint positions reached")
            except Exception as e:
                print(f"⚠️  Error returning to initial joints: {e}")
            
            # Ask user before next cycle
            print("\nPress Enter to start next cycle (Ctrl+C to exit)...")
            try:
                input()
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                raise
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown requested")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping robot and sensors...")
        if ft_thread:
            ft_thread.stop()
            ft_thread.join(timeout=2.0)
        if r is not None:
            try:
                r.stop()
            except Exception as e:
                # Robot might already be stopped or connection lost
                print(f"  ⚠️  Note: {e}")
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()

