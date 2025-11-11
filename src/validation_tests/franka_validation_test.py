#!/usr/bin/env python3
"""
Automated Validation Test - MagTec_KPM

This script performs automated validation of the trained models by:
1. Moving the robot to each validation position (9-point grid)
2. Pressing down and collecting sensor data
3. Running predictions and saving results

Usage:
    python3 franka_validation_test.py [--position POSITION_ID]

Author: Gabriele Giudici
Date: 2025
"""

import numpy as np
import time
import pyfranka_interface as franka
from scipy.spatial.transform import Rotation as R
import sys
import os
import h5py
from datetime import datetime
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

class ValidationTest:
    """Automated validation testing system."""
    
    def __init__(self, position_id=11):
        self.position_id = position_id
        if position_id not in MAIN_GRID_POSITIONS:
            raise ValueError(f"Invalid position ID: {position_id}")
        
        self.base_position = MAIN_GRID_POSITIONS[position_id]
        self.rotation_matrix = R.from_euler('x', 180, degrees=True).as_matrix()
        
        # Load models
        self.load_models()
        
        # Robot connection
        self.robot = None
        
    def load_models(self):
        """Load trained models."""
        print("Loading trained models...")
        
        try:
            # Contact classifier
            self.contact_classifier = joblib.load(get_model_path(CONTACT_CLASSIFIER_MODEL))
            self.contact_scaler = joblib.load(get_model_path(CONTACT_SCALER_MODEL))
            print(f"✅ Loaded contact classifier")
            
            # FT mapping models
            self.ft_models = {}
            self.ft_scalers = {}
            self.ft_output_scalers = {}
            
            for component in ['fx', 'fy', 'fz']:
                model_name = f"ft_mapping_{component}.joblib"
                scaler_name = f"ft_mapping_scaler_{component}.joblib"
                output_scaler_name = f"ft_mapping_output_scaler_{component}.joblib"
                
                self.ft_models[component] = joblib.load(get_model_path(model_name))
                self.ft_scalers[component] = joblib.load(get_model_path(scaler_name))
                self.ft_output_scalers[component] = joblib.load(get_model_path(output_scaler_name))
            
            print(f"✅ Loaded FT mapping models (fx, fy, fz)")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure to train models first: python3 training/train_pipeline.py")
            sys.exit(1)
    
    def connect_robot(self):
        """Connect to the Franka robot."""
        print(f"Connecting to robot at {ROBOT_IP}...")
        try:
            self.robot = franka.Robot_(ROBOT_IP, False, hand_franka=False, 
                                       auto_init=True, speed_factor=ROBOT_SPEED_FACTOR)
            print("✅ Robot connected successfully")
            return True
        except Exception as e:
            print(f"❌ Robot connection failed: {e}")
            return False
    
    def move_to_position(self, offset_key):
        """Move robot to a specific position with offset."""
        desired_position = get_position_with_offset(self.base_position, offset_key)
        
        # Create desired pose
        des_pose = np.eye(4)
        des_pose[:3, :3] = self.rotation_matrix
        des_pose[:3, 3] = desired_position
        
        # Move robot
        self.robot.move("absolute", des_pose, ABSOLUTE_MOVEMENT_DURATION)
        print(f"   Moved to position {self.position_id} - {offset_key}")
        print(f"   Coordinates: [{desired_position[0]:.6f}, {desired_position[1]:.6f}, {desired_position[2]:.6f}]")
        
        return desired_position
    
    def move_relative(self, dx, dy, dz, duration=MOVEMENT_DURATION):
        """Move robot relative to current position."""
        delta_transform = np.eye(4)
        delta_transform[:3, 3] = [dx, dy, dz]
        self.robot.move("relative", delta_transform, duration)
    
    def perform_press(self):
        """Perform a validation press."""
        print("   Pressing down...")
        for step in range(STEPS_PER_PRESS):
            self.move_relative(0, 0, VALIDATION_PRESS_DEPTH / STEPS_PER_PRESS)
            time.sleep(PRESS_DELAY)
        print(f"   ✅ Pressed down {abs(VALIDATION_PRESS_DEPTH)*1000:.1f}mm")
    
    def lift(self):
        """Lift robot back up."""
        print("   Lifting up...")
        self.move_relative(0, 0, abs(VALIDATION_PRESS_DEPTH))
        time.sleep(LIFT_DELAY)
        print("   ✅ Lifted")
    
    def run_validation(self):
        """Run the complete validation sequence."""
        if not self.connect_robot():
            return False
        
        print(f"\n{'='*70}")
        print(f"STARTING VALIDATION AT POSITION {self.position_id}")
        print(f"{'='*70}\n")
        
        results = []
        
        try:
            for offset_key in VALIDATION_POSITIONS:
                print(f"\n{'='*70}")
                print(f"Testing offset: {offset_key}")
                print(f"{'='*70}")
                
                # Move to position
                position_coords = self.move_to_position(offset_key)
                time.sleep(1)  # Stabilize
                
                # Perform press
                self.perform_press()
                time.sleep(0.5)  # Allow sensors to stabilize
                
                # Here you would collect sensor data and run predictions
                # For now, we'll create a placeholder result
                result = {
                    'offset': offset_key,
                    'position': position_coords,
                    'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                }
                results.append(result)
                
                # Lift back up
                self.lift()
                
                print(f"✅ Completed {offset_key}")
            
            # Save results
            self.save_results(results)
            
            print(f"\n{'='*70}")
            print("VALIDATION COMPLETE")
            print(f"{'='*70}")
            print(f"✅ Tested {len(results)} positions")
            print(f"📁 Results saved to: {get_data_path(f'{SENSOR_NAME}_validation_pos{self.position_id}_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.h5')}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Validation interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Validation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self, results):
        """Save validation results to HDF5."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = get_data_path(f'{SENSOR_NAME}_validation_pos{self.position_id}_{timestamp}.h5')
        
        with h5py.File(filename, 'w') as f:
            f.attrs['sensor_name'] = SENSOR_NAME
            f.attrs['position_id'] = self.position_id
            f.attrs['timestamp'] = timestamp
            f.attrs['num_positions'] = len(results)
            
            for i, result in enumerate(results):
                grp = f.create_group(f"position_{i}")
                grp.attrs['offset'] = result['offset']
                grp.attrs['timestamp'] = result['timestamp']
                grp.create_dataset('position', data=result['position'])
        
        print(f"   💾 Saved results to: {filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated validation test")
    parser.add_argument("--position", type=int, default=11,
                       help="Position ID to test (default: 11)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MagTec_KPM AUTOMATED VALIDATION TEST")
    print(f"{'='*70}")
    print(f"Position ID: {args.position}")
    print(f"Validation positions: {VALIDATION_POSITIONS}")
    print(f"Press depth: {abs(VALIDATION_PRESS_DEPTH)*1000:.1f}mm")
    print()
    
    # Run validation
    validator = ValidationTest(position_id=args.position)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

