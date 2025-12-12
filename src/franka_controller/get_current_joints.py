#!/usr/bin/env python3
"""
Script to get the current joint state of the Franka robot.

Run this script when the robot is in the desired position to obtain
the joint values to use in the initial movement command.
"""

import pyfranka_interface as franka
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

def main():
    print("="*80)
    print("READING CURRENT ROBOT JOINT STATE")
    print("="*80)
    print()
    
    try:
        # Connect to robot
        print(f"Connecting to robot at {ROBOT_IP}...")
        r = franka.Robot_(ROBOT_IP, False, hand_franka=False)
        print("✅ Robot connected")
        print()
        
        # Get current state
        cur_state = r.getState()
        
        # Extract joint positions (7 joints)
        joint_positions = np.array(cur_state.q)
        
        print("📍 Current joint positions (rad):")
        print(f"   {joint_positions.tolist()}")
        print()
        
        print("📍 Format for copying into code:")
        print(f"   INITIAL_JOINT_POSITIONS = {joint_positions.tolist()}")
        print()
        
        print("📍 Individual values:")
        for i, q in enumerate(joint_positions, 1):
            print(f"   Joint {i}: {q:.6f} rad ({np.degrees(q):.2f} deg)")
        print()
        
        # Also show current cartesian position for reference
        cur_pose = np.array(cur_state.T)
        current_position = cur_pose[:3, 3]
        print("📍 Current cartesian position (for reference):")
        print(f"   X: {current_position[0]:.6f} m")
        print(f"   Y: {current_position[1]:.6f} m")
        print(f"   Z: {current_position[2]:.6f} m")
        print()
        
        print("✅ Values ready to be used in the initial movement command")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


Script to get the current joint state of the Franka robot.

Run this script when the robot is in the desired position to obtain
the joint values to use in the initial movement command.
"""

import pyfranka_interface as franka
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

def main():
    print("="*80)
    print("READING CURRENT ROBOT JOINT STATE")
    print("="*80)
    print()
    
    try:
        # Connect to robot
        print(f"Connecting to robot at {ROBOT_IP}...")
        r = franka.Robot_(ROBOT_IP, False, hand_franka=False)
        print("✅ Robot connected")
        print()
        
        # Get current state
        cur_state = r.getState()
        
        # Extract joint positions (7 joints)
        joint_positions = np.array(cur_state.q)
        
        print("📍 Current joint positions (rad):")
        print(f"   {joint_positions.tolist()}")
        print()
        
        print("📍 Format for copying into code:")
        print(f"   INITIAL_JOINT_POSITIONS = {joint_positions.tolist()}")
        print()
        
        print("📍 Individual values:")
        for i, q in enumerate(joint_positions, 1):
            print(f"   Joint {i}: {q:.6f} rad ({np.degrees(q):.2f} deg)")
        print()
        
        # Also show current cartesian position for reference
        cur_pose = np.array(cur_state.T)
        current_position = cur_pose[:3, 3]
        print("📍 Current cartesian position (for reference):")
        print(f"   X: {current_position[0]:.6f} m")
        print(f"   Y: {current_position[1]:.6f} m")
        print(f"   Z: {current_position[2]:.6f} m")
        print()
        
        print("✅ Values ready to be used in the initial movement command")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


Script to get the current joint state of the Franka robot.

Run this script when the robot is in the desired position to obtain
the joint values to use in the initial movement command.
"""

import pyfranka_interface as franka
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

def main():
    print("="*80)
    print("READING CURRENT ROBOT JOINT STATE")
    print("="*80)
    print()
    
    try:
        # Connect to robot
        print(f"Connecting to robot at {ROBOT_IP}...")
        r = franka.Robot_(ROBOT_IP, False, hand_franka=False)
        print("✅ Robot connected")
        print()
        
        # Get current state
        cur_state = r.getState()
        
        # Extract joint positions (7 joints)
        joint_positions = np.array(cur_state.q)
        
        print("📍 Current joint positions (rad):")
        print(f"   {joint_positions.tolist()}")
        print()
        
        print("📍 Format for copying into code:")
        print(f"   INITIAL_JOINT_POSITIONS = {joint_positions.tolist()}")
        print()
        
        print("📍 Individual values:")
        for i, q in enumerate(joint_positions, 1):
            print(f"   Joint {i}: {q:.6f} rad ({np.degrees(q):.2f} deg)")
        print()
        
        # Also show current cartesian position for reference
        cur_pose = np.array(cur_state.T)
        current_position = cur_pose[:3, 3]
        print("📍 Current cartesian position (for reference):")
        print(f"   X: {current_position[0]:.6f} m")
        print(f"   Y: {current_position[1]:.6f} m")
        print(f"   Z: {current_position[2]:.6f} m")
        print()
        
        print("✅ Values ready to be used in the initial movement command")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


Script to get the current joint state of the Franka robot.

Run this script when the robot is in the desired position to obtain
the joint values to use in the initial movement command.
"""

import pyfranka_interface as franka
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_controller.config import *

def main():
    print("="*80)
    print("READING CURRENT ROBOT JOINT STATE")
    print("="*80)
    print()
    
    try:
        # Connect to robot
        print(f"Connecting to robot at {ROBOT_IP}...")
        r = franka.Robot_(ROBOT_IP, False, hand_franka=False)
        print("✅ Robot connected")
        print()
        
        # Get current state
        cur_state = r.getState()
        
        # Extract joint positions (7 joints)
        joint_positions = np.array(cur_state.q)
        
        print("📍 Current joint positions (rad):")
        print(f"   {joint_positions.tolist()}")
        print()
        
        print("📍 Format for copying into code:")
        print(f"   INITIAL_JOINT_POSITIONS = {joint_positions.tolist()}")
        print()
        
        print("📍 Individual values:")
        for i, q in enumerate(joint_positions, 1):
            print(f"   Joint {i}: {q:.6f} rad ({np.degrees(q):.2f} deg)")
        print()
        
        # Also show current cartesian position for reference
        cur_pose = np.array(cur_state.T)
        current_position = cur_pose[:3, 3]
        print("📍 Current cartesian position (for reference):")
        print(f"   X: {current_position[0]:.6f} m")
        print(f"   Y: {current_position[1]:.6f} m")
        print(f"   Z: {current_position[2]:.6f} m")
        print()
        
        print("✅ Values ready to be used in the initial movement command")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())