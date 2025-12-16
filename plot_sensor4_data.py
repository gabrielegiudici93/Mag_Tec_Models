#!/usr/bin/env python3
"""
Plot sensor data for points 4-7 (sensor 4) and points 9-10 (sensors 1, 2, 3) from test28 and test29
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
test28_file = Path("data/Multiple_Points/2.5mm_single_test28/2.5mm_single_test28_stretch_000pct.h5")
test29_file = Path("data/Multiple_Points/2.5mm_single_test29/2.5mm_single_test29_stretch_010pct.h5")

# Points and sensors to extract
# Points 4-7: Sensor 4 (index 3)
points_sensor4 = ['4', '5', '6', '7']
sensor4_idx = 3  # Sensor 4 (0-indexed: 0=1, 1=2, 2=3, 3=4)

# Points 9-10: Sensors 1, 2, 3 (indices 0, 1, 2)
points_sensors123 = ['9', '10']
sensors123_idx = [0, 1, 2]  # Sensors 1, 2, 3

def extract_sensor_data(h5_file, points, sensor_idx):
    """Extract sensor data for specific points."""
    results = {}
    
    if not h5_file.exists():
        print(f"⚠️  File not found: {h5_file}")
        return results
    
    with h5py.File(h5_file, 'r') as f:
        if 'presses' not in f:
            print(f"⚠️  No 'presses' group in {h5_file.name}")
            return results
        
        presses = f['presses']
        for press_key in sorted(presses.keys()):
            press = presses[press_key]
            
            # Get offset
            offset = press.attrs.get('offset', None)
            if offset is None:
                continue
            if isinstance(offset, bytes):
                offset = offset.decode('utf-8')
            
            # Check if this is one of our target points
            if offset not in points:
                continue
            
            # Load stretchmagtec data: [samples, 15, 3]
            if 'stretchmagtec' not in press:
                continue
            
            stretchmagtec = press['stretchmagtec'][:]  # [samples, 15, 3]
            
            # Extract sensor 4 (index 3) data: [samples, 3] (X, Y, Z)
            sensor_data = stretchmagtec[:, sensor_idx, :]  # [samples, 3]
            
            # Get timestamps
            if 'timestamps' in press:
                timestamps = press['timestamps'][:]
            else:
                timestamps = np.arange(len(sensor_data))
            
            if offset not in results:
                results[offset] = []
            
            results[offset].append({
                'press_key': press_key,
                'sensor_data': sensor_data,
                'timestamps': timestamps,
                'num_samples': len(sensor_data)
            })
    
    return results

def plot_sensor_data(test_name, test_file, points, sensor_idx, output_dir, sensor_name="Sensor 4", points_label="4-7"):
    """Plot sensor data for all points."""
    results = extract_sensor_data(test_file, points, sensor_idx)
    
    if not results:
        print(f"No data found for {test_name}")
        return
    
    # Create figure with subplots for each point
    fig, axes = plt.subplots(len(points), 3, figsize=(18, 4*len(points)))
    if len(points) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{test_name}: {sensor_name} Data (X, Y, Z) for Points {points_label}', fontsize=16, fontweight='bold')
    
    for row_idx, point in enumerate(points):
        if point not in results:
            continue
        
        point_data = results[point]
        
        # Combine all sequences for this point
        all_x = []
        all_y = []
        all_z = []
        all_t = []
        t_offset = 0
        
        for seq_idx, seq in enumerate(point_data):
            sensor_data = seq['sensor_data']  # [samples, 3]
            timestamps = seq['timestamps']
            
            # Use relative time starting from 0 for each sequence
            if len(timestamps) > 0:
                t_rel = timestamps - timestamps[0] + t_offset
            else:
                t_rel = np.arange(len(sensor_data)) + t_offset
            
            all_x.extend(sensor_data[:, 0])
            all_y.extend(sensor_data[:, 1])
            all_z.extend(sensor_data[:, 2])
            all_t.extend(t_rel)
            
            t_offset = t_rel[-1] + 0.1  # Small gap between sequences
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_z = np.array(all_z)
        all_t = np.array(all_t)
        
        # Plot X component
        axes[row_idx, 0].plot(all_t, all_x, 'b-', alpha=0.6, linewidth=0.5)
        axes[row_idx, 0].set_title(f'Point {point}: X Component\n({len(point_data)} sequences, {len(all_x)} samples)')
        axes[row_idx, 0].set_xlabel('Time (s)')
        axes[row_idx, 0].set_ylabel('X (digits)')
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # Plot Y component
        axes[row_idx, 1].plot(all_t, all_y, 'g-', alpha=0.6, linewidth=0.5)
        axes[row_idx, 1].set_title(f'Point {point}: Y Component\n({len(point_data)} sequences, {len(all_y)} samples)')
        axes[row_idx, 1].set_xlabel('Time (s)')
        axes[row_idx, 1].set_ylabel('Y (digits)')
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # Plot Z component
        axes[row_idx, 2].plot(all_t, all_z, 'r-', alpha=0.6, linewidth=0.5)
        axes[row_idx, 2].set_title(f'Point {point}: Z Component\n({len(point_data)} sequences, {len(all_z)} samples)')
        axes[row_idx, 2].set_xlabel('Time (s)')
        axes[row_idx, 2].set_ylabel('Z (digits)')
        axes[row_idx, 2].grid(True, alpha=0.3)
        axes[row_idx, 2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{sensor_name.lower().replace(" ", "_")}_{test_name.lower().replace(" ", "_")}_points_{points_label.replace("-", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_file}")
    plt.close()

def plot_multiple_sensors_data(test_name, test_file, points, sensor_indices, output_dir):
    """Plot multiple sensors data for all points."""
    # Extract data for all sensors
    all_results = {}
    for sensor_idx in sensor_indices:
        results = extract_sensor_data(test_file, points, sensor_idx)
        for point in points:
            if point not in all_results:
                all_results[point] = {}
            if point in results:
                all_results[point][sensor_idx] = results[point]
    
    if not all_results:
        print(f"No data found for {test_name}")
        return
    
    # Create figure: one row per point, one column per sensor, 3 subplots per component
    n_points = len(points)
    n_sensors = len(sensor_indices)
    
    fig, axes = plt.subplots(n_points, n_sensors * 3, figsize=(6*n_sensors*3, 4*n_points))
    if n_points == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{test_name}: Sensors 1, 2, 3 Data (X, Y, Z) for Points {", ".join(points)}', 
                 fontsize=16, fontweight='bold')
    
    for row_idx, point in enumerate(points):
        if point not in all_results:
            continue
        
        for col_sensor_idx, sensor_idx in enumerate(sensor_indices):
            sensor_num = sensor_idx + 1  # Convert 0-indexed to 1-indexed
            point_sensor_data = all_results[point].get(sensor_idx, [])
            
            if not point_sensor_data:
                continue
            
            # Combine all sequences for this point and sensor
            all_x = []
            all_y = []
            all_z = []
            all_t = []
            t_offset = 0
            
            for seq in point_sensor_data:
                sensor_data = seq['sensor_data']
                timestamps = seq['timestamps']
                
                if len(timestamps) > 0:
                    t_rel = timestamps - timestamps[0] + t_offset
                else:
                    t_rel = np.arange(len(sensor_data)) + t_offset
                
                all_x.extend(sensor_data[:, 0])
                all_y.extend(sensor_data[:, 1])
                all_z.extend(sensor_data[:, 2])
                all_t.extend(t_rel)
                
                t_offset = t_rel[-1] + 0.1
            
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_z = np.array(all_z)
            all_t = np.array(all_t)
            
            # Plot X component
            ax_x = axes[row_idx, col_sensor_idx * 3 + 0]
            ax_x.plot(all_t, all_x, 'b-', alpha=0.6, linewidth=0.5)
            ax_x.set_title(f'Point {point}, Sensor {sensor_num}: X\n({len(point_sensor_data)} seq, {len(all_x)} samples)')
            ax_x.set_xlabel('Time (s)')
            ax_x.set_ylabel('X (digits)')
            ax_x.grid(True, alpha=0.3)
            ax_x.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Plot Y component
            ax_y = axes[row_idx, col_sensor_idx * 3 + 1]
            ax_y.plot(all_t, all_y, 'g-', alpha=0.6, linewidth=0.5)
            ax_y.set_title(f'Point {point}, Sensor {sensor_num}: Y\n({len(point_sensor_data)} seq, {len(all_y)} samples)')
            ax_y.set_xlabel('Time (s)')
            ax_y.set_ylabel('Y (digits)')
            ax_y.grid(True, alpha=0.3)
            ax_y.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            
            # Plot Z component
            ax_z = axes[row_idx, col_sensor_idx * 3 + 2]
            ax_z.plot(all_t, all_z, 'r-', alpha=0.6, linewidth=0.5)
            ax_z.set_title(f'Point {point}, Sensor {sensor_num}: Z\n({len(point_sensor_data)} seq, {len(all_z)} samples)')
            ax_z.set_xlabel('Time (s)')
            ax_z.set_ylabel('Z (digits)')
            ax_z.grid(True, alpha=0.3)
            ax_z.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'sensors_123_{test_name.lower().replace(" ", "_")}_points_{"_".join(points)}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_file}")
    plt.close()

def plot_statistics_comparison(test28_file, test29_file, points, sensor_idx, output_dir):
    """Plot statistics comparison between test28 and test29."""
    results28 = extract_sensor_data(test28_file, points, sensor_idx)
    results29 = extract_sensor_data(test29_file, points, sensor_idx)
    
    fig, axes = plt.subplots(3, len(points), figsize=(4*len(points), 12))
    if len(points) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Sensor 4 Statistics Comparison: Test 28 vs Test 29', fontsize=16, fontweight='bold')
    
    components = ['X', 'Y', 'Z']
    colors = ['b', 'g', 'r']
    
    for col_idx, point in enumerate(points):
        # Collect data for test28
        data28 = {'X': [], 'Y': [], 'Z': []}
        if point in results28:
            for seq in results28[point]:
                sensor_data = seq['sensor_data']
                data28['X'].extend(sensor_data[:, 0])
                data28['Y'].extend(sensor_data[:, 1])
                data28['Z'].extend(sensor_data[:, 2])
        
        # Collect data for test29
        data29 = {'X': [], 'Y': [], 'Z': []}
        if point in results29:
            for seq in results29[point]:
                sensor_data = seq['sensor_data']
                data29['X'].extend(sensor_data[:, 0])
                data29['Y'].extend(sensor_data[:, 1])
                data29['Z'].extend(sensor_data[:, 2])
        
        for row_idx, (comp, color) in enumerate(zip(components, colors)):
            ax = axes[row_idx, col_idx]
            
            # Box plot comparison
            data_to_plot = []
            labels = []
            
            if data28[comp]:
                data_to_plot.append(data28[comp])
                labels.append('Test 28')
            
            if data29[comp]:
                data_to_plot.append(data29[comp])
                labels.append('Test 29')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_title(f'Point {point}: {comp} Component')
                ax.set_ylabel(f'{comp} (digits)')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                
                # Add statistics text
                if data28[comp]:
                    mean28 = np.mean(data28[comp])
                    std28 = np.std(data28[comp])
                    ax.text(0.5, 0.95, f'Mean: {mean28:.2f}\nStd: {std28:.2f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if data29[comp]:
                    mean29 = np.mean(data29[comp])
                    std29 = np.std(data29[comp])
                    ax.text(0.5, 0.05, f'Mean: {mean29:.2f}\nStd: {std29:.2f}', 
                           transform=ax.transAxes, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'sensor4_statistics_comparison_test28_vs_test29.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_file}")
    plt.close()

def main():
    output_dir = Path("data/Multiple_Points")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots for Sensor 4 data (points 4-7)...")
    
    # Plot Sensor 4 data for points 4-7
    plot_sensor_data("Test 28", test28_file, points_sensor4, sensor4_idx, output_dir, 
                     sensor_name="Sensor 4", points_label="4-7")
    plot_sensor_data("Test 29", test29_file, points_sensor4, sensor4_idx, output_dir,
                     sensor_name="Sensor 4", points_label="4-7")
    
    # Plot statistics comparison for Sensor 4
    plot_statistics_comparison(test28_file, test29_file, points_sensor4, sensor4_idx, output_dir)
    
    print("\nGenerating plots for Sensors 1, 2, 3 data (points 9-10)...")
    
    # Plot Sensors 1, 2, 3 data for points 9-10
    plot_multiple_sensors_data("Test 28", test28_file, points_sensors123, sensors123_idx, output_dir)
    plot_multiple_sensors_data("Test 29", test29_file, points_sensors123, sensors123_idx, output_dir)
    
    print("\n✅ All plots generated!")

if __name__ == "__main__":
    main()

