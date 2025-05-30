#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from mhe import SteeringMHE
from steering_model import export_steering_model
from steering_simulator import SteeringSimulator

def read_messages(bag_path):
    """Read messages from the rosbag."""
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_info.name: topic_info.type for topic_info in topic_types}
    
    # Filter for diagnostic topic
    topic_filter = '/control/trajectory_follower/lateral/diagnostic'
    
    timestamps = []
    commanded_steering = []
    measured_steering = []
    
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if topic_name == topic_filter:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            
            # Extract steering data from diagnostic message
            # data[1] is commanded steering (mpc-raw)
            # data[4] is measured steering
            timestamps.append(timestamp)
            commanded_steering.append(msg.data[1])
            measured_steering.append(msg.data[4])
    
    return np.array(timestamps), np.array(commanded_steering), np.array(measured_steering)

def calculate_dt_from_timestamps(timestamps):
    """Calculate the median sampling time from timestamps."""
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to calculate dt")
    
    # Convert to seconds and calculate differences
    time_diffs = np.diff(timestamps) / 1e9  # Convert nanoseconds to seconds
    
    # Remove outliers (e.g., large gaps in data)
    median_dt = np.median(time_diffs)
    valid_diffs = time_diffs[np.abs(time_diffs - median_dt) < 3 * np.std(time_diffs)]
    
    # Use median dt as it's more robust to outliers
    dt = np.median(valid_diffs)
    
    print(f"Calculated dt from timestamps:")
    print(f"  Median dt: {dt:.6f} s ({1/dt:.1f} Hz)")
    print(f"  Mean dt: {np.mean(valid_diffs):.6f} s")
    print(f"  Std dt: {np.std(valid_diffs):.6f} s")
    print(f"  Min dt: {np.min(valid_diffs):.6f} s")
    print(f"  Max dt: {np.max(valid_diffs):.6f} s")
    print(f"  Valid samples: {len(valid_diffs)}/{len(time_diffs)}")
    
    return dt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Steering System Identification using MHE')
    parser.add_argument('rosbag_path', type=str, help='Path to the rosbag file (.db3)')
    parser.add_argument('--output', '-o', type=str, default='steering_system_id_results.png', 
                        help='Output plot filename (default: steering_system_id_results.png)')
    parser.add_argument('--horizon', type=int, default=20, 
                        help='MHE horizon length (default: 20)')
    parser.add_argument('--dt', type=float, default=None, 
                        help='Sampling time in seconds (default: calculated from timestamps)')
    parser.add_argument('--delay', type=float, default=0.3, 
                        help='Delay in seconds (default: 0.3)')
    parser.add_argument('--initial-tau', type=float, default=0.2,
                        help='Initial time constant estimate in seconds (default: 0.2)')
    parser.add_argument('--topic', type=str, default='/control/trajectory_follower/lateral/diagnostic',
                        help='Topic name for diagnostic data (default: /control/trajectory_follower/lateral/diagnostic)')
    
    args = parser.parse_args()
    
    # Check if rosbag file exists
    if not os.path.exists(args.rosbag_path):
        print(f"Error: Rosbag file '{args.rosbag_path}' does not exist!")
        sys.exit(1)
    
    print(f"Processing rosbag: {args.rosbag_path}")
    
    # Process rosbag
    try:
        timestamps, commanded_steering, measured_steering = read_messages(args.rosbag_path)
    except Exception as e:
        print(f"Error reading rosbag: {e}")
        sys.exit(1)
    
    if len(timestamps) == 0:
        print(f"Error: No data found in topic '{args.topic}'")
        sys.exit(1)
    
    print(f"Found {len(timestamps)} data points")
    
    # Calculate dt from timestamps if not provided
    if args.dt is None:
        dt = calculate_dt_from_timestamps(timestamps)
    else:
        dt = args.dt
        print(f"Using provided dt: {dt:.6f} s ({1/dt:.1f} Hz)")
    
    print(f"MHE Parameters: horizon={args.horizon}, dt={dt:.6f}, delay={args.delay} s, initial_tau={args.initial_tau:.3f}s")
    
    # Create steering model and simulator for modular design
    steering_model = export_steering_model()
    simulator = SteeringSimulator(tau=args.initial_tau, dt=dt)
    
    # Initialize MHE with proper delay handling
    mhe = SteeringMHE(horizon=args.horizon, dt=dt, delay=args.delay, initial_tau=args.initial_tau)
    
    # Convert timestamps to seconds from start
    timestamps = (timestamps - timestamps[0]) / 1e9
    
    # Store results
    time_constants = []
    
    # Process data through MHE
    print("Running MHE estimation...")
    for i in range(len(timestamps)):
        mhe.update(commanded_steering[i], measured_steering[i])
        time_constants.append(mhe.get_time_constant())
        
        # Print progress occasionally
        if i % 100 == 0:
            print(f"Processed {i}/{len(timestamps)} samples, tau = {mhe.get_time_constant():.4f}")
    
    print(f"\nFinal estimates:")
    print(f"Time constant: {mhe.get_time_constant():.6f} s")
    print(f"Delay: {mhe.get_delay():.3f} s")
    
    # Validate model by simulation
    print("\nValidating model...")
    # Update simulator with final tau estimate
    simulator.tau = mhe.get_time_constant()
    
    # Apply delay to commands for simulation
    delay_samples = int(round(args.delay / dt))
    u_delayed = np.zeros_like(commanded_steering)
    for i in range(len(commanded_steering)):
        if i >= delay_samples:
            u_delayed[i] = commanded_steering[i - delay_samples]
        else:
            u_delayed[i] = 0.0
    
    # Simulate with final MHE estimate
    simulated_steering_final = simulator.simulate_trajectory(u_delayed, initial_steering=measured_steering[0], tau=mhe.get_time_constant())
    
    # Simulate with initial tau for comparison
    simulated_steering_initial = simulator.simulate_trajectory(u_delayed, initial_steering=measured_steering[0], tau=args.initial_tau)
    
    # Calculate RMSE for both
    rmse_final = np.sqrt(np.mean((simulated_steering_final - measured_steering)**2))
    rmse_initial = np.sqrt(np.mean((simulated_steering_initial - measured_steering)**2))
    print(f"Model RMSE (initial tau={args.initial_tau:.3f}): {rmse_initial:.6f} rad")
    print(f"Model RMSE (final tau={mhe.get_time_constant():.3f}): {rmse_final:.6f} rad")
    print(f"RMSE improvement: {((rmse_initial - rmse_final) / rmse_initial * 100):.1f}%")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot commanded vs measured steering
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, commanded_steering, label='Commanded Steering', alpha=0.7)
    plt.plot(timestamps, measured_steering, label='Measured Steering', alpha=0.7)
    plt.plot(timestamps, simulated_steering_initial, label=f'Simulated (Initial τ={args.initial_tau:.3f}s)', linestyle=':', alpha=0.8)
    plt.plot(timestamps, simulated_steering_final, label=f'Simulated (MHE τ={mhe.get_time_constant():.3f}s)', linestyle='--', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.legend()
    plt.title(f'Steering System Response (delay={mhe.get_delay():.3f}s)')
    plt.grid(True, alpha=0.3)
    
    # Plot estimated time constant
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, time_constants)
    plt.xlabel('Time (s)')
    plt.ylabel('Time Constant (s)')
    plt.title('Estimated Time Constant Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot error between measured and simulated
    plt.subplot(3, 1, 3)
    error_initial = simulated_steering_initial - measured_steering
    error_final = simulated_steering_final - measured_steering
    plt.plot(timestamps, error_initial, label=f'Error (Initial τ={args.initial_tau:.3f}s)', alpha=0.7)
    plt.plot(timestamps, error_final, label=f'Error (MHE τ={mhe.get_time_constant():.3f}s)', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.legend()
    plt.title(f'Model Errors - Initial RMSE: {rmse_initial:.6f}, MHE RMSE: {rmse_final:.6f} rad')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main() 