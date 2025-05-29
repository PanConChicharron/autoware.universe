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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Steering System Identification using MHE')
    parser.add_argument('rosbag_path', type=str, help='Path to the rosbag file (.db3)')
    parser.add_argument('--output', '-o', type=str, default='steering_system_id_results.png', 
                        help='Output plot filename (default: steering_system_id_results.png)')
    parser.add_argument('--horizon', type=int, default=20, 
                        help='MHE horizon length (default: 20)')
    parser.add_argument('--dt', type=float, default=0.1, 
                        help='Sampling time in seconds (default: 0.1)')
    parser.add_argument('--delay-samples', type=int, default=3, 
                        help='Delay in samples (default: 3)')
    parser.add_argument('--topic', type=str, default='/control/trajectory_follower/lateral/diagnostic',
                        help='Topic name for diagnostic data (default: /control/trajectory_follower/lateral/diagnostic)')
    
    args = parser.parse_args()
    
    # Check if rosbag file exists
    if not os.path.exists(args.rosbag_path):
        print(f"Error: Rosbag file '{args.rosbag_path}' does not exist!")
        sys.exit(1)
    
    print(f"Processing rosbag: {args.rosbag_path}")
    print(f"MHE Parameters: horizon={args.horizon}, dt={args.dt}, delay={args.delay_samples} samples")
    
    # Initialize MHE with proper delay handling
    mhe = SteeringMHE(horizon=args.horizon, dt=args.dt, delay_samples=args.delay_samples)
    
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
    simulated_steering = mhe.simulate_model(commanded_steering, initial_steering=measured_steering[0])
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((simulated_steering - measured_steering)**2))
    print(f"Model RMSE: {rmse:.6f} rad")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot commanded vs measured steering
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, commanded_steering, label='Commanded Steering', alpha=0.7)
    plt.plot(timestamps, measured_steering, label='Measured Steering', alpha=0.7)
    plt.plot(timestamps, simulated_steering, label='Simulated Steering (MHE)', linestyle='--', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.legend()
    plt.title(f'Steering System Response (τ={mhe.get_time_constant():.3f}s, delay={mhe.get_delay():.3f}s)')
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
    error = simulated_steering - measured_steering
    plt.plot(timestamps, error)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title(f'Model Error (RMSE: {rmse:.6f} rad)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main() 