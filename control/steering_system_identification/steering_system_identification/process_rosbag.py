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

def read_messages(bag_path, filter_autonomous=True):
    """Read messages from the rosbag and filter for autonomous control periods.
    
    Returns:
        timestamps, commanded_steering, measured_steering, autonomous_mask
        where autonomous_mask is a boolean array indicating autonomous periods
    """
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
    
    # Topics to read
    diagnostic_topic = '/control/trajectory_follower/lateral/diagnostic'
    operation_mode_topic = '/api/operation_mode/state'
    
    # Check if required topics exist
    available_topics = [topic_info.name for topic_info in topic_types]
    if diagnostic_topic not in available_topics:
        raise ValueError(f"Required topic '{diagnostic_topic}' not found in rosbag")
    
    if operation_mode_topic not in available_topics:
        if filter_autonomous:
            print(f"Warning: Operation mode topic '{operation_mode_topic}' not found in rosbag.")
            print("Cannot filter for autonomous control periods. Processing all data instead.")
        use_operation_mode = False
    else:
        use_operation_mode = filter_autonomous
    
    # Storage for all messages
    diagnostic_data = []
    operation_mode_data = []
    
    # Read all messages first
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        
        if topic_name == diagnostic_topic:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            diagnostic_data.append({
                'timestamp': timestamp,
                'commanded_steering': msg.data[1],  # mpc-raw
                'measured_steering': msg.data[4]
            })
        
        elif use_operation_mode and topic_name == operation_mode_topic:
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)
            operation_mode_data.append({
                'timestamp': timestamp,
                'is_autoware_control_enabled': msg.is_autoware_control_enabled
            })
    
    print(f"Read {len(diagnostic_data)} diagnostic messages")
    if use_operation_mode:
        print(f"Read {len(operation_mode_data)} operation mode messages")
        print("Will filter data to only include autonomous control periods")
    elif filter_autonomous:
        print("Autonomous filtering was requested but operation mode topic not available")
    else:
        print("Processing all data (autonomous filtering disabled)")
    
    # Create autonomous mask for all original data
    all_timestamps = np.array([d['timestamp'] for d in diagnostic_data])
    autonomous_mask_all = np.ones(len(diagnostic_data), dtype=bool)  # Default to autonomous if no filtering
    
    if use_operation_mode and len(operation_mode_data) > 0:
        autonomous_mask_all = np.zeros(len(diagnostic_data), dtype=bool)
        for i, diag in enumerate(diagnostic_data):
            # Find the most recent operation mode message before this diagnostic message
            autoware_enabled = False
            for op_mode in reversed(operation_mode_data):
                if op_mode['timestamp'] <= diag['timestamp']:
                    autoware_enabled = op_mode['is_autoware_control_enabled']
                    break
            autonomous_mask_all[i] = autoware_enabled
    
    # Filter diagnostic data based on operation mode if requested
    if filter_autonomous and use_operation_mode and len(operation_mode_data) > 0:
        filtered_data = []
        filtered_indices = []
        
        for i, diag in enumerate(diagnostic_data):
            if autonomous_mask_all[i]:
                filtered_data.append(diag)
                filtered_indices.append(i)
        
        print(f"Filtered to {len(filtered_data)} samples where Autoware control was enabled")
        print(f"Excluded {len(diagnostic_data) - len(filtered_data)} samples ({(len(diagnostic_data) - len(filtered_data))/len(diagnostic_data)*100:.1f}%)")
        
        diagnostic_data = filtered_data
        # Create mask for filtered data (all True since we filtered)
        autonomous_mask = np.ones(len(filtered_data), dtype=bool)
        # Also return the original mask for plotting
        original_autonomous_mask = autonomous_mask_all
    else:
        # No filtering applied
        autonomous_mask = autonomous_mask_all
        original_autonomous_mask = autonomous_mask_all
        filtered_indices = list(range(len(diagnostic_data)))  # All indices
    
    if len(diagnostic_data) == 0:
        raise ValueError("No valid data found after filtering")
    
    # Extract arrays
    timestamps = np.array([d['timestamp'] for d in diagnostic_data])
    commanded_steering = np.array([d['commanded_steering'] for d in diagnostic_data])
    measured_steering = np.array([d['measured_steering'] for d in diagnostic_data])
    
    return timestamps, commanded_steering, measured_steering, autonomous_mask, original_autonomous_mask, all_timestamps, filtered_indices

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

def simulate_with_transitions(simulator, commanded_steering, measured_steering, filtered_indices, original_autonomous_mask, delay_samples, tau):
    """Simulate steering with proper initial condition resets at manual-to-autonomous transitions"""
    
    # Create full simulation result array
    full_simulated = np.full(len(original_autonomous_mask), np.nan)
    
    # Apply delay to commands
    u_delayed = np.zeros_like(commanded_steering)
    for i in range(len(commanded_steering)):
        if i >= delay_samples:
            u_delayed[i] = commanded_steering[i - delay_samples]
        else:
            u_delayed[i] = 0.0
    
    # Find continuous autonomous segments
    autonomous_segments = []
    start_idx = None
    
    for i, is_autonomous in enumerate(original_autonomous_mask):
        if is_autonomous and start_idx is None:
            # Start of autonomous segment
            start_idx = i
        elif not is_autonomous and start_idx is not None:
            # End of autonomous segment
            autonomous_segments.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where data ends in autonomous mode
    if start_idx is not None:
        autonomous_segments.append((start_idx, len(original_autonomous_mask)-1))
    
    print(f"Found {len(autonomous_segments)} autonomous segments")
    
    # Simulate each autonomous segment separately
    for seg_start, seg_end in autonomous_segments:
        # Find corresponding indices in filtered data
        filtered_start = None
        filtered_end = None
        
        for i, orig_idx in enumerate(filtered_indices):
            if orig_idx == seg_start:
                filtered_start = i
            if orig_idx == seg_end:
                filtered_end = i
                break
        
        if filtered_start is None or filtered_end is None:
            continue
            
        # Get the segment data
        segment_commands = u_delayed[filtered_start:filtered_end+1]
        
        # Use measured steering at start of segment as initial condition
        initial_steering = measured_steering[filtered_start]
        
        print(f"Simulating segment {seg_start}-{seg_end} (filtered {filtered_start}-{filtered_end}) with initial steering {initial_steering:.4f}")
        
        # Simulate this segment
        segment_result = simulator.simulate_trajectory(
            segment_commands, 
            initial_steering=initial_steering, 
            tau=tau
        )
        
        # Map results back to full array
        for i, result in enumerate(segment_result):
            if filtered_start + i < len(filtered_indices):
                orig_idx = filtered_indices[filtered_start + i]
                full_simulated[orig_idx] = result
    
    return full_simulated

def calculate_rmse_with_transitions(simulated_full, measured_steering, filtered_indices, original_autonomous_mask):
    """Calculate RMSE accounting for transitions between manual and autonomous control"""
    
    # Find continuous autonomous segments
    autonomous_segments = []
    start_idx = None
    
    for i, is_autonomous in enumerate(original_autonomous_mask):
        if is_autonomous and start_idx is None:
            # Start of autonomous segment
            start_idx = i
        elif not is_autonomous and start_idx is not None:
            # End of autonomous segment
            autonomous_segments.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where data ends in autonomous mode
    if start_idx is not None:
        autonomous_segments.append((start_idx, len(original_autonomous_mask)-1))
    
    # Calculate RMSE for each segment and combine
    all_errors = []
    segment_rmses = []
    
    for seg_start, seg_end in autonomous_segments:
        # Find corresponding indices in filtered data
        filtered_start = None
        filtered_end = None
        
        for i, orig_idx in enumerate(filtered_indices):
            if orig_idx == seg_start:
                filtered_start = i
            if orig_idx == seg_end:
                filtered_end = i
                break
        
        if filtered_start is None or filtered_end is None:
            continue
        
        # Get segment data
        segment_simulated = []
        segment_measured = []
        
        for i in range(filtered_start, filtered_end + 1):
            if i < len(filtered_indices):
                orig_idx = filtered_indices[i]
                if not np.isnan(simulated_full[orig_idx]):
                    segment_simulated.append(simulated_full[orig_idx])
                    segment_measured.append(measured_steering[i])
        
        if len(segment_simulated) > 0:
            segment_errors = np.array(segment_simulated) - np.array(segment_measured)
            segment_rmse = np.sqrt(np.mean(segment_errors**2))
            segment_rmses.append(segment_rmse)
            all_errors.extend(segment_errors)
            
            print(f"  Segment {seg_start}-{seg_end}: RMSE = {segment_rmse:.6f} rad ({len(segment_simulated)} samples)")
    
    # Overall RMSE
    if len(all_errors) > 0:
        overall_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
        print(f"  Overall RMSE: {overall_rmse:.6f} rad ({len(all_errors)} total samples)")
        return overall_rmse, segment_rmses
    else:
        return float('inf'), []

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
    parser.add_argument('--no-filter-autonomous', action='store_true',
                        help='Disable filtering for autonomous control periods (process all data)')
    
    args = parser.parse_args()
    
    # Check if rosbag file exists
    if not os.path.exists(args.rosbag_path):
        print(f"Error: Rosbag file '{args.rosbag_path}' does not exist!")
        sys.exit(1)
    
    print(f"Processing rosbag: {args.rosbag_path}")
    if args.no_filter_autonomous:
        print("Autonomous control filtering: DISABLED - processing all data")
    else:
        print("Autonomous control filtering: ENABLED - only processing data when Autoware control is active")
    
    # Process rosbag
    try:
        timestamps, commanded_steering, measured_steering, autonomous_mask, original_autonomous_mask, all_timestamps, filtered_indices = read_messages(args.rosbag_path, filter_autonomous=not args.no_filter_autonomous)
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
    
    # Track previous timestamp to detect gaps
    prev_timestamp = None
    gap_threshold = 3.0 * dt  # Reset if gap is more than 3x expected dt
    
    for i in range(len(timestamps)):
        current_timestamp = timestamps[i]
        
        # Check for time gap indicating filtered manual control period
        if prev_timestamp is not None:
            time_gap = current_timestamp - prev_timestamp
            if time_gap > gap_threshold:
                print(f"Detected time gap of {time_gap:.3f}s at sample {i} (expected {dt:.3f}s)")
                print(f"Resetting MHE due to filtered manual control period")
                # Reset MHE for new continuous segment
                mhe = SteeringMHE(horizon=args.horizon, dt=dt, delay=args.delay, initial_tau=mhe.get_time_constant())
        
        mhe.update(commanded_steering[i], measured_steering[i])
        time_constants.append(mhe.get_time_constant())
        
        # Print progress occasionally
        if i % 100 == 0:
            print(f"Processed {i}/{len(timestamps)} samples, tau = {mhe.get_time_constant():.4f}")
        
        prev_timestamp = current_timestamp
    
    print(f"\nFinal estimates:")
    print(f"Time constant: {mhe.get_time_constant():.6f} s")
    print(f"Delay: {mhe.get_delay():.3f} s")
    
    # Validate model by simulation
    print("\nValidating model...")
    # Update simulator with final tau estimate
    simulator.tau = mhe.get_time_constant()
    
    # Calculate delay in samples
    delay_samples = int(round(args.delay / dt))
    
    # Simulate with proper initial condition resets at transitions
    print("Simulating with final MHE estimate...")
    simulated_steering_final_full = simulate_with_transitions(
        simulator, commanded_steering, measured_steering, filtered_indices, 
        original_autonomous_mask, delay_samples, mhe.get_time_constant()
    )
    
    print("Simulating with initial tau estimate...")
    simulated_steering_initial_full = simulate_with_transitions(
        simulator, commanded_steering, measured_steering, filtered_indices,
        original_autonomous_mask, delay_samples, args.initial_tau
    )
    
    # Calculate RMSE with transition awareness
    print("Calculating RMSE with transition awareness...")
    print("Initial tau RMSE by segment:")
    rmse_initial, initial_segment_rmses = calculate_rmse_with_transitions(
        simulated_steering_initial_full, measured_steering, filtered_indices, original_autonomous_mask
    )
    
    print("Final MHE tau RMSE by segment:")
    rmse_final, final_segment_rmses = calculate_rmse_with_transitions(
        simulated_steering_final_full, measured_steering, filtered_indices, original_autonomous_mask
    )
    
    print(f"\nModel RMSE Summary:")
    print(f"Initial tau ({args.initial_tau:.3f}s): {rmse_initial:.6f} rad")
    print(f"Final MHE tau ({mhe.get_time_constant():.3f}s): {rmse_final:.6f} rad")
    if rmse_initial != float('inf') and rmse_final != float('inf'):
        print(f"RMSE improvement: {((rmse_initial - rmse_final) / rmse_initial * 100):.1f}%")
    
    # Create full timeline for plotting with gaps during manual periods
    all_timestamps_sec = (all_timestamps - all_timestamps[0]) / 1e9
    full_time_constants = np.full(len(all_timestamps_sec), np.nan)
    full_commanded_steering = np.full(len(all_timestamps_sec), np.nan)
    full_measured_steering = np.full(len(all_timestamps_sec), np.nan)
    
    # Map all the MHE results and data back to the full timeline using the filtered indices
    for i in range(len(time_constants)):
        original_idx = filtered_indices[i]
        full_time_constants[original_idx] = time_constants[i]
        full_commanded_steering[original_idx] = commanded_steering[i]
        full_measured_steering[original_idx] = measured_steering[i]
    
    # Use the full simulation results (already mapped to original timeline)
    full_simulated_initial = simulated_steering_initial_full
    full_simulated_final = simulated_steering_final_full
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot commanded vs measured steering
    plt.subplot(3, 1, 1)
    
    # Add background shading for manual periods (non-autonomous)
    if not args.no_filter_autonomous and len(original_autonomous_mask) > 0:
        manual_periods = ~original_autonomous_mask
        if np.any(manual_periods):
            # Find continuous manual periods for shading
            manual_starts = []
            manual_ends = []
            in_manual = False
            
            for i in range(len(manual_periods)):
                if manual_periods[i] and not in_manual:
                    # Start of manual period
                    manual_starts.append(all_timestamps_sec[i])
                    in_manual = True
                elif not manual_periods[i] and in_manual:
                    # End of manual period
                    manual_ends.append(all_timestamps_sec[i-1] if i > 0 else all_timestamps_sec[i])
                    in_manual = False
            
            # Handle case where data ends in manual mode
            if in_manual:
                manual_ends.append(all_timestamps_sec[-1])
            
            # Add shading for manual periods
            for start, end in zip(manual_starts, manual_ends):
                plt.axvspan(start, end, alpha=0.2, color='red', label='Manual Control' if start == manual_starts[0] else "")
    
    plt.plot(all_timestamps_sec, full_commanded_steering, label='Commanded Steering', alpha=0.7)
    plt.plot(all_timestamps_sec, full_measured_steering, label='Measured Steering', alpha=0.7)
    plt.plot(all_timestamps_sec, full_simulated_initial, label=f'Simulated (Initial τ={args.initial_tau:.3f}s)', linestyle=':', alpha=0.8)
    plt.plot(all_timestamps_sec, full_simulated_final, label=f'Simulated (MHE τ={mhe.get_time_constant():.3f}s)', linestyle='--', alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.legend()
    plt.title(f'Steering System Response (delay={mhe.get_delay():.3f}s)')
    plt.grid(True, alpha=0.3)
    
    # Plot estimated time constant
    plt.subplot(3, 1, 2)
    
    # Add background shading for manual periods
    if not args.no_filter_autonomous and len(original_autonomous_mask) > 0 and np.any(~original_autonomous_mask):
        for start, end in zip(manual_starts, manual_ends):
            plt.axvspan(start, end, alpha=0.2, color='red')
    
    plt.plot(all_timestamps_sec, full_time_constants)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Time Constant (s)')
    plt.title('Estimated Time Constant Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot error between measured and simulated
    plt.subplot(3, 1, 3)
    
    # Add background shading for manual periods
    if not args.no_filter_autonomous and len(original_autonomous_mask) > 0 and np.any(~original_autonomous_mask):
        for start, end in zip(manual_starts, manual_ends):
            plt.axvspan(start, end, alpha=0.2, color='red')
    
    # Create full error arrays for plotting
    full_error_initial = np.full(len(all_timestamps_sec), np.nan)
    full_error_final = np.full(len(all_timestamps_sec), np.nan)
    
    # Calculate errors only for autonomous periods
    for i in range(len(measured_steering)):
        original_idx = filtered_indices[i]
        if not np.isnan(simulated_steering_initial_full[original_idx]):
            full_error_initial[original_idx] = simulated_steering_initial_full[original_idx] - measured_steering[i]
        if not np.isnan(simulated_steering_final_full[original_idx]):
            full_error_final[original_idx] = simulated_steering_final_full[original_idx] - measured_steering[i]
    
    plt.plot(all_timestamps_sec, full_error_initial, label=f'Error (Initial τ={args.initial_tau:.3f}s)', alpha=0.7)
    plt.plot(all_timestamps_sec, full_error_final, label=f'Error (MHE τ={mhe.get_time_constant():.3f}s)', alpha=0.7)
    
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