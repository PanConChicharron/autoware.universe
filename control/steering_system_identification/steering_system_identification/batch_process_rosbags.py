#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import json
import time
from datetime import datetime

def find_rosbag_files(input_paths, pattern="*.db3"):
    """Find all rosbag files from input paths (can be files or directories)"""
    bag_files = []
    
    # Handle both single path and list of paths
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    for path in input_paths:
        if os.path.isfile(path):
            # Single file provided
            if path.endswith('.db3'):
                bag_files.append(path)
        else:
            # Directory provided, search for bag files
            search_pattern = os.path.join(path, "**", pattern)
            found_files = glob.glob(search_pattern, recursive=True)
            bag_files.extend(found_files)
    
    return sorted(bag_files)

def run_process_rosbag(bag_file, delay, base_output_dir, additional_args=None):
    """Run process_rosbag.py with specified parameters"""
    
    # Create bag-specific output directory
    bag_name = Path(bag_file).stem
    bag_output_dir = os.path.join(base_output_dir, bag_name)
    os.makedirs(bag_output_dir, exist_ok=True)
    
    # Create output filename based on delay
    output_filename = f"{bag_name}_delay_{delay:.3f}s_results.png"
    output_path = os.path.join(bag_output_dir, output_filename)
    
    # Build command
    cmd = [
        sys.executable, "process_rosbag.py",
        bag_file,
        "--delay", str(delay),
        "--output", output_path
    ]
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n{'='*80}")
    print(f"Processing: {bag_name}")
    print(f"Delay: {delay:.3f}s")
    print(f"Output dir: {bag_output_dir}")
    print(f"Output file: {output_filename}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        print(f"✓ SUCCESS - Completed in {end_time - start_time:.1f}s")
        
        # Extract RMSE values from output
        rmse_info = extract_rmse_from_output(result.stdout)
        
        return {
            'bag_file': bag_file,
            'bag_name': bag_name,
            'delay': delay,
            'output_dir': bag_output_dir,
            'output_file': output_path,
            'success': True,
            'duration': end_time - start_time,
            'rmse_initial': rmse_info.get('rmse_initial'),
            'rmse_final': rmse_info.get('rmse_final'),
            'improvement_percent': rmse_info.get('improvement_percent')
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"✗ FAILED - Error after {end_time - start_time:.1f}s")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        
        return {
            'bag_file': bag_file,
            'bag_name': bag_name,
            'delay': delay,
            'output_dir': bag_output_dir,
            'output_file': output_path,
            'success': False,
            'duration': end_time - start_time,
            'error': str(e),
            'stderr': e.stderr
        }

def extract_rmse_from_output(stdout):
    """Extract RMSE values from process_rosbag.py output"""
    rmse_info = {}
    
    lines = stdout.split('\n')
    for line in lines:
        if 'Initial tau (constant' in line and 'rad' in line:
            # Extract initial RMSE
            try:
                parts = line.split(':')
                rmse_str = parts[1].strip().split()[0]
                rmse_info['rmse_initial'] = float(rmse_str)
            except:
                pass
        elif 'Time-varying MHE tau' in line and 'rad' in line:
            # Extract final RMSE
            try:
                parts = line.split(':')
                rmse_str = parts[1].strip().split()[0]
                rmse_info['rmse_final'] = float(rmse_str)
            except:
                pass
        elif 'RMSE improvement:' in line:
            # Extract improvement percentage
            try:
                parts = line.split(':')
                improvement_str = parts[1].strip().replace('%', '')
                rmse_info['improvement_percent'] = float(improvement_str)
            except:
                pass
    
    return rmse_info

def save_results_summary(results, output_dir):
    """Save a summary of all results"""
    
    # Create summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': len(results),
        'successful_runs': sum(1 for r in results if r['success']),
        'failed_runs': sum(1 for r in results if not r['success']),
        'results': results
    }
    
    # Save JSON summary
    json_path = os.path.join(output_dir, 'batch_results_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create CSV summary for easy analysis
    csv_path = os.path.join(output_dir, 'batch_results_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('bag_name,bag_file,delay,success,duration,rmse_initial,rmse_final,improvement_percent,output_dir,output_file\n')
        for result in results:
            f.write(f"{result.get('bag_name', '')},{result['bag_file']},{result['delay']},{result['success']},{result.get('duration', 0):.1f},"
                   f"{result.get('rmse_initial', '')},{result.get('rmse_final', '')},{result.get('improvement_percent', '')},"
                   f"{result.get('output_dir', '')},{result['output_file']}\n")
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful: {summary['successful_runs']}")
    print(f"Failed: {summary['failed_runs']}")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    
    # Print best results
    successful_results = [r for r in results if r['success'] and r.get('rmse_final')]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['rmse_final'])
        print(f"\nBest RMSE result:")
        print(f"  File: {best_result.get('bag_name', Path(best_result['bag_file']).name)}")
        print(f"  Delay: {best_result['delay']:.3f}s")
        print(f"  RMSE: {best_result['rmse_final']:.6f} rad")
        if best_result.get('improvement_percent'):
            print(f"  Improvement: {best_result['improvement_percent']:.1f}%")
        print(f"  Output: {best_result['output_file']}")

def main():
    parser = argparse.ArgumentParser(description='Batch process rosbag files with multiple delay values')
    parser.add_argument('input_paths', type=str, nargs='+',
                        help='Path(s) to rosbag file(s) or directory(ies) containing rosbag files')
    parser.add_argument('--delays', type=float, nargs='+', 
                        default=[0.05, 0.10, 0.24, 0.50],
                        help='List of delay values to test (default: 0.05 0.10 0.24 0.50)')
    parser.add_argument('--output-dir', '-o', type=str, default='batch_results',
                        help='Output directory for results (default: batch_results)')
    parser.add_argument('--horizon', type=int, default=20,
                        help='MHE horizon length (default: 20)')
    parser.add_argument('--initial-tau', type=float, default=0.2,
                        help='Initial time constant estimate (default: 0.2)')
    parser.add_argument('--no-filter-autonomous', action='store_true',
                        help='Disable filtering for autonomous control periods')
    parser.add_argument('--pattern', type=str, default='*.db3',
                        help='File pattern for rosbag search (default: *.db3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually running')
    
    args = parser.parse_args()
    
    # Find rosbag files
    bag_files = find_rosbag_files(args.input_paths, args.pattern)
    
    if not bag_files:
        print(f"No rosbag files found in {args.input_paths} with pattern {args.pattern}")
        sys.exit(1)
    
    print(f"Found {len(bag_files)} rosbag file(s):")
    for bag_file in bag_files:
        print(f"  {bag_file}")
    
    print(f"\nDelay values to test: {args.delays}")
    print(f"Total combinations: {len(bag_files)} bags × {len(args.delays)} delays = {len(bag_files) * len(args.delays)} runs")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    if args.dry_run:
        print("\nDRY RUN - No actual processing will be performed")
        for bag_file in bag_files:
            bag_name = Path(bag_file).stem
            bag_output_dir = os.path.join(args.output_dir, bag_name)
            for delay in args.delays:
                output_filename = f"{bag_name}_delay_{delay:.3f}s_results.png"
                print(f"Would process: {bag_name} with delay {delay:.3f}s -> {bag_output_dir}/{output_filename}")
        return
    
    # Prepare additional arguments
    additional_args = []
    if args.horizon != 20:
        additional_args.extend(['--horizon', str(args.horizon)])
    if args.initial_tau != 0.2:
        additional_args.extend(['--initial-tau', str(args.initial_tau)])
    if args.no_filter_autonomous:
        additional_args.append('--no-filter-autonomous')
    
    # Process all combinations
    results = []
    total_combinations = len(bag_files) * len(args.delays)
    current_combination = 0
    
    for bag_file in bag_files:
        for delay in args.delays:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Processing combination...")
            
            result = run_process_rosbag(bag_file, delay, args.output_dir, additional_args)
            results.append(result)
    
    # Save summary
    save_results_summary(results, args.output_dir)

if __name__ == '__main__':
    main() 