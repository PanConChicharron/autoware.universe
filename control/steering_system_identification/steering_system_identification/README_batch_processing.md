# Batch Processing for Steering System Identification

This directory contains tools for batch processing rosbag files with multiple delay values for steering system identification.

## Files

- `process_rosbag.py` - Main steering system identification script (now with time-varying tau support)
- `batch_process_rosbags.py` - Batch processing script for multiple bags and delays
- `README_batch_processing.md` - This file

## Key Improvements

The `process_rosbag.py` script has been enhanced to use **time-varying tau values** during simulation instead of a constant final tau value. This provides more accurate simulation results by using the estimated time constant at each time instant.

## Usage

### Single Bag File with Multiple Delays

```bash
python3 batch_process_rosbags.py /path/to/your/bag.db3 --delays 0.05 0.10 0.24 0.50
```

### Multiple Bag Files in a Directory

```bash
python3 batch_process_rosbags.py /path/to/bags/directory/ --delays 0.05 0.10 0.24 0.50
```

### Custom Output Directory

```bash
python3 batch_process_rosbags.py /path/to/bags/ --delays 0.05 0.10 0.24 0.50 --output-dir my_results
```

### Dry Run (Preview What Will Be Processed)

```bash
python3 batch_process_rosbags.py /path/to/bags/ --dry-run
```

## Default Parameters

- **Delays**: `[0.05, 0.10, 0.24, 0.50]` seconds
- **Output Directory**: `batch_results/`
- **MHE Horizon**: `20`
- **Initial Tau**: `0.2` seconds
- **File Pattern**: `*.db3`

## Output

The script generates:

1. **Individual Plot Files**: `{bag_name}_delay_{delay:.3f}s_results.png` for each combination
2. **Summary Files**:
   - `batch_results_summary.json` - Detailed results in JSON format
   - `batch_results_summary.csv` - Results in CSV format for easy analysis

## Example Output Structure

```
batch_results/
├── bag1_delay_0.050s_results.png
├── bag1_delay_0.100s_results.png
├── bag1_delay_0.240s_results.png
├── bag1_delay_0.500s_results.png
├── bag2_delay_0.050s_results.png
├── ...
├── batch_results_summary.json
└── batch_results_summary.csv
```

## CSV Summary Format

The CSV file contains columns:
- `bag_file` - Path to the rosbag file
- `delay` - Delay value used
- `success` - Whether processing succeeded
- `duration` - Processing time in seconds
- `rmse_initial` - RMSE with constant initial tau
- `rmse_final` - RMSE with time-varying tau
- `improvement_percent` - Percentage improvement
- `output_file` - Path to generated plot

## Advanced Options

### Custom Delay Values

```bash
python3 batch_process_rosbags.py /path/to/bags/ --delays 0.1 0.2 0.3 0.4 0.5
```

### Different MHE Parameters

```bash
python3 batch_process_rosbags.py /path/to/bags/ --horizon 30 --initial-tau 0.15
```

### Disable Autonomous Filtering

```bash
python3 batch_process_rosbags.py /path/to/bags/ --no-filter-autonomous
```

### Custom File Pattern

```bash
python3 batch_process_rosbags.py /path/to/bags/ --pattern "steering_*.db3"
```

## Time-Varying Tau Feature

The enhanced `process_rosbag.py` now uses time-varying tau values during simulation:

- **Previous behavior**: Used final estimated tau for entire simulation
- **New behavior**: Uses the estimated tau at each time instant during simulation
- **Benefits**: More accurate simulation that reflects the actual estimation process
- **Backward compatibility**: Still supports constant tau values

This improvement typically results in better RMSE values and more realistic simulation results.

## Tips

1. **Use dry-run first** to preview what will be processed
2. **Monitor disk space** - each plot is ~150KB, so large batches can use significant space
3. **Check the CSV summary** for quick analysis of which delay values work best
4. **Look for the "Best RMSE result"** printed at the end of batch processing
5. **Use time-varying tau results** for the most accurate model validation

## Example Workflow

```bash
# 1. Preview what will be processed
python3 batch_process_rosbags.py /data/steering_bags/ --dry-run

# 2. Run batch processing
python3 batch_process_rosbags.py /data/steering_bags/ --delays 0.05 0.10 0.24 0.50 -o results_2024

# 3. Analyze results
cat results_2024/batch_results_summary.csv | column -t -s,

# 4. Find best performing delay
python3 -c "
import pandas as pd
df = pd.read_csv('results_2024/batch_results_summary.csv')
best = df.loc[df['rmse_final'].idxmin()]
print(f'Best result: {best[\"bag_file\"]} with delay {best[\"delay\"]}s, RMSE: {best[\"rmse_final\"]:.6f}')
" 