import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the base directory and files
base_dir = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json'
files = [
    ('7Train_7BtrainNew.json', '7Train (Subway)'),
    ('Bus_B62_bus62_2.json', 'Bus B62'),
    ('Ferry_Ferry5.json', 'Ferry'),
    ('LIRR_Long_Island_Rail_Road.json', 'LIRR (Rail)'),
    ('Car_Car_2.json', 'Car 2')
]

plt.figure(figsize=(15, 8))
limit_seconds = 1800
all_capacities = []

for filename, label in files:
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        continue
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        patterns = []
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            patterns = data['uplink']['trace_pattern']
        elif 'downlink' in data and 'trace_pattern' in data['downlink']:
            patterns = data['downlink']['trace_pattern']
            
        if not patterns:
            continue

        # Extract capacity (kbps) -> Mbps
        # Limit to first 1800 seconds
        capacities = [p['capacity'] / 1000.0 for p in patterns[:limit_seconds]]
        
        # Add to all data for percentile calculation
        all_capacities.extend(capacities)
        
        # Time axis
        time = np.arange(len(capacities))
        
        # Plot
        plt.plot(time, capacities, label=label, alpha=0.8, linewidth=1.5)
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Calculate reasonable Y-limit
# Using 90th percentile to cut off extreme peaks
if all_capacities:
    p90 = np.percentile(all_capacities, 90)
    p95 = np.percentile(all_capacities, 95)
    
    # We choose a limit that shows most data but cuts extreme spikes
    # If 95th percentile is huge (e.g. > 100), we might want to cap it lower.
    # But usually 90-95th gives a good view of "normal" operation range.
    y_limit = p95 * 1.1 
    
    # Hard cap if it's still too crazy, e.g., if even 95% is 500Mbps
    # But let's trust the percentile first.
    # For visualization of mixed low/high traces, maybe 60 Mbps is a good baseline if high traces average 30.
    
    # Let's verify what p95 is.
    print(f"90th Percentile: {p90:.2f} Mbps")
    print(f"95th Percentile: {p95:.2f} Mbps")
    
    # Let's set a manual cap if p95 is excessively high (>100), otherwise use p95
    if p95 > 100:
        y_limit = 100
        print("Capping Y-axis at 100 Mbps (removing extreme peaks)")
    else:
        y_limit = p95
        print(f"Setting Y-axis limit to {y_limit:.2f} Mbps (95th percentile)")

    plt.ylim(0, y_limit)

plt.title(f'Throughput Comparison (First {limit_seconds}s) - Zoomed In')
plt.xlabel('Time (seconds)')
plt.ylabel('Throughput (Mbps)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, limit_seconds)

output_path = '/home/wyq/桌面/mininet-RTC/throughput_comparison_1800s_zoomed.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
