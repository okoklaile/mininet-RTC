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
window_size = 30  # Moving average window size in seconds

def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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
        
        # Apply smoothing
        smoothed_capacities = moving_average(capacities, window_size)
        
        # Add to all data for percentile calculation (using smoothed data to set limits makes sense for the plot)
        all_capacities.extend(smoothed_capacities)
        
        # Time axis (adjusted for valid convolution)
        # The 'valid' mode reduces the array size by window_size - 1
        # We shift time to align roughly with the center of the window or start
        time = np.arange(len(smoothed_capacities)) + (window_size / 2)
        
        # Plot
        plt.plot(time, smoothed_capacities, label=f"{label}", linewidth=2)
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Calculate reasonable Y-limit based on smoothed data
if all_capacities:
    p95 = np.percentile(all_capacities, 95)
    
    # Cap if too high, otherwise use percentile
    if p95 > 100:
        y_limit = 100
        print("Capping Y-axis at 100 Mbps")
    else:
        y_limit = p95 * 1.1
        print(f"Setting Y-axis limit to {y_limit:.2f} Mbps (95th percentile of smoothed data)")

    plt.ylim(0, y_limit)

plt.title(f'Throughput Comparison (First {limit_seconds}s) - Smoothed ({window_size}s moving avg)')
plt.xlabel('Time (seconds)')
plt.ylabel('Throughput (Mbps)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, limit_seconds)

output_path = '/home/wyq/桌面/mininet-RTC/throughput_comparison_1800s_smoothed.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
