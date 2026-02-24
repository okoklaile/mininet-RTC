import json
import matplotlib.pyplot as plt
import numpy as np
import os

# File configurations
files = [
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json',
        'name': '7Train (Subway)',
        'color': '#ff7f0e' # Orange
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Bus_B62_bus62_2.json',
        'name': 'Bus B62',
        'color': '#1f77b4' # Blue
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Ferry_Ferry5.json',
        'name': 'Ferry',
        'color': '#2ca02c' # Green
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_foot.json',
        'name': 'Foot (Walk)',
        'color': '#d62728' # Red
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_bicycle.json',
        'name': 'Bicycle',
        'color': '#9467bd' # Purple
    }
]

plt.figure(figsize=(10, 6))

for item in files:
    try:
        with open(item['path'], 'r') as f:
            raw = json.load(f)
            
        patterns = []
        if 'uplink' in raw and 'trace_pattern' in raw['uplink']:
            patterns = raw['uplink']['trace_pattern']
        elif 'downlink' in raw and 'trace_pattern' in raw['downlink']:
            patterns = raw['downlink']['trace_pattern']
        elif 'trace_pattern' in raw:
            patterns = raw['trace_pattern']
            
        if not patterns:
            continue
            
        # Convert to Mbps
        caps_mbps = np.array([p['capacity'] for p in patterns]) / 1000.0
        
        # Calculate CDF
        sorted_data = np.sort(caps_mbps)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        
        plt.plot(sorted_data, yvals, label=item['name'], color=item['color'], linewidth=2)
        
    except Exception as e:
        print(f"Error loading {item['name']}: {e}")

plt.title('CDF of Throughput for Different Traces')
plt.xlabel('Throughput (Mbps)')
plt.ylabel('CDF (Cumulative Probability)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Use Log Scale for X-axis to see details of both low and high bandwidth traces
plt.xscale('log') 
# Adding a small epsilon to labels if needed, but log scale handles >0 well. 
# Since we have 0 Mbps, we need to be careful. 
# Let's set x-limit from a small positive value (e.g. 0.1 Mbps) to max.
# Data points at 0 will be at -inf on log scale, effectively invisible or at the left edge.
plt.xlim(0.1, 1000) 

# Adding ticks for log scale readability
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'])

output_path = '/home/wyq/桌面/mininet-RTC/trace_cdf_logscale.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")

# Also generate a linear scale version for comparison
plt.figure(figsize=(10, 6))
for item in files:
    try:
        with open(item['path'], 'r') as f:
            raw = json.load(f)
        patterns = []
        if 'uplink' in raw: patterns = raw['uplink']['trace_pattern']
        elif 'downlink' in raw: patterns = raw['downlink']['trace_pattern']
        elif 'trace_pattern' in raw: patterns = raw['trace_pattern']
        if not patterns: continue
        caps_mbps = np.array([p['capacity'] for p in patterns]) / 1000.0
        sorted_data = np.sort(caps_mbps)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, label=item['name'], color=item['color'], linewidth=2)
    except: pass

plt.title('CDF of Throughput (Linear Scale)')
plt.xlabel('Throughput (Mbps)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 100) # Zoom in to 0-100 Mbps since tails are long
output_path_linear = '/home/wyq/桌面/mininet-RTC/trace_cdf_linear.png'
plt.savefig(output_path_linear, dpi=150)
print(f"Linear plot saved to {output_path_linear}")
