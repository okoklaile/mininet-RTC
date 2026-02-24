import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker
import matplotlib

# Set font for vector graphics compatibility (Type 42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# File configurations
files = [
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json',
        'name': 'Train',
        'color': '#ff7f0e' # Orange
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Bus_B62_bus62_2.json',
        'name': 'Bus',
        'color': '#1f77b4' # Blue
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Ferry_Ferry5.json',
        'name': 'Ferry',
        'color': '#2ca02c' # Green
    },
    {
        'path': '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_foot.json',
        'name': 'Foot',
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
        if not os.path.exists(item['path']):
            print(f"Warning: File not found {item['path']}")
            continue
            
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
        print(f"Processed {item['name']}")
        
    except Exception as e:
        print(f"Error loading {item['name']}: {e}")

# plt.title('CDF of Throughput for Different Traces') # Removed title as requested before
plt.xlabel('Throughput (Mbps)', fontsize=14)
plt.ylabel('CDF (Cumulative Probability)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Use Log Scale for X-axis
plt.xscale('log') 
plt.xlim(0.1, 1000) 

# Format ticks
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'], fontsize=12)
plt.yticks(fontsize=12)

# Save as PDF (Vector Graphic)
output_path = '/home/wyq/桌面/mininet-RTC/trace_cdf_logscale.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved to {output_path}")
