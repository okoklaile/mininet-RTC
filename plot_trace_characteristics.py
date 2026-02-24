import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math

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

data_list = []
stats_list = []

print("Loading data...")
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
            print(f"Skipping {item['name']}: No pattern found")
            continue
            
        # Convert to Mbps
        # Replace 0 with a very small number for Log Scale Box Plot (e.g., 0.001 Mbps)
        # But for stats we keep 0.
        caps_mbps = np.array([p['capacity'] / 1000.0 for p in patterns])
        
        # For box plot (log scale), avoid log(0)
        caps_log_safe = caps_mbps.copy()
        caps_log_safe[caps_log_safe == 0] = 0.01 # Treat 0 as 0.01 Mbps for visualization floor
        
        data_list.append(caps_log_safe)
        
        # Stats
        avg = np.mean(caps_mbps)
        std = np.std(caps_mbps)
        outage = (np.sum(caps_mbps == 0) / len(caps_mbps)) * 100
        cv = std / avg if avg > 0 else 0
        
        stats_list.append({
            'name': item['name'],
            'avg': avg,
            'outage': outage,
            'cv': cv,
            'color': item['color']
        })
        print(f"Loaded {item['name']}: Avg={avg:.2f}, Outage={outage:.2f}%")
        
    except Exception as e:
        print(f"Error loading {item['name']}: {e}")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: Box Plot (Bandwidth Distribution) ---
# Use log scale to show both high bandwidth and low bandwidth clearly
bplot = ax1.boxplot(data_list, 
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=[s['name'] for s in stats_list])

# Color the boxes
for patch, item in zip(bplot['boxes'], stats_list):
    patch.set_facecolor(item['color'])
    patch.set_alpha(0.6)

ax1.set_yscale('log')
ax1.set_title('Bandwidth Distribution (Log Scale)', fontsize=14)
ax1.set_ylabel('Throughput (Mbps)', fontsize=12)
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.set_xticklabels([s['name'] for s in stats_list], rotation=15)

# --- Plot 2: Bubble Chart (Scenario Classification) ---
# X-axis: Outage Probability (Reliability)
# Y-axis: Average Bandwidth (Performance)
# Size: Variability (CV) - Larger bubble = More unstable

x = [s['outage'] for s in stats_list]
y = [s['avg'] for s in stats_list]
sizes = [s['cv'] * 300 for s in stats_list] # Scale for visibility
colors = [s['color'] for s in stats_list]

scatter = ax2.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors="black", linewidth=1)

# Add labels
for i, s in enumerate(stats_list):
    ax2.annotate(s['name'], (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=11)

# Axis labels
ax2.set_xlabel('Outage Probability (%)', fontsize=12)
ax2.set_ylabel('Average Bandwidth (Mbps)', fontsize=12)
ax2.set_title('Scenario Classification\n(Bubble Size represents Instability/CV)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5)

# Add quadrant lines or reference regions if needed
# ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.3)
# ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
output_path = '/home/wyq/桌面/mininet-RTC/trace_characteristics_comparison.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
