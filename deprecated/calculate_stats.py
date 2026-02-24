import json
import os
import numpy as np
import pandas as pd

files = [
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Bus_B62_bus62_2.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Ferry_Ferry5.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_foot.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_bicycle.json'
]

def analyze_trace(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        patterns = []
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            patterns = data['uplink']['trace_pattern']
        elif 'downlink' in data and 'trace_pattern' in data['downlink']:
            patterns = data['downlink']['trace_pattern']
        elif 'trace_pattern' in data:
            patterns = data['trace_pattern']
            
        if not patterns:
            return None

        # Convert to Mbps for easier reading
        caps_kbps = np.array([p['capacity'] for p in patterns])
        caps_mbps = caps_kbps / 1000.0
        
        stats = {
            'Trace Name': os.path.basename(file_path),
            'Duration (s)': len(caps_mbps),
            'Avg Bandwidth (Mbps)': np.mean(caps_mbps),
            'Std Dev (Mbps)': np.std(caps_mbps),
            'Coeff of Var (CV)': np.std(caps_mbps) / np.mean(caps_mbps) if np.mean(caps_mbps) > 0 else 0,
            'Max (Mbps)': np.max(caps_mbps),
            'Min (Mbps)': np.min(caps_mbps),
            'Outage % (0 Mbps)': (np.sum(caps_kbps == 0) / len(caps_kbps)) * 100,
            'Low BW % (<1 Mbps)': (np.sum(caps_kbps < 1000) / len(caps_kbps)) * 100
        }
        return stats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

results = []
for f in files:
    res = analyze_trace(f)
    if res:
        results.append(res)

# Create DataFrame for nice display
df = pd.DataFrame(results)

# Format the float columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print(df.to_string(index=False))

# Also print a markdown version for the final response
print("\nMarkdown Table:")
print(df.to_markdown(index=False, floatfmt=".2f"))
