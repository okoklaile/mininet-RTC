import json
import os
import numpy as np

files = [
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Bus_B62_bus62_2.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/Ferry_Ferry5.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_foot.json',
    '/home/wyq/桌面/mininet-RTC/newtrace/merged_traces/merged_bicycle.json'
]

def analyze_file(path):
    name = os.path.basename(path)
    print(f"\nAnalyzing {name}...")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        keys = list(data.keys())
        print(f"  Root keys: {keys}")
        
        patterns = []
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            patterns = data['uplink']['trace_pattern']
        elif 'downlink' in data and 'trace_pattern' in data['downlink']:
            patterns = data['downlink']['trace_pattern']
        elif 'trace_pattern' in data:
            patterns = data['trace_pattern']
            
        if not patterns:
            print("  No trace pattern found.")
            return

        print(f"  Duration: {len(patterns)} seconds (approx)")
        
        # Check for other keys in pattern items
        first_item = patterns[0]
        item_keys = list(first_item.keys())
        print(f"  Item keys: {item_keys}")
        
        has_handover = 'handover' in item_keys or 'cell_id' in item_keys
        has_loss = 'loss' in item_keys
        
        capacities = [p['capacity'] for p in patterns]
        
        # Bandwidth Drops Analysis
        # Define drop as > 50% decrease from previous second
        drops = 0
        deep_drops = 0 # Drop to near zero (< 100 kbps)
        for i in range(1, len(capacities)):
            prev = capacities[i-1]
            curr = capacities[i]
            if prev > 0 and (prev - curr) / prev > 0.5:
                drops += 1
            if prev > 1000 and curr < 100:
                deep_drops += 1
                
        print(f"  Significant Drops (>50%): {drops} ({drops/len(capacities)*100:.1f}%)")
        print(f"  Deep Drops (to <100kbps): {deep_drops} ({deep_drops/len(capacities)*100:.1f}%)")
        print(f"  Handover info present: {has_handover}")
        print(f"  Packet loss info present: {has_loss}")

        # Check for truncation (heuristic)
        # If it ends exactly at a round number like 1800, 3600 it might be truncated.
        # But merged traces might be long.
        
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    for f in files:
        analyze_file(f)
