import json
import os
import numpy as np

file_path = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json'
output_dir = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/derived_traces'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_trace(name, patterns):
    # Construct the new trace object
    # Mirroring the original structure: Uplink has data, Downlink is empty
    
    new_data = {
        "uplink": { "trace_pattern": patterns },
        "downlink": {},
        "type": "derived" # Adding a type field just in case
    }
    
    out_path = os.path.join(output_dir, f"{name}.json")
    with open(out_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved {name} to {out_path}")

def analyze_and_extract():
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'uplink' not in data:
        print("Error: 'uplink' key not found.")
        return

    patterns = data['uplink']['trace_pattern']
    
    # Indices correspond to seconds (assuming 1000ms duration per pattern)
    
    # Define segments (start_sec, end_sec, name, description)
    segments = [
        (11560, 11660, "7Train_Stable_Medium", "Stable medium bandwidth (~9 Mbps), low jitter"),
        (10600, 10660, "7Train_High_Variable", "High bandwidth (>100 Mbps) but highly variable"),
        (10240, 10300, "7Train_Low_Stable", "Low bandwidth (~4-10 Mbps), relatively stable"),
        (11390, 11460, "7Train_Unstable_Outage", "Unstable with complete outages (0 Mbps)")
    ]
    
    print("\n--- Extracting Segments ---")
    summary = []
    for start, end, name, desc in segments:
        if start < 0 or end > len(patterns):
            print(f"Skipping {name}: range {start}-{end} out of bounds")
            continue
            
        segment_patterns = patterns[start:end]
        
        caps = [p['capacity'] for p in segment_patterns]
        avg = np.mean(caps)
        std = np.std(caps)
        min_cap = np.min(caps)
        max_cap = np.max(caps)
        
        info = f"{name}: {desc} (Avg: {avg:.0f} kbps, Range: {min_cap:.0f}-{max_cap:.0f} kbps)"
        summary.append(info)
        print(info)
        
        save_trace(name, segment_patterns)
        
    # Write summary file
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    analyze_and_extract()
