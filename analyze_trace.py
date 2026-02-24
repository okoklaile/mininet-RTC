import json
import numpy as np

file_path = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json'

def analyze_link(link_data, link_name):
    print(f"--- Analyzing {link_name} ---")
    if 'trace_pattern' not in link_data:
        print("No trace_pattern found.")
        return

    patterns = link_data['trace_pattern']
    capacities = []
    durations = []
    
    for p in patterns:
        capacities.append(p['capacity'])
        durations.append(p['duration'])

    capacities = np.array(capacities)
    durations = np.array(durations)
    
    total_duration = np.sum(durations)
    print(f"Total Duration: {total_duration/1000:.2f} s")
    print(f"Count: {len(capacities)}")
    print(f"Min Capacity: {np.min(capacities)}")
    print(f"Max Capacity: {np.max(capacities)}")
    print(f"Mean Capacity: {np.mean(capacities)}")
    print(f"Median Capacity: {np.median(capacities)}")
    print(f"Std Dev Capacity: {np.std(capacities)}")
    
    # Simple segmentation based on thresholds or windows
    # Let's try to group into 10-second chunks and print the average
    
    chunk_size_ms = 10000 # 10 seconds
    current_time = 0
    current_chunk_caps = []
    chunk_stats = []
    
    print("\n--- 10s Chunk Analysis ---")
    for i, cap in enumerate(capacities):
        dur = durations[i]
        # Assuming duration is uniform or we handle it simply. 
        # If duration > chunk_size, this loop logic is too simple, but usually duration is 1000ms.
        # Let's verify duration first.
        
    unique_durations = np.unique(durations)
    print(f"Unique durations: {unique_durations}")
    
    # If all durations are 1000ms, it's easy.
    if len(unique_durations) == 1 and unique_durations[0] == 1000:
        chunk_len = 10 # 10 samples = 10 seconds
        for i in range(0, len(capacities), chunk_len):
            chunk = capacities[i:i+chunk_len]
            avg = np.mean(chunk)
            std = np.std(chunk)
            print(f"Time {i}-{i+len(chunk)}s: Avg={avg:.1f}, Std={std:.1f}, Range=[{np.min(chunk):.1f}, {np.max(chunk):.1f}]")
    else:
        # More complex handling if variable durations
        current_dur = 0
        chunk_caps = []
        chunk_start_time = 0
        for i, cap in enumerate(capacities):
            dur = durations[i]
            chunk_caps.append(cap)
            current_dur += dur
            if current_dur >= chunk_size_ms:
                avg = np.mean(chunk_caps)
                std = np.std(chunk_caps)
                print(f"Time {chunk_start_time/1000:.0f}-{(chunk_start_time+current_dur)/1000:.0f}s: Avg={avg:.1f}, Std={std:.1f}, Range=[{np.min(chunk_caps):.1f}, {np.max(chunk_caps):.1f}]")
                chunk_caps = []
                chunk_start_time += current_dur
                current_dur = 0

with open(file_path, 'r') as f:
    data = json.load(f)

if 'uplink' in data:
    analyze_link(data['uplink'], 'Uplink')

if 'downlink' in data:
    analyze_link(data['downlink'], 'Downlink')
