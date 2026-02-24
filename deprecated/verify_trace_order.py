import json
import os

original_file = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json'
derived_dir = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/derived_traces'

def verify_segment(original_patterns, segment_file):
    with open(segment_file, 'r') as f:
        seg_data = json.load(f)
    
    seg_patterns = seg_data['uplink']['trace_pattern']
    
    # We need to find if seg_patterns exists as a contiguous sublist in original_patterns
    # Since we know where we extracted it from (roughly), we can just check if it matches the data at that location
    # But to be general and prove it's there exactly as is:
    
    len_seg = len(seg_patterns)
    print(f"Checking {os.path.basename(segment_file)} (length {len_seg})...")
    
    # Let's try to match the first element
    first_item = seg_patterns[0]
    
    found = False
    for i, item in enumerate(original_patterns):
        if item == first_item:
            # Potential match, check the rest
            if original_patterns[i:i+len_seg] == seg_patterns:
                print(f"  ✅ Match found! Corresponds to original indices {i} to {i+len_seg}")
                found = True
                break
    
    if not found:
        print("  ❌ NO match found! The segment is not a contiguous part of the original.")

def main():
    print("Loading original trace...")
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    original_patterns = original_data['uplink']['trace_pattern']
    
    files = [f for f in os.listdir(derived_dir) if f.endswith('.json')]
    for f in files:
        verify_segment(original_patterns, os.path.join(derived_dir, f))

if __name__ == "__main__":
    main()
