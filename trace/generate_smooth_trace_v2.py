import json
import numpy as np
import os

def generate_smooth_trace():
    output_path = '/home/wyq/桌面/mininet-RTC/trace/mininet.json'
    
    # Original trace pattern structure
    # We reconstruct it here to avoid reading the already modified file
    original_pattern = [
        {
            "duration": 60000,
            "capacity": 1000,
            "loss": 0,
            "rtt": 20,
            "jitter": 0
        },  
        {
            "duration": 20000,
            "capacity": 1000,
            "loss": 0,
            "rtt": 20,
            "jitter": 0
        },
        {
            "duration": 20000,
            "capacity": 500,
            "loss": 0,
            "rtt": 20,
            "jitter": 0
        },
        {
            "duration": 20000,
            "capacity": 2000,
            "loss": 0,
            "rtt": 20,
            "jitter": 0
        },
        {
            "duration": 20000,
            "capacity": 1000,
            "loss": 0,
            "rtt": 20,
            "jitter": 0
        }
    ]
    
    new_pattern = []
    
    # Transition parameters
    transition_duration = 5000  # Target transition time: 5 seconds
    step_duration = 1000        # 1 second per step
    
    # Calculate number of steps. 
    # If we want 1s steps for 5s, we want 5 intervals.
    # However, linspace(start, end, num) generates 'num' points.
    # To get 'k' intermediate steps, we generally need carefully chosen points.
    # If we have points p0, p1, p2, p3, p4, p5, p6
    # p0 = start, p6 = end.
    # intermediates: p1, p2, p3, p4, p5. (5 steps)
    # This requires 7 points in linspace.
    # steps (number of intervals) = 5
    # num_points = steps + 2? No.
    # If 1 step: start -> mid -> end. 1 intermediate. 3 points.
    # If 5 steps: start -> i1 -> i2 -> i3 -> i4 -> i5 -> end. 5 intermediates. 7 points.
    
    num_intermediate_steps = int(transition_duration / step_duration)
    # num_intermediate_steps = 5
    # We want 5 intermediate blocks.
    # So we need 5 intermediate values.
    # linspace(start, end, 5 + 2) = linspace(start, end, 7)
    # indices: 0 (start), 1, 2, 3, 4, 5, 6 (end)
    # intermediates: 1, 2, 3, 4, 5.
    
    num_points = num_intermediate_steps + 2
    
    for i in range(len(original_pattern)):
        current_stage = original_pattern[i]
        new_pattern.append(current_stage)
        
        # If there is a next stage, check if we need transition
        if i < len(original_pattern) - 1:
            next_stage = original_pattern[i+1]
            
            curr_cap = current_stage['capacity']
            next_cap = next_stage['capacity']
            
            if curr_cap != next_cap:
                # Generate intermediate steps
                caps = np.linspace(curr_cap, next_cap, num_points)
                # caps[0] is curr_cap, caps[-1] is next_cap
                # We want caps[1] to caps[-2]
                
                intermediate_caps = caps[1:-1]
                
                for cap in intermediate_caps:
                    step = current_stage.copy()
                    step['duration'] = step_duration
                    step['capacity'] = int(cap)
                    new_pattern.append(step)
                    
    # Construct the full JSON object
    data = {
        "type": "video",
        "downlink": {},
        "uplink": {
            "trace_pattern": new_pattern
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Generated smooth trace with {len(new_pattern)} stages. Step duration: {step_duration}ms.")

if __name__ == "__main__":
    generate_smooth_trace()
