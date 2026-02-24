import re
import os
import pandas as pd

# File paths
files = [
    '/home/wyq/桌面/mininet-RTC/eval_results/bus.txt',
    '/home/wyq/桌面/mininet-RTC/eval_results/bicycle.txt',
    '/home/wyq/桌面/mininet-RTC/eval_results/ferry.txt',
    '/home/wyq/桌面/mininet-RTC/eval_results/foot.txt',
    '/home/wyq/桌面/mininet-RTC/eval_results/train.txt'
]

# Output file
output_file = '/home/wyq/桌面/mininet-RTC/eval_results/extracted_metrics.txt'

def parse_report(file_path):
    scenario = os.path.basename(file_path).replace('.txt', '').capitalize()
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by algorithm sections
    # Regex to find "1. AlgoName", "2. AlgoName" etc.
    algo_sections = re.split(r'\n\d+\. ', content)
    
    results = []
    
    # Skip the header section (index 0)
    for section in algo_sections[1:]:
        lines = section.split('\n')
        algo_name = lines[0].strip()
        
        # Initialize metrics
        metrics = {
            'Scenario': scenario,
            'Algorithm': algo_name,
            'Throughput (Mbps)': 0.0,
            'Network Delay Avg (ms)': 0.0,
            'Network Delay P95 (ms)': 0.0,
            'E2E Delay Avg (ms)': 0.0,
            'E2E Delay P95 (ms)': 0.0,
            'Freeze Rate (%)': 0.0,
            'Loss Rate (%)': 0.0
        }
        
        # Extract metrics using regex
        # Throughput
        match = re.search(r'平均视频比特率:\s+([\d.]+) Mbps', section)
        if match: metrics['Throughput (Mbps)'] = float(match.group(1))
        
        # Network Delay
        match = re.search(r'网络延迟:\s+Avg:\s+([\d.]+)\s+ms\s+/\s+P95:\s+([\d.]+)\s+ms', section)
        if match:
            metrics['Network Delay Avg (ms)'] = float(match.group(1))
            metrics['Network Delay P95 (ms)'] = float(match.group(2))
            
        # E2E Delay
        match = re.search(r'端到端延迟:\s+Avg:\s+([\d.]+)\s+ms\s+/\s+P95:\s+([\d.]+)\s+ms', section)
        if match:
            metrics['E2E Delay Avg (ms)'] = float(match.group(1))
            metrics['E2E Delay P95 (ms)'] = float(match.group(2))
            
        # Freeze Rate
        match = re.search(r'卡顿率:\s+([\d.]+)\s+%', section)
        if match: metrics['Freeze Rate (%)'] = float(match.group(1))
        
        # Loss Rate
        match = re.search(r'丢包率:\s+([\d.]+)\s+%', section)
        if match: metrics['Loss Rate (%)'] = float(match.group(1))
        
        results.append(metrics)
        
    return results

# Process all files
all_data = []
for f in files:
    if os.path.exists(f):
        all_data.extend(parse_report(f))
    else:
        print(f"Warning: {f} not found.")

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Create output string
output_str = ""

# 1. Tables per Algorithm (across scenarios)
# Or per Scenario (across algorithms)?
# Request says: "为每个算法绘制一张表" (Draw a table for each algorithm)

algorithms = df['Algorithm'].unique()

for algo in algorithms:
    algo_df = df[df['Algorithm'] == algo].copy()
    
    # Format the table
    output_str += f"\n=== Algorithm: {algo} ===\n"
    # Select columns to display
    cols = ['Scenario', 'Throughput (Mbps)', 'Network Delay Avg (ms)', 'Network Delay P95 (ms)', 
            'E2E Delay Avg (ms)', 'E2E Delay P95 (ms)', 'Freeze Rate (%)', 'Loss Rate (%)']
    
    table_str = algo_df[cols].to_markdown(index=False, floatfmt=".2f")
    output_str += table_str + "\n"

# 2. Average Table (Average across all scenarios for each algorithm)
output_str += "\n=== Average Performance (Across All Scenarios) ===\n"

# Group by Algorithm and calculate mean
avg_df = df.groupby('Algorithm')[['Throughput (Mbps)', 'Network Delay Avg (ms)', 'Network Delay P95 (ms)', 
                                  'E2E Delay Avg (ms)', 'E2E Delay P95 (ms)', 'Freeze Rate (%)', 'Loss Rate (%)']].mean().reset_index()

# Sort by Throughput descending (optional, or keep original order)
# avg_df = avg_df.sort_values('Throughput (Mbps)', ascending=False)

table_str = avg_df.to_markdown(index=False, floatfmt=".2f")
output_str += table_str + "\n"

# Write to file
with open(output_file, 'w') as f:
    f.write(output_str)

print(f"Extraction complete. Results saved to {output_file}")
print(output_str)
