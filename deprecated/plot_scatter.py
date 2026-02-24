import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

input_file = '/home/wyq/桌面/mininet-RTC/eval_results/extracted_metrics.txt'
output_dir = '/home/wyq/桌面/mininet-RTC/eval_results/figures/scatter'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_data(input_path):
    with open(input_path, 'r') as f:
        content = f.read()
    
    # We will parse the "Algorithm: Name" sections
    algo_sections = re.split(r'\n\d+\. ', content) # This was for previous format
    # The new extracted_metrics.txt has tables.
    
    # Let's re-parse extracted_metrics.txt
    # Format:
    # === Algorithm: Schaferct ===
    # | Scenario | Throughput...
    
    data = []
    
    current_algo = None
    lines = content.split('\n')
    
    row_pattern = re.compile(r'^\|\s*([^|]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$')

    for line in lines:
        if "=== Algorithm:" in line:
            current_algo = line.split(":")[1].strip().replace(" ===", "")
            if "BC-GCC" in current_algo: 
                current_algo = None
                continue
            continue
            
        if "=== Average Performance" in line:
            break
            
        match = row_pattern.match(line)
        if match and current_algo:
            scenario = match.group(1).strip()
            if scenario in ["Scenario", "Algorithm"]: continue
            if "---" in line: continue
            
            throughput = float(match.group(2))
            e2e_p95 = float(match.group(6))
            
            data.append({
                'Algorithm': current_algo,
                'Scenario': scenario,
                'Throughput': throughput,
                'E2E_P95': e2e_p95
            })
            
    return pd.DataFrame(data)

def plot_scatter(df, output_folder):
    scenarios = df['Scenario'].unique()
    
    colors = plt.cm.tab10(range(len(df['Algorithm'].unique())))
    algo_color_map = dict(zip(df['Algorithm'].unique(), colors))
    
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        
        plt.figure(figsize=(7, 6))
        
        for algo in subset['Algorithm'].unique():
            row = subset[subset['Algorithm'] == algo]
            plt.scatter(row['E2E_P95'], row['Throughput'], 
                        label=algo, s=150, color=algo_color_map[algo], edgecolors='black', alpha=0.8)
            
            # Add text label
            plt.text(row['E2E_P95'], row['Throughput'], f' {algo}', 
                     fontsize=9, ha='left', va='bottom')
        
        # plt.title(f'{scenario} Scenario: Throughput vs P95 Delay')
        plt.xlabel('P95 End-to-End Delay (ms)')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Invert X axis? Usually low delay is better (left), high throughput is better (up).
        # Top-Left corner is the "Sweet Spot".
        # Let's keep normal axes but keep in mind sweet spot.
        
        plt.tight_layout()
        output_path = os.path.join(output_folder, f'scatter_{scenario}.pdf')
        plt.savefig(output_path, format='pdf')
        plt.close()
        print(f"Saved {output_path}")

if __name__ == "__main__":
    df = parse_data(input_file)
    plot_scatter(df, output_dir)
