import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import matplotlib
import numpy as np

# Set font for vector graphics compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Increase font sizes globally for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

input_file = '/home/wyq/桌面/mininet-RTC/eval_results/extracted_metrics.txt'
output_dir = '/home/wyq/桌面/mininet-RTC/eval_results/figures/scatter'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_data(input_path):
    with open(input_path, 'r') as f:
        content = f.read()
    
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
    algorithms = sorted(df['Algorithm'].unique())
    
    # Define distinct markers and colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    # Use a high-contrast colormap
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    algo_style_map = {}
    for i, algo in enumerate(algorithms):
        algo_style_map[algo] = {
            'color': colors[i],
            'marker': markers[i % len(markers)]
        }
    
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        
        # Create figure with slightly larger size
        plt.figure(figsize=(8, 6))
        
        # Plot points
        for algo in algorithms:
            if algo not in subset['Algorithm'].values:
                continue
                
            row = subset[subset['Algorithm'] == algo]
            style = algo_style_map[algo]
            
            plt.scatter(row['E2E_P95'], row['Throughput'], 
                        label=algo, 
                        s=1000,  # Increased size to 500 for plot
                        c=[style['color']], 
                        marker=style['marker'],
                        edgecolors='black', 
                        linewidth=1.5,
                        alpha=0.9,
                        zorder=5) # Ensure points are on top of grid
            
        # Add labels with better positioning
        # Simple heuristic: move label slightly away from point
        texts = []
        for _, row in subset.iterrows():
            algo = row['Algorithm']
            x = row['E2E_P95']
            y = row['Throughput']
            
            # Use adjustText if available, otherwise manual annotation
            # Since adjustText is not guaranteed, let's use a smart manual offset
            # or simply use the legend instead of labels to avoid clutter
            
            # Option 1: Labels (can be messy)
            # plt.annotate(...)
            
            # Option 2: Legend only (cleaner, but reader has to look up)
            # Let's use Legend as requested for better "identification" via shapes/colors
            pass

        # Axis labels and title
        # plt.title(f'{scenario} Scenario', fontsize=18, fontweight='bold', pad=15)
        plt.xlabel('P95 End-to-End Delay (ms)', fontsize=16, fontweight='bold')
        plt.ylabel('Throughput (Mbps)', fontsize=16, fontweight='bold')
        
        # Grid customization
        plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
        plt.minorticks_on()
        
        # Add a legend outside or in a corner if labels are messy
        # Since we have direct labels, legend might be redundant but good for "Algorithm shape" reference
        legend = plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        # Set legend marker size to 200
        for handle in legend.legendHandles:
            handle.set_sizes([200.0])
        
        # Adjust layout
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f'scatter_{scenario}.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {output_path}")

if __name__ == "__main__":
    df = parse_data(input_file)
    plot_scatter(df, output_dir)
