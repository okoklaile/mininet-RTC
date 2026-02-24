import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
import matplotlib

# Set font for vector graphics compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Increase font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

input_file = '/home/wyq/桌面/mininet-RTC/eval_results/metrics_ablation_study.txt'
output_dir = '/home/wyq/桌面/mininet-RTC/eval_results/figures/ablation'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_data(input_path):
    with open(input_path, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    data_lines = [l for l in lines if not l.strip().startswith('|:')]
    
    data = []
    header_line = data_lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]
    
    for line in data_lines[1:]:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        
        row = {}
        for i, h in enumerate(headers):
            val_str = parts[i]
            if '±' in val_str:
                mean, std = val_str.split('±')
                row[h] = float(mean.strip())
                row[h+'_std'] = float(std.strip())
            else:
                try:
                    row[h] = float(val_str)
                    row[h+'_std'] = 0.0
                except:
                    row[h] = val_str # Algorithm name
        data.append(row)
        
    return pd.DataFrame(data)

def format_label(algo_name):
    """Format algorithm name for labels"""
    if algo_name == 'Neural-GCC':
        return r'$\bf{Neural-GCC}$' # Simple bold math mode
    
    # Replace 'Neural-GCC-No' with 'w/o '
    return algo_name.replace('Neural-GCC-No', 'w/o ')

def plot_tradeoff(df, output_folder):
    # Metrics to plot: Network Delay P95 vs Throughput
    x_metric = 'Network Delay P95'
    y_metric = 'Throughput'
    
    # Define order and colors (consistent with plot_ablation_styled.py)
    desired_order = ['Neural-GCC', 'Neural-GCC-NoQoE', 'Neural-GCC-NoKL', 'Neural-GCC-NoBC']
    
    # Map colors
    # Neural-GCC -> LimeGreen
    # NoQoE -> Blue/Cyan
    # NoKL -> Red/Magenta
    # NoBC -> Yellow/Gold
    color_map = {
        'Neural-GCC': '#32CD32',
        'Neural-GCC-NoQoE': '#00BFFF',
        'Neural-GCC-NoKL': '#FF4500',
        'Neural-GCC-NoBC': '#FFD700'
    }
    
    # Map markers (consistent with plot_scatter_enhanced.py style)
    marker_map = {
        'Neural-GCC': '*',      # Star for the proposed method
        'Neural-GCC-NoQoE': 'o', # Circle
        'Neural-GCC-NoKL': 's',  # Square
        'Neural-GCC-NoBC': '^'   # Triangle
    }

    plt.figure(figsize=(8, 6))
    
    # Plot points
    for algo in desired_order:
        row = df[df['Algorithm'] == algo]
        if row.empty:
            continue
            
        x = row[x_metric].values[0]
        y = row[y_metric].values[0]
        label = format_label(algo)
        color = color_map.get(algo, 'gray')
        marker = marker_map.get(algo, 'o')
        
        plt.scatter(x, y, 
                    label=label, 
                    s=1000,  # Large size
                    c=[color], 
                    marker=marker,
                    edgecolors='black', 
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=5)

    # Axis labels
    plt.xlabel('P95 Network Delay (ms)', fontsize=16, fontweight='bold')
    plt.ylabel('Throughput (Mbps)', fontsize=16, fontweight='bold')
    
    # Grid customization
    plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4, zorder=0)
    plt.minorticks_on()
    
    # Legend
    legend = plt.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    # Set legend marker size
    for handle in legend.legendHandles:
        handle.set_sizes([200.0])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, 'ablation_tradeoff_network.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    df = parse_data(input_file)
    plot_tradeoff(df, output_dir)
