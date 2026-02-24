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
    """Format algorithm name for X-axis labels"""
    if algo_name == 'Neural-GCC':
        return r'$\bf{Neural-GCC}$' # Simple bold math mode
    
    # Replace 'Neural-GCC-No' with 'w/o '
    return algo_name.replace('Neural-GCC-No', 'w/o ')

def plot_ablation_bars(df, output_folder):
    # Metrics to plot
    metrics = [
        ('Throughput', 'Throughput (Mbps)'),
        ('Network Delay P95', 'P95 Network Delay (ms)'),
        ('E2E Delay P95', 'P95 E2E Delay (ms)'),
        ('Freeze Rate', 'Freeze Rate (%)'),
        ('Loss Rate', 'Loss Rate (%)')
    ]
    
    # Define order: Neural-GCC first, then Neural-GCC-NoQoE, then others
    # User request: "Neural-GCC" and "Neural-GCC-NoQoE" swapped?
    # Original: NoQoE (first in file), Neural-GCC, NoKL, NoBC
    # Request: "Swap NeuralGCC and QoE position", "Neural yellow"
    
    # Let's force a specific order
    desired_order = ['Neural-GCC', 'Neural-GCC-NoQoE', 'Neural-GCC-NoKL', 'Neural-GCC-NoBC']
    
    # Reorder dataframe
    df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=desired_order, ordered=True)
    df = df.sort_values('Algorithm')
    
    algorithms = df['Algorithm'].tolist()
    formatted_labels = [format_label(a) for a in algorithms]
    x = np.arange(len(algorithms))
    width = 0.6
    
    # Define vivid colors
    # Neural-GCC -> LimeGreen
    # NoQoE -> Blue/Cyan
    # NoKL -> Red/Magenta
    # NoBC -> Yellow/Gold
    colors = ['#32CD32', '#00BFFF', '#FF4500', '#FFD700']
    hatches = ['', '//', '\\\\', '..']
    
    for metric_col, ylabel in metrics:
        if metric_col not in df.columns:
            continue
            
        plt.figure(figsize=(8, 6))
        
        means = df[metric_col]
        stds = df.get(metric_col + '_std', np.zeros(len(means)))
        
        bars = plt.bar(x, means, width, yerr=stds, capsize=5, 
                       color=[colors[i % len(colors)] for i in range(len(x))],
                       edgecolor='black', alpha=0.9,
                       error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
        
        # Apply hatch patterns
        for i, bar in enumerate(bars):
            bar.set_hatch(hatches[i % len(hatches)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            y_pos = height + (stds[bars.index(bar)] if len(stds)>0 else 0)
            offset = max(means) * 0.02
            
            plt.text(bar.get_x() + bar.get_width()/2., y_pos + offset,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel(ylabel, fontweight='bold')
        plt.xticks(x, formatted_labels, rotation=0, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        safe_name = metric_col.replace(' ', '_')
        output_path = os.path.join(output_folder, f'ablation_{safe_name}.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

if __name__ == "__main__":
    df = parse_data(input_file)
    # Ensure Neural-GCC is first or in a specific order if needed
    # Currently it respects the file order
    plot_ablation_bars(df, output_dir)
