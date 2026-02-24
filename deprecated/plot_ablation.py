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
    
    # Parse header to find column indices
    # Assuming pipe table format
    # | Algorithm | Throughput | ...
    
    # Skip separator line |:---|...
    data_lines = [l for l in lines if not l.strip().startswith('|:')]
    
    # Parse using pandas read_csv with separator
    # But since it's a markdown table, let's just parse manually to be safe or use regex
    
    data = []
    
    # Header is first line
    header_line = data_lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]
    
    for line in data_lines[1:]:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        
        row = {}
        for i, h in enumerate(headers):
            val_str = parts[i]
            # Extract mean value (ignore std dev for bar height, but keep for error bar)
            # Format: "4.82 ± 0.28"
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

def plot_ablation_bars(df, output_folder):
    # Metrics to plot
    metrics = [
        ('Throughput', 'Throughput (Mbps)', 'tab:blue'),
        ('Network Delay P95', 'P95 Network Delay (ms)', 'tab:orange'),
        ('E2E Delay P95', 'P95 E2E Delay (ms)', 'tab:green'),
        ('Freeze Rate', 'Freeze Rate (%)', 'tab:red'),
        ('Loss Rate', 'Loss Rate (%)', 'tab:purple')
    ]
    
    algorithms = df['Algorithm'].tolist()
    x = np.arange(len(algorithms))
    width = 0.6  # Bar width
    
    for metric_col, ylabel, color in metrics:
        if metric_col not in df.columns:
            print(f"Warning: Metric {metric_col} not found")
            continue
            
        plt.figure(figsize=(8, 6))
        
        means = df[metric_col]
        stds = df.get(metric_col + '_std', np.zeros(len(means)))
        
        bars = plt.bar(x, means, width, yerr=stds, capsize=5, 
                       color=color, edgecolor='black', alpha=0.8,
                       error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            # Position label slightly above the error bar
            y_pos = height + (stds[bars.index(bar)] if len(stds)>0 else 0)
            
            # Smart offset for text
            offset = max(means) * 0.02
            
            plt.text(bar.get_x() + bar.get_width()/2., y_pos + offset,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel(ylabel, fontweight='bold')
        plt.xticks(x, algorithms, rotation=15, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Remove title as requested usually
        # plt.title(f'{metric_col} Comparison')
        
        plt.tight_layout()
        
        # Safe filename
        safe_name = metric_col.replace(' ', '_')
        output_path = os.path.join(output_folder, f'ablation_{safe_name}.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

if __name__ == "__main__":
    df = parse_data(input_file)
    plot_ablation_bars(df, output_dir)
