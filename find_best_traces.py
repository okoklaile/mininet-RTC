import json
import os
import numpy as np

target_dir = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json'

def get_trace_stats(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        patterns = []
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            patterns = data['uplink']['trace_pattern']
        elif 'downlink' in data and 'trace_pattern' in data['downlink']:
            patterns = data['downlink']['trace_pattern']
        
        if not patterns:
            return None

        caps = [p['capacity'] for p in patterns]
        caps = np.array(caps)
        
        if len(caps) == 0:
            return None

        avg = np.mean(caps)
        std = np.std(caps)
        cv = std / avg if avg > 0 else float('inf')
        min_cap = np.min(caps)
        max_cap = np.max(caps)
        zero_percent = np.sum(caps == 0) / len(caps) * 100
        low_percent = np.sum(caps < 1000) / len(caps) * 100 # < 1Mbps
        
        return {
            'avg': avg,
            'std': std,
            'cv': cv,
            'min': min_cap,
            'max': max_cap,
            'zero_percent': zero_percent,
            'low_percent': low_percent,
            'count': len(caps)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def categorize_file(filename):
    if filename.startswith('Bus'):
        return 'Bus'
    elif filename.startswith('Car'):
        return 'Car'
    elif filename.startswith('Ferry'):
        return 'Ferry'
    elif any(x in filename for x in ['Train', 'Subway', 'LIRR']):
        return 'Train/Subway'
    else:
        return 'Other'

def main():
    results = []
    
    for f in os.listdir(target_dir):
        if not f.endswith('.json'):
            continue
            
        full_path = os.path.join(target_dir, f)
        if os.path.isdir(full_path):
            continue
            
        stats = get_trace_stats(full_path)
        if stats:
            category = categorize_file(f)
            # Define a simple "Quality Score"
            # Higher is better.
            # Penalize outage and high variance. Reward high bandwidth.
            # Let's just use Avg Bandwidth for "Best Condition" typically, 
            # but maybe the user wants stability.
            # I will list the best by Avg Bandwidth and Best by Stability (Lowest CV).
            
            results.append({
                'file': f,
                'category': category,
                'stats': stats
            })

    # Group by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"{'Category':<15} | {'File':<30} | {'Avg (kbps)':<10} | {'Outage %':<10} | {'CV':<6} | {'Max (kbps)':<10}")
    print("-" * 95)

    overall_best_avg = None
    overall_best_stable = None

    for cat, items in categories.items():
        # Sort by Avg Bandwidth (descending)
        items.sort(key=lambda x: x['stats']['avg'], reverse=True)
        best_avg = items[0]
        
        # Sort by CV (ascending) - for stability, but ignore very low bandwidth ones if any
        # Filter out very low bandwidth (< 100 kbps) for stability check to avoid "stable dead" links
        valid_stable = [x for x in items if x['stats']['avg'] > 100]
        if valid_stable:
            valid_stable.sort(key=lambda x: x['stats']['cv'])
            best_stable = valid_stable[0]
        else:
            best_stable = items[0]

        print(f"--- {cat} ---")
        # Print top 3 by Avg
        for item in items[:3]:
             s = item['stats']
             print(f"{cat:<15} | {item['file']:<30} | {s['avg']:<10.0f} | {s['zero_percent']:<10.1f} | {s['cv']:<6.2f} | {s['max']:<10.0f}")
        
        # Update overall best
        if overall_best_avg is None or best_avg['stats']['avg'] > overall_best_avg['stats']['avg']:
            overall_best_avg = best_avg
            
        if overall_best_stable is None or (best_stable['stats']['cv'] < overall_best_stable['stats']['cv'] and best_stable['stats']['avg'] > 1000):
            overall_best_stable = best_stable

    print("\n" + "="*50)
    print("SUMMARY OF BEST CONDITIONS")
    print("="*50)
    
    if overall_best_avg:
        s = overall_best_avg['stats']
        print(f"Highest Average Bandwidth: {overall_best_avg['file']}")
        print(f"  Category: {overall_best_avg['category']}")
        print(f"  Avg: {s['avg']:.0f} kbps")
        print(f"  Outage: {s['zero_percent']:.1f}%")
        print(f"  CV: {s['cv']:.2f}")

    if overall_best_stable:
        s = overall_best_stable['stats']
        print(f"\nMost Stable (Lowest Variation): {overall_best_stable['file']}")
        print(f"  Category: {overall_best_stable['category']}")
        print(f"  Avg: {s['avg']:.0f} kbps")
        print(f"  Outage: {s['zero_percent']:.1f}%")
        print(f"  CV: {s['cv']:.2f}")
        
    # Also find the one with 0% outage and highest bandwidth
    zero_outage = [r for r in results if r['stats']['zero_percent'] == 0]
    if zero_outage:
        zero_outage.sort(key=lambda x: x['stats']['avg'], reverse=True)
        best_zero_outage = zero_outage[0]
        s = best_zero_outage['stats']
        print(f"\nBest with 0% Outage: {best_zero_outage['file']}")
        print(f"  Category: {best_zero_outage['category']}")
        print(f"  Avg: {s['avg']:.0f} kbps")
        print(f"  CV: {s['cv']:.2f}")

if __name__ == "__main__":
    main()
