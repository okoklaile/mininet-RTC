import json
import os
import numpy as np

target_dir = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json'
ref_file = '7Train_7BtrainNew.json'

def get_trace_stats(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Prefer uplink, fallback to downlink if uplink missing or empty
        patterns = []
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            patterns = data['uplink']['trace_pattern']
        elif 'downlink' in data and 'trace_pattern' in data['downlink']:
            patterns = data['downlink']['trace_pattern']
        
        if not patterns:
            return None

        caps = [p['capacity'] for p in patterns]
        caps = np.array(caps)
        
        avg = np.mean(caps)
        std = np.std(caps)
        cv = std / avg if avg > 0 else 0
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

def main():
    ref_path = os.path.join(target_dir, ref_file)
    print(f"Analyzing reference trace: {ref_file}...")
    ref_stats = get_trace_stats(ref_path)
    
    if not ref_stats:
        print("Failed to analyze reference trace.")
        return

    print(f"Reference Stats:")
    print(f"  Avg: {ref_stats['avg']:.2f} kbps")
    print(f"  Std: {ref_stats['std']:.2f} kbps")
    print(f"  CV: {ref_stats['cv']:.2f}")
    print(f"  Zero%: {ref_stats['zero_percent']:.2f}%")
    print(f"  Low (<1Mbps)%: {ref_stats['low_percent']:.2f}%")
    print("-" * 40)

    results = []
    
    for f in os.listdir(target_dir):
        if not f.endswith('.json') or f == ref_file:
            continue
            
        full_path = os.path.join(target_dir, f)
        if os.path.isdir(full_path):
            continue
            
        stats = get_trace_stats(full_path)
        if stats:
            # Calculate similarity score (lower is better)
            # Weights: Avg (1), Std (1), Zero% (2) - prioritizing outage similarity?
            # Or maybe just general profile.
            
            # Normalized differences
            diff_avg = abs(ref_stats['avg'] - stats['avg']) / (ref_stats['avg'] + 1e-6)
            diff_std = abs(ref_stats['std'] - stats['std']) / (ref_stats['std'] + 1e-6)
            diff_zero = abs(ref_stats['zero_percent'] - stats['zero_percent']) / 100.0
            
            score = diff_avg + diff_std + diff_zero
            
            results.append({
                'file': f,
                'stats': stats,
                'score': score
            })

    # Sort by score (ascending)
    results.sort(key=lambda x: x['score'])
    
    print(f"Found {len(results)} other traces. Top 5 most similar:")
    for i, res in enumerate(results[:10]):
        s = res['stats']
        print(f"{i+1}. {res['file']} (Score: {res['score']:.4f})")
        print(f"   Avg: {s['avg']:.0f}, Std: {s['std']:.0f}, CV: {s['cv']:.2f}")
        print(f"   Zero%: {s['zero_percent']:.1f}%, Low%: {s['low_percent']:.1f}%")
    
    print("\nHigh Bandwidth Traces (Avg > 10Mbps):")
    high_bw = [r for r in results if r['stats']['avg'] > 10000]
    high_bw.sort(key=lambda x: x['stats']['avg'], reverse=True)
    for r in high_bw[:5]:
        print(f"- {r['file']}: Avg {r['stats']['avg']:.0f} kbps")

if __name__ == "__main__":
    main()
