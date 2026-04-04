import numpy as np
import glob
import pandas as pd
import gudhi
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats

files = sorted(glob.glob("cleaned_nodes/*.npy"))
results = []

print(f"Universal Scan: Analyzing {len(files)} subjects...")

for file_path in files:
    try:
        data = np.load(file_path)
        sensors = data[:, :6]
        flags = data[:, 6:]
        
        # Find the top 2 flags with the most active data for THIS subject
        counts = [np.sum(flags[:, i] > 0.5) for i in range(flags.shape[1])]
        # Get indices of the two largest flags (must have at least 100 rows)
        best_indices = [i for i, count in enumerate(counts) if count > 100]
        
        if len(best_indices) >= 2:
            # We compare the "First" active state vs "Last" active state
            # Usually Baseline (low index) vs High Task (high index)
            for f_idx in [best_indices[0], best_indices[-1]]:
                active = sensors[flags[:, f_idx] > 0.5]
                clean = active[~np.isnan(active).any(axis=1)]
                
                if len(clean) > 50:
                    scaled = StandardScaler().fit_transform(clean)
                    subset = scaled[np.random.choice(len(scaled), min(len(scaled), 100), replace=False)]
                    
                    rips = gudhi.RipsComplex(points=subset, max_edge_distance=4.0)
                    stree = rips.create_simplex_tree(max_dimension=2)
                    stree.persistence()
                    
                    b1_energy = sum([p[1][1] - p[1][0] for p in stree.persistence() if p[0] == 1])
                    # Label them as 'Early State' and 'Late State'
                    state_label = "Baseline" if f_idx == best_indices[0] else "Stress"
                    results.append({'Subject': os.path.basename(file_path), 'State': state_label, 'Energy': b1_energy})
    except:
        continue

if results:
    df = pd.DataFrame(results)
    summary = df.groupby('State')['Energy'].agg(['mean', 'std', 'count'])
    
    # Paired T-Test (Comparing each subject to themselves)
    pivot = df.pivot(index='Subject', columns='State', values='Energy').dropna()
    t_stat, p_val = stats.ttest_rel(pivot['Baseline'], pivot['Stress'])
    
    print("\n" + "="*40)
    print("      THE FINAL SATISFACTION REPORT")
    print("="*40)
    print(summary)
    print("-" * 40)
    print(f"P-Value (Paired): {p_val:.5f}")
    if p_val < 0.05: print("VERDICT: SUCCESS. The loop structure changed significantly!")
    else: print("VERDICT: TRENDING. The states are different, but variance is high.")
else:
    print("Error: No subjects found with two or more active flags.")
