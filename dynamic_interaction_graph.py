"""
Dynamic Interaction Graph - Step 1 Complete
Fuse EEG, HRV, EDA into ONE time-varying graph
"""

import numpy as np
import glob
import pandas as pd
import gudhi
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
EXTRACTED_DIR = "extracted_tasks"
RESULTS_FILE = "dynamic_graph_results.csv"

FS = 10  # Hz
WINDOW_SEC = 10
WINDOW_SAMPLES = WINDOW_SEC * FS
STEP_SEC = 5
STEP_SAMPLES = STEP_SEC * FS

def build_dynamic_graph(data):
    """
    Build ONE dynamic interaction graph where:
    - Nodes: EEG (0), HRV (1), EDA (2)
    - Edges: time-varying correlation weights
    """
    eeg = data[:, 0]
    hrv = data[:, 1]
    eda = data[:, 2]

    n_samples = len(eeg)
    if n_samples < WINDOW_SAMPLES * 2:
        return None

    # Time-varying graph edges
    graph_sequence = []

    for start in range(0, n_samples - WINDOW_SAMPLES, STEP_SAMPLES):
        w_eeg = eeg[start:start + WINDOW_SAMPLES]
        w_hrv = hrv[start:start + WINDOW_SAMPLES]
        w_eda = eda[start:start + WINDOW_SAMPLES]

        # Remove NaN
        mask = ~(np.isnan(w_eeg) | np.isnan(w_hrv) | np.isnan(w_eda))
        if np.sum(mask) < WINDOW_SAMPLES // 2:
            continue

        w_eeg = w_eeg[mask]
        w_hrv = w_hrv[mask]
        w_eda = w_eda[mask]

        if len(w_eeg) < 20:
            continue

        # Build graph at this time point
        # 3 nodes: 0=EEG, 1=HRV, 2=EDA
        # 3 edges: (0,1), (0,2), (1,2)

        corr_eeg_hrv, _ = pearsonr(w_eeg, w_hrv)
        corr_eeg_eda, _ = pearsonr(w_eeg, w_eda)
        corr_hrv_eda, _ = pearsonr(w_hrv, w_eda)

        if any(np.isnan([corr_eeg_hrv, corr_eeg_eda, corr_hrv_eda])):
            continue

        # Graph metrics at this time point
        edge_weights = [abs(corr_eeg_hrv), abs(corr_eeg_eda), abs(corr_hrv_eda)]
        mean_coupling = np.mean(edge_weights)
        graph_density = np.mean(edge_weights)  # All possible edges present
        clustering = np.mean([abs(corr_eeg_hrv) * abs(corr_eeg_eda) * abs(corr_hrv_eda)]) ** (1/3)

        # Modularity: how "clique-like" is the graph?
        # High modularity = fragmented into pairs, low = fully connected
        modularity = np.std(edge_weights) / (np.mean(edge_weights) + 1e-8)

        # Integration measure: all three coupled together
        triple_coupling = np.min(edge_weights)  # Weak link determines global coupling

        graph_sequence.append({
            'time': start / FS,  # seconds
            'edge_eeg_hrv': corr_eeg_hrv,
            'edge_eeg_eda': corr_eeg_eda,
            'edge_hrv_eda': corr_hrv_eda,
            'mean_coupling': mean_coupling,
            'graph_density': graph_density,
            'clustering': clustering,
            'modularity': modularity,
            'triple_coupling': triple_coupling,
            'fragmentation': 1 - triple_coupling
        })

    return graph_sequence


def analyze_graph_topology(graph_sequence):
    """
    Analyze the topology of the dynamic graph sequence.
    """
    if len(graph_sequence) < 10:
        return None

    # Extract time series of graph metrics
    fragmentation = np.array([g['fragmentation'] for g in graph_sequence])
    modularity = np.array([g['modularity'] for g in graph_sequence])
    clustering = np.array([g['clustering'] for g in graph_sequence])
    mean_coupling = np.array([g['mean_coupling'] for g in graph_sequence])

    # Early vs late analysis (critical transition detection)
    n = len(graph_sequence)
    early = slice(0, n//3)
    late = slice(-n//3, None)

    stats = {
        'n_graphs': n,
        'duration_sec': graph_sequence[-1]['time'] - graph_sequence[0]['time'],

        # Overall graph properties
        'mean_fragmentation': np.mean(fragmentation),
        'std_fragmentation': np.std(fragmentation),
        'mean_modularity': np.mean(modularity),
        'std_modularity': np.std(modularity),
        'mean_clustering': np.mean(clustering),
        'mean_coupling': np.mean(mean_coupling),

        # Critical slowing down indicators
        'frag_early': np.mean(fragmentation[early]),
        'frag_late': np.mean(fragmentation[late]),
        'frag_trend': 'increasing' if np.polyfit(range(n), fragmentation, 1)[0] > 0 else 'stable',

        'mod_early': np.mean(modularity[early]),
        'mod_late': np.mean(modularity[late]),

        # Topology transitions
        'n_fragmentation_shifts': np.sum(np.diff(fragmentation) > 0.2),

        # Network stability (autocorrelation of graph structure)
        'graph_autocorr': np.corrcoef(fragmentation[:-1], fragmentation[1:])[0,1] if len(fragmentation) > 1 else 0,
    }

    return stats, graph_sequence


def process_subject(file_path, condition):
    """Process one subject through the dynamic graph pipeline."""
    subject_id = Path(file_path).stem.replace(f"_{condition}", "")

    try:
        data = np.load(file_path)

        if len(data) < WINDOW_SAMPLES * 2:
            return None, f"{subject_id}: Insufficient data ({len(data)} samples)"

        # Build dynamic graph
        graph_sequence = build_dynamic_graph(data)

        if graph_sequence is None or len(graph_sequence) < 10:
            return None, f"{subject_id}: Failed to build graph sequence"

        # Analyze topology
        stats, _ = analyze_graph_topology(graph_sequence)

        if stats is None:
            return None, f"{subject_id}: Failed topology analysis"

        stats['subject'] = subject_id
        stats['condition'] = condition

        return stats, f"{subject_id}: {stats['n_graphs']} graphs, " \
                     f"frag={stats['mean_fragmentation']:.3f}, " \
                     f"trend={stats['frag_trend']}"

    except Exception as e:
        return None, f"{subject_id}: ERROR - {e}"


def main():
    print("=" * 70)
    print("DYNAMIC INTERACTION GRAPH - STEP 1 COMPLETE")
    print("Fuse EEG, HRV, EDA into ONE unified time-varying graph")
    print("=" * 70)
    print(f"Nodes: 3 (EEG=0, HRV=1, EDA=2)")
    print(f"Edges: 3 (time-varying correlation weights)")
    print(f"Window: {WINDOW_SEC}s, Step: {STEP_SEC}s")
    print("-" * 70)

    # Find files
    task_files = sorted(glob.glob(f"{EXTRACTED_DIR}/*_Task.npy"))
    stress_files = sorted(glob.glob(f"{EXTRACTED_DIR}/*_Stress.npy"))

    print(f"Found {len(task_files)} Task files")
    print(f"Found {len(stress_files)} Stress files")
    print("-" * 70)

    all_results = []

    # Process Task
    print("\n🔍 Building dynamic graphs for TASK...")
    for file_path in task_files:
        result, msg = process_subject(file_path, "Task")
        print(f"  {msg}")
        if result:
            all_results.append(result)

    # Process Stress
    print("\n🔍 Building dynamic graphs for STRESS...")
    for file_path in stress_files:
        result, msg = process_subject(file_path, "Stress")
        print(f"  {msg}")
        if result:
            all_results.append(result)

    # Analysis
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\n✅ Results saved to: {RESULTS_FILE}")

        # Compare Task vs Stress
        task_data = df[df['condition'] == 'Task']
        stress_data = df[df['condition'] == 'Stress']

        common_subjects = set(task_data['subject']) & set(stress_data['subject'])
        print(f"\nSubjects with both conditions: {len(common_subjects)}")

        print("\n" + "=" * 70)
        print("DYNAMIC GRAPH TOPOLOGY: TASK vs STRESS")
        print("=" * 70)

        metrics = [
            ('Fragmentation', 'mean_fragmentation'),
            ('Modularity', 'mean_modularity'),
            ('Clustering', 'mean_clustering'),
            ('Graph Autocorr', 'graph_autocorr'),
            ('Fragmentation Shifts', 'n_fragmentation_shifts'),
        ]

        print(f"\n{'Metric':<25} {'Task':>12} {'Stress':>12} {'p-value':>12}")
        print("-" * 65)

        from scipy import stats as scipy_stats

        significant = []

        for label, metric in metrics:
            task_vals = task_data[task_data['subject'].isin(common_subjects)][metric].values
            stress_vals = stress_data[stress_data['subject'].isin(common_subjects)][metric].values

            if len(task_vals) == 0 or len(stress_vals) == 0:
                continue

            t_stat, p_val = scipy_stats.ttest_rel(stress_vals, task_vals)

            task_mean = np.mean(task_vals)
            stress_mean = np.mean(stress_vals)

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            print(f"{label:<25} {task_mean:>12.3f} {stress_mean:>12.3f} {p_val:>12.4f} {sig}")

            if p_val < 0.05:
                significant.append({
                    'metric': label,
                    'task': task_mean,
                    'stress': stress_mean,
                    'p': p_val
                })

        # Trend analysis
        print("\n" + "=" * 70)
        print("FRAGMENTATION TRENDS")
        print("=" * 70)

        task_increasing = sum(task_data['frag_trend'] == 'increasing')
        stress_increasing = sum(stress_data['frag_trend'] == 'increasing')

        print(f"\nTask:   {task_increasing}/{len(task_data)} show increasing fragmentation")
        print(f"Stress: {stress_increasing}/{len(stress_data)} show increasing fragmentation")

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if significant:
            print(f"\n✅ FOUND {len(significant)} SIGNIFICANT DIFFERENCES:")
            for finding in significant:
                direction = "higher" if finding['stress'] > finding['task'] else "lower"
                print(f"\n  • {finding['metric']}:")
                print(f"    Task:   {finding['task']:.3f}")
                print(f"    Stress: {finding['stress']:.3f} ({direction})")
                print(f"    p = {finding['p']:.4f}")

            print("\n  🎯 Step 1 Complete:")
            print("     - Fused 3 modalities into ONE dynamic graph")
            print("     - Graph topology differs between Task and Stress")
            print("     - Fragmentation signature captured")
        else:
            print("\n⚠️ No significant differences found")
            print("   Consider adjusting window sizes or trying different graph metrics")

        # Save summary
        with open("dynamic_graph_summary.txt", 'w') as f:
            f.write("DYNAMIC INTERACTION GRAPH ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write("Step 1: Multimodal Data Fusion into Dynamic Graph\n\n")
            f.write(f"Subjects: {len(common_subjects)} with both conditions\n")
            f.write(f"Nodes: 3 (EEG, HRV, EDA)\n")
            f.write(f"Edges: 3 (time-varying correlations)\n\n")

            if significant:
                f.write("SIGNIFICANT FINDINGS:\n")
                for finding in significant:
                    f.write(f"{finding['metric']}: {finding['task']:.3f} -> "
                           f"{finding['stress']:.3f} (p={finding['p']:.4f})\n")

        print(f"\nSummary saved to: dynamic_graph_summary.txt")

    else:
        print("\n❌ No results generated.")


if __name__ == "__main__":
    main()
