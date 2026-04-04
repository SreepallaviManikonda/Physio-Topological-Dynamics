import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys

# Set backend to save files without popup windows
matplotlib.use('Agg')

def plot_stress_results(csv_path):
    try:
        # 1. Try to read as CSV, then try as Tab-Separated if that fails
        df = pd.read_csv(csv_path)
        if 'condition' not in df.columns:
            df = pd.read_csv(csv_path, sep='\t')
        
        # 2. Clean column names (removes hidden spaces/tabs)
        df.columns = df.columns.str.strip()
        
        # Print columns for debugging if it fails again
        print(f"Detected columns: {list(df.columns)}")
        
        if 'condition' not in df.columns:
            print("ERROR: Could not find 'condition' column. Check your CSV header.")
            return

        # 3. Filter data
        task_data = df[df['condition'].str.strip() == 'Task']['mean_fragmentation']
        stress_data = df[df['condition'].str.strip() == 'Stress']['mean_fragmentation']
        
        if task_data.empty or stress_data.empty:
            print("ERROR: No data found for 'Task' or 'Stress'. Check capitalization.")
            return

        # 4. Create the Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- PLOT 1: Boxplot ---
        bplot = ax1.boxplot([task_data, stress_data], 
                            patch_artist=True, 
                            labels=['Task (Baseline)', 'Stress (Overload)'])
        
        colors = ['#88CCEE', '#CC6677']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            
        ax1.set_title('Physiological Fragmentation', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Fragmentation Index', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # --- PLOT 2: Slope Chart ---
        stress_df = df[df['condition'].str.strip() == 'Stress']
        
        for _, row in stress_df.iterrows():
            ax2.plot(['Early', 'Late'], [row['frag_early'], row['frag_late']], 
                     marker='o', color='gray', alpha=0.3, linewidth=1)
        
        ax2.plot(['Early', 'Late'], [stress_df['frag_early'].mean(), stress_df['frag_late'].mean()], 
                 marker='s', color='red', linewidth=3, label='Mean Trend')
        
        ax2.set_title('Internal Decay (Stress Condition)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        output_name = "stress_topology_results.png"
        plt.savefig(output_name, dpi=300)
        print(f"✅ SUCCESS: Plots saved to {output_name}")

    except Exception as e:
        print(f"failed: {e}")

if __name__ == "__main__":
    plot_stress_results("dynamic_graph_results_formatted.csv")