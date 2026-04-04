import numpy as np
import glob
import os
from pathlib import Path

# Configuration
INPUT_DIR = "cleaned_nodes"
OUTPUT_DIR = "extracted_tasks"

# Columns 1-3 are Heart/Brain sensors (indices 0-2) - reliable
# Columns 6 and 8 are Stress/Task flags (indices 6, 8 in full data)
RELIABLE_SENSOR_COLS = [0, 1, 2]  # Heart and Brain only
FLAG_COLS = {
    6: "Task",
    8: "Stress"
}

# Threshold for flag detection (handles floating point ghosting)
FLAG_THRESHOLD = 0.5

# Minimum rows required for a valid task period
MIN_ROWS = 200

def extract_task_periods():
    """Extract task/stress periods from large numpy files."""

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))
    print(f"Processing {len(files)} files...")
    print(f"Looking for flags in columns {list(FLAG_COLS.keys())}: {list(FLAG_COLS.values())}")
    print(f"Extracting reliable sensors from columns {RELIABLE_SENSOR_COLS}")
    print(f"Flag threshold: {FLAG_THRESHOLD} (avoids floating point ghosting)")
    print("-" * 50)

    total_extracted = {name: 0 for name in FLAG_COLS.values()}
    skipped_files = []

    for file_path in files:
        subject_id = os.path.basename(file_path).replace(".npy", "")

        try:
            # Memory-mapped load for large files (doesn't load all into RAM)
            data = np.load(file_path, mmap_mode='r')
            n_rows, n_cols = data.shape

            print(f"\n{subject_id}: {n_rows:,} rows")

            # Check each flag column
            for flag_col, flag_name in FLAG_COLS.items():
                # Find rows where flag is active (needle in haystack)
                # Using np.where is efficient - doesn't scan entire file row-by-row in Python
                flag_active = data[:, flag_col] > FLAG_THRESHOLD
                active_indices = np.where(flag_active)[0]

                if len(active_indices) == 0:
                    print(f"  {flag_name}: No active rows found")
                    continue

                print(f"  {flag_name}: Found {len(active_indices)} active rows (indices {active_indices[0]} to {active_indices[-1]})")

                # Extract only the reliable sensor columns for active rows
                # This avoids the NaN trap - we only care about columns 0-2
                sensor_data = data[active_indices][:, RELIABLE_SENSOR_COLS].copy()

                # Check for NaN only in the reliable columns (not the whole row)
                nan_mask = np.isnan(sensor_data).any(axis=1)
                clean_data = sensor_data[~nan_mask]
                n_dropped = len(sensor_data) - len(clean_data)

                if n_dropped > 0:
                    print(f"    Dropped {n_dropped} rows with NaN in reliable sensors")

                if len(clean_data) < MIN_ROWS:
                    print(f"    SKIPPED: Only {len(clean_data)} clean rows (min {MIN_ROWS})")
                    continue

                # Save extracted data
                output_file = f"{OUTPUT_DIR}/{subject_id}_{flag_name}.npy"
                np.save(output_file, clean_data)
                print(f"    SAVED: {clean_data.shape[0]} rows x {clean_data.shape[1]} cols -> {output_file}")
                total_extracted[flag_name] += 1

        except Exception as e:
            print(f"\n{subject_id}: ERROR - {e}")
            skipped_files.append((subject_id, str(e)))

    # Summary
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)

    for flag_name, count in total_extracted.items():
        print(f"  {flag_name}: {count} files extracted")

    if skipped_files:
        print(f"\n{len(skipped_files)} files had errors:")
        for sid, err in skipped_files[:5]:
            print(f"  {sid}: {err}")

if __name__ == "__main__":
    extract_task_periods()
