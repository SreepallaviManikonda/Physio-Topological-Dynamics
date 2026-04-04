# Preprocessed Data for Topological Analysis

## Overview
This dataset contains extracted task and stress periods from the MCL study, preprocessed for topological data analysis (TDA) of physiological synchrony.

## Data Files

### Location: `extracted_tasks/`
- **58 .npy files** (29 subjects × 2 conditions)
- Each file: `(N rows, 3 columns)` where N ≈ 1000-1600
- **Sampling rate: 10 Hz** (100ms between samples)
- **No NaN values** - all data is clean

### Naming Convention
```
{SUBJECT_ID}_{CONDITION}.npy
├── SUBJECT_ID: 0000-ACAA, 0001-AGDJ, etc.
└── CONDITION: Task or Stress
```

### Column Mapping
| Column | Sensor | Type | Notes |
|--------|--------|------|-------|
| 0 | EEG | Brain/Cortical | ~10Hz alpha-like oscillation |
| 1 | HRV | Heart/Cardiac | Heart rate variability |
| 2 | EDA | Skin/Autonomic | Electrodermal activity |

### Conditions
- **Task**: Baseline cognitive load (control condition)
- **Stress**: High cognitive load/overload periods (experimental condition)

## Processing Pipeline

### What Was Done:
1. **Extracted** only reliable sensors (cols 0-2) from raw 14-column files
2. **Identified** task/stress periods using flag columns (6 and 8)
3. **Removed** NaN values only in reliable sensor columns (avoided the "NaN trap")
4. **Saved** clean segments as individual .npy files

### What Was NOT Included:
- Eye-tracking (column 4-5): Mostly NaN, unreliable
- Speech/audio: Not in original data
- Behavioral logs: Not in original data

## Analysis Pipeline

### Running the Analysis
```bash
python3 fixed_synchrony_analysis.py
```

### What It Does:
1. **Windowing**: 10-second windows (100 samples) with 5-second overlap
2. **Feature Extraction**: Per-modality features per window
3. **Synchrony Computation**: Rolling correlations between sensor pairs:
   - EEG ↔ HRV (brain-heart coupling)
   - EEG ↔ EDA (brain-skin coupling)
   - HRV ↔ EDA (heart-skin coupling)
4. **TDA**: Persistent homology on synchrony time series topology
5. **Statistics**: Compare Task vs Stress synchrony

### Key Metrics Output:
- `mean_sync`: Average correlation strength
- `sync_stability`: 1/(1+variance) - rigidity measure
- `betti_1`: Number of topological holes (loops) in synchrony space
- `avg_lifetime_dim1`: Average persistence of coupling patterns

## Results Summary

**Significant Findings (p < 0.001):**
- EEG-HRV synchrony: Task (-0.047) → Stress (+0.084) - flips positive
- EEG-EDA synchrony: Task (-0.140) → Stress (+0.027) - flips positive
- Synchrony becomes more rigid/stable under stress

**Interpretation:**
Under cognitive overload, the brain loses autonomy and becomes entrained with autonomic responses (heart and skin). This is a topological phase transition in physiological coupling.

## Next Steps / Phase 2

Your friend can extend this to:

1. **Hypergraph Analysis** (Higher-order interactions)
   - 3-way interactions: EEG ∩ HRV ∩ EDA simultaneously
   - Use libraries: HyperNetX, PyTorch Geometric

2. **Filtration Movies**
   - Visualize how graph topology evolves over time
   - Watch the "collapse" into rigid entrainment

3. **Early Warning Detection**
   - Predict stress onset before it happens
   - Critical slowing down indicators

4. **Additional Modalities**
   - If you have speech/audio, add as column 3
   - If you have eye-tracking that works, add as column 4

## File Reference

| File | Purpose |
|------|---------|
| `fixed_synchrony_analysis.py` | Main analysis pipeline |
| `fixed_synchrony_results.csv` | Statistical results (Task vs Stress) |
| `extract_task_periods.py` | Data extraction script |
| `synchrony_analysis_summary.txt` | Text summary of findings |

## Data Quality Notes

-  All files have >200 rows (minimum for TDA)
- No NaN values in extracted data
- 28/29 subjects have usable data (0005-BQAB Stress is small)
-  Original eye-tracking had 99.9% missing data (excluded)



For questions about the preprocessing, see the original extraction script.
For questions about TDA methodology, see `fixed_synchrony_analysis.py` comments.
