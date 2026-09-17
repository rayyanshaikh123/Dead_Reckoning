# Context Continuation — SIH-IDR Workspace

## Last Updated: 2026-09-14T02:00Z

## Project Overview
SIH (Smart India Hackathon) — Inertial Dead Reckoning system. Smartphone-based vehicle position estimation during GNSS blackouts using IO-VNBD dataset.

## Current State

### Files Created/Modified
| File | Status | Purpose |
|------|--------|---------|
| `src/baseline_graphs.py` | Modified | Baseline INS processing & 8 baseline comparison plots |
| `src/idnn_model.py` | Created | IDNN v1 model (feedforward with tapped delay lines) |
| `src/idnn_pipeline.py` | Created | IDNN v1 train/eval pipeline |
| `src/idnn_model_v2.py` | Created | IDNN v2 model with BatchNorm for multi-scale features |
| `src/idnn_pipeline_v2.py` | Created | v2 pipeline: window stats + spectral + mag + orientation features |
| `src/idnn_model_v3.py` | Created | IDNN v3 model: Tapped delay lines + BatchNorm + Jerk features |
| `src/idnn_pipeline_v3.py` | Created | v3 pipeline: Multi-file training + Jerk + Kalman Filter + ZUPT on RTX 4060 GPU |
| `src/crossdrive_pipeline.py` | Created | 100% Blind Cross-Drive Pipeline (trained on external S1..S4, Vfa01; tested on blind Drive M) |
| `src/simulate_blackout.py` | **UPDATED** | Fixed interpolation clamping bug; true unclamped map matching evaluated across 5 scenarios |
| `results/plots/blackout_simulation/` | **UPDATED** | 8 comparison plots with true unclamped map matching and vector lock |

### 🚀 True Un-clamped GNSS Blackout Simulation Results

| Scenario | Duration & Distance | Model | 1D Dist Drift % | 2D Position Error (m) | 2D Drift % |
|:---|:---:|:---|:---:|:---:|:---:|
| **Straight Highway Tunnel** | 30s / 648 m | Baseline INS | 121.8% | 792.0 m | 122.2% |
| | | IDNN v3 (Phone Compass) | 24.6% | 174.2 m | 26.9% |
| | | IDNN v3 + Vector Lock (User Idea) | 24.6% | 173.9 m | 26.8% |
| | | **IDNN v3 + Map Matching** | **24.6%** | **157.3 m** | **24.3%** |
| **Short Underpass** | 30s / 219 m | Baseline INS | 431.4% | 820.3 m | 373.8% |
| | | IDNN v3 (Gyro Integrated) | 42.2% | 307.9 m | 140.3% |
| | | IDNN v3 + Vector Lock (User Idea) | 42.2% | 92.8 m | 42.3% |
| | | **IDNN v3 + Map Matching** | **42.2%** | **78.6 m** | **35.8%** |
| **Medium Tunnel** | 60s / 448 m | Baseline INS | 733.0% | 2,193.7 m | 489.7% |
| | | IDNN v3 + Vector Lock (User Idea) | 15.0% | 98.6 m | 22.0% |
| | | **IDNN v3 + Map Matching** | **15.0%** | **54.3 m** 🎯 | **12.1%** |
| **Long Mountain Tunnel** | 120s / 1,680 m | Baseline INS | 355.5% | 5,328.5 m | 317.2% |
| | | IDNN v3 (Gyro Integrated) | 5.8% | 2,174.7 m | 129.5% |
| | | IDNN v3 + Vector Lock (User Idea) | 5.8% | 1,030.8 m | 61.4% |
| | | **IDNN v3 + Map Matching** | **5.8%** 🏆 | **97.9 m** 🏆 | **5.8%** 🏆 |
| **Complex City Canyon** | 90s / 600 m | Baseline INS | 508.8% | 2,761.5 m | 460.3% |
| | | IDNN v3 + Vector Lock (User Idea) | 19.9% | 355.6 m | 59.3% |
| | | **IDNN v3 + Map Matching** | **19.9%** | **116.3 m** | **19.4%** |

### Clarification on Previous 0.0m Artifact
- In the earlier script, `gt_x` and `gt_y` were cut off at the exact exit of the tunnel. Whenever predicted distance exceeded tunnel length by even 1 meter, `np.interp` clamped to the final coordinate $(gt\_x[-1], gt\_y[-1])$, producing an artificial $0.0\text{ m}$ Euclidean error.
- Fixed: Slicing extends 1,000 samples past the tunnel exit, correctly capturing real along-track overshooting/undershooting.
