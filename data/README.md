# Oxford IO-VNBD Dataset Directory

This directory stores the **Indoor/Outdoor Vehicle Navigation Benchmark Dataset (IO-VNBD)** raw CSV files.

### Dataset Overview
- **Source**: Oxford IO-VNBD Benchmark
- **Coverage**: Over 100+ km of driving across diverse vehicles (sedans, SUVs, hatchbacks) and mount types (windshield suction cup, dashboard cradle, center console).
- **Sampling Frequency**: 10 Hz ($\Delta t = 0.1\text{ s}$)
- **Ground Truth**: Centimeter-accurate RTK-GNSS and tactical reference odometry.

### Expected Directory Layout
```
data/
├── S-M.csv        # Drive M test drive (Smartphone IMU)
├── V-M.csv        # Drive M ground truth reference (Tactical RTK-GNSS)
├── S-S1.csv       # Cross-drive training pair 1 (IMU)
├── V-S1.csv       # Cross-drive training pair 1 (Reference)
├── S-Vfa01.csv    # Cross-drive training pair 7 (IMU)
├── V-Vfa01.csv    # Cross-drive training pair 7 (Reference)
├── ...
```

*Note: Raw CSV files (> 2 GB) are excluded from the git repository via `.gitignore` to maintain fast repository operations and adhere to GitHub size limits.*
