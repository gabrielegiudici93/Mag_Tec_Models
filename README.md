# MagTec Models - Multi-Point Tactile Sensor ML Framework

This repository contains machine learning models for multi-point tactile sensing using magnetic field sensors. The system enables force regression (Fx, Fy, Fz) and classification tasks (location and stretch level) with high accuracy.

## Performance Summary

- **Fz Regression**: RMSE 0.0854 N, R² = 0.9898
- **Fx Regression**: RMSE 0.2021 N, R² = 0.9827
- **Fy Regression**: RMSE 0.1512 N, R² = 0.9900
- **Location Classification**: 99.59% accuracy
- **Stretch Classification**: 99.30% accuracy

## Repository Structure

```
Mag_Tec_Models/
├── src/
│   ├── training/              # Training scripts
│   │   ├── train_best_models.py
│   │   ├── train_separate_forces.py
│   │   └── train_location_all_methods.py
│   ├── franka_controller/     # Data collection scripts
│   └── ...
├── data/
│   └── Multiple_Points/
│       ├── 2.5mm_single_test42/    # Normal forces dataset
│       └── shear_forces_test51/    # Shear forces dataset
│           └── best_models/
│               └── best_models/
│                   └── models/     # Trained models
├── docs/
│   └── model_documentation.tex     # Technical documentation
└── README.md
```

## Quick Start

### Training Models

Train all models with optimal configurations:

```bash
python3 src/training/train_best_models.py \
    --normal-dir data/Multiple_Points/2.5mm_single_test42 \
    --shear-dir data/Multiple_Points/shear_forces_test51 \
    --run-label best_models \
    --remove-outliers
```

### Data Collection

**Normal Forces:**
```bash
python3 src/franka_controller/franka_skin_test_multiple_points.py
```

**Shear Forces:**
```bash
python3 src/franka_controller/franka_skin_test_shear_forces.py
```

## Requirements

- Python 3.8+
- NumPy, SciPy
- scikit-learn
- h5py
- joblib
- pyfranka_interface (for robot control)

## Documentation

See `docs/model_documentation.tex` for detailed technical documentation including:
- Mathematical framework
- Model architecture
- Training procedures
- Performance metrics
- Code structure and usage

## Data Format

Data is stored in HDF5 format:
- **Forces**: [N, 6] array (Fx, Fy, Fz, Tx, Ty, Tz)
- **Magnetic Field**: [N, 15, 3] array (15 taxels × 3D vectors)
- **Metadata**: Location, stretch level, timestamps

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{magtec_models,
  title={MagTec Models: Multi-Point Tactile Sensor ML Framework},
  author={MagTec Models Research Team},
  year={2024},
  url={https://github.com/yourusername/Mag_Tec_Models}
}
```
