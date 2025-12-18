# GitHub Upload Commands

## Prerequisites

1. Install Git LFS:
```bash
sudo apt-get install git-lfs
git lfs install
```

## Step-by-Step Upload

### 1. Check Repository Status
```bash
cd /home/gabriele/franka_interface/Mag_Tec_Models
git status
```

### 2. Add All Files
```bash
# Add documentation and config files first
git add docs/ README.md .gitattributes .gitignore SETUP_GITHUB.md setup_github.sh model_summary.json

# Add source code
git add src/

# Add data files (will use Git LFS)
git add data/

# Add trained models (will use Git LFS)
git add data/Multiple_Points/*/best_models/
```

### 3. Commit
```bash
git commit -m "Initial commit: Multi-point tactile sensor ML models

- Force regression models (Fz, Fx, Fy)
- Location and stretch classification models
- Complete technical documentation (LaTeX)
- Training and data collection scripts
- Full dataset (normal + shear forces)"
```

### 4. Create GitHub Repository

Go to https://github.com/new and create a new repository:
- Name: `Mag_Tec_Models`
- Description: "Multi-Point Tactile Sensor ML Framework"
- Visibility: Public or Private (your choice)
- **Do NOT** initialize with README, .gitignore, or license

### 5. Add Remote and Push
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/Mag_Tec_Models.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub (this may take time due to large files)
git push -u origin main
```

### 6. Verify Git LFS
```bash
git lfs ls-files
```

This should show all `.h5` and `.joblib` files tracked by LFS.

## Troubleshooting

### If Git LFS is not working:
```bash
# Migrate existing files to LFS
git lfs migrate import --include="*.h5,*.joblib" --everything

# Force push
git push --force origin main
```

### If push fails due to large files:
```bash
# Check file sizes
git lfs ls-files

# Ensure LFS is tracking correctly
git lfs track "*.h5"
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Configure Git LFS"
git push origin main
```

## File Size Information

- **Total repository size**: ~8-10 GB
- **HDF5 data files**: ~3-5 GB each
- **Trained models**: ~10-50 MB each
- **Source code**: ~5-10 MB

Git LFS handles large files efficiently, storing them separately from the main repository.
