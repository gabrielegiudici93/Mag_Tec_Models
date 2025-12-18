# GitHub Setup Instructions

## Prerequisites

1. Install Git LFS (for large HDF5 files):
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download from https://git-lfs.github.com/
```

2. Initialize Git LFS:
```bash
git lfs install
```

## Setting Up the Repository

### 1. Initialize Git Repository

```bash
cd /home/gabriele/franka_interface/Mag_Tec_Models
git init
git lfs install
```

### 2. Configure Git LFS for Large Files

Git LFS is already configured via `.gitattributes`:
- `*.h5` files (HDF5 data files)
- `*.joblib` files (trained models)

### 3. Create GitHub Repository

1. Go to GitHub and create a new repository (e.g., `Mag_Tec_Models`)
2. Do NOT initialize with README, .gitignore, or license (we already have these)

### 4. Add Remote and Push

```bash
# Add remote (replace with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/Mag_Tec_Models.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: Multi-point tactile sensor ML models with documentation"

# Push to GitHub
git push -u origin main
```

### 5. For Large Files (Git LFS)

If files are already committed before Git LFS setup:

```bash
# Migrate existing files to LFS
git lfs migrate import --include="*.h5,*.joblib" --everything

# Force push (if needed)
git push --force origin main
```

## File Sizes

Expected file sizes:
- HDF5 data files: ~50-130 MB each
- Trained models: ~1-10 MB each
- Total repository size: ~2-5 GB (with all data)

Git LFS handles these large files efficiently.

## Verification

After pushing, verify Git LFS is working:

```bash
git lfs ls-files
```

This should show all `.h5` and `.joblib` files tracked by LFS.

## Updating the Repository

When adding new data or models:

```bash
git add .
git commit -m "Add new training data/models"
git push origin main
```

Git LFS will automatically handle large files.

