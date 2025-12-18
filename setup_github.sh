#!/bin/bash
# Setup script for GitHub repository with Git LFS

set -e

echo "=================================================================================="
echo "GitHub Repository Setup Script"
echo "=================================================================================="

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS is not installed."
    echo "Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  macOS: brew install git-lfs"
    echo ""
    read -p "Press Enter to continue anyway (Git LFS will be needed for large files)..."
else
    echo "✅ Git LFS is installed"
    git lfs install
fi

# Check if git repository exists
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Check if .gitattributes exists
if [ ! -f ".gitattributes" ]; then
    echo "⚠️  .gitattributes not found. Creating..."
    cat > .gitattributes << 'EOF'
# Git LFS for large HDF5 files
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text

# Large model files
*.joblib filter=lfs diff=lfs merge=lfs -text
EOF
fi

# Track LFS files
if command -v git-lfs &> /dev/null; then
    echo "Configuring Git LFS for large files..."
    git lfs track "*.h5"
    git lfs track "*.hdf5"
    git lfs track "*.joblib"
fi

echo ""
echo "=================================================================================="
echo "Repository Status"
echo "=================================================================================="
git status --short | head -20

echo ""
echo "=================================================================================="
echo "Next Steps:"
echo "=================================================================================="
echo "1. Review the changes: git status"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Add ML models and documentation'"
echo ""
echo "4. Create a GitHub repository (if not exists):"
echo "   - Go to https://github.com/new"
echo "   - Create repository: Mag_Tec_Models"
echo "   - Do NOT initialize with README"
echo ""
echo "5. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/Mag_Tec_Models.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Note: Large files (*.h5, *.joblib) will be handled by Git LFS"
echo "=================================================================================="

