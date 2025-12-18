#!/bin/bash
# Compile LaTeX document

cd "$(dirname "$0")"

if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  pdflatex not found. Installing..."
    echo "Ubuntu/Debian: sudo apt-get install texlive-latex-base texlive-latex-extra"
    exit 1
fi

echo "Compiling LaTeX document..."
pdflatex -interaction=nonstopmode model_documentation.tex
pdflatex -interaction=nonstopmode model_documentation.tex  # Run twice for references

if [ -f "model_documentation.pdf" ]; then
    echo "✅ PDF generated: model_documentation.pdf"
else
    echo "❌ PDF generation failed"
    exit 1
fi

