#!/bin/bash

# Ensure pip is up to date
python -m pip install --upgrade pip

# Uninstall existing packages if they exist
packages=(
    "circuitsvis"
    "datasets"
    "einops"
    "nnsight"
    "pandas"
    "plotly"
    "tqdm"
    "zstandard"
    "wandb"
    "umap-learn"
    "llvmlite"
    "pytest"
    "tokenizers"
    "transformers"
)

echo "Uninstalling existing packages..."
for package in "${packages[@]}"; do
    pip uninstall -y "$package" 2>/dev/null || true
done

# Install packages with specific version requirements
echo "Installing required packages..."
pip install \
    "circuitsvis>=1.43.2" \
    "datasets>=2.18.0" \
    "einops>=0.7.0" \
    "nnsight>=0.3.0,<0.4.0" \
    "pandas>=2.2.1" \
    "plotly>=5.18.0" \
    "tqdm>=4.66.1" \
    "zstandard>=0.22.0" \
    "wandb>=0.12.0" \
    "umap-learn>=0.5.6" \
    "llvmlite>=0.40.0" \
    "tokenizers==0.19.1" \
    "transformers==4.43.0"

# Install dev dependencies
echo "Installing development dependencies..."
pip install "pytest>=8.3.4"

echo "Installation complete!"

# Verify installations
echo "Verifying installations..."
pip list | grep -E "circuitsvis|datasets|einops|nnsight|pandas|plotly|tqdm|zstandard|wandb|umap-learn|llvmlite|pytest|tokenizers|transformers"