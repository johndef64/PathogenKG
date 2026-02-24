conda env create -f environment.yml
conda activate graph-ml-cu128


#!/usr/bin/env bash
set -e

ENV_NAME=graph-ml-cu128

echo "Installing PyTorch..."
conda run -n $ENV_NAME pip install \
  torch==2.5.1 torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "Installing PyG..."
conda run -n $ENV_NAME pip install --no-cache-dir --only-binary=:all: \
  pyg_lib torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv termcolor torcheval \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

echo "Fix numpy/scipy/sklearn..."
conda run -n $ENV_NAME pip install --force-reinstall \
  "numpy<2.0" \
  "scipy<1.14" \
  "scikit-learn<1.5"

echo "Done."
