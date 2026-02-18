# PathogenKG Dataset and GNN Model Training

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/johndef64/PathogenKG)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.0+-3C2179?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


PathogenKG is a knowledge graph dataset focused on pathogen-related biological entities and their relationships. This project provides tools for constructing multi-organism-specific knowledge graphs from pathogen data, merging multiple pathogen graphs, and training Graph Neural Network (GNN) models for link prediction tasks. The dataset and models can be used for drug discovery, understanding pathogen-host interactions, and exploring biological relationships in infectious diseases.


## Preparing the Environment

```bash
conda create -n gnn python=3.10 -y  \
  && conda activate gnn \
  && pip install -r requirements.txt
``` 

if torch-sparse installation fails, try installing it separately:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision torchaudio
pip install --no-cache-dir --only-binary=:all: \
  pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
```

## Knowledge Graph Construction

### Full Pipeline

**/downloadres** 
- Get source datasets from STRING and DrugBank

**`generate_all_pathogenkg.sh`**
- Executes `build_pathogenkg.py` on all available pathogen datasets to generate organism-specific knowledge graphs
- Produces PathogenKG graphs for each pathogen organism

**`merge_all_pathoges.py`**
- Merges multiple PathogenKG datasets together (pathogen-only, without human KG)
- Creates integrated multi-organism knowledge graphs for training

### Pre-built Datasets
**`get_pathogenkg.py`**
- Downloads preprocessed PathogenKG datasets from [HuggingFace](https://huggingface.co/datasets/johndef64/PathogenKG)
- Skip the construction step and use existing datasets directly

### Reference Human KG (DRKG)
**DRKG is used as:**
- Structural reference for knowledge graph design
- Baseline for training and evaluation comparison
- It can be merged with PathogenKG datasets for discovering cross-species interactions

## Model Training and Evaluation

### Standard Training
**`train_and_eval.py`**
- Trains and evaluates models on link prediction tasks using PathogenKG datasets
- Supports various embedding models (CompGCN, RGCN, etc.) and evaluation metrics
- Can also be used with DRKG for baseline comparison

```bash
# Train on PathogenKG merged dataset (default)
python train_and_eval.py 

# Train on DRKG for comparison
python train_and_eval.py --task Compound-Gene --tsv dataset/drkg/drkg_reduced.tsv

# Use a different model
python train_and_eval.py --model rgcn --task Compound-Gene --tsv dataset/pathogenkg/PathogenKG_merged.tsv

# Change number of runs and epochs
python train_and_eval.py --runs 5 --epochs 200

# Training with pretraining
python train_and_eval.py --pretrain_epochs 50

# Training with focal loss parameters
python train_and_eval.py --alpha 0.25 --gamma 3.0 --alpha_adv 2.0
```

### Hyperparameter Optimization
**`parameter_tuning.py`** 
- Performs hyperparameter tuning using Weights & Biases (wandb)
- Leverages `train_and_eval.py` for systematic parameter exploration
- Logs experiments and tracks optimal configurations