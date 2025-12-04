# Neural Encoding Models & RSA Toolkit for Model–Brain Correspondence

**Author:** Leo Antonio  
**Focus:** Computational Neuroimaging • Model–Brain Alignment • Representational Geometry

---

## Overview

This repository contains two complementary NeuroAI analysis pipelines designed to study correspondences between human brain activity and artificial neural network representations.

1. **Encoding Models** — predict voxel-wise fMRI responses from deep network features.
2. **Representational Similarity Analysis (RSA)** — compare representational geometry between brain ROIs and model layers.

Both pipelines operate on publicly available fMRI datasets (primarily **Algonauts 2021**) and use pretrained convolutional neural networks (e.g., **ResNet50** and **Vision Transformers**) to extract hierarchical visual features.

The goal is to evaluate how well modern computer vision models reflect, approximate, or diverge from the representational structure of the human visual system.

---

# 1. Encoding Models

**Objective:**  
Predict voxel-level fMRI activation patterns from deep visual features using linear regression.

**Pipeline:**
- Load stimuli and ROI-specific beta patterns
- Extract features from pretrained ResNet/ViT
- Fit regularized linear models (**RidgeCV**)
- Evaluate voxel-wise prediction accuracy (R² score)

**Outputs:**
- ResNet/ViT feature matrices (n_stimuli × n_features)
- Voxel-response matrices (n_stimuli × n_voxels)
- RidgeCV models and predictions
- Performance histograms and summary statistics

**Scientific Motivation:**  
Encoding models provide direct tests of whether linear combinations of deep network features can explain measured neural responses. They are foundational in computational neuroscience, vision science, and NeuroAI model evaluation.

**Demo Notebook:**
(In Progress)

markdown
Copy code

---

# 2. Representational Similarity Analysis (RSA Tools)

**Objective:**  
Quantify similarity between representational geometry in the brain and model activations across network layers.

**Pipeline:**
- Compute Representational Dissimilarity Matrices (RDMs)
- Compare model-layer RDMs to brain-ROI RDMs
- Evaluate layer-wise correspondence using correlation metrics

**Core Scripts:**
- `compute_rdm.py` — distance-based RDMs (correlation, cosine, Euclidean)
- `rsa_compare.py` — upper-triangle similarity metrics
- `visualization.py` — RDM heatmaps and layer-correlation curves

**Outputs:**
- brain RDMs from ROI beta patterns
- model RDMs from deep feature vectors
- correlation coefficients per layer/ROI

**Scientific Motivation:**  
RSA reveals how high-level geometry of representations evolves across network depth, and whether networks recapitulate known cortical hierarchies (e.g., early layers → EVC, deeper layers → IT).

**Demo Notebook:**
(In Progress)

---

## 📁 Repository Structure
```
NeuroAI-model-brain-mapping/
│
├── encoding_models/
│   ├── data_loading.py
│   ├── feature_extraction.py
│   ├── model_fitting.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── README.md
│   └── demo_algonauts.ipynb
│
├── rsa_tools/
│   ├── compute_rdm.py
│   ├── model_features.py
│   ├── brain_data.py
│   ├── rsa_compare.py
│   ├── visualization.py
│   ├── README.md
│   └── demo_algonauts_rsa.ipynb
│
├── docs/
│   ├── reference_papers.md
│   ├── notes.md
│   └── figures/
│
├── tests/                    
│   ├── test_compute_rdm.py
│   └── test_encoding_shapes.py
│
├── scripts/                  
│   ├── run_encoding.sh
│   └── run_rsa.sh
│
├── environment.yml
├── README.md
├── LICENSE
├── CONTRIBUTING.md            
└── .gitignore
```
---

# 🧠 Dataset

## Algonauts Project 2023 – Model-to-Brain Mapping Challenge

This project uses the **Algonauts 2023 Challenge Dataset**, a publicly available benchmark designed to study correspondence between deep neural network representations and human brain activity.

The dataset provides a large-scale, standardized resource for encoding-model research that focuses on **ventral visual cortex** responses to natural images.

---

## What the Dataset Contains

Each subject folder in `encoding_models/data/train_data/` includes:

- **Stimulus Image IDs**
  - An array mapping each fMRI sample to an image index
- **ROI-specific fMRI voxel responses**
  - Multiple `.npy` files, one per cortical region (e.g., `VC`, `EVC`, `IT`, etc.)
  - Each file is shaped:
    ```
    (#stimuli, #voxels_in_ROI)
    ```
- **Train/Test split**
  - `train_data/` contains data used to fit encoding models
  - `test_data/` contains a held-out set reserved for model evaluation or leaderboard submissions

Image stimuli associated with these IDs are also included in the dataset and can be used to extract deep feature representations from pretrained models.

---

## Why Algonauts 2023?

Chosen because it offers:

- **Real fMRI signals** evoked by natural image viewing
- **Voxel-wise response matrices** aligned to consistent image IDs
- **ROI-based parcellation**
  - Enables analysis of representational differences across cortical areas
- **Subject-specific data**
  - Allows per-subject encoding or cross-subject generalization
- **Standard benchmark in model-brain alignment research**
  - Used by vision-science labs, ML groups, and neuro-AI initiatives


## ⚙️ Installation

To reproduce the analyses in this repository, create a dedicated Conda environment and install the required dependencies.

### Option 1 — Using environment.yml (recommended)

conda env create -f environment.yml
conda activate neuroai

### Option 2 — Manual setup

conda create -n neuroai python=3.10
conda activate neuroai
pip install numpy scipy scikit-learn matplotlib pillow
pip install torch torchvision
pip install nilearn

## Verify installation

python -c "import torch, sklearn, nilearn; print('Environment ready.')"

Notes:
- Python 3.10 is recommended.



