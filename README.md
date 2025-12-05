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
│   ├── algonauts_data_loading.py
│   ├── feature_extraction.py
│   ├── model_fitting.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── demo_algonauts.ipynb
│
├── rsa_tools/
│   ├── compute_rdm.py
│   ├── model_features.py
│   ├── brain_data.py
│   ├── rsa_compare.py
│   ├── visualization.py
│   └── demo_algonauts_rsa.ipynb
│
├── docs/
│   ├── reference_papers.md
│   ├── notes.md
│   └── figures/
|
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
  1. Visit the official Algonauts 2023 site (search for “Algonauts 2023 Challenge”).
  2. Follow their instructions to request and download the training data.
  3. Place the contents so that you have a structure like:

```
encoding_models/
│
├── data/
│   ├── train_data/
|       ├── subj01-009/
|            ├── subj01/
|                ├── training_split/
|                    ├── training_images/
|                        ├── train-0001_nsd-00013.png
|                        ├── train-0002_nsd-XXXXX.png
|                        └── ...
|                    ├── training_fmri/
|                        ├── lh_training_fmri.py
|                        └── rh_training_fmri.py  
│       ├── subj02-002/
|       ├── subj03-006/
|       └── ...
│            
└── .gitignore
```
The dataset provides a large-scale, standardized resource for encoding-model research that focuses on **ventral visual cortex** responses to natural images.

---

# ⚙️ Setup & Installation

This project provides a `environment.yml` file so that the full environment can be recreated reproducibly via Conda (recommended).

### 1. Create the neuroai environment

```bash
conda env create -f environment.yml
conda activate neuroai
```
### 2. Run demo as described below
--- 

# 📊 Running the Demo

  1. Ensure the Algonauts training data is in data/train_data/ as described above
  2. Start Jupyterlab
  3. Open demo_algonauts.ipynb
  4. Run all cells
The notebook will:
  1. Detect available training subjects
  2. Load fMRI data (lh + rh, concatenated across voxels)
  3. Reconstruct image paths and align them with fMRI trials
  4. Extract ResNet50 avgpool features for all training images
  5. Features are cached to data/features/<subject>_resnet50_features.npy
  6. On subsequent runs, features are loaded from disk instead of recomputed
  7. Fit a Ridge(alpha=100) encoding model from features → fMRI
  8. Compute voxel-wise R² on held-out test samples
  9. Print summary statistics and plot a histogram of R² across voxels

## Example output

For subject subj01-009 (whole brain, ResNet50 avgpool, Ridge alpha=100):

Voxel-wise R² summary:
  mean R²        : -0.0877
  median R²      : -0.1198
  % R² > 0       : 26.3%

The histogram shows a large mass of negative R² (noise voxels) and a clear right tail of voxels with positive R², indicating that ResNet50 features capture visual information represented in a subset of cortical voxels.

--- 

# Acknowledgements
* Algonauts 2023 Challenge: for providing the fMRI and stimulus data
* PyTorch / torchvision: for pretrained ResNet50 and image transforms
* scikit-learn: for Ridge regression and utility tools

Please cite the Algonauts 2023 dataset and relevant methods if you build on this work for publications or reports.

