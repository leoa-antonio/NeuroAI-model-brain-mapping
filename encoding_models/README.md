# Neural Encoding Models & RSA Toolkit for Model–Brain Correspondence

**Author:** Leo Antonio  
**Focus:** Computational Neuroimaging • Model–Brain Alignment • Representational Geometry

---

## Overview

This repository contains two complementary NeuroAI analysis pipelines designed to study correspondences between human brain activity and artificial neural network representations.

1. **Encoding Models** — predict voxel-wise fMRI responses from deep network features.
2. **Representational Similarity Analysis (RSA)** — compare representational geometry between brain ROIs and model layers.

Both pipelines operate on publicly available fMRI datasets (primarily **Algonauts 2023**) and use pretrained convolutional neural networks (e.g., **ResNet50**) to extract hierarchical visual features.

The goal is to evaluate how well modern computer vision models reflect, approximate, or diverge from the representational structure of the human visual system.

This repository serves as a portfolio project in neuroAI, beginning with encoding models and extending into RSA as the next key step in model–brain alignment.

---

# 1. Encoding Model

An end-to-end encoding model pipeline mapping deep neural network features to human fMRI responses using the **Algonauts 2023 Challenge** dataset.

The goal of this repo is to provide a clean, modular example of:

- Loading real fMRI data from Algonauts 2023
- Linking each trial to its corresponding stimulus image
- Extracting **ResNet50** features (ImageNet-pretrained, avgpool layer)
- Fitting a **Ridge regression** encoding model (features → voxel responses)
- Evaluating performance with voxel-wise **held-out R²**
- Visualizing the **distribution of voxel R²** (histogram)

# 2. Representational Similarity Analysis (RSA Tools)

Tools for computing representational similarity between model features and fMRI voxel responses using the **Algonauts 2023 Challenge** dataset.

The goal of this module is to provide a clean, modular example of:

* Computing Representational Dissimilarity Matrices (RDMs) from:
  * model-layer activations
  * ROI-specific brain voxel patterns
* Comparing model-layer RDMs to brain RDMs
* Evaluating alignment using upper-triangle similarity metrics
  (e.g., Spearman correlation)

RSA offers a complementary view to encoding models by examining whether the *geometry* of neural representations within a network mirrors geometry in human cortex.

**Status:**  
🚧 *In Progress* — notebook and core RSA utilities are currently under development.

---

### Planned Core Components

- `compute_rdm.py`
  - Generate RDMs from matrix-like inputs using correlation, cosine, or Euclidean distance

- `model_features.py`
  - Extract activations from multiple network layers for RSA comparisons

- `brain_data.py`
  - Load ROI-level voxel response matrices for brain RDM computation

- `rsa_compare.py`
  - Correlate model RDMs with brain RDMs (upper-triangle comparisons)

- `visualization.py`
  - Plot:
    - RDM heatmaps
    - layer-wise model–brain correlation curves

---

# 📁 Repository Structure
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

**Voxel-wise R² summary (subj01-009):**  
mean R²: -0.0877<br>
median R²: -0.1198<br>
% R² > 0: 26.3%

The histogram shows a large mass of negative R² (noise voxels) and a clear right tail of voxels with positive R², indicating that ResNet50 features capture visual information represented in a subset of cortical voxels.

--- 

# Acknowledgements
* Algonauts 2023 Challenge: for providing the fMRI and stimulus data
* PyTorch / torchvision: for pretrained ResNet50 and image transforms
* scikit-learn: for Ridge regression and utility tools

Please cite the Algonauts 2023 dataset and relevant methods if you build on this work for publications or reports.



