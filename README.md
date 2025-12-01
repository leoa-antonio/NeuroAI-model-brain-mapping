Neural Encoding Models & RSA Toolkit for Model–Brain Correspondence
Author: Leo Antonio
Affiliation: NYU Perception & Brain Dynamics Lab, NYU Langone
Focus: Computational Neuroimaging • Model–Brain Alignment • Representational Geometry

Overview
This repository contains two complementary NeuroAI analysis pipelines designed to study correspondences between human brain activity and artificial neural network representations.

Encoding Models — predict voxel-wise fMRI responses from deep network features.
Representational Similarity Analysis (RSA) — compare representational geometry between brain ROIs and model layers.
Both pipelines operate on publicly available fMRI datasets (primarily Algonauts 2021) and use pretrained convolutional neural networks (e.g., ResNet50 and Vision Transformers) to extract hierarchical visual features.

The goal is to evaluate how well modern computer vision models reflect, approximate, or diverge from the representational structure of the human visual system.

1. Encoding Models
Objective:
Predict voxel-level fMRI activation patterns from deep visual features using linear regression.

Pipeline:

Load stimuli and ROI-specific beta patterns
Extract features from pretrained ResNet/ViT
Fit regularized linear models (RidgeCV)
Evaluate voxel-wise prediction accuracy (R² score)
Outputs:

ResNet/ViT feature matrices (n_stimuli × n_features)
Voxel-response matrices (n_stimuli × n_voxels)
RidgeCV models and predictions
Performance histograms and summary statistics
Scientific Motivation:
Encoding models provide direct tests of whether linear combinations of deep network features can explain measured neural responses. They are foundational in computational neuroscience, vision science, and NeuroAI model evaluation.

Demo Notebook: (fill in later)

markdown Copy code

2. Representational Similarity Analysis (RSA Tools)
Objective:
Quantify similarity between representational geometry in the brain and model activations across network layers.

Pipeline:

Compute Representational Dissimilarity Matrices (RDMs)
Compare model-layer RDMs to brain-ROI RDMs
Evaluate layer-wise correspondence using correlation metrics
Core Scripts:

compute_rdm.py — distance-based RDMs (correlation, cosine, Euclidean)
rsa_compare.py — upper-triangle similarity metrics
visualization.py — RDM heatmaps and layer-correlation curves
Outputs:

brain RDMs from ROI beta patterns
model RDMs from deep feature vectors
correlation coefficients per layer/ROI
Scientific Motivation:
RSA reveals how high-level geometry of representations evolves across network depth, and whether networks recapitulate known cortical hierarchies (e.g., early layers → EVC, deeper layers → IT).

Demo Notebook: (fill in later)

📁 Repository Structure
NeuroAI/
├── encoding_models/
│   ├── data/
│   ├── data_loading.py
│   ├── feature_extraction.py
│   ├── model_fitting.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── demo_algonauts.py
│   └── README.md
│
├── rsa_tools/
│   ├── compute_rdm.py
│   ├── model_features.py
│   ├── brain_data.py
│   ├── rsa_compare.py
│   ├── visualization.py
│   ├── demo_algonauts_rsa.ipynb
│   └── README.md
|
├── docs/
│   ├── figures/
│   │   ├── example_r2_histogram.png
│   │   └── roi_layer_similarity.png
│   └── reference_papers.md
|
├── environment.yml 
└── README.md
Dataset
Algonauts 2021
Chosen for:

real fMRI beta maps per stimulus
ROI-resolved responses (EVC, LOC, FFA, etc.)
Standard benchmark for model-brain correspondence work
Data Sources:

image stimuli
voxel response matrices per ROI
subject-averaged responses
⚙️ Installation
To reproduce the analyses in this repository, create a dedicated Conda environment and install the required dependencies.

Option 1 — Using environment.yml (recommended)
conda env create -f environment.yml conda activate neuroai

Option 2 — Manual setup
conda create -n neuroai python=3.10 conda activate neuroai pip install numpy scipy scikit-learn matplotlib pillow pip install torch torchvision pip install nilearn

Verify installation
python -c "import torch, sklearn, nilearn; print('Environment ready.')"

Notes:

Python 3.10 is recommended.
All scripts and notebooks assume execution inside the neuroai environment.
GPU is optional for this project; CPU is sufficient.
