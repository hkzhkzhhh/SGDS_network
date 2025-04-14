# Dual-Stream Collaborative Road Segmentation Network

This repository contains the open-source code, experiments, and result notebooks for our paper on "Dual-Stream Collaborative Road Segmentation Network". Our model leverages a dual-stream architecture that combines local feature extraction (using multi-scale convolutions and depthwise separable convolutions) and global semantic modeling (using a Swin Transformer and Graph Attention Network) to enhance the accuracy and topology continuity of road extraction from high-resolution remote sensing images.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
    - [Experiment Scripts](#experiment-scripts)
    - [Model Training and Testing](#model-training-and-testing)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements a suite of models for remote sensing road extraction, including our novel dual-stream collaborative network as well as several popular architectures for comparison. The code is designed to work with two datasets:

- **Cheng et al. Road Detection Dataset (IEEE TGRS 2017):**
Contains 672 satellite images at 0.5 m resolution with binary masks and centerline vectors. Preprocessing includes bilinear interpolation to a unified resolution (256×256), pixel normalization, and morphological closing to repair annotation breaks.
- **DeepGlobe Road Extraction Dataset (CVPR 2018):**
Contains 6,226 training samples (1024×1024, 50 cm resolution RGB images) with corresponding binary road masks. This dataset focuses on the extraction of main road networks with particular attention to topology continuity in complex urban and rural scenes.

Our experiments cover model performance on both datasets, with evaluations on pixel-level metrics (IoU, Dice, Accuracy, Precision, Recall, F1, OA, Kappa) and topology-related metrics, along with ablation studies on key module contributions.

---

## Repository Structure

```
├── Experiment
│   ├── Ablation_experiment.py         # Script for ablation studies
│   └── Grad-CAM_Visualization.py      # Script to generate Grad-CAM visualization plots
│
├── Model
│   ├── Attention-Unet.py              # Implementation of Attention U-Net
│   ├── Dense_Unet.py                  # Implementation of Dense U-Net
│   ├── Dual-stream_collaborative_road_segmentation_network.py  # Our proposed dual-stream network
│   ├── MobileNetV2.py                 # Implementation of MobileNetV2 U-Net variant
│   ├── Residual+Transformer'Unet.py   # Implementation of Residual + Transformer U-Net model
│   ├── ResUet.py                      # Implementation of ResU-Net
│   ├── SegNet.py                      # Implementation of SegNet
│   ├── Uet++.py                      # Implementation of U-Net++ model
│   ├── Unet.py                        # Standard U-Net implementation
│   └── VNet.py                        # Implementation of V-Net
│
├── Result
│   ├── Ablation_experiment.ipynb      # Notebook showing ablation study results
│   ├── FinalExperiment(ChengPaper'sDataSet).ipynb  # Final experiment results on Cheng et al. dataset
│   └── FinalExperiment(DeepGlobeDataSet).ipynb     # Final experiment results on DeepGlobe dataset
│
├── README.md                          # This file
└── requirements.txt                   # List of required Python packages
```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/hkzhkzhhh/SGDS_network.git
cd SGDS_network
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

*Note: The code is built using TensorFlow 2.9.0, so ensure that your TensorFlow version meets this requirement.*

---

## Data

- **Cheng et al. Dataset:**
Download and extract the dataset from the corresponding source (refer to the original publication). Preprocess the images to 256×256 resolution with necessary normalization and morphological operations as described in the paper.https://www.kaggle.com/datasets/ipythonx/tgrs-road?resource=download
- **DeepGlobe Road Extraction Dataset:**
Download from the http://deepglobe.org/challenge.html and arrange the dataset following the structure expected by the experiments.

Place the downloaded datasets in appropriate directories and update the corresponding file paths in the scripts or notebooks if necessary.

---

## Usage

### Experiment Scripts

- **Ablation Experiment:**
To run the ablation study, execute the `Ablation_experiment.py` script.

```bash
python Experiment/Ablation_experiment.py
```

- **Grad-CAM Visualization:**
To generate Grad-CAM visualizations that show the evolution of feature focus across training epochs, run:

```bash
python Experiment/Grad-CAM_Visualization.py
```


### Model Training and Testing

Each model implementation is available in the `Model/` folder. You can train and test the models individually. For example, to train the proposed Dual-stream Collaborative Road Segmentation Network, run:

```bash
python Model/Dual-stream_collaborative_road_segmentation_network.py
```

Configuration parameters such as input size, batch size, training epochs, and learning rate are defined within each script. Adjust them according to your dataset and hardware specifications.

---

## Results

The `Result/` folder contains Jupyter notebooks that demonstrate the experimental outcomes:

- **Ablation_experiment.ipynb:**
Detailed results of the ablation study, analyzing the contributions of different modules (Swin Transformer, GNN, depthwise separable convolution, etc.).
- **FinalExperiment(ChengPaper'sDataSet).ipynb:**
Full experimental evaluation of the models on the Cheng et al. remote sensing dataset.
- **FinalExperiment(DeepGlobeDataSet).ipynb:**
Evaluation results on the DeepGlobe road extraction dataset, highlighting the model's performance and generalization ability.

Open these notebooks in Jupyter Notebook or JupyterLab to explore the visualizations and performance comparisons.

