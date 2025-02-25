# README

## Exploring the Viability of a Machine Learning-Based Multimodel for Quantitative Precipitation Forecast Post-Processing"* submitted to *Water Resources Research

This repository supports the work *"Exploring the Viability of a Machine Learning-Based Multimodel for Quantitative Precipitation Forecast Post-Processing"* submitted to *Water Resources Research*. The aim of this study is to assess supervised machine learning as a framework for creating a multimodel that effectively blends multiple Numerical Weather Prediction (NWP) models to enhance precipitation forecast post-processing.

### Methodology
We evaluate the performance of different machine learning architectures:
- Multi-Layer Perceptron (MLP)
- U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597))
- Residual U-Net ([Zhang et al., 2018](https://doi.org/10.1109/TMI.2018.2832618))

As a case study, this work focuses on daily cumulative Quantitative Precipitation Forecast (QPF) over Piedmont and Aosta Valley, two regions in Italy.

### Dataset
Observational data used during training, summary statistics, event classifications, and the Area of Interest domain mask utilized in this study are available at:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14923826.svg)](https://doi.org/10.5281/zenodo.14923826)

### Repository Content
This repository provides well-documented PyTorch classes for the models used in the study, along with a class for weight initialization tailored to different torch layers.

#### Files
- **`mlp.py`** : Class for Multi-Layer Perceptron (MLP)
- **`unet.py`** : Classes for U-Net and Residual U-Net
- **`initialization.py`** : Class for layer-wise weight initialization

### Usage
To use the provided models, ensure you have installed the required dependencies and simply import the relevant classes in your PyTorch-based project. Each file contains documentation to facilitate integration and customization.

For questions or contributions, feel free to open an issue or submit a pull request!
