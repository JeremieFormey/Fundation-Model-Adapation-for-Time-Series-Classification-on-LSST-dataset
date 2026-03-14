# Fundation-Model-Adapation-for-Time-Series-Classification-on-LSST-dataset
Deep Learning for Time Series : Fundation Model Adaptation for time series classification on lsst dataset

This repository contains a notebook dedicated to **multivariate time series classification** on the **LSST** dataset from the UCR/UEA archive.

The study focused on classifying astronomical light curves observed across **6 photometric bands** and **36 time steps**. The objective was to compare a strong deep learning baseline trained from scratch with an adapted **time series foundation model**.

## Notebook content

The notebook was organized into the following steps:

1. **Exploratory Data Analysis (EDA)**
   - dataset structure inspection
   - class distribution analysis
   - visualization of representative light curves

2. **Preprocessing**
   - label encoding
   - train/validation split
   - per-channel normalization
   - tensor formatting for PyTorch models

3. **Baseline modeling**
   - implementation of a **ResNet** for time series classification
   - supplementary **FCN** architecture
   - training configuration and baseline evaluation

4. **Foundation model adaptation**
   - adaptation of **Chronos** to the LSST classification task
   - integration of **LoRA** for parameter-efficient fine-tuning
   - handling class imbalance with **Focal Loss**

5. **Results analysis**
   - comparison of baseline and adapted model performances
   - discussion of class imbalance effects
   - inspection of difficult classes through light curve plots

## Dataset

The notebook used the **LSST** dataset loaded from `tslearn` through the UCR/UEA interface.

Each sample followed the shape:

- `N`: number of samples
- `T = 36`: number of time steps
- `C = 6`: number of channels

These time series corresponded to astronomical light curves recorded in different photometric bands.

## Models explored

### ResNet baseline
A ResNet architecture for multivariate time series classification was implemented as the main baseline. It relied on convolutional residual blocks and global average pooling for final classification.

### FCN baseline
A Fully Convolutional Network (FCN) was also defined as a secondary comparison point.

### Chronos-LoRA
A pre-trained **Chronos** encoder was adapted to the task using **LoRA**, along with a statistical feature branch and a classification head. Two loss settings were tested:

- **Cross-Entropy Loss**
- **Focal Loss**

## Requirements

Main libraries used in the notebook:

- `numpy`
- `matplotlib`
- `torch`
- `scikit-learn`
- `tslearn`
- `chronos-forecasting`
- `accelerate`
- `bitsandbytes`
- `peft`

## Run the notebook

Clone the repository and open the notebook in Jupyter or Google Colab:

```bash
git clone <repo-url>
cd <repo-name>
