# Gujarati Handwritten Character Recognition

This repository provides a complete pipeline for handwritten Gujarati character recognition, including data preprocessing, pixel feature extraction, and neural network model training using both custom CNN and transfer learning (ResNet-50) approaches. The workflow supports any number of classes with relevant pre-labeled data.

---

## Overview

The project can be used in two main ways:
- **Classical ML:** Convert preprocessed images to CSV of pixel values for traditional ML algorithms.
- **Deep Learning:** Train/evaluate advanced Convolutional Neural Network (CNN) architectures directly from image folders.

---

## Requirements

- Python 3.x
- numpy
- pandas
- pillow
- opencv-python
- scikit-learn
- matplotlib
- tensorflow (2.x+)

Install with:
pip install numpy pandas pillow opencv-python scikit-learn matplotlib tensorflow

---

## Sample Results

- **CNN model:** Achieved strong validation accuracy (typically 85-95% based on data quality/amount).
- **ResNet-50 transfer learning:** Higher accuracy expected with more character categories or difficult samples.
- Detailed statistics available via confusion matrix and classification_report.
