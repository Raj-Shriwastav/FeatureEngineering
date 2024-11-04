
# **Breast Cancer Dataset - Anomaly Detection Assignment**

## Overview
This repository contains the implementation of multiple anomaly detection methods applied to the Breast Cancer dataset. The methods explored in this assignment include Z-Score, Mahalanobis Distance, Local Outlier Factor (LOF), Isolation Forest, and One-Class SVM. The project demonstrates data preprocessing, anomaly detection, and result evaluation using performance metrics and visualizations.

## Setup and Imports
The following Python libraries are utilized in this project:
- `pandas` and `numpy` for data manipulation
- `scikit-learn` for data preprocessing and anomaly detection algorithms
- `matplotlib` and `seaborn` for data visualization

## Dataset
The dataset used is the **Breast Cancer Dataset** from Scikit-learn. It includes several features related to breast cancer diagnostics, making it suitable for exploring anomaly detection in healthcare data.

- **Features**: The dataset contains key features like mean radius, mean texture, mean perimeter, and others that are used for anomaly detection.
- **Target Variable**: The dataset includes a target variable to distinguish between normal and anomalous observations.

## Data Preprocessing
Feature values are standardized using `StandardScaler` from `scikit-learn`, ensuring they have a mean of 0 and a standard deviation of 1, making the dataset ready for distance-based anomaly detection methods.

## Anomaly Detection Methods
The following methods are implemented for anomaly detection in the dataset:

### a) Z-Score Based Anomaly Detection
- Outliers are flagged if their z-scores exceed a predefined threshold.

### b) Mahalanobis Distance-Based Anomaly Detection
- Anomalies are detected using the `EllipticEnvelope` model with a specified contamination level.

### c) Local Outlier Factor (LOF)
- `LocalOutlierFactor` identifies anomalies based on the local density of data points, with customizable parameters for neighbors and contamination level.

### d) Isolation Forest
- Implements `IsolationForest` for anomaly detection by isolating data points that deviate from the norm.

### e) One-Class SVM
- Uses `OneClassSVM` with a radial basis function (RBF) kernel to identify anomalies based on boundary separation.

## Evaluation and Comparison
Each method's performance is assessed using the following metrics:
- **Classification Report**: Provides metrics such as precision, recall, and F1-score for each anomaly detection method.
- **Confusion Matrix**: Summarizes true and false positives and negatives for a more comprehensive evaluation.

The comparison across methods helps reveal the consistency of results, with some methods offering more conservative or aggressive anomaly detection than others.

## Visualization
Dimensionality reduction techniques like PCA are used to create visualizations that highlight the anomalies identified by each detection method. These plots help illustrate how each method categorizes data points within the feature space.

### Visualization Setup
- **PCA**: Reduces features to two principal components for visualization purposes.
- **Seaborn**: Generates scatter plots to compare normal vs. anomalous points across each detection method.

## Results
Scatter plots for each method provide a visual representation of the differences in anomaly detection, showing how each method classifies data points as normal or anomalous in the reduced feature space.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-anomaly-detection.git
   ```
2. Navigate to the project directory and install dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script to reproduce the results.

## Dependencies
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
