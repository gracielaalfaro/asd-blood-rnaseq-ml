# Blood RNA-seq Biomarker ML for ASD (Replication + Extension)

This project reproduces and extends a published blood RNA-seq machine learning pipeline for identifying autism spectrum disorder (ASD) biomarkers. Using publicly available GEO blood transcriptomic datasets, we build predictive models that classify ASD vs. typically developing controls based on gene-expression patterns and interpret the most informative immune/inflammation-related signatures.

**Reference study:** Voinsky et al., *International Journal of Molecular Sciences* (2023).  
Replication target: blood-based RNA signature + ML diagnostic modeling.

---
## 📘 Overview
Preprocess the DESeq2 differential expression results, align diagnostic labels, engineer features using log-transformation and variance filtering, and evaluate multiple ML models:
- Logistic Regression
- Random Forest
- XGBoost
We then identify the top biomarker genes contributing to classification, generating reproducible tables and visualizations.
This repository contains all reproducible code, figures, and results for the machine learning portion of the project.

## Project Goals
1. **Reproduce** published performance using processed GEO blood RNA-seq expression matrices.
2. **Compare models** (Random Forest, SVM, XGBoost) under consistent evaluation.
3. **Identify candidate biomarkers** using feature selection and interpretability methods.
4. **Extend** the study with alternative normalization, feature selection strategies, and/or additional ML/DL models.

---

## Data
- Source: **Public GEO blood RNA-seq datasets**
- Current processed cohort:  
  - `n = 21` samples  
  - `~60,623` Ensembl gene features  
  - Labels: ASD vs Control  

> **Note on data sharing:**  
> Raw sequencing files are not redistributed in this repo.  
> Scripts are provided to download and process data directly from GEO.  
> Any included CSVs are for demonstration only.

---

## Methods Overview

### 1. Preprocessing
- Expression matrix cleaning
- Log / variance stabilization (as appropriate)
- Filtering low-expression genes
- Optional batch-effect correction if metadata allows

### 2. Feature Selection
- Differential expression (DEG-based filtering)
- Variance filtering
- Recursive feature elimination (RFECV)
- Model-based importance ranking (RF/XGB)

### 3. Modeling
Models implemented:
- Random Forest
- Logistical Regression
- XGBoost

Evaluation:
- Stratified K-Fold Cross Validation
- ROC-AUC, accuracy, precision/recall, F1
- Careful leakage prevention (feature selection inside CV)

### 4. Interpretation
- Feature importance
- SHAP values (tree models)
- Pathway enrichment on top-ranked genes (immune/inflammation focus)
- 
- ### Example Data
The repository includes a small synthetic dataset (`example_features.csv`, `example_labels.csv`)
generated to match the dimensions and structure of the real RNA-seq project. 
This data contains no biological information and is safe for public use.


## 🔬 Results (Example Synthetic Data)

This repository includes a synthetic mock dataset to demonstrate the full end-to-end pipeline structure.  
These results are *not biological* and simply confirm that the ML workflow, preprocessing pipeline, and evaluation functions operate correctly.

### 📊 Model Performance (5-fold Stratified CV)

| Model         | ROC-AUC | Accuracy | F1 Score |
|---------------|---------|----------|----------|
| XGBoost       | 0.60    | 1.00     | 1.00     |
| RandomForest  | 0.55    | 1.00     | 1.00     |
| Log.R         | 0.50    | 0.80     | 0.80     |

These scores are expected on synthetic mock data, where no biological signal exists.  
They demonstrate that the pipeline:

- loads processed RNA-seq matrices  
- applies feature selection & scaling  
- trains RF, SVM, and XGBoost classifiers  
- evaluates via leak-proof cross-validation  
- saves reproducible outputs to `results/tables/`

### 📁 Output Files

- `results/tables/model_comparison.csv`  
- `results/tables/top_genes_example.csv` (feature importance)


---
Author: Graciela Alfaro 2025



