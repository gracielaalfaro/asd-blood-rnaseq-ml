# Blood RNA-seq Biomarker ML for ASD (Replication + Extension)

This project reproduces and extends a published blood RNA-seq machine learning pipeline for identifying autism spectrum disorder (ASD) biomarkers. Using publicly available GEO blood transcriptomic datasets, we build predictive models that classify ASD vs. typically developing controls based on gene-expression patterns and interpret the most informative immune/inflammation-related signatures.

**Reference study:** Voinsky et al., *International Journal of Molecular Sciences* (2023).  
Replication target: blood-based RNA signature + ML diagnostic modeling.

---

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
- Support Vector Machine (linear/RBF)
- XGBoost

Evaluation:
- Stratified K-Fold Cross Validation
- ROC-AUC, accuracy, precision/recall, F1
- Careful leakage prevention (feature selection inside CV)

### 4. Interpretation
- Feature importance
- SHAP values (tree models)
- Pathway enrichment on top-ranked genes (immune/inflammation focus)

---

## Repo Structure
```text
asd-blood-rnaseq-ml/
├─ data/
│  ├─ raw/                 # empty; downloaded via scripts
│  ├─ processed/           # example matrices only (optional)
├─ notebooks/
│  ├─ 01_download_qc.ipynb
│  ├─ 02_preprocess_normalize.ipynb
│  ├─ 03_feature_selection.ipynb
│  ├─ 04_model_training.ipynb
│  ├─ 05_interpretation_pathways.ipynb
├─ src/
│  ├─ download.py
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ train.py
│  ├─ explain.py
├─ results/
│  ├─ figures/
│  ├─ tables/
├─ requirements.txt
└─ README.md
