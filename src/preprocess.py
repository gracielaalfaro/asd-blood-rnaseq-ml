import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

def load_processed_data(features_path: str, labels_path: str):
    """Load features and labels from CSVs."""
    X = pd.read_csv(features_path)
    y_df = pd.read_csv(labels_path)

    label_col = [c for c in y_df.columns if c.lower() != "sample_id"][0]
    y = y_df[label_col].astype(str)

    return X, y

def preprocess_matrix(X: pd.DataFrame,
                      y: pd.Series,
                      variance_threshold: float = 0.0,
                      scale: bool = True):
    """
    Impute missing values, remove constant/low-variance genes,
    optionally scale features.
    Returns (X_proc, y_enc, label_encoder, artifacts).
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    vt = VarianceThreshold(threshold=variance_threshold)
    X_vt = vt.fit_transform(X_imp)

    if scale:
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_vt)
    else:
        scaler = None
        X_scaled = X_vt

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    artifacts = {
        "imputer": imputer,
        "variance_filter": vt,
        "scaler": scaler,
        "kept_feature_mask": vt.get_support()
    }

    return X_scaled, y_enc, le, artifacts
