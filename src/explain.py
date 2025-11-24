import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def fit_rf_and_rank_genes(X: pd.DataFrame,
                          y_enc: np.ndarray,
                          variance_threshold: float = 0.0,
                          random_state: int = 42,
                          top_k: int = 50):
    """
    Fit a RandomForest on full data (for interpretability only),
    return top gene importances.
    """
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_filter", VarianceThreshold(threshold=variance_threshold)),
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestClassifier(
            n_estimators=1200,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    pipe.fit(X, y_enc)

    kept_mask = pipe.named_steps["variance_filter"].get_support()
    kept_genes = X.columns[kept_mask]

    importances = pipe.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({
        "gene": kept_genes,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return imp_df.head(top_k), pipe
