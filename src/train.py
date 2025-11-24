import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_models(random_state=42):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=800,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    return models


def evaluate_models(X, y, variance_threshold=0.0, n_splits=5, random_state=42):
    """
    Leak-safe evaluation:
    preprocessing happens INSIDE each CV fold via Pipeline.

    Accepts y as strings (e.g., 'ASD'/'Control') or numeric.
    Returns a DataFrame of metrics.
    """
    # --- Encode labels if they are strings/object ---
    if y.dtype == object or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = get_models(random_state=random_state)

    results = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(threshold=variance_threshold)),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", clf)
        ])

        aucs, accs, f1s = [], [], []

        for train_idx, test_idx in cv.split(X, y_enc):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            aucs.append(roc_auc_score(y_test, proba))
            accs.append(accuracy_score(y_test, pred))
            f1s.append(f1_score(y_test, pred))

        results.append({
            "model": name,
            "roc_auc_mean": np.mean(aucs),
            "roc_auc_std": np.std(aucs),
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "n_samples": len(y_enc),
            "n_features": X.shape[1]
        })

    return pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)
