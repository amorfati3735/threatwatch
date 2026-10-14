"""
Classical baselines on hand-crafted window statistics.

These establish the "how hard is this really?" reference point before any
deep model is trained.  Window (T, C) -> a fixed feature vector of
per-channel summary statistics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from threatwatch.config import FEATURE_CHANNELS


def _stats(x: np.ndarray) -> np.ndarray:
    """Per-channel summary stats for one window. x: (T, C) -> (5*C,)."""
    feats = [
        x.mean(axis=0),
        x.std(axis=0),
        x.min(axis=0),
        x.max(axis=0),
        np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0),  # IQR
    ]
    return np.concatenate(feats).astype(np.float32)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) -> (N, 5*C)"""
    return np.stack([_stats(w) for w in X])


def feature_names(channels: List[str] = FEATURE_CHANNELS) -> List[str]:
    names = []
    for stat in ["mean", "std", "min", "max", "iqr"]:
        names += [f"{ch}_{stat}" for ch in channels]
    return names


def build_baseline(kind: str = "random_forest", seed: int = 42):
    """
    Return a scikit-learn estimator for window classification.

    ``svm`` and ``naive_bayes`` are included specifically because the base paper
    (Wagh et al., Scientific Reports 2025, doi 10.1038/s41598-025-29146-4) reports
    its results with exactly those two classifiers.  Without them there is nothing
    to compare against — a comparison to SVM/NB that does not actually run SVM/NB
    is not a comparison.
    """
    if kind == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=seed,
            )),
        ])
    if kind in ("nb", "naive_bayes", "naivebayes"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ])
    if kind in ("rf", "random_forest"):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if kind in ("logreg", "logistic"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed,
            )),
        ])
    raise ValueError(
        f"Unknown baseline '{kind}' "
        f"(use 'random_forest', 'logistic', 'svm' or 'naive_bayes')"
    )
