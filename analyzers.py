"""
analyzers.py

Robust, production-minded dataset analyzers that compute per-feature mathematical
signals which will be passed to the LLM (Groq) for interpretation.

Key outputs per feature:
 - feature name
 - detected dtype (numeric/categorical/datetime/text)
 - missing_pct, n_unique, uniqueness_ratio
 - cardinality
 - correlation_with_target (if numeric target/numeric feature -> Pearson; if categorical target -> point-biserial-like via encoding)
 - cramers_v (for categorical vs categorical)
 - mutual_info (sklearn's MI)
 - single_feature_cv_score (AUC for binary, accuracy for multiclass, R2 for regression)
 - perm_importance (computed on a small baseline model trained on top-k univariate features; optional)
 - composite_suspiciousness (0..1) - normalized rank across features (not a label)
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import math
import logging
from dateutil.parser import parse as dateparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------- utilities ----------------
def detect_series_type(s: pd.Series, max_sample: int = 100) -> str:
    """Return one of: 'numeric', 'categorical', 'datetime', 'text'."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        # although numeric may be categorical encoded as numbers; use heuristics
        nunique = s.nunique(dropna=True)
        if nunique <= 20 and nunique / max(1, len(s)) < 0.05:
            return "categorical"
        return "numeric"
    # try parse a sample for dates
    sample = s.dropna().astype(str).head(max_sample)
    parsed = 0
    for v in sample:
        try:
            dateparse(v)
            parsed += 1
        except Exception:
            pass
    if len(sample) > 0 and parsed / len(sample) > 0.5:
        return "datetime"
    # fallback: categorical vs text
    nunique = s.nunique(dropna=True)
    if nunique < 0.05 * max(1, len(s)) or nunique <= 100:
        return "categorical"
    return "text"


def safe_label_encode(series: pd.Series) -> np.ndarray:
    """Label encode categorical or text series into integers."""
    s = series.fillna("__NA__").astype(str)
    le = LabelEncoder()
    try:
        return le.fit_transform(s)
    except Exception:
        # fallback: mapping
        uniq = {v: i for i, v in enumerate(pd.unique(s))}
        return s.map(uniq).to_numpy()


def safe_numeric(series: pd.Series) -> np.ndarray:
    """Convert to numeric, coercing; replace NaNs with column median (for modeling)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return np.zeros(len(s))
    med = s.median()
    return s.fillna(med).to_numpy()


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V with bias correction."""
    try:
        x = x.fillna("__NA__").astype(str)
        y = y.fillna("__NA__").astype(str)
        conf = pd.crosstab(x, y)
        chi2 = None
        from scipy.stats import chi2_contingency
        chi2 = chi2_contingency(conf, correction=False)[0]
        n = conf.sum().sum()
        phi2 = chi2 / n
        r, k = conf.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        denom = min((kcorr-1), (rcorr-1))
        if denom <= 0:
            return 0.0
        return float((phi2corr / denom) ** 0.5)
    except Exception:
        return 0.0


# -------------- per-feature math checks ----------------
def feature_basic_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Compute basic stats: missing%, n_unique, uniqueness_ratio, dtype_label."""
    s = df[col]
    n = len(s)
    missing_pct = float(s.isna().sum() / max(1, n))
    n_unique = int(s.nunique(dropna=False))
    uniqueness_ratio = n_unique / max(1, n)
    dtype = detect_series_type(s)
    cardinality = n_unique
    return {
        "feature": col,
        "dtype": dtype,
        "missing_pct": round(missing_pct, 4),
        "n_unique": n_unique,
        "uniqueness_ratio": round(uniqueness_ratio, 4),
        "cardinality": cardinality,
    }


def _univariate_correlation(df: pd.DataFrame, feature: str, target: str, target_dtype: str) -> Tuple[float, float]:
    """Return (correlation, mutual_info). Correlation uses numeric encoding when needed."""
    try:
        fseries = df[feature]
        tseries = df[target]
        # mutual info
        if target_dtype == "classification":
            # mutual_info_classif needs 1d arrays
            try:
                mi = mutual_info_classif(
                    safe_label_encode(fseries).reshape(-1, 1),
                    safe_label_encode(tseries),
                    discrete_features=True,
                    random_state=0
                )
                mi_val = float(mi[0]) if hasattr(mi, "__len__") else float(mi)
            except Exception:
                mi_val = 0.0
        else:
            try:
                mi = mutual_info_regression(
                    safe_numeric(fseries).reshape(-1, 1),
                    pd.to_numeric(tseries, errors="coerce").fillna(0).to_numpy(),
                    random_state=0
                )
                mi_val = float(mi[0])
            except Exception:
                mi_val = 0.0

        # correlation: numeric vs numeric -> Pearson; else fallback to encoding and Pearson
        try:
            fx = None
            tx = None
            if pd.api.types.is_numeric_dtype(fseries) and target_dtype == "regression":
                fx = safe_numeric(fseries)
                tx = pd.to_numeric(tseries, errors="coerce").fillna(0).to_numpy()
            else:
                fx = safe_label_encode(fseries)
                tx = safe_label_encode(tseries)
            if len(fx) < 3:
                corr = 0.0
            else:
                corr = float(np.corrcoef(fx, tx)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
        except Exception:
            corr = 0.0

        return round(corr, 4), round(mi_val, 6)
    except Exception as e:
        logger.debug("univariate_correlation error for %s: %s", feature, e)
        return 0.0, 0.0


def _single_feature_cv_score(df: pd.DataFrame, feature: str, target: str, target_dtype: str, cv: int = 3) -> float:
    """Return a cross-validated single-feature predictive score:
       - classification (binary): mean AUC (if possible) else accuracy
       - multiclass: accuracy
       - regression: R2
    """
    try:
        X_raw = df[feature]
        y_raw = df[target]
        n = len(df)
        if target_dtype == "classification":
            y = safe_label_encode(y_raw)
            # need at least 2 classes
            if len(np.unique(y)) < 2:
                return 0.0
            # use logistic regression for binary; for >2 use RandomForest
            if len(np.unique(y)) == 2:
                X = safe_numeric(X_raw).reshape(-1, 1)
                clf = LogisticRegression(max_iter=300)
                # stratified CV
                cv_obj = StratifiedKFold(n_splits=min(cv, max(2, len(np.unique(y)))), shuffle=True, random_state=0)
                try:
                    scores = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv_obj, n_jobs=1)
                    return float(np.nanmean(scores))
                except Exception:
                    scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv_obj, n_jobs=1)
                    return float(np.nanmean(scores))
            else:
                # multiclass
                X = safe_label_encode(X_raw).reshape(-1, 1)
                clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
                cv_obj = StratifiedKFold(n_splits=min(cv, max(2, len(np.unique(y)))), shuffle=True, random_state=0)
                scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv_obj, n_jobs=1)
                return float(np.nanmean(scores))
        else:
            # regression
            X = safe_numeric(X_raw).reshape(-1, 1)
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()
            if np.all(y == y[0]):
                return 0.0
            clf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
            cv_obj = KFold(n_splits=min(cv, 5), shuffle=True, random_state=0)
            scores = cross_val_score(clf, X, y, scoring="r2", cv=cv_obj, n_jobs=1)
            return float(np.nanmean(scores))
    except Exception as e:
        logger.debug("single_feature_cv_score error for %s: %s", feature, e)
        return 0.0


def compute_permutation_importance(df: pd.DataFrame, target: str, feature_list: List[str],
                                   target_dtype: str, random_state: int = 0, n_repeats: int = 5) -> Dict[str, float]:
    """Train a light baseline model on feature_list and compute permutation importance.

    Returns a mapping {feature: importance_mean}.
    """
    try:
        if not feature_list:
            return {}
        X = pd.DataFrame()
        for c in feature_list:
            X[c] = safe_numeric(df[c]) if detect_series_type(df[c]) == "numeric" else safe_label_encode(df[c])
        y_raw = df[target]
        if target_dtype == "classification":
            y = safe_label_encode(y_raw)
            model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
        else:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()
            model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=1)

        # small train (fit on full X)
        model.fit(X, y)
        res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=None, n_jobs=1)
        importance_means = {feature_list[i]: float(res.importances_mean[i]) for i in range(len(feature_list))}
        return importance_means
    except Exception as e:
        logger.debug("perm importance failed: %s", e)
        return {f: 0.0 for f in feature_list}


# analyzers.py
# Same robust math analyzers from v2 with only one change: permutation importance on full feature set.

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import logging
from dateutil.parser import parse as dateparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Utilities and detect_series_type, safe_label_encode, safe_numeric, cramers_v etc same as v2
# ... (copy all utility functions and _univariate_correlation, _single_feature_cv_score) ...

# Main orchestrator
def analyze_dataset(
    df: pd.DataFrame,
    target: str,
    sample_frac: float = 1.0,
    n_jobs: int = 1,
    cv: int = 3,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Compute math-based metrics for all features, including full permutation importance.
    """
    if target not in df.columns:
        raise ValueError(f"target {target} not found in dataframe")

    nrows = len(df)
    if sample_frac < 1.0:
        df_proc = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    else:
        df_proc = df.reset_index(drop=True)

    # determine target type
    tseries = df_proc[target]
    if pd.api.types.is_numeric_dtype(tseries) and tseries.nunique(dropna=True) > max(10, 0.01 * nrows):
        target_dtype = "regression"
    elif tseries.nunique(dropna=True) <= 10 and not pd.api.types.is_float_dtype(tseries):
        target_dtype = "classification"
    else:
        target_dtype = "classification"

    features = [c for c in df_proc.columns if c != target]
    basic_stats_list = [feature_basic_stats(df_proc, c) for c in features]

    # Compute univariate stats in parallel
    def _compute_for_col(col: str) -> Dict[str, Any]:
        basic = next((b for b in basic_stats_list if b["feature"] == col), None)
        corr, mi = _univariate_correlation(df_proc, col, target, target_dtype)
        sf_score = _single_feature_cv_score(df_proc, col, target, target_dtype, cv=cv)
        return {**(basic or {}), "correlation": corr, "mutual_info": mi, "single_feature_score": round(float(sf_score), 6)}

    parallel = Parallel(n_jobs=n_jobs, prefer="threads")
    results = parallel(delayed(_compute_for_col)(col) for col in features)

    # Permutation importance on full feature set
    try:
        perm_importances = compute_permutation_importance(df_proc, target, features, target_dtype, random_state=random_state)
    except Exception as e:
        logger.debug("perm importance step failed: %s", e)
        perm_importances = {f: 0.0 for f in features}

    for r in results:
        r["perm_importance"] = round(float(perm_importances.get(r["feature"], 0.0)), 6)

    # Composite suspiciousness
    def _normalize(arr: List[float]) -> List[float]:
        arr = np.array(arr, dtype=float)
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        if math.isclose(mx, mn):
            return [0.0 for _ in arr]
        return list((arr - mn) / (mx - mn))

    abs_corrs = [abs(r.get("correlation", 0.0)) for r in results]
    mis = [r.get("mutual_info", 0.0) for r in results]
    sfs = [r.get("single_feature_score", 0.0) for r in results]
    perms = [r.get("perm_importance", 0.0) for r in results]
    unqs = [r.get("uniqueness_ratio", 0.0) for r in results]

    norm_corr = _normalize(abs_corrs)
    norm_mi = _normalize(mis)
    norm_sfs = _normalize(sfs)
    norm_perms = _normalize(perms)
    norm_unq = _normalize(unqs)

    w = {"sfs": 0.4, "perm": 0.3, "corr": 0.15, "mi": 0.1, "uniq": 0.05}
    for i, r in enumerate(results):
        composite = (w["sfs"] * norm_sfs[i] + w["perm"] * norm_perms[i]
                     + w["corr"] * norm_corr[i] + w["mi"] * norm_mi[i] + w["uniq"] * norm_unq[i])
        r["composite_score"] = round(float(composite), 6)

    results_sorted = sorted(results, key=lambda x: x["composite_score"], reverse=True)

    metadata = {"n_rows": nrows, "n_features": len(features), "sample_frac": sample_frac, "target_dtype": target_dtype, "notes": []}
    if nrows < 200:
        metadata["notes"].append("Small dataset (n < 200); statistics may be unstable.")

    return {"metadata": metadata, "results": results_sorted}