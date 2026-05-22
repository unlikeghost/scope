import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List

from scope import SCoPE # noqa
from scope.prediction import Prediction
from scope.utils.eval_metrics import make_evaluation_report, Report

try:
    from sklearn.svm import SVC
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from scipy.stats import loguniform, randint
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, ParameterSampler
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    raise ImportError(
        "Please install the sklearn package. "
        "If you are using uv: uv sync --group experiments"
    )

META_MODELS = {
    "lr": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),)
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            probability=True,
            random_state=42,
            class_weight="balanced"
        )),
    ]),
    "rf": Pipeline([
        ("clf", RandomForestClassifier(
            random_state=42,
        ))
    ]),
}

META_PARAM_DISTRIBUTIONS = {
    "lr": {
        "clf__solver": ["saga", "lbfgs", "liblinear", "sag"],
        "clf__penalty": ["l1", "l2", "elasticnet"],
        "clf__l1_ratio": loguniform(0.01, 0.9),
        "clf__tol": loguniform(1e-4, 1e-2),
        "clf__C": loguniform(1e-3, 100),
    },
    "svm": {
        "clf__kernel": ["rbf", "poly", "sigmoid"],
        "clf__gamma": ["scale", "auto"],
        "clf__C": loguniform(1e-3, 100),
    },
    "rf": {
        "clf__class_weight": ["balanced", "balanced_subsample"],
        "clf__n_estimators": randint(50, 200),
        "clf__max_depth": randint(2, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 12),
        "clf__max_features": ["sqrt", "log2", None],
    },
}

@dataclass
class FoldResult:
    fold:   int
    score:  float
    report: Report


@dataclass
class TrialResult:
    params:       Dict[str, Any]
    mean_score:   float
    std_score:    float
    fold_results: List[FoldResult] = field(default_factory=list)


def scores_to_features(predictions: list[Prediction]) -> np.ndarray:
    features = []
    for p in predictions:
        classes = sorted(p.scores.keys())
        # score = [p.scores[c] for c in classes]

        geo_features = []
        for c in classes:
            geo = p.geometry_metrics[c]
            geo_features.extend([
                geo.normalized_distance,
                geo.fraction_outside,
                geo.iou,
            ])

        features.append(geo_features)

    return np.array(features)


def _run_fold(
    fold_idx:  int,
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    x:         np.ndarray,
    y_true:    np.ndarray,
    meta:      Any,
    metric:    str,
) -> FoldResult:
    try:
        meta.fit(x[train_idx], y_true[train_idx])

        y_pred  = meta.predict(x[test_idx])
        y_score = meta.predict_proba(x[test_idx])

        report = make_evaluation_report(
            y_true  = y_true[test_idx],
            y_pred  = y_pred,
            y_score = y_score,
        )
        return FoldResult(
            fold=fold_idx,
            score=report[metric],
            report=report
        )
    except ValueError as error:
        return FoldResult(fold=fold_idx, score=0.0, report=None)

def random_search_meta_kfold(
    scope_model:  SCoPE,
    queries:      list[str],
    supports:     list[dict],
    y_true:       list[int],
    meta_model:   Any,
    param_distributions: Dict[str, Any],
    n_iter:       int = 20,
    metric:       str = "balanced_accuracy",
    n_splits:     int = 5,
    random_state: int = 42,
) -> list[TrialResult]:

    y_true = np.array(y_true)

    predictions = scope_model.predict(
        queries=queries, kw_samples=supports
    )

    X = scores_to_features(predictions)

    kfold  = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    splits = list(kfold.split(X, y_true))

    sampler = ParameterSampler(
        param_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )
    combos = list(sampler)

    print(f"Random search: {metric} | {n_iter} trials | {n_splits}-fold CV")
    results: list[TrialResult] = []

    pbar = tqdm(combos, desc="Random search", unit="trial")

    for params in pbar:

        meta = clone(meta_model)
        meta.set_params(**params) # noqa

        fold_results = [
            _run_fold(
                fold_idx  = k,
                train_idx = train_idx,
                test_idx  = test_idx,
                x         = X,
                y_true    = y_true,
                meta      = clone(meta),
                metric    = metric,
            )
            for k, (train_idx, test_idx) in enumerate(splits)
        ]

        scores = [fr.score for fr in fold_results]
        trial  = TrialResult(
            params       = params,
            mean_score   = float(np.mean(scores)),
            std_score    = float(np.std(scores)),
            fold_results = fold_results,
        )
        results.append(trial)
        # tqdm.write(f"{params}  →  {metric}={trial.mean_score:.4f} ± {trial.std_score:.4f}")
        pbar.set_postfix(
            {
                metric: f"{trial.mean_score:.4f} ± {trial.std_score:.4f}",
            }
        )

    results.sort(
        key=lambda r: r.mean_score - r.std_score, reverse=True
    )

    return results