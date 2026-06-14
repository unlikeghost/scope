import numpy as np
from tqdm import tqdm
from itertools import product
from dataclasses import dataclass, field
from typing import Any, Dict, List

from scope import SCoPEPolygon as SCoPE
from scope.utils.eval_metrics import predictions_to_report, Report

from sklearn.model_selection import StratifiedKFold

@dataclass
class FoldResult:
    fold: int
    score:  float
    report: Report


@dataclass
class TrialResult:
    params:       Dict[str, Any]
    mean_score:   float
    std_score:    float
    results: List[FoldResult] = field(default_factory=list)


def _scope_run_(
    fold_idx:   int,
    test_idx:  np.ndarray,
    queries:      list[str],
    supports:     list[dict],
    y_true:       np.ndarray,
    model:        SCoPE,
    metric:       str,
) -> FoldResult:

    queries = np.array(queries)[test_idx].tolist()
    supports = np.array(supports)[test_idx].tolist()
    y_true = np.array(y_true)[test_idx].tolist()

    predictions = model.predict(
        queries=queries,
        kw_samples=supports,
    )

    report = predictions_to_report(
        predictions=predictions,
        y_true=y_true,
    )

    return FoldResult(
        fold=fold_idx,
        score=report[metric],
        report=report
    )


def grid_search(
    queries:      list[str],
    supports:     list[dict],
    y_true:       list[int],
    param_grid:   Dict[str, list],
    metric:       str = "balanced_accuracy",
    n_splits:     int = 5,
    random_seed: int = 42,
) -> list[TrialResult]:
    y_true = np.array(y_true)
    indices = np.arange(len(queries))

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results: list[TrialResult] = []

    print(
        f"Grid search: {metric} over {len(combos)} combinations of {keys}"
    )

    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed
    )
    splits = list(kfold.split(indices, y_true))

    pbar = tqdm(combos, desc="Grid search", unit="trial")

    for combo in pbar:

        params = dict(zip(keys, combo))

        model = SCoPE(
            **params
        )

        fold_result = [
            _scope_run_(
                fold_idx=k,
                test_idx=test_idx,
                queries=queries,
                supports=supports,
                y_true=y_true,
                model=model,
                metric=metric,
            )
            for k, (test_idx, _) in enumerate(splits)
        ]

        scores = [fr.score for fr in fold_result]
        
        trial  = TrialResult(
            params=params,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
        )

        results.append(trial)
        pbar.set_postfix(
            {
                metric: f"{trial.mean_score:.4f} ± {trial.std_score:.4f}",
            }
        )

    results.sort(
        key=lambda r: r.mean_score - r.std_score, reverse=True
    )

    return results