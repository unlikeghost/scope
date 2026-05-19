import numpy as np
from tqdm import tqdm
from itertools import product
from dataclasses import dataclass, field
from typing import Any, Dict, List

from scope import SCoPE
from scope.utils.eval_metrics import predictions_to_report, Report


@dataclass
class ScopeResult:
    score:  float
    report: Report


@dataclass
class TrialResult:
    params:       Dict[str, Any]
    score:   float
    results: List[ScopeResult] = field(default_factory=list)


def _scope_run_(
    queries:      list[str],
    supports:     list[dict],
    y_true:       np.ndarray,
    model:        SCoPE,
    metric:       str,
) -> ScopeResult:

    predictions = model.predict(
        queries=queries,
        kw_samples=supports,
    )

    report = predictions_to_report(
        predictions=predictions,
        y_true=y_true,
    )

    return ScopeResult(
        score=report[metric],
        report=report
    )


def grid_search(
    queries:      list[str],
    supports:     list[dict],
    y_true:       list[int],
    param_grid:   Dict[str, list],
    metric:       str = "balanced_accuracy",
) -> list[TrialResult]:
    y_true = np.array(y_true)

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results: list[TrialResult] = []

    print(
        f"Grid search: {metric} over {len(combos)} combinations of {keys}"
    )

    pbar = tqdm(combos, desc="Grid search", unit="trial")

    for combo in pbar:

        params = dict(zip(keys, combo))

        model = SCoPE(
            **params
        )

        run_results = _scope_run_(
            queries=queries,
            supports=supports,
            y_true=y_true,
            model=model,
            metric=metric,
        )
        
        trial  = TrialResult(
            params=params,
            score=run_results.score,
        )

        results.append(trial)
        pbar.set_postfix(
            {
                metric: f"{trial.score:.4f}",
                "compressor": f"{params['compressors']}"
            }
        )

    results.sort(key=lambda r: r.score, reverse=True)

    return results