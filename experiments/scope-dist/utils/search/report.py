import os
import json
from typing import Dict
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

def serialize_param_space(param_space: dict) -> dict:
    result = {}
    for key, val in param_space.items():
        if isinstance(val, list):
            result[key] = [
                str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                for v in val
            ]
        elif hasattr(val, "a") and hasattr(val, "b"):  # loguniform, randint
            result[key] = {
                "type": type(val).__name__,
                "low":  val.a,
                "high": val.b,
            }
        else:
            result[key] = str(val)
    return result

def save_search_results(
    results:     list,
    path:        str | Path,
    metric:      str = "balanced_accuracy",
    param_space: Dict[str, list] | None = None,  # ← agregado
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metric":     metric,
        "n_trials":   len(results),
        "param_space": param_space or {},        # ← guardado
        "best": {
            "score":  results[0].mean_score if hasattr(results[0], "mean_score") else results[0].score,
            "std_score": results[0].std_score if hasattr(results[0], "std_score") else None,
            "params": results[0].params,
        },
        "trials": [
            {
                "rank": i + 1,
                "score": trial.mean_score if hasattr(trial, "mean_score") else trial.score,
                "std_score": trial.std_score if hasattr(trial, "std_score") else None,
                "params": trial.params,
                "reports": [
                    fr.report if isinstance(fr.report, dict) else asdict(fr.report)
                    for fr in trial.fold_results
                    if fr.report is not None
                ] if hasattr(trial, "fold_results") else [
                    r.report if isinstance(r.report, dict) else asdict(r.report)
                    for r in trial.results
                    if r is not None
                ],
            }
            for i, trial in enumerate(results)
        ],
    }

    os.makedirs(path, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(path, f"search_report_{stamp}.json")

    with open(filepath, "w") as fh:
        json.dump(data, fh, indent=4, default=str)

    print(f"Saved {len(results)} trials → {path}")

