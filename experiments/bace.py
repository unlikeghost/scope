import os.path
import random
import warnings
import numpy as np
from dataclasses import replace

from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

from scope import SCoPE
from scope.utils.eval_metrics import make_evaluation_report, predictions_to_report

from utils.settings import GetSettings
from utils.combinations import all_subsets_str
from utils.dataset import load_dataset, build_dataset_by_sample
from utils.plots import plot_confusion_matrix, plot_correct_predictions_by_class, plot_auc_roc

from utils.search.report import save_search_results, serialize_param_space
from utils.search.scope import grid_search
from utils.search.meta import (
    META_MODELS,
    META_PARAM_DISTRIBUTIONS,
    scores_to_features,
    random_search_meta_kfold
)

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_dataset(
    settings: GetSettings,
    search_size: float
):
    search, test = load_dataset(
        dataset_path=settings.dataset.file_path,
        smiles_column=settings.dataset.smiles_column,
        label_columns=settings.dataset.label_column,
        min_length=settings.dataset.min_length,
        search_size=search_size,
        random_seed=random_seed,
    )

    search_queries, search_y, search_supports = build_dataset_by_sample(
        dataframe=search,
        sample_sizes=sample_sizes,
        random_seed=random_seed
    )

    test_queries, test_y, test_supports = build_dataset_by_sample(
        dataframe=test,
        sample_sizes=sample_sizes,
        random_seed=random_seed
    )

    search_queries = search_queries[sample_sizes[0]]
    search_y = search_y[sample_sizes[0]]
    search_supports = search_supports[sample_sizes[0]]

    test_queries = test_queries[sample_sizes[0]]
    test_y = test_y[sample_sizes[0]]
    test_supports = test_supports[sample_sizes[0]]

    return (search_queries, search_y, search_supports), (test_queries, test_y, test_supports)


def print_results(report) -> None:
    print(f"\nResultados finales en test:")
    print(f"  auc_roc              : {report['auc_roc']}")
    print(f"  ap_score              : {report['ap_score']}")
    print(f"  f1              : {report['f1']}")
    print(f"  mcc              : {report['mcc']}")
    print(f"  balanced_accuracy              : {report['balanced_accuracy']}")

    cm = report['confusion_matrix']
    print(f"  {'confusion_matrix'}:")
    print("        pred 0  pred 1")
    print(f"true 0    {cm[0][0]:>4}    {cm[0][1]:>4}")
    print(f"true 1    {cm[1][0]:>4}    {cm[1][1]:>4}")

    cm = report['confusion_matrix_normalized']
    print(f"  {'confusion_matrix_normalized'}:")
    print("        pred 0  pred 1")
    print(f"true 0    {cm[0][0]:>4}    {cm[0][1]:>4}")
    print(f"true 1    {cm[1][0]:>4}    {cm[1][1]:>4}")


def search_scope(
    settings: GetSettings,
    search_dataset,
    test_dataset,
) -> SCoPE:

    search_queries, search_y, search_supports = search_dataset
    test_queries, test_y, test_supports = test_dataset

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(search_y),
        y=search_y,
    )
    class_weights = dict(enumerate(weights))

    metric = settings.optimization.scope.metric
    report_path = os.path.join(
        settings.optimization_path,
        "scope"
    )
    plot_path = os.path.join(
        settings.plots_path,
        "scope"
    )

    all_compressors = [
        selection.split(",")
        for selection in all_subsets_str(
            ["bz2", "gzip", "zlib"]
        )
    ]

    param_grid = {
        "compressors": all_compressors,
        "join_string": [" "],
        "keep_similar": [True, False],
        "class_weights": [None, class_weights],
        "dissimilarity_metric_names": [
            ["ncd", "cdm"],
            ["ncd", "clm"],
            ["cdm", "clm"],
        ],
    }

    results = grid_search(
        queries=search_queries,
        supports=search_supports,
        y_true=search_y,
        param_grid=param_grid,
        metric=metric,
    )

    save_search_results(
        results=results,
        path=report_path,
        metric=metric,
        param_space=param_grid,
    )

    best_params = results[0].params
    print(f"\nMejor configuración: {best_params}")
    print(f"Score (CV):          {results[0].score:.4f}")

    model = SCoPE(**best_params)

    predictions = model.predict(
        kw_samples=test_supports,
        queries=test_queries,
    )

    report = predictions_to_report(
        predictions=predictions,
        y_true=test_y,
        save_path=report_path,
    )
    plot_confusion_matrix(
        report["confusion_matrix"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix.png")
    )
    plot_confusion_matrix(
        report["confusion_matrix_normalized"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix_normalized.png"),
        show=True,
        normalize=True,
    )
    plot_correct_predictions_by_class(
        predictions=predictions,
        y_true=test_y,
        n_per_class=sample_sizes[0],
        save_dir=plot_path,
    )

    print_results(report)
    plot_auc_roc(
        report,
        save_path=os.path.join(plot_path, "auc_roc.png"),
        show=False,
    )

    return model


def search_meta(
    meta_name: str,
    scope_model: SCoPE,
    settings: GetSettings,
    search_dataset,
    test_dataset,
):
    meta_model = META_MODELS[meta_name]
    param_distributions = META_PARAM_DISTRIBUTIONS[meta_name]

    search_queries, search_y, search_supports = search_dataset
    test_queries, test_y, test_supports = test_dataset

    n_splits = settings.optimization.meta.n_splits
    metric = settings.optimization.meta.metric
    n_iter = settings.optimization.meta.n_iter

    report_path = os.path.join(
        settings.optimization_path,
        "meta",
        meta_name,
    )
    plot_path = os.path.join(
        settings.plots_path,
        "meta",
        meta_name,
    )

    results = random_search_meta_kfold(
        scope_model=scope_model,
        queries=search_queries,
        supports=search_supports,
        y_true=search_y,
        meta_model=meta_model,
        param_distributions=param_distributions,
        metric=metric,
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_seed,
    )

    best_params = results[0].params
    print(f"\nMejor configuración: {best_params}")
    print(f"Score (CV):          {results[0].mean_score:.4f} ± {results[0].std_score:.4f}")

    predictions_search = scope_model.predict(
        queries=search_queries,
        kw_samples=search_supports,
    )

    X_train = scores_to_features(predictions_search)

    final_meta = clone(meta_model)
    final_meta.set_params(**best_params) # noqa
    final_meta.fit(X_train, np.array(search_y)) # noqa

    predictions_test = scope_model.predict(
        queries=test_queries,
        kw_samples=test_supports,
    )
    X_test = scores_to_features(predictions_test)
    y_pred = final_meta.predict(X_test) # noqa
    y_score = final_meta.predict_proba(X_test) # noqa

    predictions_test = [
        replace(prediction, predicted_class=int(pred))
        for prediction, pred in zip(predictions_test, y_pred)
    ]

    report = make_evaluation_report(
        y_true=np.array(test_y),
        y_pred=y_pred,
        y_score=y_score,
        save_path=report_path,
    )

    save_search_results(
        results=results,
        path=report_path,
        metric=metric,
        param_space=serialize_param_space(param_distributions),
    )

    print_results(report)

    plot_confusion_matrix(
        report["confusion_matrix"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix.png")
    )
    plot_confusion_matrix(
        report["confusion_matrix_normalized"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix_normalized.png"),
        show=True,
        normalize=True,
    )
    plot_correct_predictions_by_class(
        predictions = predictions_test,
        y_true      = test_y,
        n_per_class = sample_sizes[0],
        save_dir    = plot_path,
    )
    plot_auc_roc(
        report,
        save_path=os.path.join(plot_path, "auc_roc.png"),
        show=False,
    )


if __name__ == "__main__":
    settings = GetSettings("settings/bace.toml")

    random_seed = settings.experiment.random_seed
    sample_sizes = settings.experiment.sample_sizes

    set_seed(random_seed)

    search, test = get_dataset(
        settings,
        search_size=settings.optimization.scope.test_size,
    )

    scope_model = search_scope(
        settings=settings,
        search_dataset=search,
        test_dataset=test,
    )

    search, test = get_dataset(
        settings,
        search_size=settings.optimization.meta.test_size,
    )

    for meta_name in META_MODELS:
        print("="*60)
        print(f"{meta_name}")
        search_meta(
            meta_name=meta_name,
            scope_model=scope_model,
            settings=settings,
            search_dataset=search,
            test_dataset=test,
        )
        print("=" * 60)
        print("\n")