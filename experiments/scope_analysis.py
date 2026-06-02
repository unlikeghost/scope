import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#FCFCFC",
    "figure.facecolor": "#FFFFFF",
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})


def parse_all_reports_comprehensive(root_dir):
    search_records = []
    eval_records = []

    search_regex = re.compile(r'results/([^/]+)/search_results/scope/(\d+)')
    eval_regex = re.compile(r'results/([^/]+)/search_results/scope/(\d+)')

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith('.json') and file.startswith('search_report'):
                match = search_regex.search(file_path)
                if not match: continue
                dataset, samples = match.group(1), int(match.group(2))

                try:
                    with open(file_path, 'r') as f:
                        report = json.load(f)

                    native_metric = report.get("metric", "balanced_accuracy")

                    for trial in report.get("trials", []):
                        params = trial.get("params", {})
                        compressors = "+".join(sorted(params.get("compressors", [])))
                        metrics = "+".join(sorted(params.get("dissimilarity_metrics", [])))

                        search_records.append({
                            "dataset": dataset,
                            "samples": samples,
                            "rank": trial.get("rank"),
                            "native_metric": native_metric,
                            "score": trial.get("score"),
                            "std_score": trial.get("std_score"),
                            "compressors": compressors,
                            "metrics": metrics,
                            "keep_similar": str(params.get("keep_similar"))
                        })
                except Exception as e:
                    print(f"Error leyendo {file}: {e}")

            elif file.endswith('.json') and file.startswith('evaluation_report'):
                match = eval_regex.search(file_path)
                if not match: continue
                dataset, samples = match.group(1), int(match.group(2))

                try:
                    with open(file_path, 'r') as f:
                        report = json.load(f)

                    cm = report.get("confusion_matrix", [[0, 0], [0, 0]])
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

                    eval_records.append({
                        "dataset": dataset,
                        "samples": samples,
                        "auc_roc": report.get("auc_roc"),
                        "f1_score": report.get("f1"),
                        "balanced_accuracy": report.get("balanced_accuracy"),
                        "mcc": report.get("mcc"),
                        "fp_rate": fp / (tn + fp) if (tn + fp) > 0 else 0,
                        "fn_rate": fn / (fn + tp) if (fn + tp) > 0 else 0
                    })
                except Exception as e:
                    print(f"Error leyendo {file}: {e}")

    return pd.DataFrame(search_records), pd.DataFrame(eval_records)


# ─── GRÁFICA 1: CURVA DE APRENDIZAJE PURA ──────────────────────────────────
def plot_learning_curve(
    df_search,
    output_dir: str,
    dataset_name: str,
    figsize: tuple=(10, 5),
):
    plt.figure(figsize=figsize)
    best_trials = df_search[df_search["rank"] == 1].sort_values("samples")

    base_color = sns.color_palette("flare", 3)[1]

    sns.lineplot(
        data=best_trials, x="samples", y="score",
        marker="o", linewidth=3, markersize=8, color=base_color
    )

    plt.fill_between(
        best_trials["samples"],
        best_trials["score"] - best_trials["std_score"],
        best_trials["score"] + best_trials["std_score"],
        color=base_color, alpha=0.15
    )

    plt.xlim(2, 21)
    plt.xticks(np.arange(3, 21, 1))
    plt.title(
        f"[{dataset_name.capitalize()}] Parameter Search Validation: Best Score vs Sample Size",
        fontsize=12, fontweight='bold', pad=15
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Top Validation Score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_learning_curve.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 2: ESTUDIO DE ABLACIÓN (KEEP_SIMILAR) ─────────────────────────
def plot_keep_similar_impact(
    df_search,
    output_dir: str,
    dataset_name: str,
    figsize: tuple=(10, 5),
):
    plt.figure(figsize=figsize)
    flare_colors = sns.color_palette("flare", 5)
    ks_palette = {"True": flare_colors[3], "False": flare_colors[0]}

    sns.boxplot(
        data=df_search,
        x="samples", y="score", hue="keep_similar",
        palette=ks_palette, linewidth=1.5, width=0.6, fliersize=3
    )

    plt.title(
        f"[{dataset_name.capitalize()}] Ablation Study: Impact of 'keep_similar' (Parameter Search Validation)",
        fontsize=12, fontweight='bold', pad=15
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Validation Score Distribution (All Search Trials)", fontsize=11)
    plt.legend(
        title="keep_similar",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "keep_similar_ablation.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 3: EVALUACIÓN FINAL DE TEST (MULTIMÉTRICA) ─────────────────────
def plot_final_test_metrics(
    df_eval,
    output_dir: str,
    dataset_name: str,
    figsize: tuple = (10, 8)
):
    if df_eval.empty: return

    df_long = df_eval.melt(
        id_vars=["samples"],
        value_vars=["auc_roc", "f1_score", "balanced_accuracy", "mcc"],
        var_name="Metric", value_name="Value"
    )

    g = sns.FacetGrid(
        df_long,
        col="Metric",
        col_wrap=2,
        sharey=False
    )

    g.figure.set_size_inches(figsize)

    g.map(
        sns.lineplot,
        "samples", "Value",
        marker="o", linewidth=2.5,
        color=sns.color_palette("flare", 3)[1]
    )

    g.set_titles(template="Test: {col_name}")
    g.set_axis_labels("Number of Samples", "Test Score / Value")

    for ax in g.axes.flat:
        ax.set_xlim(2, 21)
        ax.set_xticks(np.arange(3, 21, 2))

    plt.suptitle(
        f"Dataset: {dataset_name.capitalize()} (Final Test Evaluation)",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    g.figure.subplots_adjust(top=0.90)

    g.savefig(os.path.join(output_dir, "final_test_evaluation.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 4: COMPARATIVA BÚSQUEDA VS TEST (OVERFITTING GAP) ────────────
def plot_validation_vs_test(
    df_search,
    df_eval,
    output_dir: str,
    dataset_name: str,
    target_metric: str,
    figsize: tuple=(10, 5)
):
    if df_search.empty or df_eval.empty: return

    test_metric_col = "balanced_accuracy" if target_metric == "balanced_accuracy" else target_metric
    if test_metric_col not in df_eval.columns: return

    val_best = df_search[df_search["rank"] == 1][["samples", "score"]].rename(
        columns={"score": "Parameter Search Validation"}
    )
    test_results = df_eval[["samples", test_metric_col]].rename(columns={test_metric_col: "Final Test Evaluation"})

    merged = pd.merge(val_best, test_results, on=["samples"])
    if merged.empty: return

    merged_long = merged.melt(
        id_vars=["samples"], value_vars=["Parameter Search Validation", "Final Test Evaluation"],
        var_name="Phase", value_name="Value"
    )

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=merged_long, x="samples", y="Value", style="Phase", hue="Phase",
        markers=True, linewidth=2.5, palette="flare"
    )

    plt.xlim(2, 21)
    plt.xticks(np.arange(3, 21, 1))
    plt.title(
        f"[{dataset_name.capitalize()}] Generalization Gap: Parameter Search vs Final Test ({target_metric})",
        fontsize=12, fontweight='bold', pad=15
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel(f"Score ({target_metric})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "search_vs_test_gap.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 5: EVOLUCIÓN DE ERRORES (FP VS FN EN TEST) ──────────────
def plot_error_rates_evolution(
    df_eval,
    output_dir: str,
    dataset_name: str,
    figsize: tuple=(9, 5.5)
):
    if df_eval.empty: return

    df_errors = df_eval.melt(
        id_vars=["samples"],
        value_vars=["fp_rate", "fn_rate"],
        var_name="Error_Type", value_name="Rate"
    )

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=df_errors, x="samples", y="Rate", hue="Error_Type", style="Error_Type",
        markers=True, linewidth=2.5, palette="flare"
    )

    plt.xlim(2, 21)
    plt.xticks(np.arange(3, 21, 1))
    plt.title(
        f"[{dataset_name.capitalize()}] Error Analysis: False Positive vs False Negative Rates (Final Test Evaluation)",
        fontsize=12,
        fontweight='bold', pad=15
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Error Rate", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_error_rates.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 6: SENSIBILIDAD Y VARIANZA (SCATTER DE ESTABILIDAD) ─────
def plot_stability_scatter(
    df_search,
    output_dir: str,
    dataset_name: str,
    figsize: tuple=(9, 5.5)
):
    viable_df = df_search[df_search["keep_similar"] == "True"]
    if viable_df.empty: return

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=viable_df, x="score", y="std_score", hue="samples", size="samples",
        palette="flare", alpha=0.7, sizes=(40, 200)
    )

    plt.title(
        f"[{dataset_name.capitalize()}] Optimizer Stability: Validation Score vs Standard Deviation",
        fontsize=12, fontweight='bold',
        pad=15
    )
    plt.xlabel("Parameter Search Validation Score", fontsize=11)
    plt.ylabel("Standard Deviation (std_score)", fontsize=11)
    plt.legend(title="Sample Size", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimizer_stability_scatter.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 7: MAPA DE INTERACCIÓN DIRECCIONAL (COMPRESOR × MÉTRICA) ───
def plot_compressor_metric_heatmap(
    df_search,
    output_dir: str,
    dataset_name: str,
    figsize: tuple = (10, 6.5)
):
    viable_df = df_search[df_search["keep_similar"] == "True"]
    if viable_df.empty: return

    pivot_table = viable_df.pivot_table(
        values="score",
        index="compressors",
        columns="metrics",
        aggfunc="mean"
    )

    dynamic_width = max(figsize[0], pivot_table.shape[1] * 1.3 + 2)
    dynamic_height = max(figsize[1], pivot_table.shape[0] * 0.8 + 2)

    plt.figure(figsize=(dynamic_width, dynamic_height))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="flare",
        linewidths=.5,
        square=False,
        cbar_kws={'label': 'Mean Validation Score', 'shrink': 0.8}
    )

    plt.title(
        f"[{dataset_name.capitalize()}] Synergy Matrix: Compressor vs Metric Interaction (Parameter Search)",
        fontsize=12, fontweight='bold', pad=20
    )

    plt.xlabel("Dissimilarity Metrics", fontsize=11, labelpad=10)
    plt.ylabel("Compressor Pipelines", fontsize=11, labelpad=10)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compressor_metric_synergy_heatmap.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 8: HISTOGRAMA DE CONFIGURACIONES GANADORAS GLOBAL ─────
def plot_global_top_parameters_census(
    df_search,
    output_dir: str,
    figsize: tuple = (10, 6)
):
    best_configs = df_search[df_search["rank"] == 1].copy()
    if best_configs.empty: return

    best_configs["combo"] = best_configs["compressors"] + "\n(" + best_configs["metrics"] + ")"
    counts = best_configs["combo"].value_counts().reset_index()
    counts.columns = ["Configuration", "Count"]

    calculated_height = max(figsize[1], len(counts) * 0.7 + 2)
    plt.figure(figsize=(figsize[0], calculated_height))

    sns.barplot(
        data=counts, x="Count", y="Configuration",
        palette="flare_r", edgecolor=".2", width=0.8
    )

    plt.title(f"Global Parameter Census: Most Frequent Rank-1 Configs (Search Parameters)",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    plt.xlabel("Number of Wins (Across all Datasets & Samples)", fontsize=11)
    plt.ylabel("Configuration (Compressor + Metric)", fontsize=11)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_top_parameters_census.png"), dpi=300)
    plt.close()


# ─── GRÁFICA 9: MATRIZ DE CONFIGURACIONES ÓPTIMAS (DATASET × SAMPLES) ──
def plot_best_param_matrix(
    df_search,
    output_dir: str,
    target_metric: str,
    figsize: tuple=(14, 6)
):
    best_configs = df_search[df_search["rank"] == 1].copy()
    if best_configs.empty: return

    best_configs["label"] = (
        best_configs["compressors"] + "\n" +
        best_configs["metrics"] + "\n" +
        best_configs["score"].map(lambda x: f"({x:.3f})")
    )

    matrix_df = best_configs.pivot(index="dataset", columns="samples", values="label").fillna("N/A")
    score_pivot = best_configs.pivot(index="dataset", columns="samples", values="score").fillna(0)

    calculated_height = max(figsize[1], len(matrix_df.index) * 1.8 + 2)
    plt.figure(figsize=(figsize[0], calculated_height))

    sns.heatmap(
        score_pivot, annot=matrix_df.values, fmt="", cmap="flare",
        linewidths=.8, cbar_kws={'label': f'Search Validation Score ({target_metric})'},
        annot_kws={"fontsize": 9, "fontweight": "medium"}
    )

    plt.title(
        f"Optimal Parameter Landscape: Rank 1 Combinations ({target_metric})",
        fontsize=13, fontweight='bold',
        pad=15
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Datasets", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimal_parameter_landscape_matrix.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    TARGET_METRIC = "balanced_accuracy"

    ROOT_DATA_DIR = "./results"
    BASE_OUTPUT_DIR = os.path.join(ROOT_DATA_DIR, "plots_analisis")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    df_search_all, df_eval_all = parse_all_reports_comprehensive(ROOT_DATA_DIR)

    datasets_search = df_search_all["dataset"].unique() if not df_search_all.empty else []
    datasets_eval = df_eval_all["dataset"].unique() if not df_eval_all.empty else []
    all_datasets = sorted(list(set(datasets_search) | set(datasets_eval)))

    if not all_datasets:
        print("No se encontraron registros validos mapeados en el directorio raiz.")
    else:
        if not df_search_all.empty:
            plot_global_top_parameters_census(df_search_all, BASE_OUTPUT_DIR)
            plot_best_param_matrix(df_search_all, BASE_OUTPUT_DIR, TARGET_METRIC)

        for current_dataset in all_datasets:
            dataset_plot_dir = os.path.join(BASE_OUTPUT_DIR, current_dataset)
            os.makedirs(dataset_plot_dir, exist_ok=True)

            df_search = df_search_all[
                df_search_all["dataset"] == current_dataset] if not df_search_all.empty else pd.DataFrame()
            df_eval = df_eval_all[
                df_eval_all["dataset"] == current_dataset] if not df_eval_all.empty else pd.DataFrame()

            if not df_search.empty:
                plot_learning_curve(df_search, dataset_plot_dir, current_dataset)
                plot_keep_similar_impact(df_search, dataset_plot_dir, current_dataset)
                plot_stability_scatter(df_search, dataset_plot_dir, current_dataset)
                plot_compressor_metric_heatmap(df_search, dataset_plot_dir, current_dataset)

            if not df_eval.empty:
                plot_final_test_metrics(df_eval, dataset_plot_dir, current_dataset)
                plot_error_rates_evolution(df_eval, dataset_plot_dir, current_dataset)

            if not df_search.empty and not df_eval.empty:
                plot_validation_vs_test(df_search, df_eval, dataset_plot_dir, current_dataset, TARGET_METRIC)