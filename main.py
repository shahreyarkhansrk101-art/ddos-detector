"""
End-to-end pipeline: load data -> train candidate models -> evaluate ->
save metrics, figures, and models.

Run: python main.py
"""
from pathlib import Path

from src.evaluate import evaluate_all, evaluate_model, save_metrics
from src.preprocessing import load_data
from src.train import CANDIDATE_MODELS, make_pipeline, save_model, train_test_data
from src.visualize import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_label_distribution,
    plot_model_comparison,
    plot_packet_rate_by_label,
    plot_roc_curve,
    plot_unique_src_ips,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "network_flows.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Generating EDA figures...")
    plot_label_distribution(df, FIGURES_DIR / "label_distribution.png")
    plot_packet_rate_by_label(df, FIGURES_DIR / "packet_rate_by_label.png")
    plot_unique_src_ips(df, FIGURES_DIR / "unique_src_ips.png")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_data(DATA_PATH)

    print("Training candidate models...")
    fitted_models = {}
    for name, model in CANDIDATE_MODELS.items():
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        save_model(pipeline, MODELS_DIR / f"{name}.joblib")

    print("Evaluating models...")
    results_df = evaluate_all(fitted_models, X_test, y_test)
    print(results_df.to_string(index=False))

    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_model_name]
    y_pred = best_pipeline.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, best_model_name, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curve(best_pipeline, X_test, y_test, best_model_name, FIGURES_DIR / "roc_curve.png")
    plot_feature_importance(best_pipeline, FIGURES_DIR / "feature_importance.png")

    detailed_metrics = {
        name: evaluate_model(pipeline, X_test, y_test) for name, pipeline in fitted_models.items()
    }
    detailed_metrics["best_model"] = best_model_name
    save_metrics(detailed_metrics, RESULTS_DIR / "metrics.json")

    print(f"\nBest model: {best_model_name}")
    print(f"Figures saved to {FIGURES_DIR}")
    print(f"Metrics saved to {RESULTS_DIR / 'metrics.json'}")
    print("\nEnsure 'random_forest.joblib' exists in models/ for the API and live detector to work.")


if __name__ == "__main__":
    main()
