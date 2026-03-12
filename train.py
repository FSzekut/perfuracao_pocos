from pathlib import Path

import pandas as pd

from src.business import (
    BOOTSTRAP_SAMPLE_SIZE,
    BOOTSTRAP_SAMPLES,
    INVESTMENT_BUDGET,
    LOSS_RISK_LIMIT,
    PRICE_PER_THOUSAND_BARRELS,
    SELECTED_WELLS,
    bootstrap_profit,
    minimum_volume_per_well,
    profit_from_top_predictions,
)
from src.data import load_region_data
from src.modeling import evaluate_region_model


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
REGION_FILES = {
    "region_0": DATA_DIR / "geo_data_0.csv",
    "region_1": DATA_DIR / "geo_data_1.csv",
    "region_2": DATA_DIR / "geo_data_2.csv",
}


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)


def build_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    break_even_volume = minimum_volume_per_well(
        budget=INVESTMENT_BUDGET,
        selected_wells=SELECTED_WELLS,
        price_per_unit=PRICE_PER_THOUSAND_BARRELS,
    )

    for region_name, csv_path in REGION_FILES.items():
        dataset = load_region_data(csv_path)
        model_result = evaluate_region_model(dataset.features, dataset.target)

        top_volume, top_profit = profit_from_top_predictions(
            predictions=model_result.validation_predictions,
            targets=model_result.validation_target,
            top_n=SELECTED_WELLS,
            budget=INVESTMENT_BUDGET,
            price_per_unit=PRICE_PER_THOUSAND_BARRELS,
        )

        bootstrap_result = bootstrap_profit(
            predictions=model_result.validation_predictions,
            targets=model_result.validation_target,
            sample_size=BOOTSTRAP_SAMPLE_SIZE,
            top_n=SELECTED_WELLS,
            n_samples=BOOTSTRAP_SAMPLES,
            budget=INVESTMENT_BUDGET,
            price_per_unit=PRICE_PER_THOUSAND_BARRELS,
            random_state=12345,
        )

        model_rows.append(
            {
                "region": region_name,
                "rows": dataset.row_count,
                "feature_count": len(dataset.feature_names),
                "predicted_mean": model_result.prediction_mean,
                "actual_mean": model_result.target_mean,
                "rmse": model_result.rmse,
                "baseline_rmse": model_result.baseline_rmse,
                "rmse_gain": model_result.baseline_rmse - model_result.rmse,
                "break_even_volume": break_even_volume,
                "top200_volume": top_volume,
                "top200_avg_volume": top_volume / SELECTED_WELLS,
                "top200_profit_usd": top_profit,
            }
        )

        bootstrap_rows.append(
            {
                "region": region_name,
                "mean_profit_usd": bootstrap_result.mean_profit,
                "ci_lower_usd": bootstrap_result.confidence_interval[0],
                "ci_upper_usd": bootstrap_result.confidence_interval[1],
                "loss_risk_pct": bootstrap_result.loss_risk_pct,
                "approved_by_risk": bootstrap_result.loss_risk_pct < LOSS_RISK_LIMIT,
            }
        )

    return pd.DataFrame(model_rows), pd.DataFrame(bootstrap_rows)


def print_summary(model_report: pd.DataFrame, bootstrap_report: pd.DataFrame) -> None:
    print("=== Model comparison by region ===")
    print(
        model_report[
            [
                "region",
                "predicted_mean",
                "actual_mean",
                "rmse",
                "baseline_rmse",
                "top200_profit_usd",
            ]
        ].round(2).to_string(index=False)
    )

    print("\n=== Bootstrap risk analysis ===")
    print(
        bootstrap_report[
            [
                "region",
                "mean_profit_usd",
                "ci_lower_usd",
                "ci_upper_usd",
                "loss_risk_pct",
                "approved_by_risk",
            ]
        ].round(2).to_string(index=False)
    )

    eligible = bootstrap_report[bootstrap_report["approved_by_risk"]]
    if eligible.empty:
        print("\nNo region met the maximum loss risk threshold.")
        return

    best_region = eligible.sort_values("mean_profit_usd", ascending=False).iloc[0]
    print(
        f"\nRecommended region: {best_region['region']} "
        f"(mean profit ${best_region['mean_profit_usd']:.2f}, "
        f"loss risk {best_region['loss_risk_pct']:.2f}%)."
    )


def save_reports(model_report: pd.DataFrame, bootstrap_report: pd.DataFrame) -> None:
    model_report.to_csv(REPORTS_DIR / "model_report.csv", index=False)
    bootstrap_report.to_csv(REPORTS_DIR / "bootstrap_report.csv", index=False)


def main() -> None:
    ensure_dirs()
    model_report, bootstrap_report = build_report()
    save_reports(model_report, bootstrap_report)
    print_summary(model_report, bootstrap_report)


if __name__ == "__main__":
    main()
