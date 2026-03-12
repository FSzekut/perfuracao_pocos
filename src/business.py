from dataclasses import dataclass

import numpy as np
import pandas as pd


INVESTMENT_BUDGET = 100_000_000
SELECTED_WELLS = 200
BOOTSTRAP_SAMPLE_SIZE = 500
BOOTSTRAP_SAMPLES = 1000
PRICE_PER_BARREL = 4.5
PRICE_PER_THOUSAND_BARRELS = 4_500
LOSS_RISK_LIMIT = 2.5


@dataclass
class BootstrapResult:
    mean_profit: float
    confidence_interval: tuple[float, float]
    loss_risk_pct: float


def minimum_volume_per_well(
    budget: float,
    selected_wells: int,
    price_per_unit: float,
) -> float:
    return budget / selected_wells / price_per_unit


def profit_from_top_predictions(
    predictions: np.ndarray,
    targets: pd.Series,
    top_n: int,
    budget: float,
    price_per_unit: float,
) -> tuple[float, float]:
    results = pd.DataFrame(
        {
            "prediction": predictions,
            "target": pd.Series(targets).reset_index(drop=True),
        }
    )
    top_wells = results.sort_values("prediction", ascending=False).head(top_n)
    total_volume = float(top_wells["target"].sum())
    profit = total_volume * price_per_unit - budget
    return total_volume, float(profit)


def bootstrap_profit(
    predictions: np.ndarray,
    targets: pd.Series,
    sample_size: int,
    top_n: int,
    n_samples: int,
    budget: float,
    price_per_unit: float,
    random_state: int,
) -> BootstrapResult:
    rng = np.random.RandomState(random_state)
    results = pd.DataFrame(
        {
            "prediction": predictions,
            "target": pd.Series(targets).reset_index(drop=True),
        }
    )

    profits: list[float] = []
    for _ in range(n_samples):
        sample = results.sample(n=sample_size, replace=True, random_state=rng)
        _, profit = profit_from_top_predictions(
            predictions=sample["prediction"].to_numpy(),
            targets=sample["target"],
            top_n=top_n,
            budget=budget,
            price_per_unit=price_per_unit,
        )
        profits.append(profit)

    lower, upper = np.percentile(profits, [2.5, 97.5])
    return BootstrapResult(
        mean_profit=float(np.mean(profits)),
        confidence_interval=(float(lower), float(upper)),
        loss_risk_pct=float((np.array(profits) < 0).mean() * 100),
    )
