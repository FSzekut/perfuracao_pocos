from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class RegionModelResult:
    model: LinearRegression
    validation_predictions: np.ndarray
    validation_target: pd.Series
    rmse: float
    baseline_rmse: float
    prediction_mean: float
    target_mean: float


def evaluate_region_model(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.25,
    random_state: int = 12345,
) -> RegionModelResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    baseline_predictions = np.repeat(y_train.mean(), len(y_valid))

    rmse = root_mean_squared_error(y_valid, predictions)
    baseline_rmse = root_mean_squared_error(y_valid, baseline_predictions)

    return RegionModelResult(
        model=model,
        validation_predictions=predictions,
        validation_target=y_valid.reset_index(drop=True),
        rmse=rmse,
        baseline_rmse=baseline_rmse,
        prediction_mean=float(np.mean(predictions)),
        target_mean=float(y_valid.mean()),
    )


def root_mean_squared_error(target: pd.Series, predictions: np.ndarray) -> float:
    return float(mean_squared_error(target, predictions) ** 0.5)
