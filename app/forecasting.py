from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "xgboost_electricity_forecasting_model.pkl"
)

TARGET_COLUMN = "Zone 1 Power Consumption"

FEATURE_COLUMNS = [
    "Hour",
    "DayOfWeek",
    "Day",
    "Month",
    "IsWeekend",
    "Lag_1",
    "Lag_24",
    "Lag_168",
    "RollingMean_24",
    "RollingStd_24"
]


def load_forecasting_model():
    """
    Load the trained XGBoost forecasting model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Forecasting model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_consumption(model, feature_df):
    """
    Generate predictions for prepared feature data.
    """

    X = feature_df[FEATURE_COLUMNS]

    predictions = model.predict(X)

    return predictions


def forecast_next_24_hours(model, hourly_df):
    """
    Forecast the next 24 hours using recursive prediction.
    """

    history = hourly_df[
        [
            "DateTime",
            TARGET_COLUMN
        ]
    ].dropna().copy()

    if len(history) < 168:
        raise ValueError(
            "At least 168 hourly records are required "
            "to generate a 24-hour forecast."
        )

    forecast_rows = []

    for _ in range(24):

        next_time = (
            history["DateTime"].iloc[-1]
            + pd.Timedelta(hours=1)
        )

        values = history[TARGET_COLUMN]

        feature_row = {
            "Hour": next_time.hour,
            "DayOfWeek": next_time.dayofweek,
            "Day": next_time.day,
            "Month": next_time.month,
            "IsWeekend": int(
                next_time.dayofweek >= 5
            ),
            "Lag_1": values.iloc[-1],
            "Lag_24": values.iloc[-24],
            "Lag_168": values.iloc[-168],
            "RollingMean_24": values.iloc[-24:].mean(),
            "RollingStd_24": values.iloc[-24:].std()
        }

        feature_df = pd.DataFrame(
            [feature_row],
            columns=FEATURE_COLUMNS
        )

        prediction = model.predict(
            feature_df
        )[0]

        forecast_rows.append(
            {
                "DateTime": next_time,
                "Predicted Consumption": prediction
            }
        )

        new_row = pd.DataFrame(
            [
                {
                    "DateTime": next_time,
                    TARGET_COLUMN: prediction
                }
            ]
        )

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    return forecast_df