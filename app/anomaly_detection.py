from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "isolation_forest_anomaly_model.pkl"
)


def load_anomaly_model():
    """
    Load the trained Isolation Forest model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def calculate_residuals(
    actual,
    predicted
):
    """
    Calculate residual and absolute residual.
    """

    residual = actual - predicted
    absolute_residual = np.abs(residual)

    return residual, absolute_residual


def calculate_iqr_threshold(
    absolute_residual
):
    """
    Calculate anomaly threshold using IQR.
    """

    q1 = np.quantile(
        absolute_residual,
        0.25
    )

    q3 = np.quantile(
        absolute_residual,
        0.75
    )

    iqr = q3 - q1

    threshold = q3 + (1.5 * iqr)

    return threshold


def detect_anomalies(
    model,
    actual,
    predicted
):
    """
    Detect anomalies using:

    1. Residual + IQR
    2. Isolation Forest

    Final anomaly is recorded only
    when both methods agree.
    """

    residual, absolute_residual = (
        calculate_residuals(
            actual,
            predicted
        )
    )

    threshold = calculate_iqr_threshold(
        absolute_residual
    )

    iqr_anomaly = (
        absolute_residual > threshold
    )

    isolation_input = pd.DataFrame(
        {
            "Absolute_Residual":
                absolute_residual
        }
    )

    isolation_prediction = (
        model.predict(
            isolation_input
        )
    )

    isolation_anomaly = (
        isolation_prediction == -1
    )

    final_anomaly = (
        iqr_anomaly
        & isolation_anomaly
    )

    results = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predicted,
            "Residual": residual,
            "Absolute_Residual":
                absolute_residual,
            "IQR_Anomaly":
                iqr_anomaly,
            "Isolation_Anomaly":
                isolation_anomaly,
            "Final_Anomaly":
                final_anomaly
        }
    )

    results["Anomaly_Type"] = np.where(
        results["Final_Anomaly"]
        & (results["Residual"] > 0),
        "High Consumption",
        np.where(
            results["Final_Anomaly"]
            & (results["Residual"] < 0),
            "Low Consumption",
            "Normal"
        )
    )

    return results, threshold