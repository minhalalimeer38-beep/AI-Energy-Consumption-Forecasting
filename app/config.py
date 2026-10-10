from pathlib import Path
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR = PROJECT_DIR / "Data"
MODELS_DIR = PROJECT_DIR / "Models"
RESULTS_DIR = PROJECT_DIR / "Results"

RAW_DATA_PATH = (
    DATA_DIR / "Tetuan City power consumption.csv"
)

HOURLY_DATA_PATH = (
    DATA_DIR / "hourly_electricity_data.csv"
)

FORECAST_RESULTS_PATH = (
    RESULTS_DIR / "forecast_results.csv"
)

ANOMALY_RESULTS_PATH = (
    RESULTS_DIR / "anomaly_results.csv"
)


FORECAST_MODEL_PATH = (
    MODELS_DIR
    / "xgboost_electricity_forecasting_model.pkl"
)

ANOMALY_MODEL_PATH = (
    MODELS_DIR
    / "isolation_forest_anomaly_model.pkl"
)

ANOMALY_CONFIG_PATH = (
    MODELS_DIR
    / "anomaly_detection_config.pkl"
)


TARGET_COLUMN = "Zone 1 Power Consumption"

FORECAST_HORIZON = 24

MINIMUM_HISTORY_HOURS = 168

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