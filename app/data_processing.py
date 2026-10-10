import pandas as pd


TARGET_COLUMN = "usage_kw"


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


def load_csv(file):
    """
    Load electricity consumption CSV file.
    Automatically detects common CSV separators.
    """

    separators = [",", ";", "\t", "|"]

    last_error = None

    for separator in separators:
        try:
            file.seek(0)

            df = pd.read_csv(
                file,
                sep=separator,
                encoding="utf-8-sig"
            )

            if len(df.columns) > 1:
                break

        except Exception as e:
            last_error = e

    else:
        try:
            file.seek(0)

            df = pd.read_csv(
                file,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

        except Exception as e:
            raise ValueError(
                f"Could not read the CSV file. {e}"
            ) from e

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def detect_datetime_column(df):

    preferred_names = [
        "datetime",
        "date_time",
        "timestamp",
        "date",
        "time",
        "localminute",
        "local_minute"
    ]

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in preferred_names:
            return column

    best_column = None
    best_ratio = 0

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_ratio = parsed.notna().mean()

        if valid_ratio > 0.8 and valid_ratio > best_ratio:
            best_ratio = valid_ratio
            best_column = column

    return best_column


def detect_consumption_column(df):

    keywords = {
        "consumption": 100,
        "electricity": 100,
        "usage": 95,
        "load": 90,
        "power": 85,
        "energy": 80,
        "watt": 75,
        "kw": 70,
        "kwh": 70
    }

    excluded_words = [
        "temperature",
        "humidity",
        "wind",
        "voltage",
        "current",
        "frequency",
        "pressure",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "id"
    ]

    candidates = []

    for column in df.columns:

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        numeric_ratio = numeric_values.notna().mean()

        if numeric_ratio < 0.7:
            continue

        name = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        score = 0

        for keyword, value in keywords.items():

            if keyword in name:
                score += value

        for word in excluded_words:

            if word in name:
                score -= 100

        candidates.append(
            (column, score)
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if candidates and candidates[0][1] > 0:
        return candidates[0][0]

    numeric_columns = []

    for column in df.columns:

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if numeric_values.notna().mean() >= 0.7:
            numeric_columns.append(column)

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    return None


def normalize_columns(df):

    df = df.copy()

    datetime_column = detect_datetime_column(df)
    consumption_column = detect_consumption_column(df)

    rename_map = {}

    if (
        datetime_column is not None
        and "DateTime" not in df.columns
    ):
        rename_map[datetime_column] = "DateTime"

    if (
        consumption_column is not None
        and "usage_kw" not in df.columns
    ):
        rename_map[consumption_column] = "usage_kw"

    df = df.rename(
        columns=rename_map
    )

    return df


def validate_data(df):

    missing_columns = []

    if "DateTime" not in df.columns:
        missing_columns.append("DateTime")

    if "usage_kw" not in df.columns:
        missing_columns.append("electricity consumption")

    if missing_columns:

        available_columns = ", ".join(
            str(column)
            for column in df.columns
        )

        raise ValueError(
            "The uploaded file could not be identified "
            "as an electricity consumption dataset.\n\n"
            f"Missing: {', '.join(missing_columns)}\n\n"
            f"Available columns: {available_columns}\n\n"
            "The file should contain a date/time column "
            "and an electricity usage, power, load, "
            "energy, consumption, kW, kWh, or similar column."
        )

    return True

def prepare_hourly_data(df):
    """
    Convert raw electricity data into
    hourly electricity consumption data.
    """

    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce"
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "DateTime",
            TARGET_COLUMN
        ]
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)

    hourly_df = (
            df
            .set_index("DateTime")
            .resample("1h")[TARGET_COLUMN]
            .mean()
            .reset_index()
        )

    hourly_df["Zone 1 Power Consumption"] = hourly_df[TARGET_COLUMN]

    return hourly_df


def create_features(df):
    """
    Create forecasting features.

    These features match the feature structure
    expected by the forecasting model.
    """

    df = df.copy()

    df["Hour"] = (
        df["DateTime"].dt.hour
    )

    df["DayOfWeek"] = (
        df["DateTime"].dt.dayofweek
    )

    df["Day"] = (
        df["DateTime"].dt.day
    )

    df["Month"] = (
        df["DateTime"].dt.month
    )

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    df["Lag_1"] = (
        df[TARGET_COLUMN]
        .shift(1)
    )

    df["Lag_24"] = (
        df[TARGET_COLUMN]
        .shift(24)
    )

    df["Lag_168"] = (
        df[TARGET_COLUMN]
        .shift(168)
    )

    df["RollingMean_24"] = (
        df[TARGET_COLUMN]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["RollingStd_24"] = (
        df[TARGET_COLUMN]
        .shift(1)
        .rolling(24)
        .std()
    )

    return df


def prepare_user_data(file):
    """
    Complete preprocessing pipeline
    for uploaded electricity data.
    """

    df = load_csv(file)

    df = normalize_columns(df)

    validate_data(df)

    hourly_df = prepare_hourly_data(
        df
    )

    feature_df = create_features(
        hourly_df
    )

    return feature_df