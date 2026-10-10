import pandas as pd


def get_consumption_summary(df, target_column):
    """
    Create a compact summary of electricity consumption data.
    """

    if df.empty:
        return "No electricity data is available."

    total_records = len(df)

    average_consumption = df[target_column].mean()
    maximum_consumption = df[target_column].max()
    minimum_consumption = df[target_column].min()

    max_row = df.loc[
        df[target_column].idxmax()
    ]

    min_row = df.loc[
        df[target_column].idxmin()
    ]

    summary = f"""
Electricity Consumption Summary

Total records: {total_records}
Average consumption: {average_consumption:.2f}
Maximum consumption: {maximum_consumption:.2f}
Maximum consumption time: {max_row['DateTime']}
Minimum consumption: {minimum_consumption:.2f}
Minimum consumption time: {min_row['DateTime']}
"""

    return summary.strip()


def get_anomaly_summary(anomaly_df):
    """
    Create a compact anomaly summary.
    """

    if anomaly_df.empty:
        return "No anomalies were detected."

    if "Final_Anomaly" in anomaly_df.columns:

        anomaly_count = int(
            anomaly_df["Final_Anomaly"].sum()
        )

    else:

        anomaly_count = len(anomaly_df)

    if (
        "Anomaly_Type" in anomaly_df.columns
        and anomaly_count > 0
    ):

        anomaly_types = (
            anomaly_df[
                anomaly_df["Final_Anomaly"]
            ]["Anomaly_Type"]
            .value_counts()
            .to_dict()
        )

    else:

        anomaly_types = {}

    summary = f"""
Anomaly Detection Summary

Total detected anomalies: {anomaly_count}
Anomaly types: {anomaly_types}
"""

    return summary.strip()


def get_peak_usage_summary(df, target_column):
    """
    Find the average consumption by hour.
    """

    if df.empty:
        return "No data is available for peak analysis."

    temp_df = df.copy()

    temp_df["Hour"] = (
        temp_df["DateTime"].dt.hour
    )

    hourly_average = (
        temp_df
        .groupby("Hour")[target_column]
        .mean()
    )

    peak_hour = int(
        hourly_average.idxmax()
    )

    peak_value = hourly_average.max()

    summary = f"""
Peak Usage Summary

Peak average hour: {peak_hour:02d}:00
Average consumption during peak hour: {peak_value:.2f}
"""

    return summary.strip()


def build_ai_context(
    consumption_df,
    anomaly_df,
    target_column
):
    """
    Combine important project information
    into a single context for the AI assistant.
    """

    consumption_summary = (
        get_consumption_summary(
            consumption_df,
            target_column
        )
    )

    anomaly_summary = (
        get_anomaly_summary(
            anomaly_df
        )
    )

    peak_summary = (
        get_peak_usage_summary(
            consumption_df,
            target_column
        )
    )

    context = f"""
{consumption_summary}

{peak_summary}

{anomaly_summary}
"""

    return context.strip()