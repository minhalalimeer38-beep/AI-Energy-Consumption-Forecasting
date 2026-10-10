import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DATA_DIR,
    HOURLY_DATA_PATH,
    FORECAST_RESULTS_PATH,
    ANOMALY_RESULTS_PATH,
    TARGET_COLUMN
)

from styles import load_css

from data_processing import (
    prepare_user_data,
    FEATURE_COLUMNS
)

from forecasting import (
    load_forecasting_model,
    predict_consumption,
    forecast_next_24_hours
)

from anomaly_detection import (
    load_anomaly_model,
    detect_anomalies
)


st.set_page_config(
    page_title="AI Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

@st.cache_resource
def get_models():

    forecast_model = load_forecasting_model()
    anomaly_model = load_anomaly_model()

    return forecast_model, anomaly_model


try:

    forecast_model, anomaly_model = get_models()

except Exception as e:

    st.error(f"Could not load models: {e}")
    st.stop()

@st.cache_data
def load_saved_results():

    forecast_df = pd.DataFrame()
    anomaly_df = pd.DataFrame()

    if FORECAST_RESULTS_PATH.exists():

        forecast_df = pd.read_csv(
            FORECAST_RESULTS_PATH
        )

        if "DateTime" in forecast_df.columns:

            forecast_df["DateTime"] = pd.to_datetime(
                forecast_df["DateTime"]
            )

    if ANOMALY_RESULTS_PATH.exists():

        anomaly_df = pd.read_csv(
            ANOMALY_RESULTS_PATH
        )

        if "DateTime" in anomaly_df.columns:

            anomaly_df["DateTime"] = pd.to_datetime(
                anomaly_df["DateTime"]
            )

    return forecast_df, anomaly_df


forecast_results, anomaly_results = (
    load_saved_results()
)

@st.cache_data
def load_hourly_data():

    if not HOURLY_DATA_PATH.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        HOURLY_DATA_PATH
    )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"]
    )

    return df


hourly_data = load_hourly_data()


if "user_data" not in st.session_state:

    st.session_state.user_data = None


if "user_anomalies" not in st.session_state:

    st.session_state.user_anomalies = None


if "user_forecast" not in st.session_state:

    st.session_state.user_forecast = None


if "ai_messages" not in st.session_state:

    st.session_state.ai_messages = []


if "ai_agent" not in st.session_state:

    st.session_state.ai_agent = None


with st.sidebar:

    st.markdown(
        """
        ## ⚡ AI Energy Analytics

        Smart Electricity Monitoring
        """
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyze Your Data",
            "Forecast",
            "Anomaly Detection",
            "Peak Usage",
            "AI Assistant"
        ]
    )

    st.divider()

    st.caption("Forecasting")
    st.write("XGBoost")

    st.caption("Anomaly Detection")
    st.write("IQR + Isolation Forest")

    st.caption("AI")
    st.write("RAG + Agentic AI")


if page == "Dashboard":

    st.title(
        "Electricity Consumption Dashboard"
    )

    st.caption(
        "AI-powered electricity forecasting, anomaly detection "
        "and energy analytics."
    )

    if forecast_results.empty:

        st.warning(
            "Forecast results are not available."
        )

    else:

        average_consumption = (
            forecast_results["Actual"].mean()
        )

        peak_consumption = (
            forecast_results["Actual"].max()
        )

        if (
            not anomaly_results.empty
            and "Final_Anomaly" in anomaly_results.columns
        ):

            anomaly_count = int(
                anomaly_results["Final_Anomaly"].sum()
            )

        else:

            anomaly_count = 0

        latest_date = (
            forecast_results["DateTime"].max()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Average Consumption",
                f"{average_consumption:,.0f}"
            )

        with col2:

            st.metric(
                "Peak Consumption",
                f"{peak_consumption:,.0f}"
            )

        with col3:

            st.metric(
                "Anomalies Detected",
                anomaly_count
            )

        with col4:

            st.metric(
                "Latest Reading",
                latest_date.strftime("%d %b")
            )

        st.divider()


        st.subheader(
            "Actual vs Predicted Consumption"
        )

        period = st.selectbox(
            "Display period",
            [
                "Last 7 Days",
                "Last 14 Days",
                "Last 30 Days",
                "Full Test Period"
            ]
        )

        if period == "Last 7 Days":

            chart_df = forecast_results.tail(
                24 * 7
            )

        elif period == "Last 14 Days":

            chart_df = forecast_results.tail(
                24 * 14
            )

        elif period == "Last 30 Days":

            chart_df = forecast_results.tail(
                24 * 30
            )

        else:

            chart_df = forecast_results

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_df["DateTime"],
                y=chart_df["Actual"],
                mode="lines",
                name="Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_df["DateTime"],
                y=chart_df["Predicted"],
                mode="lines",
                name="Predicted"
            )
        )

        fig.update_layout(
            height=450,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Power Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.subheader(
            "Forecasting Model Performance"
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "MAE",
                "903.72"
            )

        with m2:

            st.metric(
                "RMSE",
                "1217.75"
            )

        with m3:

            st.metric(
                "MAPE",
                "3.15%"
            )


elif page == "Analyze Your Data":

    st.title(
        "Analyze Your Electricity Data"
    )

    st.caption(
        "Upload historical electricity data to generate "
        "forecasts and detect unusual consumption."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    st.info(
        "Required columns: DateTime and "
        "Zone 1 Power Consumption"
    )

    if uploaded_file is not None:

        if st.button(
            "Analyze Data",
            type="primary"
        ):

            try:


                user_df = prepare_user_data(
                    uploaded_file
                )

                model_df = user_df.dropna(
                    subset=FEATURE_COLUMNS
                ).copy()

                if len(model_df) < 200:

                    st.warning(
                        "Please upload more historical data. "
                        "At least 200 hourly records are recommended."
                    )

                    st.stop()


                predictions = predict_consumption(
                    forecast_model,
                    model_df
                )

                model_df["Predicted"] = predictions

                anomaly_output, threshold = (
                    detect_anomalies(
                        anomaly_model,
                        model_df[TARGET_COLUMN].values,
                        model_df["Predicted"].values
                    )
                )

                anomaly_output["DateTime"] = (
                    model_df["DateTime"].values
                )

                future_forecast = (
                    forecast_next_24_hours(
                        forecast_model,
                        user_df
                    )
                )

                st.session_state.user_data = model_df

                st.session_state.user_anomalies = (
                    anomaly_output
                )

                st.session_state.user_forecast = (
                    future_forecast
                )

                # Reset AI agent because data changed

                st.session_state.ai_agent = None
                st.session_state.ai_messages = []

                st.success(
                    "Data analyzed successfully."
                )

            except Exception as e:

                st.error(
                    f"Could not analyze the uploaded file: {e}"
                )

    if st.session_state.user_data is not None:

        user_df = (
            st.session_state.user_data
        )

        user_anomalies = (
            st.session_state.user_anomalies
        )

        user_forecast = (
            st.session_state.user_forecast
        )

        anomaly_count = int(
            user_anomalies["Final_Anomaly"].sum()
        )

        total_records = len(
            user_anomalies
        )

        anomaly_rate = (
            anomaly_count / total_records
        ) * 100

        st.divider()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Processed Records",
                f"{total_records:,}"
            )

        with c2:

            st.metric(
                "Anomalies Detected",
                anomaly_count
            )

        with c3:

            st.metric(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%"
            )

        st.subheader(
            "Consumption and Anomalies"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=user_df["DateTime"],
                y=user_df[TARGET_COLUMN],
                mode="lines",
                name="Actual Consumption"
            )
        )

        anomalies = user_anomalies[
            user_anomalies["Final_Anomaly"]
        ]

        if not anomalies.empty:

            fig.add_trace(
                go.Scatter(
                    x=anomalies["DateTime"],
                    y=anomalies["Actual"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        size=9,
                        symbol="x"
                    )
                )
            )

        fig.update_layout(
            height=450,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Power Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Detected Anomalies"
        )

        if anomalies.empty:

            st.success(
                "No potential anomalies were detected."
            )

        else:

            anomaly_table = anomalies[
                [
                    "DateTime",
                    "Actual",
                    "Predicted",
                    "Residual",
                    "Absolute_Residual",
                    "Anomaly_Type"
                ]
            ].sort_values(
                "Absolute_Residual",
                ascending=False
            )

            st.dataframe(
                anomaly_table,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Anomaly Report",
                anomaly_table.to_csv(
                    index=False
                ),
                file_name="user_anomaly_report.csv",
                mime="text/csv"
            )

        st.subheader(
            "Next 24 Hours Forecast"
        )

        st.dataframe(
            user_forecast,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download 24-Hour Forecast",
            user_forecast.to_csv(
                index=False
            ),
            file_name="next_24_hours_forecast.csv",
            mime="text/csv"
        )

        forecast_fig = go.Figure()

        recent_history = user_df.tail(168)

        forecast_fig.add_trace(
            go.Scatter(
                x=recent_history["DateTime"],
                y=recent_history[TARGET_COLUMN],
                mode="lines",
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=user_forecast["DateTime"],
                y=user_forecast[
                    "Predicted Consumption"
                ],
                mode="lines+markers",
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            height=450,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Power Consumption"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

elif page == "Forecast":

    st.title(
        "Electricity Forecast"
    )

    st.caption(
        "XGBoost forecasting results on the test period."
    )

    if forecast_results.empty:

        st.warning(
            "Forecast results are not available."
        )

    else:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=forecast_results["DateTime"],
                y=forecast_results["Actual"],
                mode="lines",
                name="Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_results["DateTime"],
                y=forecast_results["Predicted"],
                mode="lines",
                name="Predicted"
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Power Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "MAE",
                "903.72"
            )

        with c2:

            st.metric(
                "RMSE",
                "1217.75"
            )

        with c3:

            st.metric(
                "MAPE",
                "3.15%"
            )

elif page == "Anomaly Detection":

    st.title(
        "Anomaly Detection"
    )

    st.caption(
        "Potentially unusual electricity consumption events."
    )

    if anomaly_results.empty:

        st.warning(
            "Anomaly results are not available."
        )

    else:

        if "Final_Anomaly" in anomaly_results.columns:

            final_anomalies = anomaly_results[
                anomaly_results["Final_Anomaly"]
            ].copy()

        else:

            final_anomalies = anomaly_results.copy()

        st.metric(
            "Detected Anomalies",
            len(final_anomalies)
        )

        if final_anomalies.empty:

            st.success(
                "No anomalies were detected."
            )

        else:

            st.dataframe(
                final_anomalies,
                use_container_width=True,
                hide_index=True
            )

elif page == "Peak Usage":

    st.title(
        "Peak Usage Analysis"
    )

    st.caption(
        "Average electricity consumption by hour of day."
    )

    if hourly_data.empty:

        st.warning(
            "Hourly data is not available."
        )

    else:

        temp_df = hourly_data.copy()

        temp_df["Hour"] = (
            temp_df["DateTime"].dt.hour
        )

        hourly_average = (
            temp_df
            .groupby("Hour")[TARGET_COLUMN]
            .mean()
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=hourly_average["Hour"],
                y=hourly_average[TARGET_COLUMN],
                name="Average Consumption"
            )
        )

        fig.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Hour of Day",
            yaxis_title="Average Consumption",
            xaxis=dict(
                tickmode="linear",
                dtick=1
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        peak_row = hourly_average.loc[
            hourly_average[TARGET_COLUMN].idxmax()
        ]

        st.info(
            f"Highest average consumption occurs around "
            f"{int(peak_row['Hour']):02d}:00."
        )

elif page == "AI Assistant":

    st.title(
        "AI Energy Assistant"
    )

    st.caption(
        "Ask questions about your electricity consumption, "
        "anomalies, peak usage and energy-management knowledge."
    )

    st.info(
        "The assistant responds in the same language you use, "
        "including English and Roman Urdu."
    )

    if st.session_state.user_data is not None:

        ai_consumption_df = (
            st.session_state.user_data
        )

        ai_anomaly_df = (
            st.session_state.user_anomalies
        )

        data_source = "Uploaded user data"

    else:

        ai_consumption_df = hourly_data

        ai_anomaly_df = anomaly_results

        data_source = "Project dataset"

    st.caption(
        f"Current AI data source: {data_source}"
    )

    for message in st.session_state.ai_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask about your electricity consumption..."
    )

    if question:

        st.session_state.ai_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        try:

            # Import only when AI Assistant is used.
            from agent import (
                create_energy_agent,
                ask_energy_agent
            )

            if st.session_state.ai_agent is None:

                st.session_state.ai_agent = (
                    create_energy_agent(
                        consumption_df=ai_consumption_df,
                        anomaly_df=ai_anomaly_df
                    )
                )

            with st.chat_message("assistant"):

                with st.spinner(
                    "Analyzing..."
                ):

                    answer = ask_energy_agent(
                        st.session_state.ai_agent,
                        question
                    )

                st.markdown(
                    answer
                )

            st.session_state.ai_messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            error_message = (
                "AI Assistant could not respond. "
                "Please check your Gemini API key and "
                f"AI dependencies.\n\nError: {e}"
            )

            with st.chat_message("assistant"):

                st.error(
                    error_message
                )

            st.session_state.ai_messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )