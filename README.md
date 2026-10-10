# AI-Based Electricity Consumption Forecasting and Anomaly Detection System

An AI-powered electricity analytics system that analyzes historical electricity consumption, forecasts future electricity usage, detects unusual consumption patterns, identifies peak-usage periods, and provides an AI assistant for natural-language, data-driven energy insights.

---

## Project Overview

Electricity consumption changes over time due to daily routines, weekly patterns, weather conditions, and other factors. Traditional electricity monitoring mainly shows how much electricity has already been consumed, but it does not automatically explain future usage or identify unusual consumption behavior.

This project combines Machine Learning, anomaly detection, interactive visualization, Retrieval-Augmented Generation (RAG), and Agentic AI into one electricity analytics application.

The system is designed to answer practical questions such as:

* How much electricity is normally consumed?
* What is the expected electricity consumption for the next 24 hours?
* When does peak electricity usage usually occur?
* Was a particular consumption event unusual?
* How different was the actual consumption from the expected pattern?
* What energy-efficiency information is relevant to the user's question?

---

# Problem Statement

Electricity users and energy managers may have access to historical consumption records, but raw data alone is difficult to interpret.

The main problems addressed by this project are:

1. Lack of future electricity consumption forecasting.
2. Difficulty identifying unusual consumption patterns.
3. Lack of clear peak-usage analysis.
4. Difficulty interpreting large deviations from normal consumption.
5. Lack of a simple natural-language interface for asking questions about electricity usage.
6. Separation between numerical electricity analytics and general energy-management knowledge.

The proposed system addresses these issues through an integrated AI-based analytics platform.

---

# Proposed Solution

The system follows this overall architecture:

```text
Historical Electricity Data
            |
            v
     Data Preprocessing
            |
            v
     Feature Engineering
            |
            +----------------------+
            |                      |
            v                      v
      XGBoost Forecasting     Anomaly Detection
            |                      |
            +----------+-----------+
                       |
                       v
               Analytics Layer
                       |
                       v
              Streamlit Dashboard
                       |
                       v
                AI Assistant
                 /         \
                v           v
               RAG       Data Tools
                |           |
                +-----+-----+
                      |
                      v
                 Gemini LLM
```

---

# Main Objectives

The project aims to:

* Analyze historical electricity consumption data.
* Convert high-frequency readings into an hourly forecasting dataset.
* Create time-based, lag, and rolling statistical features.
* Forecast future electricity consumption.
* Detect potentially abnormal consumption.
* Identify peak electricity usage periods.
* Provide interactive visualizations and reports.
* Allow users to upload compatible electricity consumption data.
* Provide a natural-language AI assistant.
* Retrieve information from energy-related documents using RAG.
* Use an agentic workflow to select appropriate analysis or retrieval tools.

---

# Core Features

## Electricity Consumption Analysis

The system provides:

* Average consumption
* Peak consumption
* Consumption trends
* Historical consumption visualization
* Hourly peak-usage analysis

---

## Electricity Consumption Forecasting

The main forecasting model is an **XGBoost Regressor**.

The model uses engineered features based on time and historical consumption.

### Features Used

#### Time Features

```text
Hour
DayOfWeek
Day
Month
IsWeekend
```

#### Lag Features

```text
Lag_1
Lag_24
Lag_168
```

Where:

* `Lag_1` = previous hour consumption
* `Lag_24` = same hour on the previous day
* `Lag_168` = same hour during the previous week

#### Rolling Features

```text
RollingMean_24
RollingStd_24
```

The rolling features use previous observations and exclude the current target value to prevent data leakage.

---

## Anomaly Detection

The system identifies consumption that is unusually different from the expected model behavior.

Two anomaly-detection signals are used:

1. Forecast residual + IQR threshold
2. Isolation Forest

A final anomaly is recorded when both methods agree.

### Important Interpretation

An anomaly means:

> Actual electricity consumption was unusually different from the expected pattern.

It does **not** automatically prove that:

* an appliance is faulty,
* electricity was stolen,
* a specific appliance caused the event,
* or a specific physical cause exists.

Additional sensor or appliance-level information would be required to determine the actual cause.

---

## Peak Usage Analysis

The system calculates average electricity consumption for each hour of the day.

This allows the application to identify:

* Peak hour
* Low-usage periods
* Daily consumption patterns
* Potential high-demand periods

---

## User Data Upload

The Streamlit application allows a user to upload electricity consumption data.

The current compatible minimum schema is:

```text
DateTime
Zone 1 Power Consumption
```

The application then:

```text
CSV Upload
    |
    v
Validation
    |
    v
DateTime Processing
    |
    v
Hourly Aggregation
    |
    v
Feature Engineering
    |
    v
Forecasting
    |
    v
Anomaly Detection
    |
    v
24-Hour Forecast
```

---

## AI Energy Assistant

The planned AI assistant provides a natural-language interface for:

* Consumption questions
* Peak-usage questions
* Anomaly questions
* Energy-efficiency questions
* Document-based energy questions

The assistant is designed to respond in the same language used by the user, including:

* English
* Urdu
* Roman Urdu

Example:

```text
Why was my electricity consumption unusually high?
```

or:

```text
Kal meri electricity consumption itni zyada kyun thi?
```

---

# Dataset

The project was initially developed using the **UCI Power Consumption of Tetouan City** dataset.

The dataset contains high-frequency electricity consumption measurements together with environmental and time-related information.

The original dataset contains electricity measurements at 10-minute intervals and provides consumption data for multiple zones.

For this project:

* The original readings were converted to hourly observations.
* `Zone 1 Power Consumption` was selected as the primary target.
* Time-based and historical lag features were engineered.

The same architecture can later be adapted to another electricity dataset if it follows a compatible input format.

---

# Why Was the Data Converted from 10-Minute to Hourly?

The original dataset contains observations such as:

```text
10:00
10:10
10:20
10:30
10:40
10:50
```

These readings were aggregated into hourly observations.

The reason is that this project focuses on:

* upcoming hours,
* upcoming days,
* peak periods,
* and practical electricity forecasting.

Hourly data reduces complexity while preserving the major consumption patterns needed for the forecasting task.

---

# Data Preprocessing

The preprocessing pipeline includes:

1. CSV loading
2. Required-column validation
3. DateTime conversion
4. Chronological sorting
5. Hourly aggregation
6. Feature generation
7. Removal of rows where lag/rolling features are not yet available

The project avoids blindly filling lag/rolling values with artificial data because those values represent real temporal relationships.

---

# Feature Engineering

Feature engineering is a core part of the forecasting system.

### Time-Based Features

```text
Hour
DayOfWeek
Day
Month
IsWeekend
```

### Historical Lag Features

```text
Lag_1
Lag_24
Lag_168
```

These represent short-term, daily, and weekly historical patterns.

### Rolling Features

```text
RollingMean_24
RollingStd_24
```

These describe the recent 24-hour consumption level and variation.

The rolling calculations use previous observations only to avoid target leakage.

---

# Forecasting Model

## XGBoost Regressor

XGBoost was selected as the primary forecasting model because the project uses structured time-based and lag-based features.

Advantages relevant to this project include:

* Strong performance on tabular data
* Ability to learn nonlinear relationships
* Good handling of mixed numerical features
* Practical training time
* Easy deployment as a saved model

The trained model is stored as:

```text
Models/xgboost_electricity_forecasting_model.pkl
```

---

# Baseline Comparison

A simple `Lag_24` baseline was created for comparison.

The baseline predicts electricity consumption using the same hour from the previous day.

### Evaluation Results

| Model           |        MAE |        RMSE |      MAPE |
| --------------- | ---------: | ----------: | --------: |
| Lag-24 Baseline |     968.40 |     1455.87 |     3.35% |
| **XGBoost**     | **903.72** | **1217.75** | **3.15%** |

The XGBoost model produced lower MAE, RMSE, and MAPE than the Lag-24 baseline on the evaluated held-out test period.

---

# Forecast Evaluation Metrics

The project uses three primary metrics.

## MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted consumption.

Lower MAE means smaller average prediction error.

---

## RMSE — Root Mean Squared Error

RMSE gives higher weight to large prediction errors.

It is useful for identifying whether a model occasionally produces larger mistakes.

---

## MAPE — Mean Absolute Percentage Error

MAPE represents prediction error as a percentage of the actual value.

Current test-period result:

```text
MAE  = 903.72
RMSE = 1217.75
MAPE = 3.15%
```

These values describe the evaluated test period and should not be interpreted as guaranteed performance on every future dataset or household.

---

# XGBoost Feature Importance

The trained model showed the following main feature-importance pattern:

| Feature        | Importance |
| -------------- | ---------: |
| `Lag_24`       |     60.22% |
| `Lag_168`      |     24.30% |
| `Lag_1`        |     10.97% |
| `Hour`         |      1.88% |
| Other features |  < 1% each |

The results indicate that previous-day and previous-week consumption patterns were particularly informative for the selected dataset.

This does not mean that the lower-ranked features are useless. Some information may already be captured indirectly through the lag features.

---

# Anomaly Detection Methodology

The system first calculates the forecasting residual:

```text
Residual = Actual Consumption - Predicted Consumption
```

Then:

```text
Absolute Residual = |Residual|
```

Large residuals indicate that actual consumption was substantially different from what the forecasting model expected.

---

## IQR-Based Detection

The threshold is calculated using the Interquartile Range:

```text
Q1 = 25th percentile
Q3 = 75th percentile

IQR = Q3 - Q1

Upper Threshold = Q3 + 1.5 × IQR
```

For the evaluated test period:

```text
Q1                = 297.75
Q3                = 1272.63
IQR               = 974.88
Anomaly Threshold = 2734.96
```

This method identified:

```text
65 potential anomalies
```

---

## Isolation Forest

Isolation Forest was used as a second anomaly-detection method.

Its purpose is to identify observations that are unusual relative to the residual-based representation.

The contamination parameter was calibrated using the IQR-based anomaly rate.

For the evaluated test period:

```text
Residual + IQR anomalies    = 65
Isolation Forest anomalies  = 65
Both methods agree          = 65
```

The final anomaly label is created from the agreement between both methods.

### Important Technical Note

Because the Isolation Forest contamination rate was calibrated from the IQR anomaly rate, the agreement should be interpreted as a consistency check rather than an independent accuracy validation.

---

# AI Assistant and RAG

The AI layer is designed to extend the numerical analytics system with natural-language interaction.

## Retrieval-Augmented Generation

RAG means **Retrieval-Augmented Generation**.

The RAG pipeline is:

```text
Energy Documents
       |
       v
Document Loading
       |
       v
Text Chunking
       |
       v
Gemini Embeddings
       |
       v
Chroma Vector Database
       |
       v
Semantic Retrieval
       |
       v
Relevant Context
       |
       v
Gemini
```

The knowledge base is intended to contain Pakistan-relevant energy documents, including official material related to:

* Energy efficiency
* Energy conservation
* Electricity policy
* Building energy efficiency

RAG is used for knowledge retrieval.

It does **not** train the electricity forecasting model.

---

# Agentic AI

The AI agent is designed to choose the appropriate tool according to the user's question.

Potential tools include:

```text
Consumption Analysis Tool
Anomaly Analysis Tool
Peak Usage Tool
RAG Knowledge Search Tool
```

Example:

```text
"What was my peak electricity usage?"
            |
            v
     Peak Usage Tool
            |
            v
      Data-Based Answer
```

Another example:

```text
"Which electricity events were unusual?"
            |
            v
       Anomaly Tool
            |
            v
      Data-Based Answer
```

Another example:

```text
"How can household electricity consumption be reduced?"
            |
            v
      RAG Knowledge Search
            |
            v
      Relevant Documents
            |
            v
       Gemini Response
```

The objective is to create an assistant that can use both **project data** and **external knowledge documents** instead of behaving like a simple fixed chatbot.

---

# Technology Stack

## Programming

* Python

## Data Processing

* Pandas
* NumPy

## Machine Learning

* XGBoost
* Scikit-learn

## Model Persistence

* Joblib

## Dashboard

* Streamlit
* Plotly
* Custom CSS through Streamlit

## Generative AI

* Google Gemini

## RAG

* LangChain
* Chroma
* Gemini Embeddings
* PyPDF

## Agentic AI

* LangChain Agent API
* LangGraph-based agent architecture

## Configuration and Security

* python-dotenv
* `.env`
* `.gitignore`

---

# Project Structure

```text
AI_Electricity_FYP/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── Data/
│   ├── Tetuan City power consumption.csv
│   └── hourly_electricity_data.csv
│
├── Models/
│   ├── xgboost_electricity_forecasting_model.pkl
│   ├── isolation_forest_anomaly_model.pkl
│   └── anomaly_detection_config.pkl
│
├── Notebook/
│   ├── 01_EDA_and_Forecasting.ipynb
│   └── 02_Anomaly_Detection.ipynb
│
├── Results/
│   ├── forecast_results.csv
│   ├── anomaly_results.csv
│   └── final_anomalies.csv
│
├── Knowledge/
│   └── *.pdf
│
└── app/
    ├── app.py
    ├── styles.py
    ├── data_processing.py
    ├── forecasting.py
    ├── anomaly_detection.py
    ├── ai_assistant.py
    ├── config.py
    ├── rag.py
    └── agent.py
```

---

# How to Run the Project

## Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_Electricity_FYP
```

---

## Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Never publish the real API key.

The `.env` file must remain excluded from Git.

---

## Run the Application

```bash
streamlit run app/app.py
```

The Streamlit dashboard will open in the browser.

---

# How to Use the Application

## Dashboard

The Dashboard provides:

* Average consumption
* Peak consumption
* Number of detected anomalies
* Latest available observation
* Actual vs predicted consumption
* Forecast model metrics

---

## Analyze Your Data

1. Open **Analyze Your Data**.
2. Upload a compatible CSV file.
3. Make sure the file contains:

```text
DateTime
Zone 1 Power Consumption
```

4. Click **Analyze Data**.
5. The system processes the dataset.
6. Forecasting is performed.
7. Potential anomalies are detected.
8. A 24-hour forecast is generated.
9. Forecast and anomaly results can be downloaded.

---

## Forecast

The Forecast page displays:

* Actual consumption
* Predicted consumption
* MAE
* RMSE
* MAPE

---

## Anomaly Detection

The Anomaly Detection page displays detected abnormal observations and their related residual information.

---

## Peak Usage

The Peak Usage page displays average consumption by hour and identifies the highest average usage hour.

---

## AI Assistant

The AI Assistant can answer questions such as:

```text
What was my peak consumption?
```

```text
Which events were unusual?
```

```text
Why was my electricity consumption unusually high?
```

```text
How can household electricity consumption be reduced?
```

The assistant can combine user/project data with the RAG knowledge base.

---

# Input Data Format

A minimal compatible CSV can look like:

```csv
DateTime,Zone 1 Power Consumption
2017-01-01 00:00:00,23000
2017-01-01 00:10:00,23250
2017-01-01 00:20:00,22800
```

The application aggregates high-frequency records to hourly observations before generating forecasts.

For the current recursive 24-hour forecast, the system requires enough history to calculate the 168-hour weekly lag.

The current application therefore recommends providing sufficient historical data, preferably several days or more.

---

# Security

Never commit your API key to GitHub.

The project uses:

```text
.env
```

for secret configuration.

The `.gitignore` should contain:

```gitignore
.env
__pycache__/
*.pyc
.venv/
```

If an API key is accidentally exposed, revoke it and create a new one.

---

# Limitations

The current system has several limitations:

1. The primary forecasting model was trained using the Tetouan City dataset and Zone 1 consumption.
2. A model trained on one geographical/data distribution should not automatically be assumed to be accurate for every household or region.
3. Anomaly detection identifies unusual statistical behavior but does not prove the physical cause of an event.
4. The current upload workflow expects the Zone 1 target-column schema.
5. Forecast performance depends on the quantity and quality of historical data.
6. RAG response quality depends on the quality, relevance, and coverage of the uploaded knowledge documents.
7. More detailed explanations of appliance-level causes would require additional information such as appliance measurements, occupancy, weather, tariffs, or outage data.

---

# Future Enhancements

Possible future extensions include:

* Household-specific model retraining
* LSTM-based forecasting
* Transformer-based time-series models
* Weather-aware forecasting
* Electricity bill estimation
* Smart-meter integration
* Real-time IoT data
* Appliance-level consumption analysis
* More detailed anomaly classification
* Multi-user database support
* Automated energy reports
* Cloud deployment
* Advanced multi-tool AI agents

---

# Project Outcome

The system transforms raw electricity consumption data into useful analytical information:

```text
Historical Consumption
        |
        v
Understand Consumption Patterns
        |
        v
Forecast Future Usage
        |
        v
Detect Unusual Events
        |
        v
Identify Peak Periods
        |
        v
Visualize Results
        |
        v
Ask Natural-Language Questions
        |
        v
Generate AI-Based Insights
```

---

#  Final Summary

This project is more than a simple electricity prediction model.

It is an integrated AI-based electricity analytics system combining:

* Data preprocessing
* Time-series feature engineering
* XGBoost forecasting
* Residual-based anomaly detection
* Isolation Forest
* Interactive Streamlit visualization
* RAG
* Agentic AI
* Gemini
* Natural-language interaction

The system is designed to transform historical electricity consumption data into:

* Future forecasts
* Anomaly alerts
* Peak-usage insights
* Interactive visualizations
* AI-generated explanations
* Energy-management knowledge

---

#  Academic Use

This project was developed as an academic Final Year Project.

External datasets, official documents, libraries, models, and third-party content remain subject to their respective licenses and terms. Original sources should be credited when redistributing external material.

---

## One-Line Project Description

> **An AI-powered electricity analytics system that forecasts future consumption, detects unusual usage patterns, identifies peak periods, and provides RAG-based Agentic AI insights through an interactive Streamlit dashboard.**
