import streamlit as st


def load_css():

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f5f7fa;
        }

        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }

        /* SIDEBAR */

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        section[data-testid="stSidebar"] .stRadio > div {
            gap: 4px;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
            display: none;
        }

        section[data-testid="stSidebar"] .stRadio label {
            font-size: 14px;
            padding: 8px 10px;
            border-radius: 7px;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            background-color: #1f2937;
        }

        /* BRAND */

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 2px 14px 2px;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background-color: #f59e0b;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .brand-title {
            font-size: 17px;
            font-weight: 700;
            color: #ffffff;
        }

        .brand-subtitle {
            font-size: 9px;
            color: #9ca3af;
            letter-spacing: 0.8px;
            margin-top: 2px;
        }

        /* SIDEBAR STATUS */

        .status-card {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 13px;
            margin-bottom: 18px;
        }

        .status-label {
            color: #9ca3af;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .status-title {
            color: #ffffff;
            font-size: 14px;
            font-weight: 650;
            margin-top: 4px;
        }

        .status-text {
            color: #9ca3af;
            font-size: 11px;
            margin-top: 3px;
        }

        .sidebar-section {
            color: #9ca3af;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            margin: 15px 0 8px 0;
        }

        /* TOP HEADER */

        .top-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #6b7280;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 25px;
        }

        .top-header span {
            color: #d1d5db;
            margin: 0 5px;
        }

        /* PAGE */

        .page-eyebrow {
            color: #6b7280;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.2px;
            margin-bottom: 8px;
        }

        .page-title {
            color: #111827;
            font-size: 32px;
            font-weight: 750;
            letter-spacing: -0.7px;
            line-height: 1.15;
            margin: 0;
        }

        .page-description {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.6;
            max-width: 760px;
            margin-top: 9px;
            margin-bottom: 26px;
        }

        /* EMPTY STATE */

        .empty-state {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 55px 30px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 18px;
        }

        .empty-icon {
            font-size: 34px;
            margin-bottom: 10px;
        }

        .empty-state h2 {
            color: #111827;
            font-size: 21px;
            margin-bottom: 7px;
        }

        .empty-state p {
            color: #6b7280;
            font-size: 13px;
            margin: 0;
        }

        /* SECTIONS */

        .section-title {
            color: #111827;
            font-size: 19px;
            font-weight: 650;
            margin-top: 24px;
            margin-bottom: 10px;
        }

        /* BUTTONS */

        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 40px;
        }

        /* UPLOADER */

        section[data-testid="stFileUploader"] {
            background-color: #ffffff;
            border: 1px dashed #cbd5e1;
            border-radius: 10px;
            padding: 8px;
        }

        /* DATAFRAME */

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }

        /* CHAT */

        div[data-testid="stChatMessage"] {
            border-radius: 10px;
            margin-bottom: 8px;
        }

        div[data-testid="stChatInput"] {
            border-radius: 10px;
        }

        /* DIVIDER */

        hr {
            border-color: #e5e7eb;
        }

        /* RESPONSIVE */

        @media (max-width: 768px) {

            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .page-title {
                font-size: 25px;
            }

            .page-description {
                font-size: 13px;
            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )