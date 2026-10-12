import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.connect_snowflake import get_connection


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

TABLE_NAME = "HR_ATTRITION.RAW.HR_ATTRITION_FINAL"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "attrition_model.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# Optimized threshold from your threshold analysis
FINAL_THRESHOLD = 0.25


# ============================================================
# THEMES
# ============================================================

THEMES = {

    "Midnight": {
        "background": "#080D1A",
        "surface": "#111827",
        "surface2": "#172033",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "primary": "#6366F1",
        "secondary": "#8B5CF6",
        "accent": "#22D3EE",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "border": "#263247"
    },

    "Ocean": {
        "background": "#071A26",
        "surface": "#0B2535",
        "surface2": "#10364A",
        "text": "#F0FDFA",
        "muted": "#94A3B8",
        "primary": "#06B6D4",
        "secondary": "#0EA5E9",
        "accent": "#2DD4BF",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#F43F5E",
        "border": "#164E63"
    },

    "Emerald": {
        "background": "#071B13",
        "surface": "#0C281C",
        "surface2": "#123B29",
        "text": "#F0FDF4",
        "muted": "#9CA3AF",
        "primary": "#10B981",
        "secondary": "#14B8A6",
        "accent": "#84CC16",
        "success": "#22C55E",
        "warning": "#EAB308",
        "danger": "#EF4444",
        "border": "#14532D"
    },

    "Royal": {
        "background": "#120B24",
        "surface": "#1D1235",
        "surface2": "#2A1B4D",
        "text": "#FAF5FF",
        "muted": "#A78BFA",
        "primary": "#A855F7",
        "secondary": "#7C3AED",
        "accent": "#EC4899",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#F43F5E",
        "border": "#4C1D95"
    },

    "Professional Light": {
        "background": "#F1F5F9",
        "surface": "#FFFFFF",
        "surface2": "#F8FAFC",
        "text": "#0F172A",
        "muted": "#64748B",
        "primary": "#2563EB",
        "secondary": "#4F46E5",
        "accent": "#0891B2",
        "success": "#16A34A",
        "warning": "#D97706",
        "danger": "#DC2626",
        "border": "#E2E8F0"
    }
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🏢 HR Intelligence")
    st.caption("Attrition Analytics Platform")

    st.divider()

    theme_name = st.selectbox(
        "🎨 Dashboard Theme",
        list(THEMES.keys()),
        index=0
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "📊 Attrition Analytics",
            "🔮 Predict Attrition",
            "👥 Employee Explorer",
            "🧠 Model Intelligence"
        ]
    )

    st.divider()

    st.caption("❄️ Data Source: Snowflake")
    st.caption("🌲 Algorithm: Random Forest")
    st.caption("🎯 Decision Threshold: 0.25")


THEME = THEMES[theme_name]


# ============================================================
# STREAMLIT STYLING
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background:
            radial-gradient(
                circle at 85% 0%,
                {THEME["surface2"]} 0%,
                {THEME["background"]} 45%
            );

        color: {THEME["text"]};
    }}

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                {THEME["surface"]},
                {THEME["background"]}
            );

        border-right:
            1px solid {THEME["border"]};
    }}

    section[data-testid="stSidebar"] * {{
        color: {THEME["text"]};
    }}

    h1, h2, h3, h4 {{
        color: {THEME["text"]} !important;
    }}

    div[data-testid="stMetric"] {{
        background:
            linear-gradient(
                145deg,
                {THEME["surface"]},
                {THEME["surface2"]}
            );

        border:
            1px solid {THEME["border"]};

        border-radius:
            16px;

        padding:
            18px;

        box-shadow:
            0 8px 28px rgba(0,0,0,0.15);
    }}

    div[data-testid="stMetricValue"] {{
        font-size:
            2.15rem !important;

        font-weight:
            800 !important;
    }}

    div[data-testid="stMetricLabel"] {{
        font-size:
            0.95rem !important;
    }}

    .stButton > button {{
        border-radius:
            12px;

        border:
            1px solid {THEME["primary"]};

        background:
            linear-gradient(
                90deg,
                {THEME["primary"]},
                {THEME["secondary"]}
            );

        color:
            white;

        font-weight:
            700;
    }}

    .stDownloadButton > button {{
        border-radius:
            12px;

        border:
            1px solid {THEME["primary"]};

        background:
            {THEME["surface2"]};

        color:
            {THEME["text"]};

        font-weight:
            700;
    }}

    div[data-testid="stAlert"] {{
        border-radius:
            12px;
    }}

    hr {{
        border-color:
            {THEME["border"]};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA FROM SNOWFLAKE
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def load_data():

    conn = None
    cursor = None

    try:

        conn = get_connection()
        cursor = conn.cursor()

        query = f"""
            SELECT *
            FROM {TABLE_NAME}
        """

        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

        data = pd.DataFrame(
            rows,
            columns=columns
        )

        return data

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LOAD OUTPUT FILE
# ============================================================

def load_output_csv(filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    df = load_data()

    model = load_model()

    metrics_df = load_output_csv(
        "tuned_random_forest_metrics.csv"
    )

    if metrics_df.empty:

        metrics_df = load_output_csv(
            "random_forest_metrics.csv"
        )

    threshold_df = load_output_csv(
        "threshold_optimization_results.csv"
    )

except Exception as e:

    st.error(
        "Unable to load dashboard resources."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

df["ATTRITION_BINARY"] = (
    df["ATTRITION"]
    .map(
        {
            "No": 0,
            "Yes": 1
        }
    )
)


COLOR_MAP = {
    "No": THEME["success"],
    "Yes": THEME["danger"]
}


# ============================================================
# PLOT STYLE FUNCTION
# ============================================================

def style_plot(
    fig,
    height=430
):

    fig.update_layout(

        height=height,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font={
            "color": THEME["text"]
        },

        margin={
            "l": 25,
            "r": 25,
            "t": 65,
            "b": 30
        },

        legend={
            "bgcolor": "rgba(0,0,0,0)"
        }
    )

    fig.update_xaxes(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"]
    )

    fig.update_yaxes(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"]
    )

    return fig


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        rf_model = model.named_steps[
            "model"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importance_values = (
            rf_model
            .feature_importances_
        )

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values
            }
        )

        result = (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(12)
        )

        return result

    except Exception:

        return pd.DataFrame()


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(
    probability
):

    if probability >= 0.60:
        return "HIGH"

    elif probability >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# RISK FACTORS
# ============================================================

def get_risk_factors(
    employee
):

    factors = []

    if str(
        employee["OVERTIME"]
    ).lower() == "yes":

        factors.append(
            "Overtime workload"
        )

    if employee[
        "JOB_SATISFACTION"
    ] <= 2:

        factors.append(
            "Low job satisfaction"
        )

    if employee[
        "DISTANCE_FROM_HOME"
    ] >= 20:

        factors.append(
            "Long distance from home"
        )

    if employee[
        "YEARS_AT_COMPANY"
    ] <= 2:

        factors.append(
            "Short tenure"
        )

    if employee[
        "PERFORMANCE_RATING"
    ] <= 2:

        factors.append(
            "Low performance rating"
        )

    if employee[
        "MONTHLY_INCOME"
    ] < df[
        "MONTHLY_INCOME"
    ].median():

        factors.append(
            "Below-median monthly income"
        )

    if not factors:

        factors.append(
            "No major rule-based risk signals detected"
        )

    return factors


# ============================================================
# PAGE 1
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.title(
        "🏢 HR Attrition Intelligence"
    )

    st.caption(
        "A data-driven workforce analytics platform "
        "powered by Snowflake and Random Forest Machine Learning."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_employees = len(df)

    attrition_cases = int(
        (
            df["ATTRITION"]
            == "Yes"
        ).sum()
    )

    attrition_rate = (
        attrition_cases
        /
        total_employees
        *
        100
    )

    avg_income = (
        df["MONTHLY_INCOME"]
        .mean()
    )

    avg_tenure = (
        df["YEARS_AT_COMPANY"]
        .mean()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Total Employees",
        f"{total_employees:,}"
    )

    c2.metric(
        "🚨 Attrition Cases",
        f"{attrition_cases:,}"
    )

    c3.metric(
        "📉 Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    c4.metric(
        "💰 Avg Monthly Income",
        f"₹{avg_income:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # WORKFORCE SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "📊 Workforce Snapshot"
    )

    c1, c2 = st.columns(2)

    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    with c1:

        department_data = (
            df.groupby(
                [
                    "DEPARTMENT",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            department_data,
            x="DEPARTMENT",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Attrition by Department"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # ATTRITION SPLIT
    # --------------------------------------------------------

    with c2:

        pie_data = (
            df["ATTRITION"]
            .value_counts()
            .reset_index()
        )

        pie_data.columns = [
            "ATTRITION",
            "Employees"
        ]

        fig = px.pie(
            pie_data,
            names="ATTRITION",
            values="Employees",
            hole=0.55,
            color="ATTRITION",
            color_discrete_map=COLOR_MAP,
            title="Overall Attrition Split"
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # OVERTIME + AGE
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        overtime_data = (
            df.groupby(
                [
                    "OVERTIME",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            overtime_data,
            x="OVERTIME",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Attrition by Overtime"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    with c2:

        fig = px.histogram(
            df,
            x="AGE",
            color="ATTRITION",
            nbins=20,
            marginal="box",
            color_discrete_map=COLOR_MAP,
            title="Age Distribution by Attrition"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # INCOME + SATISFACTION
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        fig = px.violin(
            df,
            x="ATTRITION",
            y="MONTHLY_INCOME",
            color="ATTRITION",
            box=True,
            points="outliers",
            color_discrete_map=COLOR_MAP,
            title="Monthly Income vs Attrition"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    with c2:

        satisfaction_data = (
            df.groupby(
                [
                    "JOB_SATISFACTION",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            satisfaction_data,
            x="JOB_SATISFACTION",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Job Satisfaction vs Attrition"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "💡 Workforce Summary"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Tenure",
        f"{avg_tenure:.1f} years"
    )

    c2.metric(
        "Employees Remaining",
        f"{total_employees - attrition_cases:,}"
    )

    c3.metric(
        "Employees Leaving",
        f"{attrition_cases:,}"
    )


# ============================================================
# PAGE 2
# ATTRITION ANALYTICS
# ============================================================

elif page == "📊 Attrition Analytics":

    st.title(
        "📊 Attrition Analytics"
    )

    st.caption(
        "Interactive workforce analysis with department and tenure filtering."
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    f1, f2 = st.columns(2)

    with f1:

        departments = sorted(
            df[
                "DEPARTMENT"
            ]
            .dropna()
            .unique()
        )

        dept_filter = st.multiselect(
            "🏢 Filter by Department",
            options=departments,
            default=departments
        )

    with f2:

        min_years = int(
            df[
                "YEARS_AT_COMPANY"
            ].min()
        )

        max_years = int(
            df[
                "YEARS_AT_COMPANY"
            ].max()
        )

        yrs_range = st.slider(
            "⏳ Tenure Range",
            min_years,
            max_years,
            (
                min_years,
                max_years
            )
        )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    fdf = df[
        df[
            "DEPARTMENT"
        ].isin(
            dept_filter
        )
        &
        df[
            "YEARS_AT_COMPANY"
        ].between(
            yrs_range[0],
            yrs_range[1]
        )
    ].copy()

    st.info(
        f"Showing {len(fdf):,} employees."
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total = len(fdf)

    left = int(
        (
            fdf["ATTRITION"]
            == "Yes"
        ).sum()
    )

    rate = (
        left / total * 100
        if total
        else 0
    )

    avg_income = (
        fdf[
            "MONTHLY_INCOME"
        ].mean()
        if total
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Employees",
        f"{total:,}"
    )

    c2.metric(
        "Employees Left",
        f"{left:,}"
    )

    c3.metric(
        "Attrition Rate",
        f"{rate:.1f}%"
    )

    c4.metric(
        "Avg Monthly Income",
        f"₹{avg_income:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # DEPARTMENT CHART
    # --------------------------------------------------------

    st.subheader(
        "🏢 Department Analysis"
    )

    dept_counts = (
        fdf.groupby(
            [
                "DEPARTMENT",
                "ATTRITION"
            ]
        )
        .size()
        .reset_index(
            name="Employees"
        )
    )

    fig = px.bar(
        dept_counts,
        x="DEPARTMENT",
        y="Employees",
        color="ATTRITION",
        barmode="group",
        text="Employees",
        color_discrete_map=COLOR_MAP,
        title="Attrition by Department"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        style_plot(fig),
        use_container_width=True
    )

    # --------------------------------------------------------
    # OVERTIME + AGE
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        overtime_data = (
            fdf.groupby(
                [
                    "OVERTIME",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            overtime_data,
            x="OVERTIME",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Attrition by Overtime"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    with c2:

        fig = px.histogram(
            fdf,
            x="AGE",
            color="ATTRITION",
            nbins=20,
            marginal="box",
            color_discrete_map=COLOR_MAP,
            title="Age Distribution"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # INCOME + SATISFACTION
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        fig = px.violin(
            fdf,
            x="ATTRITION",
            y="MONTHLY_INCOME",
            color="ATTRITION",
            box=True,
            points="outliers",
            color_discrete_map=COLOR_MAP,
            title="Monthly Income vs Attrition"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    with c2:

        satisfaction_data = (
            fdf.groupby(
                [
                    "JOB_SATISFACTION",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            satisfaction_data,
            x="JOB_SATISFACTION",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Job Satisfaction vs Attrition"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # GENDER + MARITAL STATUS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        gender_data = (
            fdf.groupby(
                [
                    "GENDER",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            gender_data,
            x="GENDER",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Gender vs Attrition"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    with c2:

        marital_data = (
            fdf.groupby(
                [
                    "MARITAL_STATUS",
                    "ATTRITION"
                ]
            )
            .size()
            .reset_index(
                name="Employees"
            )
        )

        fig = px.bar(
            marital_data,
            x="MARITAL_STATUS",
            y="Employees",
            color="ATTRITION",
            barmode="group",
            text="Employees",
            color_discrete_map=COLOR_MAP,
            title="Marital Status vs Attrition"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_plot(fig),
            use_container_width=True
        )

    # --------------------------------------------------------
    # JOB ROLE
    # --------------------------------------------------------

    st.subheader(
        "💼 Job Role Analysis"
    )

    role_data = (
        fdf.groupby(
            [
                "JOB_ROLE",
                "ATTRITION"
            ]
        )
        .size()
        .reset_index(
            name="Employees"
        )
    )

    fig = px.bar(
        role_data,
        x="Employees",
        y="JOB_ROLE",
        color="ATTRITION",
        orientation="h",
        barmode="group",
        text="Employees",
        color_discrete_map=COLOR_MAP,
        title="Attrition by Job Role"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        style_plot(fig, 520),
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🔑 Top Factors Driving Attrition"
    )

    st.caption(
        "Model-based feature importance from the trained Random Forest."
    )

    importance_df = get_feature_importance()

    if not importance_df.empty:

        importance_df = (
            importance_df
            .sort_values(
                "Importance"
            )
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues",
            title="Random Forest Feature Importance"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            style_plot(fig, 520),
            use_container_width=True
        )


# ============================================================
# PAGE 3
# PREDICT ATTRITION
# ============================================================

elif page == "🔮 Predict Attrition":

    st.title(
        "🔮 Predict Employee Attrition Risk"
    )

    st.caption(
        "Estimate attrition probability using the trained Random Forest model."
    )

    st.divider()

    if model is None:

        st.error(
            "Trained model not found."
        )

        st.info(
            "Run: python src\\train_model.py"
        )

        st.stop()

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form(
        "prediction_form"
    ):

        st.subheader(
            "👤 Employee Profile"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30
            )

            department = st.selectbox(
                "Department",
                sorted(
                    df[
                        "DEPARTMENT"
                    ]
                    .dropna()
                    .unique()
                )
            )

            job_role = st.selectbox(
                "Job Role",
                sorted(
                    df[
                        "JOB_ROLE"
                    ]
                    .dropna()
                    .unique()
                )
            )

        with c2:

            gender = st.selectbox(
                "Gender",
                sorted(
                    df[
                        "GENDER"
                    ]
                    .dropna()
                    .unique()
                )
            )

            marital_status = st.selectbox(
                "Marital Status",
                sorted(
                    df[
                        "MARITAL_STATUS"
                    ]
                    .dropna()
                    .unique()
                )
            )

            overtime = st.selectbox(
                "Overtime",
                sorted(
                    df[
                        "OVERTIME"
                    ]
                    .dropna()
                    .unique()
                )
            )

        with c3:

            performance = st.slider(
                "Performance Rating",
                1,
                4,
                3
            )

            satisfaction = st.slider(
                "Job Satisfaction",
                1,
                4,
                3
            )

            distance = st.number_input(
                "Distance From Home",
                min_value=1,
                max_value=100,
                value=10
            )

        st.subheader(
            "💼 Employment & Compensation"
        )

        c1, c2 = st.columns(2)

        with c1:

            income = st.number_input(
                "Monthly Income",
                min_value=10000,
                max_value=200000,
                value=int(
                    df[
                        "MONTHLY_INCOME"
                    ].median()
                ),
                step=1000
            )

        with c2:

            years = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=50,
                value=5
            )

        submitted = st.form_submit_button(
            "🔮 Predict Attrition Risk",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        input_df = pd.DataFrame(
            [
                {
                    "AGE": age,
                    "DEPARTMENT": department,
                    "JOB_ROLE": job_role,
                    "GENDER": gender,
                    "MARITAL_STATUS": marital_status,
                    "PERFORMANCE_RATING":
                        performance,
                    "OVERTIME": overtime,
                    "DISTANCE_FROM_HOME":
                        distance,
                    "MONTHLY_INCOME":
                        income,
                    "YEARS_AT_COMPANY":
                        years,
                    "JOB_SATISFACTION":
                        satisfaction
                }
            ]
        )

        try:

            probability = float(
                model
                .predict_proba(
                    input_df
                )[0][1]
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # OPTIMIZED THRESHOLD
        # ----------------------------------------------------

        prediction = (
            probability
            >=
            FINAL_THRESHOLD
        )

        risk = get_risk_category(
            probability
        )

        risk_factors = get_risk_factors(
            input_df.iloc[0]
        )

        st.divider()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if risk == "HIGH":

            st.error(
                f"🚨 HIGH ATTRITION RISK — "
                f"{probability * 100:.1f}% probability"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"⚠️ MEDIUM ATTRITION RISK — "
                f"{probability * 100:.1f}% probability"
            )

        else:

            st.success(
                f"✅ LOW ATTRITION RISK — "
                f"{probability * 100:.1f}% probability"
            )

        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction",
            "ATTRITION"
            if prediction
            else
            "NO ATTRITION"
        )

        c2.metric(
            "Probability",
            f"{probability * 100:.1f}%"
        )

        c3.metric(
            "Threshold",
            "25%"
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        st.subheader(
            "🎯 Attrition Probability"
        )

        st.progress(
            min(
                max(
                    int(
                        probability * 100
                    ),
                    0
                ),
                100
            )
        )

        # ----------------------------------------------------
        # GAUGE
        # ----------------------------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 34
                    }
                },

                title={
                    "text":
                        "Predicted Attrition Probability"
                },

                gauge={
                    "axis": {
                        "range": [
                            0,
                            100
                        ]
                    },

                    "bar": {
                        "color":
                            THEME["primary"]
                    },

                    "steps": [
                        {
                            "range": [
                                0,
                                30
                            ],
                            "color":
                                "rgba(34,197,94,0.18)"
                        },

                        {
                            "range": [
                                30,
                                60
                            ],
                            "color":
                                "rgba(245,158,11,0.18)"
                        },

                        {
                            "range": [
                                60,
                                100
                            ],
                            "color":
                                "rgba(239,68,68,0.18)"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            font={
                "color":
                    THEME["text"]
            }
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ----------------------------------------------------
        # RISK FACTORS
        # ----------------------------------------------------

        st.subheader(
            "🔍 Potential Risk Signals"
        )

        for factor in risk_factors:

            st.info(
                f"⚡ {factor}"
            )

        st.warning(
            "This prediction is a statistical estimate "
            "and should support HR analysis and human judgment, "
            "not replace it."
        )


# ============================================================
# PAGE 4
# EMPLOYEE EXPLORER
# ============================================================

elif page == "👥 Employee Explorer":

    st.title(
        "👥 Employee Explorer"
    )

    st.caption(
        "Search and filter employee records loaded directly from Snowflake."
    )

    st.divider()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Search Employee ID"
    )

    explorer_df = df.copy()

    if search:

        explorer_df = explorer_df[
            explorer_df[
                "EMP_ID"
            ]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        attrition_filter = st.selectbox(
            "Attrition",
            [
                "All",
                "Yes",
                "No"
            ]
        )

    with c2:

        department_filter = st.selectbox(
            "Department",
            [
                "All"
            ]
            +
            sorted(
                df[
                    "DEPARTMENT"
                ]
                .dropna()
                .unique()
            )
        )

    with c3:

        overtime_filter = st.selectbox(
            "Overtime",
            [
                "All",
                "Yes",
                "No"
            ]
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    if attrition_filter != "All":

        explorer_df = explorer_df[
            explorer_df[
                "ATTRITION"
            ]
            ==
            attrition_filter
        ]

    if department_filter != "All":

        explorer_df = explorer_df[
            explorer_df[
                "DEPARTMENT"
            ]
            ==
            department_filter
        ]

    if overtime_filter != "All":

        explorer_df = explorer_df[
            explorer_df[
                "OVERTIME"
            ]
            ==
            overtime_filter
        ]

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    st.metric(
        "Matching Employees",
        f"{len(explorer_df):,}"
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    display_columns = [
        "EMP_ID",
        "AGE",
        "DEPARTMENT",
        "JOB_ROLE",
        "GENDER",
        "MARITAL_STATUS",
        "PERFORMANCE_RATING",
        "OVERTIME",
        "DISTANCE_FROM_HOME",
        "ATTRITION",
        "MONTHLY_INCOME",
        "YEARS_AT_COMPANY",
        "JOB_SATISFACTION"
    ]

    st.dataframe(
        explorer_df[
            display_columns
        ],
        use_container_width=True,
        height=550,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = (
        explorer_df[
            display_columns
        ]
        .to_csv(
            index=False
        )
    )

    st.download_button(
        "📥 Download Filtered Employee Data",
        data=csv_data,
        file_name="hr_attrition_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# PAGE 5
# MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.title(
        "🧠 Model Intelligence"
    )

    st.caption(
        "Performance, configuration and methodology of the attrition prediction model."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Performance"
    )

    if not metrics_df.empty:

        metrics = metrics_df.iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{float(metrics['Accuracy']) * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{float(metrics['Precision']) * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{float(metrics['Recall']) * 100:.2f}%"
        )

        c4.metric(
            "F1 Score",
            f"{float(metrics['F1']) * 100:.2f}%"
        )

        c5.metric(
            "ROC-AUC",
            f"{float(metrics['ROC-AUC']) * 100:.2f}%"
        )

    else:

        st.warning(
            "Model metrics file was not found."
        )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    st.subheader(
        "🎯 Threshold Optimization"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Final Threshold",
            "0.25"
        )

    with c2:

        st.metric(
            "Best F1 Score",
            "49.20%"
        )

    st.info(
        "The classification threshold was optimized from "
        "the default 0.50 to 0.25. At 0.25, the model achieved "
        "approximately 90.44% recall and 49.20% F1 score "
        "on the evaluated test set."
    )

    # --------------------------------------------------------
    # THRESHOLD GRAPH
    # --------------------------------------------------------

    if not threshold_df.empty:

        threshold_column = None
        f1_column = None

        for column in threshold_df.columns:

            if column.lower() == "threshold":
                threshold_column = column

            if column.lower() == "f1":
                f1_column = column

        if (
            threshold_column
            and
            f1_column
        ):

            fig = px.line(
                threshold_df,
                x=threshold_column,
                y=f1_column,
                markers=True,
                title="Threshold vs F1 Score"
            )

            fig.add_vline(
                x=FINAL_THRESHOLD,
                line_dash="dash",
                annotation_text="Selected 0.25"
            )

            st.plotly_chart(
                style_plot(fig),
                use_container_width=True
            )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.subheader(
        "🌲 Random Forest Configuration"
    )

    model_info = pd.DataFrame(
        {
            "Component": [
                "Data Source",
                "Algorithm",
                "Trees",
                "Numeric Imputation",
                "Categorical Imputation",
                "Categorical Encoding",
                "Class Balancing",
                "Decision Threshold"
            ],

            "Configuration": [
                "Snowflake",
                "Random Forest",
                "300",
                "Median",
                "Most Frequent",
                "One-Hot Encoding",
                "SMOTE",
                "0.25"
            ]
        }
    )

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🔑 Top Model Features"
    )

    importance_df = get_feature_importance()

    if not importance_df.empty:

        importance_df = (
            importance_df
            .sort_values(
                "Importance"
            )
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues",
            title="Random Forest Feature Importance"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            style_plot(fig, 520),
            use_container_width=True
        )

    # --------------------------------------------------------
    # ML WORKFLOW
    # --------------------------------------------------------

    st.subheader(
        "🔬 End-to-End Machine Learning Workflow"
    )

    workflow = pd.DataFrame(
        {
            "Step": [
                1,
                2,
                3,
                4,
                5,
                6,
                7
            ],

            "Stage": [
                "Snowflake Data Extraction",
                "Data Preprocessing",
                "One-Hot Encoding",
                "SMOTE",
                "Random Forest Training",
                "Model Evaluation",
                "Threshold Optimization"
            ],

            "Purpose": [
                "Load employee records",
                "Handle missing values",
                "Encode categorical variables",
                "Handle class imbalance",
                "Train attrition classifier",
                "Evaluate prediction quality",
                "Optimize classification threshold"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # BUSINESS APPLICATION
    # --------------------------------------------------------

    st.subheader(
        "💡 Business Application"
    )

    st.info(
        "The system can be used as an early-warning tool "
        "for HR teams to identify employees with elevated "
        "statistical attrition risk and support retention planning."
    )

    st.warning(
        "Predictions should support human review and business "
        "context rather than being used as the sole basis for "
        "employment decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Attrition Intelligence • "
    "Snowflake • Random Forest • Streamlit"
)