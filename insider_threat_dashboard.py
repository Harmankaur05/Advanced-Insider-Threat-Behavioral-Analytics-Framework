
# ============================================================
# INSIDER THREAT BEHAVIORAL ANALYTICS FRAMEWORK
# Streamlit Dashboard
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insider Threat Behavioral Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1,h2,h3{
    color:#003366;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border-left:6px solid #0A66C2;
    box-shadow:0px 2px 6px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_files():

    original = pd.read_csv("ehr_access_log.csv")

    processed = pd.read_csv("processed_employee_data.csv")

    report = pd.read_csv("Anomaly_Report.csv")

    return original, processed, report


original_df, processed_df, report_df = load_files()
# ============================================================
# FIX DATATYPE OF USER ID
# ============================================================

original_df["user_id"] = original_df["user_id"].astype(str)

processed_df["user_id"] = processed_df["user_id"].astype(str)

report_df["user_id"] = report_df["user_id"].astype(str)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load("RandomForest_Model.pkl")

    scaler = joblib.load("StandardScaler.pkl")

    encoder = joblib.load("LabelEncoder.pkl")

    return model, scaler, encoder


rf_model, scaler, encoder = load_models()

# ============================================================
# FIX DATATYPES BEFORE MERGING
# ============================================================

processed_df["user_id"] = (
    processed_df["user_id"]
    .astype(int)
    .astype(str)
    .str.zfill(4)
    .radd("USR")
)
report_df["user_id"] = report_df["user_id"].astype(str).str.strip()

dashboard_df = pd.merge(
    processed_df,
    report_df[
        [
            "user_id",
            "Risk_Score",
            "Threat_Level",
            "Isolation_Prediction",
            "Timestamp",
            "Detection_Reason"
        ]
    ],
    on="user_id",
    how="left"
)
# ============================================================
# RESTORE ORIGINAL IDENTIFIER & CATEGORICAL COLUMNS
# ============================================================

dashboard_df["event_id"] = original_df["event_id"].astype(str)
dashboard_df["username"] = original_df["username"].astype(str)
dashboard_df["department"] = original_df["department"].astype(str)
dashboard_df["role"] = original_df["role"].astype(str)

# ============================================================
# CONVERT TIMESTAMP
# ============================================================

if "Timestamp" in dashboard_df.columns:

    dashboard_df["Timestamp"] = pd.to_datetime(
        dashboard_df["Timestamp"],
        errors="coerce"
    )

if "timestamp" in dashboard_df.columns:

    dashboard_df["timestamp"] = pd.to_datetime(
        dashboard_df["timestamp"],
        errors="coerce"
    )

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🛡 Insider Threat Behavioral Analytics Framework")

st.markdown("""
### Continuous Employee Behaviour Monitoring using Machine Learning

This dashboard demonstrates:

- Isolation Forest Anomaly Detection
- Random Forest Classification
- Behavioural Analytics
- Insider Threat Monitoring
- Employee Risk Profiling
""")

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/security-checked.png",
    width=80
)

st.sidebar.title("Dashboard Filters")

# ------------------------------------------------------------
# Department Filter
# ------------------------------------------------------------

departments = sorted(
    dashboard_df["department"]
    .dropna()
    .astype(str)
    .unique()
)

department = st.sidebar.selectbox(

    "Department",

    ["All"] + departments

)

# ------------------------------------------------------------
# Role Filter
# ------------------------------------------------------------

roles = sorted(

    dashboard_df["role"]

    .dropna()

    .astype(str)

    .unique()

)

role = st.sidebar.selectbox(

    "Role",

    ["All"] + roles

)

# ------------------------------------------------------------
# Threat Filter
# ------------------------------------------------------------

threats = sorted(

    dashboard_df["Threat_Level"]

    .fillna("Unknown")

    .astype(str)

    .unique()

)

threat = st.sidebar.selectbox(

    "Threat Level",

    ["All"] + threats

)

# ------------------------------------------------------------
# User Search
# ------------------------------------------------------------

# ------------------------------------------------------------
# EVENT TIME FILTER
# ------------------------------------------------------------

event_times = sorted(
    dashboard_df["Timestamp"]
    .dropna()
    .dt.strftime("%Y-%m-%d %H:%M:%S")
    .unique()
)

selected_time = st.sidebar.selectbox(
    "Select Event Time",
    ["All"] + list(event_times)
)

# ------------------------------------------------------------
# USERNAME SEARCH
# ------------------------------------------------------------

# ------------------------------------------------------------
# EVENT ID FILTER
# ------------------------------------------------------------

event_ids = sorted(
    dashboard_df["event_id"]
    .dropna()
    .astype(str)
    .unique()
)

# ------------------------------------------------------------
# SEARCH EVENT ID
# ------------------------------------------------------------

search_event = st.sidebar.text_input(
    "Search Event ID"
)

# ------------------------------------------------------------
# USERNAME SEARCH
# ------------------------------------------------------------

search_name = st.sidebar.text_input(
    "Search Username"
)

st.sidebar.markdown("---")
st.sidebar.success("Dashboard Ready")
# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = dashboard_df.copy()

if department != "All":

    filtered_df = filtered_df[
        filtered_df["department"] == department
    ]

if role != "All":

    filtered_df = filtered_df[
        filtered_df["role"] == role
    ]

if threat != "All":

    filtered_df = filtered_df[
        filtered_df["Threat_Level"] == threat
    ]

if selected_time != "All":

    filtered_df = filtered_df[
        filtered_df["Timestamp"]
        .dt.strftime("%Y-%m-%d %H:%M:%S") == selected_time
    ]
if search_event:

    filtered_df = filtered_df[
        filtered_df["event_id"]
        .astype(str)
        .str.contains(search_event, case=False, na=False)
    ]
if search_event:

    filtered_df = filtered_df[
        filtered_df["event_id"]
        .astype(str)
        .str.contains(search_event, case=False)
    ]

if search_name:

    filtered_df = filtered_df[
        filtered_df["username"]
        .astype(str)
        .str.contains(search_name, case=False)
    ]

# ============================================================
# KPI CALCULATIONS
# ============================================================

total_employees = dashboard_df["user_id"].nunique()

suspicious_employees = report_df["user_id"].nunique()

threat_percentage = round(

    suspicious_employees /
    total_employees * 100,

    2

)

average_risk = round(

    report_df["Risk_Score"].mean(),

    2

)

critical_threats = len(

    report_df[
        report_df["Threat_Level"] == "Critical"
    ]

)

# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📈 Key Performance Indicators")

col1,col2,col3,col4,col5 = st.columns(5)

with col1:

    st.metric(

        "Employees",

        total_employees

    )

with col2:

    st.metric(

        "Suspicious",

        suspicious_employees

    )

with col3:

    st.metric(

        "Threat %",

        f"{threat_percentage}%"

    )

with col4:

    st.metric(

        "Average Risk",

        average_risk

    )

with col5:

    st.metric(

        "Critical Threats",

        critical_threats

    )

st.markdown("---")
# ============================================================
# THREAT ANALYTICS
# ============================================================

st.header("📊 Threat Analytics")

col1, col2 = st.columns(2)

# ============================================================
# THREAT DISTRIBUTION
# ============================================================

with col1:

    st.subheader("Threat Level Distribution")

    threat_data = (
        filtered_df["Threat_Level"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    threat_data.columns = [
        "Threat_Level",
        "Count"
    ]

    fig = px.pie(

        threat_data,

        names="Threat_Level",

        values="Count",

        hole=0.45,

        title="Threat Distribution"

    )

    st.plotly_chart(
    fig,
    use_container_width=True,
    key="threat_distribution"
    )
# ============================================================
# DEPARTMENT THREATS
# ============================================================

with col2:

    st.subheader("Department-wise Threats")

    dept = (

        filtered_df

        .groupby("department")

        .size()

        .reset_index(name="Threat_Count")

    )

    fig = px.bar(

        dept,

        x="department",

        y="Threat_Count",

        color="Threat_Count",

        text="Threat_Count",

        title="Threats by Department"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="department_chart"
    )

st.markdown("---")

# ============================================================
# ROLE ANALYTICS
# ============================================================

col3, col4 = st.columns(2)

with col3:

    st.subheader("Role-wise Threats")

    role_df = (

        filtered_df

        .groupby("role")

        .size()

        .reset_index(name="Threat_Count")

    )

    fig = px.bar(

        role_df,

        x="role",

        y="Threat_Count",

        color="Threat_Count",

        text="Threat_Count",

        title="Threats by Role"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="role_chart"
    )

# ============================================================
# RISK SCORE HISTOGRAM
# ============================================================

with col4:

    st.subheader("Risk Score Distribution")

    fig = px.histogram(

        filtered_df,

        x="Risk_Score",

        nbins=30,

        title="Risk Score Histogram"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="risk_histogram"
    )

st.markdown("---")

# ============================================================
# LOGIN ANALYTICS
# ============================================================

st.header("🔐 Login Behaviour Analysis")

col5, col6 = st.columns(2)

# ============================================================
# FAILED LOGIN ANALYSIS
# ============================================================

with col5:

    failed = (

        original_df

        .groupby("department")["failed_auth_attempts"]

        .mean()

        .reset_index()

    )

    fig = px.bar(

        failed,

        x="department",

        y="failed_auth_attempts",

        color="failed_auth_attempts",

        title="Average Failed Authentication"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="failed_login_chart"
    )

# ============================================================
# VPN USAGE
# ============================================================

with col6:

    vpn = (

        original_df["vpn_usage"]

        .value_counts()

        .reset_index()

    )

    vpn.columns = [

        "VPN",

        "Employees"

    ]

    fig = px.pie(

        vpn,

        names="VPN",

        values="Employees",

        hole=0.45,

        title="VPN Usage"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="vpn_chart"
    )

st.markdown("---")

# ============================================================
# EMPLOYEE ACTIVITY
# ============================================================

st.header("👨‍💻 Employee Behaviour")

col7, col8 = st.columns(2)

# ============================================================
# OFF HOURS ACCESS
# ============================================================

with col7:

    off = (

        original_df["off_hours_access"]

        .value_counts()

        .reset_index()

    )

    off.columns = [

        "Off Hours",

        "Employees"

    ]

    fig = px.pie(

        off,

        names="Off Hours",

        values="Employees",

        hole=0.45,

        title="Off-Hours Access"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="offhours_access_chart"
    )

# ============================================================
# BULK DOWNLOAD
# ============================================================

with col8:

    bulk = (

        original_df["bulk_download_flag"]

        .value_counts()

        .reset_index()

    )

    bulk.columns = [

        "Bulk Download",

        "Employees"

    ]

    fig = px.bar(

        bulk,

        x="Bulk Download",

        y="Employees",

        color="Employees",

        title="Bulk Download Activity"

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="bulk_download_chart"
    )

st.markdown("---")

# ============================================================
# TIMELINE
# ============================================================

st.header("📅 Threat Timeline")

timeline_df = filtered_df.dropna(subset=["Timestamp"]).copy()

if len(timeline_df) > 0:

    timeline = (
        timeline_df
        .groupby(timeline_df["Timestamp"].dt.date)
        .size()
        .reset_index(name="Threats")
    )

    timeline.columns = ["Date", "Threats"]

    fig = px.line(
        timeline,
        x="Date",
        y="Threats",
        markers=True,
        title="Timeline of Detected Threats"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="timeline_chart"
    )

else:

    st.info("No timeline data available.")
# ============================================================
# EMPLOYEE RISK PROFILE
# ============================================================

st.header("👤 Employee Risk Profile")

employee_list = sorted(filtered_df["user_id"].astype(str).unique())

selected_user = st.selectbox(
    "Select User ID",
    employee_list
)

employee = filtered_df[
    filtered_df["user_id"].astype(str) == selected_user
]

if not employee.empty:

    col1, col2 = st.columns([1,2])

    with col1:

        st.metric(
            "Risk Score",
            round(float(employee["Risk_Score"].iloc[0]),2)
        )

        st.metric(
            "Threat Level",
            employee["Threat_Level"].iloc[0]
        )

    with col2:

        st.write("### Employee Details")

        st.write(f"**User ID :** {employee['user_id'].iloc[0]}")
        st.write(f"**Username :** {employee['username'].iloc[0]}")
        st.write(f"**Department :** {employee['department'].iloc[0]}")
        st.write(f"**Role :** {employee['role'].iloc[0]}")
        st.write(f"**Detection Reason :** {employee['Detection_Reason'].iloc[0]}")

st.markdown("---")

# ============================================================
# TOP HIGH RISK EMPLOYEES
# ============================================================

st.header("🚨 Top High Risk Employees")

top_risk = filtered_df.sort_values(
    by="Risk_Score",
    ascending=False
)

st.dataframe(

    top_risk[
        [
            "user_id",
            "username",
            "department",
            "role",
            "Risk_Score",
            "Threat_Level",
            "Detection_Reason"
        ]
    ],

    use_container_width=True

)

st.markdown("---")

# ============================================================
# COMPLETE ANOMALY REPORT
# ============================================================

st.header("📋 Complete Anomaly Report")

total_records = len(filtered_df)

if total_records == 0:

    st.warning("No records found for the selected filters.")

elif total_records == 1:

    st.info("1 record found.")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

else:

    records = st.slider(
        "Number of Records",
        min_value=1,
        max_value=total_records,
        value=min(25, total_records)
    )

    st.dataframe(
        filtered_df.head(records),
        use_container_width=True
    )

# ============================================================
# DETECTION REASONS
# ============================================================

st.header("📌 Threat Detection Reasons")

reason = (

    filtered_df["Detection_Reason"]

    .value_counts()

    .reset_index()

)

reason.columns = [

    "Reason",

    "Count"

]

fig = px.bar(

    reason,

    x="Reason",

    y="Count",

    color="Count",

    title="Reasons Behind Insider Threat Detection"

)

st.plotly_chart(

    fig,

    use_container_width=True,
    key="Reason_chart"

)

st.markdown("---")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("🌳 Random Forest Feature Importance")

try:

    # Remove non-feature columns
    features = processed_df.drop(
        columns=[
            "Target",
            "user_id",
            "username"
        ],
        errors="ignore"
    )

    # Match feature names with model feature importances
    feature_names = list(features.columns)
    feature_scores = list(rf_model.feature_importances_)

    min_len = min(len(feature_names), len(feature_scores))

    importance = pd.DataFrame({
        "Feature": feature_names[:min_len],
        "Importance": feature_scores[:min_len]
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Top 15 Important Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="feature_importance_chart"
    )

except Exception as e:

    st.error(f"Feature Importance Error: {e}")

st.markdown("---")

# ============================================================
# RISK SCORE TABLE
# ============================================================

st.header("📈 Employee Risk Scores")

risk_table = filtered_df[[

    "user_id",

    "username",

    "department",

    "role",

    "Risk_Score",

    "Threat_Level"

]]

st.dataframe(

    risk_table,

    use_container_width=True

)

st.markdown("---")

# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.header("📥 Download Report")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(

    "Download Anomaly Report",

    csv,

    "Anomaly_Report.csv",

    "text/csv"

)

st.markdown("---")
# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("🤖 Machine Learning Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info("""
### Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm.

Purpose:
- Detect unusual employee behaviour
- Identify insider threats
- Generate anomaly labels
""")

with col2:

    st.success("""
### Random Forest

Random Forest is used to classify employees based on anomaly labels.

Purpose:
- Predict insider threat level
- Calculate employee risk score
- Identify suspicious behaviour
""")

st.markdown("---")

# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📂 Dataset Summary")

dataset_summary = pd.DataFrame({

    "Metric":[
        "Total Employees",
        "Processed Records",
        "Threat Records",
        "Departments",
        "Roles",
        "Features Used"
    ],

    "Value":[
        dashboard_df["user_id"].nunique(),
        len(processed_df),
        len(report_df),
        dashboard_df["department"].nunique(),
        dashboard_df["role"].nunique(),
        processed_df.shape[1]-3
    ]

})

st.dataframe(
    dataset_summary,
    use_container_width=True
)

st.markdown("---")

# ============================================================
# LIVE DASHBOARD STATUS
# ============================================================

st.header("🟢 Dashboard Status")

status1, status2, status3 = st.columns(3)

status1.success("Dataset Loaded")

status2.success("Models Loaded")

status3.success("Dashboard Running")

st.markdown("---")

# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.header("⚙️ Project Workflow")

workflow = """

1️⃣ Data Collection

↓

2️⃣ Data Cleaning

↓

3️⃣ Exploratory Data Analysis

↓

4️⃣ Feature Engineering

↓

5️⃣ Encoding

↓

6️⃣ Feature Selection

↓

7️⃣ Train-Test Split

↓

8️⃣ Feature Scaling

↓

9️⃣ Isolation Forest

↓

🔟 Random Forest

↓

11️⃣ Model Evaluation

↓

12️⃣ Risk Scoring

↓

13️⃣ Insider Threat Detection

↓

14️⃣ Streamlit Dashboard

"""

st.code(workflow)

st.markdown("---")

# ============================================================
# LAST REFRESH
# ============================================================

st.header("🕒 Dashboard Information")

st.write("Current Time : ", datetime.now())

st.write("Dashboard Status : Running")

st.write("Machine Learning Model : Random Forest")

st.write("Anomaly Detection : Isolation Forest")

st.write("Monitoring : Continuous Employee Behaviour")

st.markdown("---")

# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.header("📖 Project Description")

st.write("""

This project demonstrates an Advanced Insider Threat Behavioural Analytics Framework.

The framework continuously monitors employee activities using behavioural analytics.

The project applies:

• Isolation Forest for anomaly detection

• Random Forest for threat classification

• Behavioural feature engineering

• Risk score generation

• Insider threat monitoring

The dashboard provides:

✔ Employee Risk Profiles

✔ Department Analysis

✔ Threat Analytics

✔ Behaviour Monitoring

✔ Risk Score Distribution

✔ Feature Importance

✔ Downloadable Reports

""")

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
---
<center>

# 🛡 Insider Threat Behavioral Analytics Framework

### Advanced Machine Learning based Insider Threat Detection

Developed using

Python • Scikit-Learn • Isolation Forest • Random Forest • Streamlit • Plotly

---

College Capstone Project

Continuous Employee Behaviour Monitoring using Behavioural Analytics

</center>
""",
unsafe_allow_html=True
)
