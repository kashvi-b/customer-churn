# ============================================================
# STEP 6: Streamlit Dashboard (Portfolio Showpiece)
# ============================================================
# pip install streamlit plotly shap joblib pandas
# Run with: streamlit run src/dashboard.py


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
import sys
import os

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if project_root not in sys.path:
    sys.path.append(project_root)

from explainability.shap_utils import get_top_features
from recommendations.retention_engine import retention_recommendation
# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📉",
    layout="wide"
)
from analytics.business_metrics import calculate_business_impact
st.title("📉 Telco Customer Churn Risk Dashboard")
st.caption("Powered by XGBoost + SHAP — IBM Telco Dataset")

# ── Load model & data ──────────────────────────────────────
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

@st.cache_resource
def load_model():

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "best_model.pkl"
    )

    return joblib.load(model_path)

@st.cache_data
def load_test_data():

    x_path = os.path.join(
        BASE_DIR,
        "data",
        "X_test.csv"
    )

    y_path = os.path.join(
        BASE_DIR,
        "data",
        "y_test.csv"
    )

    X = pd.read_csv(x_path)

    y = pd.read_csv(y_path).squeeze()

    return X, y

model      = load_model()
X_test, y_test = load_test_data()
y_proba    = model.predict_proba(X_test)[:, 1]

results = X_test.copy()

results['churn_probability'] = y_proba

results['actual_churn'] = y_test.values

results['risk_tier'] = pd.cut(
    y_proba,
    bins=[0, 0.4, 0.7, 1.0],
    labels=['Low', 'Medium', 'High']
)


results["recommendation"] = (
    results.apply(
        retention_recommendation,
        axis=1
    )
)

# ── Sidebar: threshold slider ──────────────────────────────
# ── Sidebar ─────────────────────────────

st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "Risk threshold",
    0.2,
    0.8,
    0.4,
    0.05
)

st.sidebar.markdown("---")
# ===================================================
# FILTERS
# ===================================================

st.sidebar.subheader("🔎 Filters")

# Senior Citizen

senior_filter = st.sidebar.selectbox(
    "Senior Citizen",
    ["All", "Yes", "No"],
    key="senior"
)

# Contract Type

contract_filter = st.sidebar.selectbox(
    "Contract Type",
    ["All", "Month-to-month", "One year", "Two year"],
    key="contract"
)

# Internet Service

internet_filter = st.sidebar.selectbox(
    "Internet Service",
    ["All", "Fiber optic", "DSL", "No"],
    key="internet"
)

# Payment Method

payment_filter = st.sidebar.selectbox(
    "Payment Method",
    [
        "All",
        "Electronic check",
        "Credit card",
        "Mailed check",
        "Bank transfer"
    ],
    key="payment"
)


# ====================================
# START WITH ALL CUSTOMERS
# ====================================

filtered_df = results.copy()


# ====================================
# SENIOR CITIZEN FILTER
# ====================================

if senior_filter == "Yes":

    filtered_df = filtered_df[
        filtered_df["SeniorCitizen"] == 1
    ]

elif senior_filter == "No":

    filtered_df = filtered_df[
        filtered_df["SeniorCitizen"] == 0
    ]


# ====================================
# CONTRACT FILTER
# ====================================

if contract_filter == "One year":

    filtered_df = filtered_df[
        filtered_df["Contract_One year"] == 1
    ]

elif contract_filter == "Two year":

    filtered_df = filtered_df[
        filtered_df["Contract_Two year"] == 1
    ]

elif contract_filter == "Month-to-month":

    filtered_df = filtered_df[

        (filtered_df["Contract_One year"] == 0)

        &

        (filtered_df["Contract_Two year"] == 0)

    ]


# ====================================
# INTERNET FILTER
# ====================================

if internet_filter == "Fiber optic":

    filtered_df = filtered_df[
        filtered_df["InternetService_Fiber optic"] == 1
    ]

elif internet_filter == "DSL":

    filtered_df = filtered_df[

        (filtered_df["InternetService_Fiber optic"] == 0)

        &

        (filtered_df["InternetService_No"] == 0)

    ]

elif internet_filter == "No":

    filtered_df = filtered_df[
        filtered_df["InternetService_No"] == 1
    ]


# ====================================
# PAYMENT FILTER
# ====================================

if payment_filter == "Electronic check":

    filtered_df = filtered_df[
        filtered_df["PaymentMethod_Electronic check"] == 1
    ]

elif payment_filter == "Credit card":

    filtered_df = filtered_df[
        filtered_df["PaymentMethod_Credit card (automatic)"] == 1
    ]

elif payment_filter == "Mailed check":

    filtered_df = filtered_df[
        filtered_df["PaymentMethod_Mailed check"] == 1
    ]

elif payment_filter == "Bank transfer":

    # Since your dataset doesn't contain
    # PaymentMethod_Bank transfer (automatic)

    filtered_df = filtered_df[

        (filtered_df["PaymentMethod_Credit card (automatic)"] == 0)

        &

        (filtered_df["PaymentMethod_Electronic check"] == 0)

        &

        (filtered_df["PaymentMethod_Mailed check"] == 0)

    ]


# ====================================
# CREATE HIGH RISK CUSTOMERS
# ====================================

high_risk_df = filtered_df.query(
    "risk_tier == 'High'"
).copy()


# ====================================
# SIDEBAR RISK LEGEND
# ====================================

st.sidebar.markdown("---")

st.sidebar.markdown("**Risk tiers**")

st.sidebar.markdown("🔴 HIGH: ≥ 70%")

st.sidebar.markdown("🟡 MEDIUM: 40–70%")

st.sidebar.markdown("🟢 LOW: < 40%")


# ── KPI metrics ────────────────────────────────────────────
y_pred    = (y_proba >= threshold).astype(int)
flagged = len(filtered_df)
high_risk = len(high_risk_df)
from sklearn.metrics import roc_auc_score, f1_score
auc_score = roc_auc_score(y_test, y_proba)
f1        = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC",         f"{auc_score:.3f}", "↑ Target: 0.85+")
col2.metric("F1 Score",        f"{f1:.3f}",        f"threshold={threshold}")
col3.metric(
    "Customers flagged",

    flagged,

    f"of {len(filtered_df)} shown"
)
col4.metric("High-risk (≥70%)", f"{high_risk}",    "immediate action")

impact = calculate_business_impact(filtered_df)

st.subheader(

    "💰 Business Impact"

)

b1,b2,b3 = st.columns(3)

b1.metric(

    "Total Customers",

    impact["total_customers"]

)

b2.metric(

    "High Risk Customers",

    impact["high_risk_customers"]

)

if impact["total_customers"] > 0:

    high_risk_rate = (

        impact["high_risk_customers"]

        / impact["total_customers"]

    )

else:

    high_risk_rate = 0

b3.metric(
    "High-Risk Rate",
    f"{high_risk_rate:.1%}"
)
st.markdown("---")

# ── Score distribution ─────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:

    st.subheader("Churn probability distribution")

    fig = go.Figure()

    retained = filtered_df[
        filtered_df["actual_churn"] == 0
    ]

    churned = filtered_df[
        filtered_df["actual_churn"] == 1
    ]

    fig.add_trace(

        go.Histogram(

            x=retained["churn_probability"],

            name="Retained",

            marker_color="#4A90D9",

            opacity=0.7,

            nbinsx=40

        )

    )

    fig.add_trace(

        go.Histogram(

            x=churned["churn_probability"],

            name="Churned",

            marker_color="#E05C5C",

            opacity=0.7,

            nbinsx=40

        )

    )

    fig.add_vline(

        x=threshold,

        line_dash="dash",

        line_color="orange",

        annotation_text=f"threshold={threshold}"

    )

    fig.update_layout(

        barmode="overlay",

        height=320,

        xaxis_title="Churn probability",

        yaxis_title="Count",

        legend_title="Actual"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

with col_b:
    st.subheader("Risk tier breakdown")
    tier_counts = (

    filtered_df["risk_tier"]

    .value_counts()

    .reset_index()

)

    tier_counts.columns = [

        "Tier",

        "Count"

    ]
    
    fig2 = px.pie(tier_counts, values='Count', names='Tier',
                  color='Tier',
                  color_discrete_map={'Low': '#5CB85C',
                                      'Medium': '#F0AD4E',
                                      'High': '#D9534F'})
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

# ── SHAP feature importance ────────────────────────────────
# ── Feature Importance ────────────────────────────────
st.markdown("---")

st.subheader("🔍 Top Churn Drivers")

top_features = get_top_features(
    model,
    filtered_df.drop(
        columns=[
            "churn_probability",
            "actual_churn",
            "risk_tier",
            "recommendation"
        ],
        errors="ignore"
    )
)

st.dataframe(

    top_features.head(10),

    use_container_width=True

)

fig = px.bar(

    top_features.head(10),

    x="Importance",

    y="Feature",

    orientation="h"

)

fig.update_layout(

    height=500

)

st.plotly_chart(

    fig,

    use_container_width=True

)
# ── High-risk customer table ───────────────────────────────
st.markdown("---")
st.subheader("🔴 Top high-risk customers")

top_n = st.slider("Show top N customers", 5, 50, 20)

top_risk = (

    high_risk_df

    .sort_values(

        "churn_probability",

        ascending=False

    )

    .head(top_n)

    [[

    "tenure",

    "MonthlyCharges",

    "is_monthly",

    "no_support",

    "service_count",

    "churn_probability",

    "risk_tier",

    "recommendation"

    ]]

)

def colour_risk(val):
    if val == 'High':   return 'background-color: #ffe0e0'
    if val == 'Medium': return 'background-color: #fff5cc'
    return ''

display_cols = [
    "tenure",
    "MonthlyCharges",
    "churn_probability",
    "recommendation"
]

display_df = top_risk[display_cols].copy()

display_df["churn_probability"] = (
    display_df["churn_probability"] * 100
).round(1)

display_df["churn_probability"] = (
    display_df["churn_probability"].astype(str) + "%"
)

if display_df.empty:

    st.warning(
        "No customers match the selected filters."
    )

else:

    st.dataframe(

        display_df,

        use_container_width=True

    )

    csv = display_df.to_csv(
        index=False
    )

    st.download_button(

        "📥 Download High-Risk Customers",

        data=csv,

        file_name="high_risk_customers.csv",

        mime="text/csv"

    )


# ── Individual prediction explorer ────────────────────────
# ── Individual prediction explorer ──

st.markdown("---")

st.subheader("🧪 Individual customer explorer")

if filtered_df.empty:

    st.warning(
        "No customers available for the selected filters."
    )

else:

    customer_pool = filtered_df.reset_index()

    idx = st.number_input(

        "Customer index",

        0,

        len(customer_pool)-1,

        0,

        1

    )

    cust = customer_pool.iloc[idx]

    prob = cust["churn_probability"]

    actual = cust["actual_churn"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Churn probability",
        f"{prob:.1%}"
    )

    c2.metric(
        "Predicted",
        "Churn ⚠️"
        if prob >= threshold
        else "Retain ✅"
    )

    c3.metric(
        "Actual",
        "Churned 🔴"
        if actual == 1
        else "Retained 🟢"
    )

    feature_cols = X_test.columns

    with st.expander(

        "Show all features for this customer"

    ):

        st.dataframe(

            cust[feature_cols]

            .to_frame()

            .T,

            use_container_width=True

        )

st.markdown("---")

st.caption(
    "Built by Kashvi Bhardwaj | XGBoost | Streamlit | SHAP | IBM Telco Dataset"
)