import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import API_PORT

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Customer Churn Intelligence")
st.caption("Real-time churn prediction powered by XGBoost + SHAP")

# ── Sidebar — customer profile ───────────────────────────────
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 3)
monthly = st.sidebar.number_input(
    "Monthly Charges ($)", 20.0, 150.0, 89.5, step=0.5)
contract = st.sidebar.selectbox(
    "Contract type",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Month-to-month",
        1: "One year",
        2: "Two year"
    }[x]
)
internet = st.sidebar.selectbox(
    "Internet Service",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "No internet",
        1: "DSL",
        2: "Fiber optic"
    }[x]
)
num_services = st.sidebar.slider("Number of services", 0, 8, 2)
num_protective = st.sidebar.slider(
    "Protective services (security/support)", 0, 4, 0)
payment = st.sidebar.selectbox(
    "Payment method",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Electronic check",
        1: "Mailed check",
        2: "Bank transfer",
        3: "Credit card"
    }[x]
)
senior = st.sidebar.selectbox(
    "Senior citizen", options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# ── Build payload ────────────────────────────────────────────
payload = {
    "tenure":                tenure,
    "MonthlyCharges":        monthly,
    "TotalCharges":          round(monthly * tenure, 2),
    "Contract":              contract,
    "InternetService":       internet,
    "PaymentMethod":         payment,
    "NumServices":           num_services,
    "NumProtectiveServices": num_protective,
    "IsNewCustomer":         int(tenure < 12),
    "SeniorCitizen":         senior
}

# ── Call API ─────────────────────────────────────────────────
try:
    resp = requests.post(
        f"http://localhost:{API_PORT}/predict",
        json=payload,
        params={"customer_id": "dashboard-user"},
        timeout=5
    ).json()
    api_ok = True
except Exception as e:
    api_ok = False
    st.error(f"API not reachable: {e}. Is api/main.py running?")
    st.stop()

prob  = resp["churn_probability"]
risk  = resp["risk_level"]
pred  = resp["churn_prediction"]
reasons = resp["top_churn_reasons"]

# ── Risk colour ──────────────────────────────────────────────
risk_color = {
    "high":   "#D85A30",
    "medium": "#BA7517",
    "low":    "#1D9E75"
}[risk]

# ── Top metrics row ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Churn Probability",
        f"{prob:.1%}",
        delta=f"{prob - 0.265:+.1%} vs avg",
        delta_color="inverse"
    )
with col2:
    st.metric("Risk Level", risk.upper())
with col3:
    st.metric(
        "Prediction",
        "Will Churn" if pred else "Will Stay"
    )
with col4:
    st.metric("Tenure", f"{tenure} months")

st.divider()

# ── Main content: two columns ────────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    # Probability gauge
    st.subheader("Churn risk meter")
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh([""], [1], color="#F1EFE8", height=0.5)
    ax.barh([""], [prob], color=risk_color, height=0.5)
    ax.axvline(x=0.265, color="#888780",
               linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.265, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "Avg\n26.5%", "50%", "75%", "100%"],
                       fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(prob, 0, f" {prob:.1%}",
            va="center", fontsize=11,
            fontweight="500", color=risk_color)
    st.pyplot(fig)
    plt.close()

    # SHAP reasons
    st.subheader("Why this prediction?")
    for i, reason in enumerate(reasons, 1):
        direction = "increases" in reason
        color = "#D85A30" if direction else "#1D9E75"
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;"
            f"border-left:3px solid {color};"
            f"border-radius:0 8px 8px 0;"
            f"font-size:14px'>"
            f"{i}. {reason}</div>",
            unsafe_allow_html=True
        )

with right:
    # ── What-if simulator ────────────────────────────────────
    st.subheader("What-if simulator")
    st.caption("Change an intervention — see the impact instantly")

    new_contract = st.selectbox(
        "Offer this contract",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Month-to-month (no change)",
            1: "One year",
            2: "Two year"
        }[x],
        index=contract
    )
    new_services = st.slider(
        "Bundle protective services",
        0, 4, num_protective
    )
    new_monthly = st.number_input(
        "Offer discounted monthly rate ($)",
        20.0, 150.0, float(monthly), step=1.0
    )

    # Build what-if payload
    whatif_payload = {
        **payload,
        "Contract":              new_contract,
        "NumProtectiveServices": new_services,
        "MonthlyCharges":        new_monthly,
        "TotalCharges":          round(new_monthly * tenure, 2)
    }

    whatif_resp = requests.post(
        f"http://localhost:{API_PORT}/predict",
        json=whatif_payload,
        params={"customer_id": "whatif"},
        timeout=5
    ).json()

    new_prob    = whatif_resp["churn_probability"]
    delta       = prob - new_prob
    new_risk    = whatif_resp["risk_level"]

    # Result card
    if delta > 0.01:
        st.success(
            f"Risk drops {prob:.1%} → {new_prob:.1%}  "
            f"({delta:.1%} reduction)"
        )
    elif delta < -0.01:
        st.warning(
            f"Risk increases {prob:.1%} → {new_prob:.1%}  "
            f"({abs(delta):.1%} increase)"
        )
    else:
        st.info("No significant change with this intervention.")

    # Before/after bar
    fig2, ax2 = plt.subplots(figsize=(5, 2.5))
    bars = ax2.bar(
        ["Current risk", "After intervention"],
        [prob, new_prob],
        color=[risk_color,
               {"high":"#D85A30","medium":"#BA7517",
                "low":"#1D9E75"}[new_risk]],
        width=0.5
    )
    ax2.bar_label(bars,
                  labels=[f"{prob:.1%}", f"{new_prob:.1%}"],
                  padding=4, fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Churn probability")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    st.pyplot(fig2)
    plt.close()

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost | AUC-ROC: 0.87 | "
    "Trained on Telco Customer Churn dataset | "
    "Explanations by SHAP TreeExplainer"
)