import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# NOTE: Ensure to run this from the root folder: `streamlit run src/app.py`
# CONFIG
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "churn_xgb_model.pkl"
model_path = Path(os.getenv("CHURN_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()

if not os.path.exists(model_path):
    st.error(f"Moldel not fdound at: {model_path}")
    st.info("Did you run the 'Save Model' cell in notebook 02?")
    st.stop()

# Load model
model = joblib.load(model_path)

st.set_page_config(page_title="Churn Risk Predictor", layout="centered")

st.title("Customer Retention Predictor")
st.write(
    "Enter customer behavior data to predict if they will **Stay (Retain)** or **Leave (Churn)**."
)

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        days_since = st.number_input("Days Since Last Order", 0, 365, 30)
        total_orders = st.number_input("Total Orders (Lifetime)", 1, 100, 3)
        total_spend = st.number_input("Total Spend ($)", 0.0, 5000.0, 150.0)

    with col2:
        # Calculate AOV automatically
        avg_order = total_spend / total_orders if total_orders > 0 else 0
        st.metric("Avg Order Value", f"${avg_order:.2f}")

        returned = st.number_input("Returned Orders", 0, 50, 0)

        # Calculate return rate automatically
        # NOTE: we don't ask for "return_rate" directly, we calculate it just like the model expects
        return_rate = returned / total_orders if total_orders > 0 else 0
        st.metric("Return Rate", f"{return_rate:.1%}")

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Prepare data (must match the columns X_train exactly)
    input_data = pd.DataFrame(
        {
            "days_since_last_order": [days_since],
            "total_orders": [total_orders],
            "total_spend": [total_spend],
            "avg_order_value": [avg_order],
            "returned_orders": [returned],
            "return_rate": [return_rate],
        }
    )

    # Predict
    # NOTE: Class 1 = Retained; Class 0 = Churned
    prediction = model.predict(input_data)[0]
    prob_retained = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"✅ **Safe!** Probabiliy of Retention: {prob_retained:.1%}")
        st.write("Recommendation: No immediate action needed.")
    else:
        st.error(f"⚠️ **At Risk!** Probability of Retention: {prob_retained:.1%}")
        st.write(f"**Action:** Send 'We Miss You' campaign immediately.")

