import streamlit as st
import pandas as pd

from model import train_model


st.set_page_config(
    page_title="PayPredict",
    page_icon="📊"
)

st.title("📊 PayPredict")
st.subheader("AI-Powered Customer Churn & Retention Advisor")

st.write(
    "PayPredict helps businesses identify customers at risk "
    "of churn and take proactive retention actions."
)

DATA_FILE = "customer_churn.csv"


@st.cache_resource
def load_model():
    model, features, accuracy, report = train_model(DATA_FILE)
    return model, features, accuracy, report


model, features, accuracy, report = load_model()


# Model performance
st.divider()
st.subheader("Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)


# Customer selection
st.divider()
st.subheader("Customer Churn Prediction")

data = pd.read_csv(DATA_FILE)

customer_id = st.selectbox(
    "Select a customer",
    data["customerID"].tolist()
)

customer = data[
    data["customerID"] == customer_id
].iloc[0]


if st.button("Predict Churn Risk"):

    st.write("### Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Tenure:** {customer['tenure']} months")
        st.write(f"**Contract:** {customer['Contract']}")

    with col2:
        st.write(
            f"**Monthly Charges:** "
            f"${customer['MonthlyCharges']}"
        )
        st.write(
            f"**Internet Service:** "
            f"{customer['InternetService']}"
        )

    # Prepare customer data
    input_df = pd.DataFrame([customer])

    input_df = input_df.drop(
        ["customerID", "Churn"],
        axis=1
    )

    input_df["TotalCharges"] = pd.to_numeric(
        input_df["TotalCharges"],
        errors="coerce"
    )

    input_df["TotalCharges"] = input_df[
        "TotalCharges"
    ].fillna(0)

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction")

    if probability >= 0.7:

        st.error(
            f"🔴 High Churn Risk — "
            f"{probability * 100:.1f}%"
        )

        recommendation = (
            "Prioritize this customer for retention. "
            "Consider a personalized offer, loyalty benefit, "
            "or proactive customer support."
        )

    elif probability >= 0.4:

        st.warning(
            f"🟡 Medium Churn Risk — "
            f"{probability * 100:.1f}%"
        )

        recommendation = (
            "Monitor this customer and consider targeted "
            "engagement or a personalized promotion."
        )

    else:

        st.success(
            f"🟢 Low Churn Risk — "
            f"{probability * 100:.1f}%"
        )

        recommendation = (
            "The customer currently shows low churn risk. "
            "Focus on continued engagement and upselling."
        )

    st.subheader("💡 Retention Recommendation")

    st.info(recommendation)
