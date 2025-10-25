import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
st.set_page_config(page_title="Expresso Churn Prediction", layout="wide")
st.title("EXPRESSO CHURN PREDICTION")
st.subheader("Predicting customer churn for Expresso using machine learning")

st.sidebar.title("About 🎯")
st.sidebar.info(f"This application predicts customer churn(stop using this service) for Expresso using a trained machine learning model.")
st.sidebar.title("Contact Information 📧")
st.sidebar.info("Email ✉️: chizeyfrancis@gmail.com")
st.sidebar.info("Phone 📞: +234 8135874318")
st.sidebar.title("Help 🆘")
st.sidebar.info("For any assistance or enquirery, kindly refer to the documentation or contact support.")

st.sidebar.title("Dataset Summary 📊")
st.sidebar.info("The dataset contains customer information such as region, tenure, revenue, and usage patterns.")

# Feature description
st.sidebar.title("Feature Description 📋")
st.sidebar.info("""
- REGION: Customer's region
- TENURE: Duration of customer relationship
- MONTANT: Amount spent
- FREQUENCE_RECH: Recharge frequency
- REVENUE: Revenue generated
- ARPU_SEGMENT: ARPU segment
- FREQUENCE: Frequency of usage
- DATA_VOLUME: Data volume used
- ON_NET: On-net calls
- ORANGE: Calls to Orange network
- TIGO: Calls to Tigo network
- ZONE1: Calls to Zone 1
- ZONE2: Calls to Zone 2
- MRG: Margin
- REGULARITY: Regularity of usage
- TOP_PACK: Top pack used
- FREQ_TOP_PACK: Frequency of top pack usage
""")

col1, col2, col3 = st.columns(3)

with col1:
    REGION = st.text_input("REGION")
    TENURE = st.number_input("TENURE", min_value=0)
    MONTANT = st.number_input("MONTANT", min_value=0.0)
    FREQUENCE_RECH = st.number_input("FREQUENCE_RECH", min_value=0)
    REVENUE = st.number_input("REVENUE", min_value=0.0)
    ARPU_SEGMENT = st.text_input("ARPU_SEGMENT")

with col2:
    FREQUENCE = st.number_input("FREQUENCE", min_value=0)
    DATA_VOLUME = st.number_input("DATA_VOLUME", min_value=0.0)
    ON_NET = st.number_input("ON_NET", min_value=0)
    ORANGE = st.number_input("ORANGE", min_value=0)
    TIGO = st.number_input("TIGO", min_value=0)
    ZONE1 = st.number_input("ZONE1", min_value=0)

with col3:
    ZONE2 = st.number_input("ZONE2", min_value=0)
    MRG = st.number_input("MRG", min_value=0)
    REGULARITY = st.number_input("REGULARITY", min_value=0)
    TOP_PACK = st.text_input("TOP_PACK")
    FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", min_value=0)

st.button("Predict")
if not REGION or not ARPU_SEGMENT or not TOP_PACK:
    st.error("Please fill out all required fields.")
else:
    try:
    # Create a DataFrame for the input values
        input_data = pd.DataFrame({
        "REGION": [REGION],
        "TENURE": [TENURE],
        "MONTANT": [MONTANT],
        "FREQUENCE_RECH": [FREQUENCE_RECH],
        "REVENUE": [REVENUE],
        "ARPU_SEGMENT": [ARPU_SEGMENT],
        "FREQUENCE": [FREQUENCE],
        "DATA_VOLUME": [DATA_VOLUME],
        "ON_NET": [ON_NET],
        "ORANGE": [ORANGE],
        "TIGO": [TIGO],
        "ZONE1": [ZONE1],
        "ZONE2": [ZONE2],
        "MRG": [MRG],
        "REGULARITY": [REGULARITY],
        "TOP_PACK": [TOP_PACK],
        "FREQ_TOP_PACK": [FREQ_TOP_PACK]
        })    

            # Make prediction
        prediction = model.predict(input_data)
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

            # Model explanation
        st.subheader("Model Explanation")
        st.write("The model uses the input features to predict whether a customer will churn or not. The prediction is based on patterns learned from historical data.")

            # Visualization
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        features = input_data.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title("Feature Importance")
        st.pyplot(plt)

    except Exception as e:
            st.error(f"An error occurred: {e}")
            # Documentation, resources, and feedback
st.title("Documentation 📚")
st.info("For detailed documentation, kindly refer to the user manual or visit our website.")

# Sidebar links
st.sidebar.title("Useful Links 🔗")
st.sidebar.markdown("[Streamlit Documentation](https://docs.streamlit.io)")
st.sidebar.markdown("[Machine Learning Tutorials](https://www.kaggle.com/learn/overview)")

st.title("Resources 🔗")
st.info("Check out our resources for more information and tutorials.")

st.title("Feedback 📝")
st.info("We value your feedback! Please let us know how we can improve.")
st.text_area("kindly type in Your Feedback")
st.button("Submit Feedback")

# Footer
st.info("© 2025 De'Expresso. All rights reserved.")