import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

# Load the trained model and preprocessor
model = tf.keras.models.load_model("friend_lending_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

# Streamlit UI
st.title("Friend Loan Repayment Predictor 💰😆")
st.write("Ever wondered if your friend will return the money? Let's find out!")

# User Inputs
amount_lent = st.number_input("Loan Amount (INR)", min_value=1, step=100)
relationship = st.selectbox("Relationship with Borrower", ["Close Friend", "Family", "Acquaintance"])
repayment_history = st.selectbox("Has the friend repaid past loans?", ["Yes", "No"])
financial_situation = st.selectbox("Friend's Financial Situation", ["Good", "Average", "Poor"])
trust_factor = st.slider("Social Trust Factor (1-5)", 1, 5, 3)

# Calculate Loan Amount to Trust Ratio
loan_trust_ratio = amount_lent / trust_factor

# Define column names (must match training dataset)
columns = ["Amount_Lent_INR", "Relationship_with_Borrower", "Repayment_History", 
           "Borrower_Financial_Situation", "Social_Trust_Factors", "Loan_Amount_to_Trust_Ratio"]
# Predict Button
if st.button("Predict 🚀"):
    # Prepare input data
    #input_data = np.array([[amount_lent, relationship, repayment_history, financial_situation, trust_factor, loan_trust_ratio]])
    input_data = pd.DataFrame([[amount_lent, relationship, repayment_history, 
                                financial_situation, trust_factor, loan_trust_ratio]], 
                              columns=columns)
    # Transform input data using preprocessor
    input_preprocessed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_preprocessed)
    result = "Yes" if prediction[0][0] > 0.5 else "No"

    # Funny Response
    if result == "Yes":
        st.success("✅ Your friend **WILL** return the money! 🎉 Maybe they value the friendship! 😍")
        st.image("yes.png", width=300)
    else:
        st.error("❌ Uh-oh! Your friend **WON'T** return the money! 🤦‍♂️ Maybe start chasing them now! 🏃💨")
        st.image("no.png", width=300)
st.write("Remember, this is just AI, not your friend's **real intentions**! 😂")
