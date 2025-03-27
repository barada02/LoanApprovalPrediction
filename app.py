import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .prediction-approved {
        font-size: 1.8rem;
        color: #4CAF50;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .prediction-rejected {
        font-size: 1.8rem;
        color: #F44336;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(244, 67, 54, 0.1);
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'models/loan_approval_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found. Please run train_model.py first.")
        return None
    return joblib.load(model_path)

# Load feature importances if available
@st.cache_data
def load_feature_importances():
    fi_path = 'models/feature_importances.csv'
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None

# Function to make predictions
def predict_loan_approval(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode loan purpose
    purpose_columns = ['Purpose_Business', 'Purpose_Debt_Consolidation', 'Purpose_Education', 'Purpose_Home', 'Purpose_Personal']
    for col in purpose_columns:
        input_df[col] = 0
    
    # Set the selected purpose to 1
    selected_purpose = f"Purpose_{input_data['Loan_Purpose']}"
    if selected_purpose in purpose_columns:
        input_df[selected_purpose] = 1
    
    # Drop the original Loan_Purpose column
    input_df = input_df.drop('Loan_Purpose', axis=1)
    
    # Make prediction
    probability = model.predict_proba(input_df)[0, 1]
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability

# Function to display feature importance visualization
def display_feature_importance(feature_importances):
    if feature_importances is not None:
        st.subheader("Feature Importance")
        fig = px.bar(
            feature_importances.head(10), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Features Affecting Loan Approval',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Function to display confusion matrix if available
def display_confusion_matrix():
    cm_path = 'models/confusion_matrix.png'
    if os.path.exists(cm_path):
        st.subheader("Model Performance")
        st.image(cm_path, caption="Confusion Matrix of the Model")

# Main function
def main():
    # Load model
    model = load_model()
    if model is None:
        st.warning("Please run the following commands to generate data and train the model:")
        st.code("python data_generation.py\npython train_model.py")
        return
    
    # Load feature importances
    feature_importances = load_feature_importances()
    
    # Sidebar
    st.sidebar.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "This application uses machine learning to predict loan approval based on "
        "applicant information. Enter your details in the form and get an instant prediction."
    )
    
    st.sidebar.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "The prediction model is trained on synthetic data that simulates real loan "
        "applications. The model considers factors such as credit score, income, "
        "loan amount, and other financial indicators to make predictions."
    )
    
    # Main content
    st.markdown("<h1 class='main-header'>Loan Approval Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Fill in your details to check if your loan application would be approved</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Loan Application", "Model Insights"])
    
    with tab1:
        # Create columns for form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Personal Information</h3>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=60000, step=1000)
            employment_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            married = st.selectbox("Marital Status", options=["Single", "Married"], index=0)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        
        with col2:
            st.markdown("<h3>Loan Details</h3>", unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=150000, step=1000)
            loan_term = st.selectbox("Loan Term (Years)", options=[3, 5, 7, 10, 15, 20, 30], index=1)
            loan_purpose = st.selectbox(
                "Loan Purpose", 
                options=["Home", "Education", "Personal", "Debt_Consolidation", "Business"],
                index=0
            )
            
            st.markdown("<h3>Credit Information</h3>", unsafe_allow_html=True)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
            existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=10, value=1)
            debt_to_income = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            payment_history = st.selectbox("Payment History", options=["Poor", "Good"], index=1)
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Income': income,
            'Employment_Years': employment_years,
            'Married': 1 if married == "Married" else 0,
            'Dependents': dependents,
            'Loan_Amount': loan_amount,
            'Loan_Term': loan_term,
            'Loan_Purpose': loan_purpose,
            'Credit_Score': credit_score,
            'Existing_Loans': existing_loans,
            'Debt_to_Income': debt_to_income,
            'Payment_History': 1 if payment_history == "Good" else 0
        }
        
        # Predict button
        if st.button("Predict Loan Approval"):
            prediction, probability = predict_loan_approval(model, input_data)
            
            st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"<div class='prediction-approved'>Congratulations! Your loan is likely to be APPROVED with {probability:.1%} confidence.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-rejected'>We're sorry. Your loan is likely to be REJECTED with {1-probability:.1%} confidence.</div>", unsafe_allow_html=True)
            
            # Display probability gauge
            fig = px.pie(values=[probability, 1-probability], names=['Approval Probability', 'Rejection Probability'],
                         hole=0.7, color_discrete_sequence=['#4CAF50', '#F44336'])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide some explanation
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("<h3>Key Factors Affecting This Decision:</h3>", unsafe_allow_html=True)
            
            factors = []
            if credit_score < 650:
                factors.append("Your credit score is below the recommended threshold.")
            if debt_to_income > 0.4:
                factors.append("Your debt-to-income ratio is higher than recommended.")
            if loan_amount > income * 3:
                factors.append("The requested loan amount is high relative to your income.")
            if existing_loans > 2:
                factors.append("You have multiple existing loans.")
            if employment_years < 2:
                factors.append("Your employment history is relatively short.")
            if payment_history == "Poor":
                factors.append("Your payment history indicates past issues.")
                
            if not factors:
                if prediction == 1:
                    factors.append("Your overall financial profile is strong.")
                else:
                    factors.append("Multiple factors combined led to this decision.")
            
            for factor in factors:
                st.markdown(f"• {factor}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            display_feature_importance(feature_importances)
        
        with col2:
            display_confusion_matrix()
            
            st.subheader("Tips to Improve Approval Chances")
            st.markdown("""
            - **Improve Credit Score**: Pay bills on time and reduce outstanding debt
            - **Lower Debt-to-Income Ratio**: Pay down existing debts or increase income
            - **Stable Employment**: Longer employment history improves chances
            - **Appropriate Loan Amount**: Request an amount appropriate for your income
            - **Good Payment History**: Maintain a good record of on-time payments
            """)

if __name__ == "__main__":
    main()
