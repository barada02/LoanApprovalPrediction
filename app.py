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
    page_icon="ud83dudcb0",
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

# Load data for EDA
@st.cache_data
def load_data_for_eda():
    train_path = 'data/loan_train.csv'
    test_path = 'data/loan_test.csv'
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    all_data = pd.concat([train_data, test_data], axis=0)
    return all_data

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

# Function to display model comparison and insights
def display_model_comparison():
    # Model comparison data
    models = {
        'Logistic Regression': {
            'Accuracy': 0.8450,
            'Precision': 0.8525,
            'Recall': 0.9870,
            'F1': 0.9148,
            'ROC AUC': 0.7986,
            'Description': 'A linear model that estimates the probability of loan approval based on a linear combination of features.',
            'Strengths': ['Simple and interpretable', 'Provides probability estimates', 'Less prone to overfitting with regularization'],
            'Weaknesses': ['May not capture complex non-linear relationships', 'Limited by linearity assumption', 'Feature scaling is important']
        },
        'Random Forest': {
            'Accuracy': 0.8530,
            'Precision': 0.8566,
            'Recall': 0.9917,
            'F1': 0.9192,
            'ROC AUC': 0.8273,
            'Description': 'An ensemble learning method that builds multiple decision trees and merges their predictions.',
            'Strengths': ['Handles non-linear relationships well', 'Provides feature importance scores', 'Robust to outliers and noise', 'Less prone to overfitting'],
            'Weaknesses': ['Less interpretable than linear models', 'Can be computationally intensive', 'May overfit on noisy data if not tuned properly']
        },
        'Gradient Boosting': {
            'Accuracy': 0.8420,
            'Precision': 0.8594,
            'Recall': 0.9715,
            'F1': 0.9120,
            'ROC AUC': 0.8217,
            'Description': 'An ensemble technique that builds trees sequentially, with each tree correcting the errors of its predecessors.',
            'Strengths': ['Often achieves state-of-the-art performance', 'Handles mixed data types well', 'Provides feature importance scores'],
            'Weaknesses': ['More prone to overfitting than Random Forest', 'Requires careful tuning', 'Sequential nature makes it harder to parallelize']
        }
    }
    
    # Create dataframe for metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    comparison_data = []
    
    for model_name, model_info in models.items():
        model_metrics = {metric: model_info[metric] for metric in metrics}
        model_metrics['Model'] = model_name
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display model comparison
    st.subheader("Model Comparison")
    
    # Metrics comparison chart
    fig = px.bar(
        comparison_df, 
        x='Model', 
        y=metrics,
        barmode='group',
        title='Performance Metrics Across Models',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for model comparison
    fig = px.line_polar(
        comparison_df.melt(id_vars='Model', value_vars=metrics), 
        r='value',
        theta='variable',
        line_close=True,
        color='Model',
        title='Model Performance Comparison (Radar Chart)',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.subheader("Model Details")
    
    tabs = st.tabs(list(models.keys()))
    
    for i, (model_name, model_info) in enumerate(models.items()):
        with tabs[i]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Description**: {model_info['Description']}")
                
                st.markdown("**Strengths**:")
                for strength in model_info['Strengths']:
                    st.markdown(f"- {strength}")
                
                st.markdown("**Weaknesses**:")
                for weakness in model_info['Weaknesses']:
                    st.markdown(f"- {weakness}")
            
            with col2:
                # Create a DataFrame for the metrics
                metrics_df = pd.DataFrame({
                    'Metric': metrics,
                    'Value': [model_info[metric] for metric in metrics]
                })
                
                # Plot the metrics
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    title=f'{model_name} Performance Metrics',
                    color='Value',
                    color_continuous_scale='Blues',
                    text_auto='.3f'
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
    
    # Selected model explanation
    st.subheader("Selected Model: Random Forest")
    st.info(
        "The Random Forest model was selected as the best model based on its ROC AUC score (0.8273), which was the highest among all models. "
        "It achieved the best overall balance of precision and recall, with particularly strong recall (99.17%), which is important for not missing good applicants. "
        "The model also provides feature importance scores that help understand which factors most strongly influence loan approval decisions."
    )

# Function to display data insights and EDA
def display_data_insights(data):
    if data is None:
        st.warning("Data files not found. Please run data_generation.py first.")
        return
    
    st.subheader("Dataset Overview")
    st.write(f"Total samples: {len(data)}")
    st.write(f"Approval rate: {data['Loan_Approval'].mean():.2%}")
    
    # Data summary
    with st.expander("View Data Summary"):
        st.dataframe(data.describe())
    
    # Sample data
    with st.expander("View Sample Data"):
        st.dataframe(data.head(10))
    
    # Distribution of loan approval
    st.subheader("Loan Approval Distribution")
    fig = px.pie(
        values=[data['Loan_Approval'].sum(), len(data) - data['Loan_Approval'].sum()],
        names=['Approved', 'Rejected'],
        title='Loan Approval Distribution',
        color_discrete_sequence=['#4CAF50', '#F44336'],
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Numeric features
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        fig = px.histogram(
            data, 
            x="Credit_Score", 
            color="Loan_Approval",
            color_discrete_map={0: '#F44336', 1: '#4CAF50'},
            marginal="box",
            title="Credit Score Distribution",
            labels={"Loan_Approval": "Loan Approved"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Income Distribution
        fig = px.histogram(
            data, 
            x="Income", 
            color="Loan_Approval",
            color_discrete_map={0: '#F44336', 1: '#4CAF50'},
            marginal="box",
            title="Income Distribution",
            labels={"Loan_Approval": "Loan Approved"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loan Amount Distribution
        fig = px.histogram(
            data, 
            x="Loan_Amount", 
            color="Loan_Approval",
            color_discrete_map={0: '#F44336', 1: '#4CAF50'},
            marginal="box",
            title="Loan Amount Distribution",
            labels={"Loan_Approval": "Loan Approved"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Debt to Income Distribution
        fig = px.histogram(
            data, 
            x="Debt_to_Income", 
            color="Loan_Approval",
            color_discrete_map={0: '#F44336', 1: '#4CAF50'},
            marginal="box",
            title="Debt-to-Income Ratio Distribution",
            labels={"Loan_Approval": "Loan Approved"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features
    st.subheader("Categorical Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loan Purpose
        purpose_counts = data.filter(like='Purpose_').sum().reset_index()
        purpose_counts.columns = ['Loan Purpose', 'Count']
        purpose_counts['Loan Purpose'] = purpose_counts['Loan Purpose'].str.replace('Purpose_', '')
        
        fig = px.bar(
            purpose_counts,
            x='Loan Purpose',
            y='Count',
            title='Loan Purpose Distribution',
            color='Count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Marital Status
        married_approval = data.groupby('Married')['Loan_Approval'].mean().reset_index()
        married_approval['Marital Status'] = married_approval['Married'].map({0: 'Single', 1: 'Married'})
        
        fig = px.bar(
            married_approval,
            x='Marital Status',
            y='Loan_Approval',
            title='Approval Rate by Marital Status',
            color='Loan_Approval',
            color_continuous_scale='Viridis',
            labels={"Loan_Approval": "Approval Rate"}
        )
        st.plotly_chart(fig, use_container_width=True)

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
    
    # Load data for EDA
    data = load_data_for_eda()
    
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
    tab1, tab2, tab3 = st.tabs(["Loan Application", "Data Insights", "Model Insights"])
    
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
                st.markdown(f"u2022 {factor}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Display data insights and EDA
        display_data_insights(data)
    
    with tab3:
        st.subheader("Model Insights and Comparison")
        st.markdown(
            "This section provides insights into the machine learning models used for loan approval prediction. "
            "We compare three different models: Logistic Regression, Random Forest, and Gradient Boosting."
        )
        
        # Display model comparison
        display_model_comparison()
        
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
