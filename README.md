# Loan Approval Prediction System

A machine learning project that predicts loan approval based on applicant information. The system uses a trained ML model to make predictions and provides a user-friendly Streamlit interface for input and results.

## Project Components

1. **Data Generation**: Synthetic data generation for model training
2. **Model Training**: ML pipeline for preprocessing and model training
3. **Streamlit Frontend**: User interface for inputting loan application details
4. **Prediction Service**: Backend service to process inputs and return predictions

## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate synthetic data and train the model:
   ```
   python data_generation.py
   python train_model.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Features

- User-friendly form for loan application details
- Instant prediction results
- Visualization of key factors affecting the decision
