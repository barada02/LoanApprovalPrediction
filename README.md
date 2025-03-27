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

## Machine Learning Models

The system trains and evaluates three different classification models:
1. **Logistic Regression**: A linear model for binary classification
2. **Random Forest**: An ensemble method using multiple decision trees (best performer)
3. **Gradient Boosting**: A sequential ensemble technique

The best model is selected based on ROC AUC score and saved for use in the application.

## Repository Structure

```
├── app.py                    # Streamlit frontend application
├── data_generation.py        # Script to generate synthetic loan data
├── train_model.py            # Model training and evaluation script
├── model_documentation.md    # Detailed documentation of ML approach
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview (this file)
├── data/                     # Directory for generated data
│   ├── loan_train.csv        # Training dataset
│   └── loan_test.csv         # Testing dataset
└── models/                   # Directory for trained models
    ├── loan_approval_model.pkl  # Saved model (excluded from git)
    ├── confusion_matrix.png     # Model evaluation visualization
    └── feature_importances.csv  # Feature importance scores
```

## Version Control Notes

This project uses `.gitignore` to exclude certain files from version control:

- **Trained Models**: Model files (*.pkl, *.joblib) are excluded due to their size. Run `train_model.py` to generate these files locally.
- **Virtual Environments**: venv/, env/ directories are excluded
- **Python Bytecode**: __pycache__/ and *.pyc files are excluded
- **IDE Files**: .vscode/, .idea/ and similar IDE-specific directories are excluded

## Detailed Documentation

For more detailed information about the machine learning approach, model performance, and technical implementation, see [model_documentation.md](model_documentation.md).
