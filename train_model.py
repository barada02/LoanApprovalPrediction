import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def load_data(data_dir='data'):
    """
    Load the training and testing data from CSV files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) - Features and target variables for training and testing
    """
    # Load training and testing data
    train_df = pd.read_csv(os.path.join(data_dir, 'loan_train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'loan_test.csv'))
    
    # Separate features and target
    X_train = train_df.drop('Loan_Approval', axis=1)
    y_train = train_df['Loan_Approval']
    X_test = test_df.drop('Loan_Approval', axis=1)
    y_test = test_df['Loan_Approval']
    
    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print performance metrics.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        True target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot
    if not os.path.exists('models'):
        os.makedirs('models')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    return metrics

def train_and_select_model(X_train, y_train, X_test, y_test):
    """
    Train multiple models, evaluate them, and select the best one.
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : training and testing data
        
    Returns:
    --------
    tuple
        (best_model, feature_importances) - The best model and its feature importances
    """
    # Define models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Create a preprocessing pipeline
    preprocessor = StandardScaler()
    
    # Train and evaluate each model
    best_score = 0
    best_model_name = None
    best_model = None
    results = {}
    
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        
        # Check if this model is the best so far (based on ROC AUC)
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = name
            best_model = pipeline
    
    print(f"\nBest model: {best_model_name} with ROC AUC: {best_score:.4f}")
    
    # Get feature importances (if available)
    feature_importances = None
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.named_steps['classifier'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
        plt.title(f'Top 15 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig('models/feature_importances.png')
        plt.close()
    
    return best_model, feature_importances

def save_model(model, output_dir='models'):
    """
    Save the trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    output_dir : str
        Directory to save the model
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the model
    model_path = os.path.join(output_dir, 'loan_approval_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    # Check if data exists, if not generate it
    if not os.path.exists('data/loan_train.csv'):
        print("Data files not found. Generating synthetic data...")
        from data_generation import main as generate_data
        generate_data()
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # Display data information
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Approval rate (train): {y_train.mean():.2f}")
    print(f"Approval rate (test): {y_test.mean():.2f}")
    
    # Train and select the best model
    best_model, feature_importances = train_and_select_model(X_train, y_train, X_test, y_test)
    
    # Save the model
    save_model(best_model)
    
    # Save feature importances if available
    if feature_importances is not None:
        feature_importances.to_csv('models/feature_importances.csv', index=False)
        print("Feature importances saved to models/feature_importances.csv")
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()
