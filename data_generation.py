import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def generate_synthetic_loan_data(n_samples=1000, random_state=42):
    """
    Generate synthetic loan application data for model training.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic loan application data
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        # Personal Information
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.normal(60000, 25000, n_samples),
        'Employment_Years': np.random.normal(8, 5, n_samples),
        'Married': np.random.choice([0, 1], n_samples),
        'Dependents': np.random.choice([0, 1, 2, 3, 4], n_samples),
        
        # Loan Details
        'Loan_Amount': np.random.normal(150000, 75000, n_samples),
        'Loan_Term': np.random.choice([3, 5, 7, 10, 15, 20, 30], n_samples),
        'Loan_Purpose': np.random.choice(['Home', 'Education', 'Personal', 'Debt_Consolidation', 'Business'], n_samples),
        
        # Credit History
        'Credit_Score': np.random.normal(700, 100, n_samples),
        'Existing_Loans': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.07, 0.03]),
        'Debt_to_Income': np.random.normal(0.3, 0.15, n_samples),
        'Payment_History': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),  # 0: Poor, 1: Good
    }
    
    # Convert categorical features to one-hot encoding
    df = pd.DataFrame(data)
    
    # Convert Loan_Purpose to one-hot encoding
    loan_purpose_dummies = pd.get_dummies(df['Loan_Purpose'], prefix='Purpose')
    df = pd.concat([df.drop('Loan_Purpose', axis=1), loan_purpose_dummies], axis=1)
    
    # Apply some constraints to make the data more realistic
    df['Income'] = np.maximum(df['Income'], 0)  # No negative income
    df['Employment_Years'] = np.maximum(df['Employment_Years'], 0)  # No negative employment years
    df['Loan_Amount'] = np.maximum(df['Loan_Amount'], 1000)  # Minimum loan amount
    df['Credit_Score'] = np.clip(df['Credit_Score'], 300, 850)  # Valid credit score range
    df['Debt_to_Income'] = np.clip(df['Debt_to_Income'], 0, 1)  # Valid DTI range
    
    # Generate target variable (Loan_Approval) based on features
    # Higher probability of approval for good credit, higher income, lower debt, etc.
    approval_probability = (
        0.4 +
        0.3 * (df['Credit_Score'] > 700) +
        0.2 * (df['Income'] > 50000) +
        0.2 * (df['Debt_to_Income'] < 0.3) +
        0.1 * (df['Payment_History'] == 1) +
        0.1 * (df['Employment_Years'] > 5) -
        0.2 * (df['Existing_Loans'] > 2) -
        0.1 * (df['Loan_Amount'] > 200000)
    )
    
    # Clip probabilities to [0, 1] range
    approval_probability = np.clip(approval_probability, 0, 1)
    
    # Generate binary approval decisions
    df['Loan_Approval'] = np.random.binomial(1, approval_probability)
    
    return df

def save_data(df, output_dir='data'):
    """
    Save the generated data to CSV files for training and testing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing synthetic loan application data
    output_dir : str
        Directory to save the data files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split data into training and testing sets (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, 'loan_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'loan_test.csv'), index=False)
    
    print(f"Saved {len(train_df)} training samples and {len(test_df)} testing samples.")
    print(f"Approval rate in training data: {train_df['Loan_Approval'].mean():.2f}")
    print(f"Approval rate in testing data: {test_df['Loan_Approval'].mean():.2f}")

def main():
    # Generate synthetic data
    print("Generating synthetic loan application data...")
    loan_data = generate_synthetic_loan_data(n_samples=5000)
    
    # Display data summary
    print("\nData summary:")
    print(f"Total samples: {len(loan_data)}")
    print(f"Approval rate: {loan_data['Loan_Approval'].mean():.2f}")
    print("\nFeature statistics:")
    print(loan_data.describe().round(2))
    
    # Save data to files
    save_data(loan_data)
    
    print("\nData generation complete!")

if __name__ == "__main__":
    main()
