import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_preprocessing_pipeline(csv_path, models_dir):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # 2 & 3. Handle missing values and duplicates
    # Removing duplicates
    df = df.drop_duplicates()
    
    # 1. Remove identifiers
    identifiers = ['UDI', 'Product ID']
    df = df.drop(columns=[col for col in identifiers if col in df.columns], errors='ignore')
    
    # 5, 6, 7. Separate X and y, drop leakage columns
    target_col = 'Target'
    # Failure Type is the main leakage column in this dataset format, but we will exclude any known Kaggle original labels as well just in case
    leakage_cols = ['Failure Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    y = df[target_col]
    X = df.drop(columns=[target_col] + [col for col in leakage_cols if col in df.columns], errors='ignore')
    
    # 9, 10. Split into train, val, test (70% train, 15% val, 15% test) using stratified splitting
    # First split to hold out 15% for testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Then split the remaining 85% into train (70%) and val (15%)
    # val_fraction = 15 / 85
    val_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=42, stratify=y_train_val
    )
    
    # Define features
    categorical_features = ['Type']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    # 4 & 8. Preprocessing steps (Encode categories, scale numerics)
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 11. Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = numeric_features + list(cat_feature_names)
    
    # 12. Save the preprocessor
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    joblib.dump(final_feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    
    return (X_train_processed, X_val_processed, X_test_processed, 
            y_train, y_val, y_test, final_feature_names)

def preprocess_new_data(X_new, models_dir='models'):
    """
    Reusable function for future predictions.
    Loads the saved preprocessor and transforms new data.
    """
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    # Ensure identifiers/leakage are dropped if passed
    identifiers = ['UDI', 'Product ID']
    leakage_cols = ['Failure Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X_clean = X_new.drop(columns=[col for col in identifiers + leakage_cols if col in X_new.columns], errors='ignore')
    
    X_processed = preprocessor.transform(X_clean)
    return X_processed

def generate_notebook():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Predictive Maintenance Data Preprocessing\n",
                    "This notebook implements the preprocessing pipeline for the AI4I dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import joblib\n",
                    "import os\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "\n",
                    "df = pd.read_csv('../dataset/predictive_maintenance.csv')\n",
                    "df = df.drop_duplicates()\n",
                    "print(f'Shape after dropping duplicates: {df.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Drop identifiers and leakage columns\n",
                    "identifiers = ['UDI', 'Product ID']\n",
                    "leakage_cols = ['Failure Type']\n",
                    "target_col = 'Target'\n",
                    "\n",
                    "X = df.drop(columns=identifiers + leakage_cols + [target_col], errors='ignore')\n",
                    "y = df[target_col]\n",
                    "print('Features shape:', X.shape)\n",
                    "X.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Stratified Train, Validation, Test Split (70%, 15%, 15%)\n",
                    "X_train_val, X_test, y_train_val, y_test = train_test_split(\n",
                    "    X, y, test_size=0.15, random_state=42, stratify=y\n",
                    ")\n",
                    "\n",
                    "val_fraction = 0.15 / 0.85\n",
                    "X_train, X_val, y_train, y_val = train_test_split(\n",
                    "    X_train_val, y_train_val, test_size=val_fraction, random_state=42, stratify=y_train_val\n",
                    ")\n",
                    "print(f'X_train shape: {X_train.shape}')\n",
                    "print(f'X_val shape: {X_val.shape}')\n",
                    "print(f'X_test shape: {X_test.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Build preprocessing pipeline\n",
                    "categorical_features = ['Type']\n",
                    "numeric_features = [col for col in X.columns if col not in categorical_features]\n",
                    "\n",
                    "preprocessor = ColumnTransformer(\n",
                    "    transformers=[\n",
                    "        ('num', StandardScaler(), numeric_features),\n",
                    "        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)\n",
                    "    ])\n",
                    "\n",
                    "# Fit ONLY on training data\n",
                    "X_train_processed = preprocessor.fit_transform(X_train)\n",
                    "X_val_processed = preprocessor.transform(X_val)\n",
                    "X_test_processed = preprocessor.transform(X_test)\n",
                    "\n",
                    "cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)\n",
                    "final_feature_names = numeric_features + list(cat_names)\n",
                    "print('Final features:', final_feature_names)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save preprocessor for reuse\n",
                    "os.makedirs('../models', exist_ok=True)\n",
                    "joblib.dump(preprocessor, '../models/preprocessor.pkl')\n",
                    "joblib.dump(final_feature_names, '../models/feature_names.pkl')\n",
                    "print('Preprocessor saved successfully.')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/02_Preprocessing.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)

if __name__ == '__main__':
    csv_file = 'dataset/predictive_maintenance.csv'
    models_dir = 'models'
    
    print("Running Preprocessing Pipeline...")
    X_train, X_val, X_test, y_train, y_val, y_test, features = train_preprocessing_pipeline(csv_file, models_dir)
    
    print("\n--- DATA SHAPES ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_validation shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    print("\n--- CLASS DISTRIBUTION ---")
    print("Train set Target distribution:")
    print(y_train.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    print(f"Failures: {y_train.sum()} / {len(y_train)}")
    
    print("\nValidation set Target distribution:")
    print(y_val.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    print(f"Failures: {y_val.sum()} / {len(y_val)}")
    
    print("\nTest set Target distribution:")
    print(y_test.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    print(f"Failures: {y_test.sum()} / {len(y_test)}")
    
    print("\n--- FINAL FEATURE NAMES ---")
    print(features)
    
    print("\nGenerating 02_Preprocessing.ipynb...")
    generate_notebook()
    print("Done.")
