import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def get_feature_columns(df, target_col='Fine_Category'):
    """
    Identifies numerical and categorical columns for the model.
    Excludes target column and other non-feature columns.
    """
    # Exclude target and potential leakage/irrelevant columns
    exclude_cols = [target_col, 'Fine Amount $', 'Summons Number', 'Plate ID', 'Issue Date', 
                    'Violation Time', 'Vehicle Expiration Date', 'Violation Description',
                    'Violation Location', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code',
                    'Issuer Command', 'Issuer Squad', 'Time First Observed', 'Law Section',
                    'Sub Division', 'Violation Legal Code', 'Days Parking In Effect    ',
                    'Unregistered Vehicle?', 'Meter Number', 'Feet From Curb', 'Violation Post Code',
                    'No Standing or Stopping Violation', 'Hydrant Violation', 'Double Parking Violation',
                    'Latitude', 'Longitude', 'Community Board', 'Community Council ', 'Census Tract',
                    'BIN', 'BBL', 'NTA', 'VIOLATION CODE', 'VIOLATION DESCRIPTION', 'ticket_year', 'vehicle_age', 'age_group']
    

    num_cols = ['Vehicle Year', 'Hour_sin', 'Hour_cos', 'Vehicle Expiration_year', 
                'Vehicle Expiration_month', 'Vehicle Expiration_day', 'Issue Date_year', 
                'Issue Date_month', 'Issue Date_day']
    
    cat_cols = ['Registration State', 'Plate Type', 'Vehicle Body Type', 'Vehicle Make', 
                'Violation County', 'Violation In Front Of Or Opposite']
    
    # Ensure these columns exist in df
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    return num_cols, cat_cols

def create_pipeline(num_cols, cat_cols):
    """
    Creates the training pipeline with preprocessing and XGBoost classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='multi:softprob',
            n_jobs=-1,
            random_state=42,
            tree_method='hist' # Faster training
        ))
    ])
    
    return pipeline

def train_model(df, target_col='Fine_Category', test_size=0.3, tune_hyperparameters=False, max_samples=None):
    """
    Trains the model. Optionally performs GridSearchCV.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        test_size: Test set proportion
        tune_hyperparameters: Whether to perform GridSearchCV
        max_samples: If specified, randomly sample this many rows for faster training
    """
    num_cols, cat_cols = get_feature_columns(df, target_col)
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Sample data if max_samples is specified
    if max_samples is not None and len(df) > max_samples:
        print(f"Sampling {max_samples} rows from {len(df)} total rows for faster training...")
        df = df.sample(n=max_samples, random_state=42)
    
    X = df[num_cols + cat_cols]
    y = df[target_col]
    
    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42)
    
    pipeline = create_pipeline(num_cols, cat_cols)
    
    if tune_hyperparameters:
        # Reduced param grid for speed in this refactoring demo
        param_grid = {
            'classifier__max_depth': [6, 8, 10],
            'classifier__learning_rate': [0.05],
            'classifier__n_estimators': [800],
            'classifier__subsample': [0.8],
            'classifier__colsample_bytree': [0.8, 0.9],
            'classifier__min_child_weight': [1, 3]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        model = pipeline
        model.fit(X_train, y_train)
        
    return model, X_test, y_test, le

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the model and prints metrics.
    """
    y_pred = model.predict(X_test)
    
    # Decode labels for reporting if needed, or just use encoded
    # y_test_decoded = label_encoder.inverse_transform(y_test)
    # y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("Model Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return acc, f1
