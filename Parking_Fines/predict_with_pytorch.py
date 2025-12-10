"""
Example script for making predictions with trained PyTorch models

This script demonstrates how to load trained models and make predictions
on new parking violation data.
"""

import torch
import pickle
import pandas as pd
import numpy as np
from pytorch_model import FineAmountRegressor, FineCategoryClassifier


def load_models():
    """Load trained models and preprocessor"""
    
    print("Loading models...")
    
    # Load preprocessor
    try:
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("✓ Preprocessor loaded")
    except FileNotFoundError:
        print("Error: Preprocessor not found. Please train models first using train_pytorch_model.py")
        return None, None, None, None
    
    # Load regression model
    try:
        reg_checkpoint = torch.load('models/pytorch_regressor.pth', map_location='cpu')
        reg_model = FineAmountRegressor(input_dim=reg_checkpoint['input_dim'])
        reg_model.load_state_dict(reg_checkpoint['model_state_dict'])
        reg_model.eval()
        print("✓ Regression model loaded")
        print(f"  - MAE: {reg_checkpoint['metrics']['mae']:.2f}")
        print(f"  - RMSE: {reg_checkpoint['metrics']['rmse']:.2f}")
        print(f"  - R²: {reg_checkpoint['metrics']['r2']:.4f}")
    except FileNotFoundError:
        print("Error: Regression model not found. Please train models first.")
        return None, None, None, None
    
    # Load classification model
    try:
        class_checkpoint = torch.load('models/pytorch_classifier.pth', map_location='cpu')
        class_model = FineCategoryClassifier(
            input_dim=class_checkpoint['input_dim'],
            num_classes=class_checkpoint['num_classes']
        )
        class_model.load_state_dict(class_checkpoint['model_state_dict'])
        class_model.eval()
        label_encoder = class_checkpoint['label_encoder']
        print("✓ Classification model loaded")
        print(f"  - Accuracy: {class_checkpoint['metrics']['accuracy']:.4f}")
        print(f"  - F1 Score: {class_checkpoint['metrics']['f1_macro']:.4f}")
    except FileNotFoundError:
        print("Error: Classification model not found. Please train models first.")
        return None, None, None, None
    
    return reg_model, class_model, preprocessor, label_encoder


def create_sample_data():
    """Create sample parking violation data for demonstration"""
    
    # Example parking violation records
    sample_data = pd.DataFrame({
        'Registration State': ['NY', 'NJ', 'NY', 'PA'],
        'Plate Type': ['PAS', 'PAS', 'COM', 'PAS'],
        'Vehicle Body Type': ['SUBN', 'SDN', 'VAN', 'SDN'],
        'Vehicle Make': ['TOYOTA', 'HONDA', 'FORD', 'CHEVR'],
        'Violation County': ['NY', 'NY', 'BX', 'K'],
        'Violation In Front Of Or Opposite': ['F', 'O', 'F', 'F'],
        'Plate ID': ['ABC1234', 'XYZ5678', 'COM999', 'TEST123'],
        'Vehicle Expiration_year': [2015, 2016, 2015, 2015],
        'Vehicle Expiration_month': [12, 6, 9, 3],
        'Vehicle Expiration_day': [31, 15, 1, 20],
        'Vehicle Year': [2012, 2010, 2008, 2013],
        'Hour_sin': [0.5, 0.866, -0.5, 0.0],
        'Hour_cos': [0.866, 0.5, 0.866, 1.0],
        'Issue Date_year': [2015, 2015, 2015, 2015],
        'Issue Date_month': [5, 5, 6, 4],
        'Issue Date_day': [15, 20, 1, 10]
    })
    
    return sample_data


def predict(reg_model, class_model, preprocessor, label_encoder, X):
    """
    Make predictions using trained models
    
    Args:
        reg_model: Trained regression model
        class_model: Trained classification model
        preprocessor: Fitted preprocessor
        label_encoder: Label encoder for classification
        X: Features dataframe
    
    Returns:
        DataFrame with predictions
    """
    print("\nMaking predictions...")
    
    # Preprocess features
    X_processed = preprocessor.transform(X)
    X_tensor = torch.FloatTensor(X_processed)
    
    # Make predictions
    with torch.no_grad():
        # Regression predictions
        fine_amounts = reg_model(X_tensor).numpy()
        
        # Classification predictions
        class_outputs = class_model(X_tensor)
        _, class_indices = torch.max(class_outputs, 1)
        fine_categories = label_encoder.inverse_transform(class_indices.numpy())
    
    # Create results dataframe
    results = X.copy()
    results['Predicted_Fine_Amount'] = fine_amounts
    results['Predicted_Fine_Category'] = fine_categories
    
    return results


def main():
    """Main prediction pipeline"""
    
    print("="*60)
    print("PyTorch Parking Fines Prediction Demo")
    print("="*60)
    
    # Load models
    reg_model, class_model, preprocessor, label_encoder = load_models()
    
    if reg_model is None:
        print("\nPlease train the models first by running:")
        print("  python train_pytorch_model.py")
        return
    
    # Create sample data
    print("\nCreating sample parking violation data...")
    sample_data = create_sample_data()
    print(f"Created {len(sample_data)} sample records")
    
    # Make predictions
    predictions = predict(reg_model, class_model, preprocessor, label_encoder, sample_data)
    
    # Display results
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    
    display_columns = [
        'Registration State', 'Vehicle Make', 'Vehicle Body Type',
        'Predicted_Fine_Amount', 'Predicted_Fine_Category'
    ]
    
    print(predictions[display_columns].to_string(index=False))
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Average predicted fine: ${predictions['Predicted_Fine_Amount'].mean():.2f}")
    print(f"Min predicted fine: ${predictions['Predicted_Fine_Amount'].min():.2f}")
    print(f"Max predicted fine: ${predictions['Predicted_Fine_Amount'].max():.2f}")
    print("\nCategory distribution:")
    print(predictions['Predicted_Fine_Category'].value_counts())


if __name__ == "__main__":
    main()
