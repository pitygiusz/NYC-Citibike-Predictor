import os
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline
from src.eda import run_eda
from src.modeling import train_model, evaluate_model

def run_training_pipeline(violations_path, codes_path, output_dir='plots', run_eda_flag=True, tune_hyperparameters=False):
    """
    Orchestrates the entire workflow: Data Loading -> Preprocessing -> EDA -> Modeling.
    """
    print("Starting Training Pipeline...")
    
    # 1. Data Loading & Preprocessing
    # preprocess_pipeline handles loading and cleaning internally
    print("Step 1: Loading and Preprocessing Data...")
    df = preprocess_pipeline(violations_path, codes_path)
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    
    # 2. Exploratory Data Analysis
    if run_eda_flag:
        print("Step 2: Running Exploratory Data Analysis...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        run_eda(df, output_dir=output_dir)
        print(f"EDA completed. Plots saved to {output_dir}/")
    else:
        print("Skipping EDA...")
        
    # 3. Modeling
    print("Step 3: Training Model...")
    # Note: train_model handles splitting and encoding
    model, X_test, y_test, label_encoder = train_model(df, target_col='Fine_Category', tune_hyperparameters=tune_hyperparameters)
    print("Model training completed.")
    
    # 4. Evaluation
    print("Step 4: Evaluating Model...")
    evaluate_model(model, X_test, y_test, label_encoder)
    
    print("Pipeline Finished Successfully.")
    return model
