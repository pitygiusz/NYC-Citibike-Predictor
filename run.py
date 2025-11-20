import argparse
import os
from pipelines.training_pipeline import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="NYC Parking Violations Project Runner")
    
    # Default paths based on the user's workspace structure
    default_violations_path = 'archive/Parking_Violations_Issued_-_Fiscal_Year_2015.csv'
    default_codes_path = 'ParkingViolationCodes.csv'
    
    parser.add_argument('--violations', type=str, default=default_violations_path, 
                        help='Path to the Parking Violations CSV file')
    parser.add_argument('--codes', type=str, default=default_codes_path, 
                        help='Path to the Parking Violation Codes CSV file')
    parser.add_argument('--no-eda', action='store_true', 
                        help='Skip Exploratory Data Analysis')
    parser.add_argument('--tune', action='store_true', 
                        help='Enable hyperparameter tuning (GridSearchCV)')
    parser.add_argument('--output', type=str, default='plots', 
                        help='Directory to save EDA plots')
    
    args = parser.parse_args()
    
    # Construct absolute paths if they are relative
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # If running from project root, base_dir is project root.
    # Adjust if necessary. Assuming run.py is in project root.
    
    # Check if files exist
    if not os.path.exists(args.violations):
        # Try relative to data directory if not found directly
        # But user structure has data in root/archive or root
        print(f"Warning: File {args.violations} not found. Please check the path.")
        # We will let the pipeline fail if it can't find it, or we could try to be smarter here.
        
    run_training_pipeline(
        violations_path=args.violations,
        codes_path=args.codes,
        output_dir=args.output,
        run_eda_flag=not args.no_eda,
        tune_hyperparameters=args.tune
    )

if __name__ == "__main__":
    main()
