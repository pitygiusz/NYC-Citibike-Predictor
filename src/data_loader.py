import pandas as pd
import os

def load_violation_codes(filepath='ParkingViolationCodes.csv'):
    """Loads the parking violation codes data."""
    return pd.read_csv(filepath)

def load_parking_violations(filepath='archive/Parking_Violations_Issued_-_Fiscal_Year_2015.csv'):
    """Loads the parking violations data."""
    # Using pyarrow engine for faster reading if available, otherwise default
    try:
        return pd.read_csv(filepath, engine='pyarrow')
    except ImportError:
        return pd.read_csv(filepath)

def load_data(violations_path, codes_path):
    """Loads both datasets and returns them."""
    violations = load_parking_violations(violations_path)
    codes = load_violation_codes(codes_path)
    return violations, codes
