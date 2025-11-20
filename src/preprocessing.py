import pandas as pd
import numpy as np

def clean_data(df, fine_amount_df):
    """
    Performs basic data cleaning and merging.
    """
    # Join with fine amounts
    df = df.join(fine_amount_df.set_index('VIOLATION CODE'), on='Violation Code', how='inner')
    
    # Columns to drop
    columns_to_drop = [
        'Street Code1', 'Street Code2', 'Street Code3', 'Violation Location', 'Violation Precinct',
        'Issuer Precinct', 'Issuer Code', 'Issuer Command', 'Issuer Squad', 'Time First Observed',
        'Law Section', 'Issuing Agency',
        'Sub Division', 'Violation Legal Code', 'Days Parking In Effect    ',
        'Unregistered Vehicle?', 'Meter Number',
        'Feet From Curb', 'Violation Post Code', 'Violation Description',
        'No Standing or Stopping Violation', 'Hydrant Violation',
        'Double Parking Violation', 'Latitude', 'Longitude', 'Community Board',
        'Community Council ', 'Census Tract', 'BIN', 'BBL', 'NTA', 'VIOLATION DESCRIPTION'
    ]
    
    # Drop columns if they exist
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    
    return df

def feature_engineering(df):
    """
    Performs feature engineering.
    """
    # Convert 'Violation Time' to standardized datetime format
    vt = df['Violation Time'].str.upper().str.replace('A', 'AM').str.replace('P', 'PM')
    vt = vt.str[:2] + ':' + vt.str[2:4] + vt.str[4:]
    vt = vt.str.replace('^00', '12', regex=True)
    df['Violation Hour'] = pd.to_datetime(vt, format='%I:%M%p', errors='coerce')
    df['Violation_Hour_Num'] = df['Violation Hour'].dt.hour
    
    # Convert 'Issue Date' to datetime
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], format="%m/%d/%Y", errors='coerce')
    
    # Convert 'Vehicle Expiration Date' to datetime
    df['Vehicle Expiration'] = df['Vehicle Expiration Date'].str[:10]
    df['Vehicle Expiration'] = pd.to_datetime(df['Vehicle Expiration'], format="%m/%d/%Y", errors='coerce')
    
    # Add Day of week and Month
    df['Day of week'] = df['Issue Date'].dt.dayofweek # 0=monday, 6=sunday
    df['Month'] = df['Issue Date'].dt.month
    
    # Cyclical features for time
    import numpy as np
    df['Hour_sin'] = np.sin(2 * np.pi * df['Violation_Hour_Num']/24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Violation_Hour_Num']/24.0)
    
    # Extract date components
    df['Vehicle Expiration_year'] = df['Vehicle Expiration'].dt.year
    df['Vehicle Expiration_month'] = df['Vehicle Expiration'].dt.month
    df['Vehicle Expiration_day'] = df['Vehicle Expiration'].dt.day
    
    df['Issue Date_year'] = df['Issue Date'].dt.year
    df['Issue Date_month'] = df['Issue Date'].dt.month
    df['Issue Date_day'] = df['Issue Date'].dt.day
    
    # Create Target Variable: Fine_Category
    # Logic from notebook: bins = [0, 50, 100, np.inf], labels = ['small', 'medium', 'large']
    bins = [0, 50, 100, np.inf]
    labels = ['small', 'medium', 'large']
    df['Fine_Category'] = pd.cut(df['Fine Amount $'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    return df

def preprocess_pipeline(violations_path, codes_path):
    from src.data_loader import load_data
    
    print("Loading data...")
    df, codes = load_data(violations_path, codes_path)
    
    print("Cleaning data...")
    df = clean_data(df, codes)
    
    print("Feature engineering...")
    df = feature_engineering(df)
    
    return df
