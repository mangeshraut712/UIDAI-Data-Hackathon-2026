"""
Data preprocessing utilities for UIDAI Hackathon
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various formats with error handling
    
    Args:
        filepath: Path to data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, **kwargs)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath, **kwargs)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f" Loaded {len(df):,} rows and {len(df.columns)} columns from {filepath}")
        return df
    except Exception as e:
        print(f" Error loading {filepath}: {e}")
        raise


def data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> Dict:
    """
    Generate comprehensive data quality report
    
    Args:
        df: Input dataframe
        name: Dataset name for reporting
        
    Returns:
        Dict: Quality metrics
    """
    report = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_cells': df.isna().sum().sum(),
        'missing_percent': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicates': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
    }
    
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT: {name}")
    print(f"{'='*60}")
    print(f"Rows: {report['rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Memory: {report['memory_mb']:.2f} MB")
    print(f"Missing cells: {report['missing_cells']:,} ({report['missing_percent']:.2f}%)")
    print(f"Duplicates: {report['duplicates']:,}")
    print(f"Numeric columns: {report['numeric_columns']}")
    print(f"Categorical columns: {report['categorical_columns']}")
    print(f"Datetime columns: {report['datetime_columns']}")
    print(f"{'='*60}\n")
    
    return report


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names (lowercase, replace spaces with underscores)
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'report') -> pd.DataFrame:
    """
    Handle missing values with various strategies
    
    Args:
        df: Input dataframe
        strategy: 'report', 'drop', 'fill_zero', 'fill_median', 'fill_mode'
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    if strategy == 'report':
        missing = df.isna().sum()
        if missing.sum() > 0:
            print("\nMissing Values Report:")
            print(missing[missing > 0].sort_values(ascending=False))
        return df
    
    elif strategy == 'drop':
        return df.dropna()
    
    elif strategy == 'fill_zero':
        return df.fillna(0)
    
    elif strategy == 'fill_median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        return df
    
    elif strategy == 'fill_mode':
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        df: Input dataframe
        column: Column name to check
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    n_outliers = outliers.sum()
    
    if n_outliers > 0:
        print(f"\n{column}: Found {n_outliers:,} outliers ({n_outliers/len(df)*100:.2f}%)")
        print(f"  Lower bound: {lower_bound:,.2f}")
        print(f"  Upper bound: {upper_bound:,.2f}")
    
    return outliers


def normalize_state_names(df: pd.DataFrame, state_column: str = 'state') -> pd.DataFrame:
    """
    Normalize state names to fix inconsistencies like:
    - 'WESTBENGAL' -> 'West Bengal'
    - 'Westbengal' -> 'West Bengal'  
    - 'WEST BENGAL' -> 'West Bengal'
    - 'West Bangal' -> 'West Bengal' (typo fix)
    - 'andhra pradesh' -> 'Andhra Pradesh' (case fix)
    - Remove invalid numeric entries like '100000'
    
    Args:
        df: Input dataframe
        state_column: Name of state column (default 'state')
        
    Returns:
        pd.DataFrame: DataFrame with normalized state names
    """
    # State name mapping for normalization (comprehensive)
    STATE_MAPPING = {
        # West Bengal variations (including typos)
        'WESTBENGAL': 'West Bengal',
        'Westbengal': 'West Bengal',
        'WEST BENGAL': 'West Bengal',
        'westbengal': 'West Bengal',
        'West Bangal': 'West Bengal',  # Typo
        'west bengal': 'West Bengal',
        'West  Bengal': 'West Bengal',  # Double space
        
        # Odisha variations
        'ODISHA': 'Odisha',
        'Orissa': 'Odisha',
        'ORISSA': 'Odisha',
        'odisha': 'Odisha',
        
        # Andhra Pradesh variations
        'ANDHRA PRADESH': 'Andhra Pradesh',
        'andhra pradesh': 'Andhra Pradesh',
        'Andhra pradesh': 'Andhra Pradesh',
        'ANDHRAPRADESH': 'Andhra Pradesh',
        
        # Other common variations
        'MAHARASHTRA': 'Maharashtra',
        'maharashtra': 'Maharashtra',
        'KARNATAKA': 'Karnataka',
        'karnataka': 'Karnataka',
        'TAMIL NADU': 'Tamil Nadu',
        'TAMILNADU': 'Tamil Nadu',
        'tamil nadu': 'Tamil Nadu',
        'TELANGANA': 'Telangana',
        'telangana': 'Telangana',
        'KERALA': 'Kerala',
        'kerala': 'Kerala',
        'UTTAR PRADESH': 'Uttar Pradesh',
        'UTTARPRADESH': 'Uttar Pradesh',
        'uttar pradesh': 'Uttar Pradesh',
        'MADHYA PRADESH': 'Madhya Pradesh',
        'MADHYAPRADESH': 'Madhya Pradesh',
        'madhya pradesh': 'Madhya Pradesh',
        'RAJASTHAN': 'Rajasthan',
        'rajasthan': 'Rajasthan',
        'GUJARAT': 'Gujarat',
        'gujarat': 'Gujarat',
        'BIHAR': 'Bihar',
        'bihar': 'Bihar',
        'JHARKHAND': 'Jharkhand',
        'jharkhand': 'Jharkhand',
        'CHHATTISGARH': 'Chhattisgarh',
        'CHATTISGARH': 'Chhattisgarh',
        'chhattisgarh': 'Chhattisgarh',
        'PUNJAB': 'Punjab',
        'punjab': 'Punjab',
        'HARYANA': 'Haryana',
        'haryana': 'Haryana',
        'HIMACHAL PRADESH': 'Himachal Pradesh',
        'himachal pradesh': 'Himachal Pradesh',
        'UTTARAKHAND': 'Uttarakhand',
        'UTTARANCHAL': 'Uttarakhand',
        'uttarakhand': 'Uttarakhand',
        'JAMMU AND KASHMIR': 'Jammu And Kashmir',
        'JAMMU & KASHMIR': 'Jammu And Kashmir',
        'jammu and kashmir': 'Jammu And Kashmir',
        'GOA': 'Goa',
        'goa': 'Goa',
        'ASSAM': 'Assam',
        'assam': 'Assam',
        'TRIPURA': 'Tripura',
        'tripura': 'Tripura',
        'MEGHALAYA': 'Meghalaya',
        'meghalaya': 'Meghalaya',
        'MANIPUR': 'Manipur',
        'manipur': 'Manipur',
        'NAGALAND': 'Nagaland',
        'nagaland': 'Nagaland',
        'MIZORAM': 'Mizoram',
        'mizoram': 'Mizoram',
        'ARUNACHAL PRADESH': 'Arunachal Pradesh',
        'arunachal pradesh': 'Arunachal Pradesh',
        'SIKKIM': 'Sikkim',
        'sikkim': 'Sikkim',
        'DELHI': 'Delhi',
        'delhi': 'Delhi',
        'NCT OF DELHI': 'Delhi',
        'NCT Delhi': 'Delhi',
        'CHANDIGARH': 'Chandigarh',
        'chandigarh': 'Chandigarh',
        'PUDUCHERRY': 'Puducherry',
        'PONDICHERRY': 'Puducherry',
        'puducherry': 'Puducherry',
        'ANDAMAN AND NICOBAR': 'Andaman And Nicobar Islands',
        'ANDAMAN & NICOBAR': 'Andaman And Nicobar Islands',
        'ANDAMAN AND NICOBAR ISLANDS': 'Andaman And Nicobar Islands',
        'LAKSHADWEEP': 'Lakshadweep',
        'lakshadweep': 'Lakshadweep',
        'LADAKH': 'Ladakh',
        'ladakh': 'Ladakh',
        'DADRA AND NAGAR HAVELI': 'Dadra And Nagar Haveli',
        'DAMAN AND DIU': 'Daman And Diu',
        'DAMAN & DIU': 'Daman And Diu',
        'Dadra and Nagar Haveli and Daman and Diu': 'Dadra And Nagar Haveli And Daman And Diu',
    }
    
    # Invalid entries to filter out (numeric codes, etc.)
    INVALID_STATES = ['100000', '0', 'NA', 'N/A', 'None', 'null', '', ' ']
    
    df = df.copy()
    
    # Convert state column to string for safety
    df[state_column] = df[state_column].astype(str)
    
    # Strip whitespace and normalize multiple spaces
    df[state_column] = df[state_column].str.strip()
    df[state_column] = df[state_column].str.replace(r'\s+', ' ', regex=True)
    
    # Filter out invalid entries
    original_len = len(df)
    df = df[~df[state_column].isin(INVALID_STATES)]
    filtered_count = original_len - len(df)
    
    if filtered_count > 0:
        print(f"    Filtered {filtered_count:,} rows with invalid state entries")
    
    # Apply state name mapping
    df[state_column] = df[state_column].replace(STATE_MAPPING)
    
    # Title case for remaining states (basic normalization)
    # Only apply to all-caps entries that weren't in the mapping
    mask = df[state_column].str.isupper()
    df.loc[mask, state_column] = df.loc[mask, state_column].str.title()
    
    # Also fix all-lowercase entries
    mask_lower = df[state_column].str.islower()
    df.loc[mask_lower, state_column] = df.loc[mask_lower, state_column].str.title()
    
    return df


def create_age_groups(ages: pd.Series) -> pd.Series:
    """
    Create age group categories
    
    Args:
        ages: Series of ages
        
    Returns:
        pd.Series: Age group categories
    """
    bins = [0, 5, 17, 25, 35, 45, 55, 65, 100]
    labels = ['0-5', '6-17', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    return pd.cut(ages, bins=bins, labels=labels, include_lowest=True)


def aggregate_by_time(df: pd.DataFrame, date_column: str, freq: str = 'M') -> pd.DataFrame:
    """
    Aggregate data by time period
    
    Args:
        df: Input dataframe
        date_column: Name of date column
        freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
        pd.DataFrame: Aggregated data
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df.groupby(pd.Grouper(key=date_column, freq=freq))


def save_processed_data(df: pd.DataFrame, filepath: str, format: str = 'parquet'):
    """
    Save processed data in efficient format
    
    Args:
        df: DataFrame to save
        filepath: Output filepath
        format: 'parquet', 'csv', or 'excel'
    """
    try:
        if format == 'parquet':
            df.to_parquet(filepath, index=False, compression='gzip')
        elif format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        
        print(f" Saved {len(df):,} rows to {filepath}")
    except Exception as e:
        print(f" Error saving to {filepath}: {e}")
        raise


if __name__ == "__main__":
    print("Data preprocessing utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - load_data()")
    print("  - data_quality_report()")
    print("  - clean_column_names()")
    print("  - handle_missing_values()")
    print("  - detect_outliers_iqr()")
    print("  - normalize_state_names()  # NEW: Fix duplicate state names")
    print("  - create_age_groups()")
    print("  - aggregate_by_time()")
    print("  - save_processed_data()")

