"""
Data preprocessing utilities for YugenAI project
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Create necessary directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

def list_available_datasets() -> List[str]:
    """
    List all available raw datasets in the data/raw directory
    
    Returns:
        List of dataset filenames
    """
    if not RAW_DATA_DIR.exists():
        return []
    return [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]

def get_dataset_info(file_path: str, sample_size: int = 5) -> dict:
    """
    Get basic information about a dataset
    
    Args:
        file_path: Path to the dataset file
        sample_size: Number of rows to include in the sample
        
    Returns:
        Dictionary containing dataset information including a sample
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:  # Excel files
            df = pd.read_excel(file_path)
            
        # Get a sample of the data
        sample = df.head(sample_size) if len(df) > sample_size else df.copy()
        
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'columns_list': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample': sample
        }
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        return {}


def preprocess_housing_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Preprocess housing dataset for model training

    Args:
        input_path: Path to raw housing data CSV file
        output_path: Path to save preprocessed data (optional)

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading housing data from: {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df)} rows with missing values")

    # Target column
    target_col = "median_house_value"

    # Handle categorical columns (ocean_proximity)
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:
        logger.info(f"Applying one-hot encoding to: {categorical_columns}")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Get numerical columns (excluding target)
    numerical_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != target_col
    ]

    # Scale numerical features
    if numerical_columns:
        logger.info(f"Scaling numerical features: {numerical_columns}")
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Ensure target column is the last column
    if target_col in df.columns:
        columns = [col for col in df.columns if col != target_col] + [target_col]
        df = df[columns]

        # Save preprocessed data if output path provided
        saved_path = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with the same extension as input
            if str(input_path).endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
                
            saved_path = str(output_path.absolute())
            logger.info(f"Preprocessed data saved to: {saved_path}")

        return df, saved_path


def preprocess_iris_data(input_path: str, output_path: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess iris dataset for model training

    Args:
        input_path: Path to raw iris data file (CSV or Excel)
        output_path: Path to save preprocessed data (optional)

    Returns:
        Tuple of (preprocessed DataFrame, output file path)
    """
    logger.info(f"Loading iris data from: {input_path}")
    saved_path = None

    try:
        # Load data based on file extension
        if str(input_path).endswith('.csv'):
            df = pd.read_csv(input_path)
        else:  # Assume Excel
            df = pd.read_excel(input_path)
            
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        if initial_rows > len(df):
            logger.info(f"Removed {initial_rows - len(df)} rows with missing values")

        # Encode target variable (Species)
        if "Species" in df.columns:
            label_encoder = LabelEncoder()
            df["Species"] = label_encoder.fit_transform(df["Species"])
            logger.info(f"Encoded target variable with classes: {label_encoder.classes_}")

        # Scale features
        feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        # Only scale columns that exist in the dataframe
        existing_columns = [col for col in feature_columns if col in df.columns]
        if existing_columns:
            scaler = StandardScaler()
            df[existing_columns] = scaler.fit_transform(df[existing_columns])
            logger.info(f"Scaled feature columns: {existing_columns}")

        # Save preprocessed data if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with the same extension as input
            if str(input_path).endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
                
            saved_path = str(output_path.absolute())
            logger.info(f"Preprocessed data saved to: {saved_path}")

    except Exception as e:
        logger.error(f"Error preprocessing iris data: {str(e)}")
        raise

    return df, saved_path


def validate_dataframe(df: pd.DataFrame, expected_columns: list = None) -> bool:
    """
    Validate DataFrame for training

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names

    Returns:
        True if validation passes, False otherwise
    """
    if df.empty:
        logger.error("DataFrame is empty")
        return False

    if df.isnull().any().any():
        logger.error("DataFrame contains missing values")
        return False

    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False

    logger.info("DataFrame validation passed")
    return True
