"""Data loading utilities for insurance analytics."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """Load and preprocess insurance claims data."""
    
    def __init__(self, data_path: str = "data/insurance_data.csv"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load insurance data from CSV.
        
        Returns:
            DataFrame with insurance data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
            return self.data
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_path}")
            raise
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: DataFrame to clean (uses self.data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.data.copy() if self.data is not None else self.load_data()
        
        logger.info("Starting data cleaning...")
        
        # Convert date columns
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        
        if 'VehicleIntroDate' in df.columns:
            df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Create derived metrics
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
            df['Margin'] = df['TotalPremium'] - df['TotalClaims']
            
            # Cap LossRatio at reasonable values
            df['LossRatio'] = df['LossRatio'].clip(upper=5)
        
        logger.info("Data cleaning completed")
        return df
    
    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Generate a comprehensive data summary.
        
        Args:
            df: DataFrame to summarize (uses self.data if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            df = self.data if self.data is not None else self.load_data()
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        return summary
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None) -> Tuple[bool, list]:
        """
        Validate data quality and return warnings.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        if df is None:
            df = self.data if self.data is not None else self.load_data()
        
        warnings = []
        
        # Check for missing values in critical columns
        critical_cols = ['TotalPremium', 'TotalClaims', 'Province', 'Gender']
        for col in critical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                warnings.append(f"Missing values in {col}: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)")
        
        # Check for negative premiums
        if 'TotalPremium' in df.columns and (df['TotalPremium'] < 0).any():
            warnings.append(f"Negative TotalPremium values: {(df['TotalPremium'] < 0).sum()}")
        
        # Check for unrealistic claim amounts
        if 'TotalClaims' in df.columns:
            extreme_claims = (df['TotalClaims'] > df['TotalClaims'].quantile(0.99)).sum()
            if extreme_claims > 0:
                warnings.append(f"Extreme claim values (top 1%): {extreme_claims} records")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings