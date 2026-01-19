"""
Data Imputer - Handles missing data imputation
Supports multiple imputation strategies: mean, median, forward fill, backward fill, mode, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DataImputer:
    """Handles missing data imputation for healthcare data"""
    
    def __init__(self):
        self.imputation_report = {}
    
    def impute_dataframe(self, df: pd.DataFrame, strategy: str = 'auto', data_type: str = 'generic') -> pd.DataFrame:
        """
        Main imputation function - applies imputation based on strategy
        
        Args:
            df: Input dataframe with missing values
            strategy: Imputation strategy ('auto', 'mean', 'median', 'forward_fill', 'backward_fill', 'mode', 'drop')
            data_type: Type of data for context-aware imputation
        
        Returns:
            Dataframe with imputed values
        """
        logger.info(f"Starting data imputation for {data_type} using {strategy} strategy")
        
        initial_nulls = df.isnull().sum().sum()
        
        if strategy == 'auto':
            df = self._auto_impute(df, data_type)
        elif strategy == 'mean':
            df = self._impute_mean(df)
        elif strategy == 'median':
            df = self._impute_median(df)
        elif strategy == 'forward_fill':
            df = self._impute_forward_fill(df)
        elif strategy == 'backward_fill':
            df = self._impute_backward_fill(df)
        elif strategy == 'mode':
            df = self._impute_mode(df)
        elif strategy == 'drop':
            df = df.dropna()
        
        final_nulls = df.isnull().sum().sum()
        imputed_count = initial_nulls - final_nulls
        
        logger.info(f"Imputation complete: {imputed_count} values imputed, {final_nulls} remaining nulls")
        self.imputation_report[data_type] = {
            'initial_nulls': initial_nulls,
            'final_nulls': final_nulls,
            'imputed_count': imputed_count,
            'strategy': strategy
        }
        
        return df
    
    def _auto_impute(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Automatically select imputation strategy based on column type and data type"""
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            null_pct = df[col].isnull().sum() / len(df) * 100
            
            # Skip columns with >50% missing data
            if null_pct > 50:
                logger.warning(f"Skipping {col}: {null_pct:.1f}% missing data")
                continue
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for skewed distributions (healthcare data often skewed)
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Imputed {col} with median")
            
            # Date columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else pd.Timestamp.now())
                logger.info(f"Imputed {col} with mode date")
            
            # Categorical columns
            else:
                # Use mode (most frequent value)
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    logger.info(f"Imputed {col} with mode")
                else:
                    df[col] = df[col].fillna('Unknown')
                    logger.info(f"Imputed {col} with 'Unknown'")
        
        return df
    
    def _impute_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute numeric columns with mean"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        return df
    
    def _impute_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute numeric columns with median"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        return df
    
    def _impute_forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill - use previous value"""
        return df.fillna(method='ffill')
    
    def _impute_backward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward fill - use next value"""
        return df.fillna(method='bfill')
    
    def _impute_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute all columns with mode (most frequent value)"""
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
        return df
    
    def impute_by_group(self, df: pd.DataFrame, group_col: str, target_col: str, strategy: str = 'median') -> pd.DataFrame:
        """
        Impute missing values within groups
        Useful for patient-level or facility-level imputation
        
        Args:
            df: Input dataframe
            group_col: Column to group by (e.g., 'patient_id', 'facility_id')
            target_col: Column to impute
            strategy: Imputation strategy
        
        Returns:
            Dataframe with group-based imputation
        """
        if strategy == 'median':
            df[target_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.fillna(x.median())
            )
        elif strategy == 'mean':
            df[target_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.fillna(x.mean())
            )
        elif strategy == 'mode':
            df[target_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.iloc[0])
            )
        
        return df
    
    def get_imputation_report(self) -> Dict:
        """Get imputation report"""
        return self.imputation_report
    
    def get_missing_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of missing data"""
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        return missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
