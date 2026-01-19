"""
Data Cleaner - Handles data cleaning operations
Removes duplicates, handles outliers, validates data types, standardizes formats
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and validates healthcare data"""
    
    def __init__(self):
        self.cleaning_report = {}
    
    def clean_dataframe(self, df: pd.DataFrame, data_type: str = "generic") -> pd.DataFrame:
        """
        Main cleaning function - applies all cleaning operations
        
        Args:
            df: Input dataframe
            data_type: Type of data (patients, encounters, claims, etc.)
        
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting data cleaning for {data_type}")
        initial_rows = len(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle data types
        df = self._standardize_data_types(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Standardize formats
        df = self._standardize_formats(df, data_type)
        
        # Validate data
        df = self._validate_data(df, data_type)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Cleaning complete: {removed_rows} rows removed, {final_rows} rows remaining")
        self.cleaning_report[data_type] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'removed_rows': removed_rows
        }
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        
        # Identify primary key columns
        pk_cols = self._identify_primary_keys(df)
        
        if pk_cols:
            df = df.drop_duplicates(subset=pk_cols, keep='first')
        else:
            df = df.drop_duplicates(keep='first')
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def _identify_primary_keys(self, df: pd.DataFrame) -> List[str]:
        """Identify likely primary key columns"""
        pk_candidates = []
        
        for col in df.columns:
            if '_id' in col.lower() or col.lower() in ['id', 'key']:
                if df[col].nunique() == len(df):
                    pk_candidates.append(col)
        
        return pk_candidates[:1] if pk_candidates else []
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types"""
        for col in df.columns:
            # Handle date columns
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            # Handle numeric columns
            elif 'amount' in col.lower() or 'cost' in col.lower() or 'rate' in col.lower():
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            
            # Handle boolean columns
            elif 'flag' in col.lower() or 'is_' in col.lower() or 'has_' in col.lower():
                try:
                    df[col] = df[col].astype(bool)
                except:
                    pass
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip ID columns
            if '_id' in col.lower() or col.lower() == 'id':
                continue
            
            # Use IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            initial_count = len(df)
            df = df[(df[col] >= lower_bound) | (df[col].isna())]
            df = df[(df[col] <= upper_bound) | (df[col].isna())]
            
            removed = initial_count - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} outliers from {col}")
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Standardize data formats based on data type"""
        
        # Standardize ID formats (uppercase, remove spaces)
        id_cols = [col for col in df.columns if '_id' in col.lower()]
        for col in id_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.upper().str.strip()
        
        # Standardize string columns (trim whitespace)
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].str.strip()
        
        # Data type specific formatting
        if data_type == 'patients':
            if 'gender' in df.columns:
                df['gender'] = df['gender'].str.upper()
            if 'race' in df.columns:
                df['race'] = df['race'].str.title()
        
        elif data_type == 'encounters':
            if 'encounter_type' in df.columns:
                df['encounter_type'] = df['encounter_type'].str.upper()
        
        elif data_type == 'claims':
            # Ensure amounts are positive
            amount_cols = [col for col in df.columns if 'amount' in col.lower()]
            for col in amount_cols:
                df[col] = df[col].abs()
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate data based on business rules"""
        
        initial_count = len(df)
        
        # Remove rows with critical null values
        critical_cols = self._get_critical_columns(data_type)
        df = df.dropna(subset=critical_cols, how='any')
        
        # Data type specific validation
        if data_type == 'patients':
            # Age should be between 0 and 120
            if 'age' in df.columns:
                df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        
        elif data_type == 'encounters':
            # Encounter date should be reasonable
            if 'encounter_date' in df.columns:
                df = df[df['encounter_date'] >= pd.Timestamp('2014-01-01')]
        
        elif data_type == 'claims':
            # Claim amounts should be positive
            amount_cols = [col for col in df.columns if 'amount' in col.lower()]
            for col in amount_cols:
                df = df[df[col] >= 0]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows during validation")
        
        return df
    
    def _get_critical_columns(self, data_type: str) -> List[str]:
        """Get critical columns that cannot be null"""
        critical_cols = {
            'patients': ['patient_id'],
            'encounters': ['encounter_id', 'patient_id', 'encounter_date'],
            'claims': ['claim_id', 'patient_id', 'claim_date'],
            'providers': ['provider_id'],
            'facilities': ['facility_id'],
            'registry': ['registry_id', 'patient_id'],
            'cms_measures': ['facility_id', 'measure_code'],
            'hai_data': ['facility_id']
        }
        return critical_cols.get(data_type, [])
    
    def get_cleaning_report(self) -> Dict:
        """Get cleaning report"""
        return self.cleaning_report
