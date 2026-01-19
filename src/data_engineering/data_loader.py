"""
Data Loader - Orchestrates complete data engineering pipeline
Loads, cleans, imputes, joins, and transforms healthcare data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

from .data_cleaner import DataCleaner
from .data_imputer import DataImputer
from .data_joiner import DataJoiner
from .data_transformer import DataTransformer
from .data_quality import DataQualityChecker

logger = logging.getLogger(__name__)


class DataLoader:
    """Orchestrates complete data engineering pipeline"""
    
    def __init__(self, data_dir: str = 'data/landing_zone'):
        self.data_dir = Path(data_dir)
        self.cleaner = DataCleaner()
        self.imputer = DataImputer()
        self.joiner = DataJoiner()
        self.transformer = DataTransformer()
        self.quality_checker = DataQualityChecker()
        
        self.raw_data = {}
        self.cleaned_data = {}
        self.imputed_data = {}
        self.fact_tables = {}
        self.dim_tables = {}
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all healthcare data files
        
        Returns:
            Dictionary with all loaded dataframes
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        data_files = {
            'patients': 'patients.csv',
            'providers': 'providers.csv',
            'facilities': 'facilities.csv',
            'encounters': 'encounters.csv',
            'claims': 'claims.csv',
            'registry': 'registry.csv',
            'cms_measures': 'cms_measures.csv',
            'hai_data': 'hai_data.csv'
        }
        
        for data_type, filename in data_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    self.raw_data[data_type] = df
                    logger.info(f"Loaded {data_type}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {str(e)}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        return self.raw_data
    
    def clean_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Clean all loaded data
        
        Returns:
            Dictionary with cleaned dataframes
        """
        logger.info("Starting data cleaning phase")
        
        for data_type, df in self.raw_data.items():
            try:
                self.cleaned_data[data_type] = self.cleaner.clean_dataframe(df, data_type)
            except Exception as e:
                logger.error(f"Failed to clean {data_type}: {str(e)}")
                self.cleaned_data[data_type] = df
        
        return self.cleaned_data
    
    def impute_all_data(self, strategy: str = 'auto') -> Dict[str, pd.DataFrame]:
        """
        Impute missing values in all data
        
        Args:
            strategy: Imputation strategy
        
        Returns:
            Dictionary with imputed dataframes
        """
        logger.info(f"Starting data imputation phase (strategy: {strategy})")
        
        for data_type, df in self.cleaned_data.items():
            try:
                self.imputed_data[data_type] = self.imputer.impute_dataframe(df, strategy, data_type)
            except Exception as e:
                logger.error(f"Failed to impute {data_type}: {str(e)}")
                self.imputed_data[data_type] = df
        
        return self.imputed_data
    
    def create_fact_and_dimension_tables(self) -> Tuple[Dict, Dict]:
        """
        Create fact and dimension tables through joins
        
        Returns:
            Tuple of (fact_tables, dimension_tables)
        """
        logger.info("Creating fact and dimension tables")
        
        try:
            # Get imputed data
            patients = self.imputed_data.get('patients', pd.DataFrame())
            encounters = self.imputed_data.get('encounters', pd.DataFrame())
            claims = self.imputed_data.get('claims', pd.DataFrame())
            providers = self.imputed_data.get('providers', pd.DataFrame())
            facilities = self.imputed_data.get('facilities', pd.DataFrame())
            
            # Create fact tables
            if not encounters.empty and not patients.empty:
                self.fact_tables['fact_encounters'] = self.joiner.create_fact_encounters(
                    patients, encounters, providers, facilities
                )
            
            if not claims.empty and not patients.empty:
                self.fact_tables['fact_claims'] = self.joiner.create_fact_claims(
                    patients, claims, providers, facilities
                )
            
            # Create dimension tables
            if not patients.empty:
                self.dim_tables['dim_patient'] = self.joiner.create_dim_patient(patients)
            
            if not providers.empty:
                self.dim_tables['dim_provider'] = self.joiner.create_dim_provider(providers)
            
            if not facilities.empty:
                self.dim_tables['dim_facility'] = self.joiner.create_dim_facility(facilities)
            
            if not encounters.empty:
                self.dim_tables['dim_diagnosis'] = self.joiner.create_dim_diagnosis(encounters)
            
        except Exception as e:
            logger.error(f"Failed to create fact/dimension tables: {str(e)}")
        
        return self.fact_tables, self.dim_tables
    
    def transform_data(self) -> Dict[str, pd.DataFrame]:
        """
        Apply transformations and aggregations
        
        Returns:
            Dictionary with transformed data
        """
        logger.info("Starting data transformation phase")
        
        transformed_data = {}
        
        try:
            encounters = self.imputed_data.get('encounters', pd.DataFrame())
            claims = self.imputed_data.get('claims', pd.DataFrame())
            patients = self.imputed_data.get('patients', pd.DataFrame())
            
            if not encounters.empty:
                # Calculate readmission metrics
                encounters = self.transformer.calculate_readmission_metrics(encounters)
                
                # Calculate length of stay
                encounters = self.transformer.calculate_length_of_stay(encounters)
                
                transformed_data['encounters_with_metrics'] = encounters
            
            if not claims.empty:
                # Calculate cost metrics
                claims = self.transformer.calculate_cost_metrics(claims)
                transformed_data['claims_with_metrics'] = claims
            
            if not encounters.empty and not claims.empty:
                # Calculate cost per encounter
                transformed_data['encounters_with_costs'] = self.transformer.calculate_cost_per_encounter(
                    encounters, claims
                )
            
            if not encounters.empty:
                # Aggregate by provider
                transformed_data['provider_metrics'] = self.transformer.aggregate_by_provider(encounters)
                
                # Aggregate by facility
                transformed_data['facility_metrics'] = self.transformer.aggregate_by_facility(encounters)
                
                # Aggregate by diagnosis
                transformed_data['diagnosis_metrics'] = self.transformer.aggregate_by_diagnosis(encounters)
                
                # Create time series
                transformed_data['daily_metrics'] = self.transformer.create_time_series(encounters, 'D')
                transformed_data['monthly_metrics'] = self.transformer.create_time_series(encounters, 'M')
            
            if not patients.empty and not encounters.empty:
                # Calculate patient risk scores
                transformed_data['patient_risk_scores'] = self.transformer.calculate_patient_risk_score(
                    patients, encounters
                )
        
        except Exception as e:
            logger.error(f"Failed to transform data: {str(e)}")
        
        return transformed_data
    
    def check_data_quality(self) -> Dict:
        """
        Check quality of all data
        
        Returns:
            Quality report
        """
        logger.info("Checking data quality")
        
        quality_report = {}
        
        for data_type, df in self.imputed_data.items():
            try:
                quality_report[data_type] = self.quality_checker.check_data_quality(df, data_type)
            except Exception as e:
                logger.error(f"Failed to check quality of {data_type}: {str(e)}")
        
        return quality_report
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run complete data engineering pipeline
        
        Returns:
            Dictionary with all pipeline outputs
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DATA ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        self.load_all_data()
        
        # Clean data
        self.clean_all_data()
        
        # Impute missing values
        self.impute_all_data()
        
        # Create fact and dimension tables
        self.create_fact_and_dimension_tables()
        
        # Transform data
        transformed_data = self.transform_data()
        
        # Check quality
        quality_report = self.check_data_quality()
        
        logger.info("=" * 60)
        logger.info("DATA ENGINEERING PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return {
            'raw_data': self.raw_data,
            'cleaned_data': self.cleaned_data,
            'imputed_data': self.imputed_data,
            'fact_tables': self.fact_tables,
            'dim_tables': self.dim_tables,
            'transformed_data': transformed_data,
            'quality_report': quality_report
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all data"""
        summary = {}
        
        for data_type, df in self.imputed_data.items():
            summary[data_type] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        return summary
