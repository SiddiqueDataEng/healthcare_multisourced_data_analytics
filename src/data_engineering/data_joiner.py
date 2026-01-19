"""
Data Joiner - Handles data joins and relationships
Creates fact and dimension tables through intelligent joins
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataJoiner:
    """Handles joining healthcare data tables"""
    
    def __init__(self):
        self.join_report = {}
    
    def join_patient_encounters(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join patients with encounters
        Creates enriched encounter records with patient demographics
        """
        logger.info("Joining patients with encounters")
        
        initial_rows = len(encounters_df)
        
        # Left join - keep all encounters
        result = encounters_df.merge(
            patients_df[['patient_id', 'age', 'gender', 'race', 'comorbidity_count']],
            on='patient_id',
            how='left'
        )
        
        final_rows = len(result)
        logger.info(f"Patient-Encounter join: {initial_rows} → {final_rows} rows")
        
        self.join_report['patient_encounters'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'join_type': 'left'
        }
        
        return result
    
    def join_encounters_claims(self, encounters_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join encounters with claims
        Links clinical encounters to financial claims
        """
        logger.info("Joining encounters with claims")
        
        initial_rows = len(encounters_df)
        
        # Left join - keep all encounters, add claims if available
        result = encounters_df.merge(
            claims_df[['patient_id', 'encounter_id', 'claim_id', 'claim_amount', 'paid_amount', 'adjudication_status']],
            on=['patient_id', 'encounter_id'],
            how='left'
        )
        
        final_rows = len(result)
        logger.info(f"Encounter-Claims join: {initial_rows} → {final_rows} rows")
        
        self.join_report['encounter_claims'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'join_type': 'left'
        }
        
        return result
    
    def join_provider_facility(self, providers_df: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join providers with facilities
        Links providers to their affiliated facilities
        """
        logger.info("Joining providers with facilities")
        
        initial_rows = len(providers_df)
        
        # Left join - keep all providers
        result = providers_df.merge(
            facilities_df[['facility_id', 'facility_name', 'facility_type', 'state']],
            on='facility_id',
            how='left'
        )
        
        final_rows = len(result)
        logger.info(f"Provider-Facility join: {initial_rows} → {final_rows} rows")
        
        self.join_report['provider_facility'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'join_type': 'left'
        }
        
        return result
    
    def create_fact_encounters(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame, 
                               providers_df: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fact_encounters table with all relevant dimensions
        """
        logger.info("Creating fact_encounters table")
        
        # Start with encounters
        fact = encounters_df.copy()
        
        # Join with patient demographics
        fact = fact.merge(
            patients_df[['patient_id', 'age', 'gender', 'race', 'comorbidity_count']],
            on='patient_id',
            how='left'
        )
        
        # Join with provider info
        fact = fact.merge(
            providers_df[['provider_id', 'specialty']],
            on='provider_id',
            how='left'
        )
        
        # Join with facility info
        fact = fact.merge(
            facilities_df[['facility_id', 'facility_name', 'facility_type']],
            on='facility_id',
            how='left'
        )
        
        logger.info(f"Fact encounters created: {len(fact)} rows")
        
        return fact
    
    def create_fact_claims(self, patients_df: pd.DataFrame, claims_df: pd.DataFrame,
                          providers_df: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fact_claims table with all relevant dimensions
        """
        logger.info("Creating fact_claims table")
        
        # Start with claims
        fact = claims_df.copy()
        
        # Join with patient demographics
        fact = fact.merge(
            patients_df[['patient_id', 'age', 'gender', 'race']],
            on='patient_id',
            how='left'
        )
        
        # Join with provider info
        fact = fact.merge(
            providers_df[['provider_id', 'specialty']],
            on='provider_id',
            how='left'
        )
        
        # Join with facility info
        fact = fact.merge(
            facilities_df[['facility_id', 'facility_name', 'facility_type']],
            on='facility_id',
            how='left'
        )
        
        logger.info(f"Fact claims created: {len(fact)} rows")
        
        return fact
    
    def create_dim_patient(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Create patient dimension table"""
        logger.info("Creating dim_patient")
        
        dim = patients_df[[
            'patient_id', 'age', 'gender', 'race', 'comorbidity_count'
        ]].drop_duplicates()
        
        logger.info(f"Dim patient created: {len(dim)} rows")
        return dim
    
    def create_dim_provider(self, providers_df: pd.DataFrame) -> pd.DataFrame:
        """Create provider dimension table"""
        logger.info("Creating dim_provider")
        
        dim = providers_df[[
            'provider_id', 'specialty', 'facility_id'
        ]].drop_duplicates()
        
        logger.info(f"Dim provider created: {len(dim)} rows")
        return dim
    
    def create_dim_facility(self, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """Create facility dimension table"""
        logger.info("Creating dim_facility")
        
        dim = facilities_df[[
            'facility_id', 'facility_name', 'facility_type', 'state'
        ]].drop_duplicates()
        
        logger.info(f"Dim facility created: {len(dim)} rows")
        return dim
    
    def create_dim_diagnosis(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """Create diagnosis dimension table"""
        logger.info("Creating dim_diagnosis")
        
        # Extract unique diagnoses
        diagnoses = []
        if 'primary_diagnosis' in encounters_df.columns:
            diagnoses.extend(encounters_df['primary_diagnosis'].dropna().unique())
        if 'secondary_diagnosis' in encounters_df.columns:
            diagnoses.extend(encounters_df['secondary_diagnosis'].dropna().unique())
        
        dim = pd.DataFrame({
            'diagnosis_code': list(set(diagnoses))
        }).drop_duplicates()
        
        logger.info(f"Dim diagnosis created: {len(dim)} rows")
        return dim
    
    def get_join_report(self) -> Dict:
        """Get join report"""
        return self.join_report
