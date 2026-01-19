"""
Data Transformer - Handles data transformations and aggregations
Creates derived metrics, aggregations, and feature engineering
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms and aggregates healthcare data"""
    
    def __init__(self):
        self.transformation_report = {}
    
    def calculate_readmission_metrics(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate readmission metrics
        Identifies patients readmitted within 30 days
        """
        logger.info("Calculating readmission metrics")
        
        # Sort by patient and date
        encounters_sorted = encounters_df.sort_values(['patient_id', 'encounter_date'])
        
        # Calculate days between encounters
        encounters_sorted['days_to_next_encounter'] = encounters_sorted.groupby('patient_id')['encounter_date'].diff().dt.days
        
        # Flag readmissions (within 30 days)
        encounters_sorted['is_readmission'] = encounters_sorted['days_to_next_encounter'].between(1, 30)
        
        # Calculate readmission rate
        total_encounters = len(encounters_sorted)
        readmissions = encounters_sorted['is_readmission'].sum()
        readmission_rate = (readmissions / total_encounters * 100) if total_encounters > 0 else 0
        
        logger.info(f"Readmission rate: {readmission_rate:.2f}% ({readmissions}/{total_encounters})")
        
        self.transformation_report['readmission_metrics'] = {
            'total_encounters': total_encounters,
            'readmissions': readmissions,
            'readmission_rate': readmission_rate
        }
        
        return encounters_sorted
    
    def calculate_length_of_stay(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate length of stay (LOS) metrics
        """
        logger.info("Calculating length of stay metrics")
        
        encounters = encounters_df.copy()
        
        # Calculate LOS if discharge date available
        if 'discharge_date' in encounters.columns and 'encounter_date' in encounters.columns:
            encounters['length_of_stay'] = (
                pd.to_datetime(encounters['discharge_date']) - 
                pd.to_datetime(encounters['encounter_date'])
            ).dt.days
        else:
            # Default to 1 day if not available
            encounters['length_of_stay'] = 1
        
        # Calculate statistics
        avg_los = encounters['length_of_stay'].mean()
        median_los = encounters['length_of_stay'].median()
        max_los = encounters['length_of_stay'].max()
        
        logger.info(f"LOS - Avg: {avg_los:.1f}, Median: {median_los:.1f}, Max: {max_los:.0f} days")
        
        self.transformation_report['los_metrics'] = {
            'average_los': avg_los,
            'median_los': median_los,
            'max_los': max_los
        }
        
        return encounters
    
    def calculate_cost_metrics(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cost and financial metrics
        """
        logger.info("Calculating cost metrics")
        
        claims = claims_df.copy()
        
        # Ensure numeric types
        for col in ['claim_amount', 'paid_amount', 'allowed_amount']:
            if col in claims.columns:
                claims[col] = pd.to_numeric(claims[col], errors='coerce')
        
        # Calculate metrics
        total_claims = claims['claim_amount'].sum()
        total_paid = claims['paid_amount'].sum()
        avg_claim = claims['claim_amount'].mean()
        
        # Calculate denial rate
        if 'adjudication_status' in claims.columns:
            denied = (claims['adjudication_status'] == 'Denied').sum()
            denial_rate = (denied / len(claims) * 100) if len(claims) > 0 else 0
        else:
            denial_rate = 0
        
        logger.info(f"Cost metrics - Total: ${total_claims:,.0f}, Paid: ${total_paid:,.0f}, Avg: ${avg_claim:,.0f}")
        
        self.transformation_report['cost_metrics'] = {
            'total_claims': total_claims,
            'total_paid': total_paid,
            'average_claim': avg_claim,
            'denial_rate': denial_rate
        }
        
        return claims
    
    def calculate_cost_per_encounter(self, encounters_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cost per encounter
        """
        logger.info("Calculating cost per encounter")
        
        # Group claims by encounter
        encounter_costs = claims_df.groupby('encounter_id').agg({
            'paid_amount': 'sum',
            'claim_amount': 'sum'
        }).reset_index()
        
        encounter_costs.columns = ['encounter_id', 'total_paid', 'total_claimed']
        
        # Join with encounters
        result = encounters_df.merge(encounter_costs, on='encounter_id', how='left')
        result['total_paid'] = result['total_paid'].fillna(0)
        result['total_claimed'] = result['total_claimed'].fillna(0)
        
        avg_cost = result['total_paid'].mean()
        logger.info(f"Average cost per encounter: ${avg_cost:,.0f}")
        
        return result
    
    def aggregate_by_provider(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics by provider
        """
        logger.info("Aggregating metrics by provider")
        
        provider_metrics = encounters_df.groupby('provider_id').agg({
            'encounter_id': 'count',
            'patient_id': 'nunique',
            'length_of_stay': ['mean', 'median'],
            'is_readmission': 'sum' if 'is_readmission' in encounters_df.columns else 'count'
        }).reset_index()
        
        provider_metrics.columns = ['provider_id', 'total_encounters', 'unique_patients', 
                                    'avg_los', 'median_los', 'readmissions']
        
        logger.info(f"Provider aggregation: {len(provider_metrics)} providers")
        
        return provider_metrics
    
    def aggregate_by_facility(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics by facility
        """
        logger.info("Aggregating metrics by facility")
        
        facility_metrics = encounters_df.groupby('facility_id').agg({
            'encounter_id': 'count',
            'patient_id': 'nunique',
            'length_of_stay': ['mean', 'median'],
            'is_readmission': 'sum' if 'is_readmission' in encounters_df.columns else 'count'
        }).reset_index()
        
        facility_metrics.columns = ['facility_id', 'total_encounters', 'unique_patients',
                                    'avg_los', 'median_los', 'readmissions']
        
        logger.info(f"Facility aggregation: {len(facility_metrics)} facilities")
        
        return facility_metrics
    
    def aggregate_by_diagnosis(self, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate metrics by diagnosis
        """
        logger.info("Aggregating metrics by diagnosis")
        
        diagnosis_metrics = encounters_df.groupby('primary_diagnosis').agg({
            'encounter_id': 'count',
            'patient_id': 'nunique',
            'length_of_stay': ['mean', 'median'],
            'is_readmission': 'sum' if 'is_readmission' in encounters_df.columns else 'count'
        }).reset_index()
        
        diagnosis_metrics.columns = ['diagnosis', 'total_encounters', 'unique_patients',
                                     'avg_los', 'median_los', 'readmissions']
        
        logger.info(f"Diagnosis aggregation: {len(diagnosis_metrics)} diagnoses")
        
        return diagnosis_metrics
    
    def create_time_series(self, encounters_df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """
        Create time series aggregations
        
        Args:
            encounters_df: Encounters dataframe
            frequency: 'D' for daily, 'W' for weekly, 'M' for monthly
        """
        logger.info(f"Creating time series aggregation (frequency: {frequency})")
        
        encounters = encounters_df.copy()
        encounters['encounter_date'] = pd.to_datetime(encounters['encounter_date'])
        
        time_series = encounters.set_index('encounter_date').resample(frequency).agg({
            'encounter_id': 'count',
            'patient_id': 'nunique',
            'length_of_stay': 'mean',
            'is_readmission': 'sum' if 'is_readmission' in encounters.columns else 'count'
        }).reset_index()
        
        time_series.columns = ['date', 'encounters', 'unique_patients', 'avg_los', 'readmissions']
        
        logger.info(f"Time series created: {len(time_series)} periods")
        
        return time_series
    
    def calculate_patient_risk_score(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate patient risk scores based on multiple factors
        """
        logger.info("Calculating patient risk scores")
        
        # Aggregate patient-level metrics
        patient_metrics = encounters_df.groupby('patient_id').agg({
            'encounter_id': 'count',
            'length_of_stay': 'mean',
            'is_readmission': 'sum' if 'is_readmission' in encounters_df.columns else 'count'
        }).reset_index()
        
        patient_metrics.columns = ['patient_id', 'encounter_count', 'avg_los', 'readmission_count']
        
        # Join with patient demographics
        result = patients_df.merge(patient_metrics, on='patient_id', how='left')
        
        # Calculate risk score (0-100)
        # Higher age = higher risk
        age_risk = (result['age'] / result['age'].max() * 30).fillna(0)
        
        # Higher comorbidities = higher risk
        comorbidity_risk = (result['comorbidity_count'] / result['comorbidity_count'].max() * 30).fillna(0)
        
        # Higher readmissions = higher risk
        readmission_risk = (result['readmission_count'] / result['readmission_count'].max() * 40).fillna(0)
        
        result['risk_score'] = (age_risk + comorbidity_risk + readmission_risk).round(2)
        
        logger.info(f"Risk scores calculated for {len(result)} patients")
        
        return result
    
    def get_transformation_report(self) -> Dict:
        """Get transformation report"""
        return self.transformation_report
