"""
Cohort Analysis for Healthcare Data
Analyzes specific patient cohorts: Diabetic, Hypertensive, Chronic Disease, Comorbidity patterns
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class CohortAnalyzer:
    """Analyze healthcare cohorts and their characteristics"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def identify_cohorts(self, patients_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Identify and extract patient cohorts
        
        Cohorts:
            - Diabetic patients
            - Hypertensive patients
            - Chronic disease management (2+ conditions)
            - High comorbidity (3+ conditions)
            - Cardiopulmonary patients
            - Medicare (age 65+)
        """
        self.logger.info("Identifying patient cohorts")
        
        cohorts = {}
        
        # Diabetic cohort
        if 'is_diabetic' in patients_df.columns:
            cohorts['diabetic'] = patients_df[patients_df['is_diabetic'] == True].copy()
        else:
            cohorts['diabetic'] = patients_df[patients_df['comorbidities'].str.contains('Diabetes', na=False)].copy()
        
        # Hypertensive cohort
        if 'is_hypertensive' in patients_df.columns:
            cohorts['hypertensive'] = patients_df[patients_df['is_hypertensive'] == True].copy()
        else:
            cohorts['hypertensive'] = patients_df[patients_df['comorbidities'].str.contains('Hypertension', na=False)].copy()
        
        # Chronic disease management (2+ conditions)
        if 'comorbidity_count' in patients_df.columns:
            cohorts['chronic_disease'] = patients_df[patients_df['comorbidity_count'] >= 2].copy()
        else:
            patients_df['comorbidity_count'] = patients_df['comorbidities'].str.split('|').str.len()
            cohorts['chronic_disease'] = patients_df[patients_df['comorbidity_count'] >= 2].copy()
        
        # High comorbidity (3+ conditions)
        cohorts['high_comorbidity'] = patients_df[patients_df['comorbidity_count'] >= 3].copy()
        
        # Cardiopulmonary
        cohorts['cardiopulmonary'] = patients_df[
            patients_df['comorbidities'].str.contains('Heart Disease|COPD', na=False, regex=True)
        ].copy()
        
        # Medicare (age 65+)
        cohorts['medicare'] = patients_df[patients_df['age'] >= 65].copy()
        
        # High-cost patients
        if 'is_high_cost' in patients_df.columns:
            cohorts['high_cost'] = patients_df[patients_df['is_high_cost'] == True].copy()
        
        self.logger.info(f"Identified {len(cohorts)} cohorts")
        for name, df in cohorts.items():
            self.logger.info(f"  - {name}: {len(df)} patients ({len(df)/len(patients_df)*100:.1f}%)")
        
        return cohorts
    
    def cohort_demographics(self, cohort_df: pd.DataFrame, cohort_name: str) -> Dict[str, Any]:
        """
        Analyze demographics of a cohort
        
        Returns:
            - Age distribution
            - Gender distribution
            - Race distribution
            - Comorbidity patterns
        """
        self.logger.info(f"Analyzing demographics for {cohort_name} cohort")
        
        result = {
            'cohort_name': cohort_name,
            'total_patients': int(len(cohort_df)),
            
            # Age statistics
            'age_mean': float(cohort_df['age'].mean()),
            'age_median': float(cohort_df['age'].median()),
            'age_std': float(cohort_df['age'].std()),
            'age_min': int(cohort_df['age'].min()),
            'age_max': int(cohort_df['age'].max()),
            
            # Age groups
            'age_groups': self._calculate_age_groups(cohort_df),
            
            # Gender distribution
            'gender_distribution': cohort_df['gender'].value_counts().to_dict() if 'gender' in cohort_df.columns else {},
            
            # Race distribution
            'race_distribution': cohort_df['race'].value_counts().to_dict() if 'race' in cohort_df.columns else {},
            
            # Comorbidity statistics
            'avg_comorbidities': float(cohort_df['comorbidity_count'].mean()) if 'comorbidity_count' in cohort_df.columns else 0,
            'comorbidity_distribution': self._analyze_comorbidities(cohort_df),
            
            # Risk score
            'avg_risk_score': float(cohort_df['risk_score'].mean()) if 'risk_score' in cohort_df.columns else None,
        }
        
        return result
    
    def cohort_outcomes(self, cohort_df: pd.DataFrame, encounters_df: pd.DataFrame, 
                       claims_df: pd.DataFrame, cohort_name: str) -> Dict[str, Any]:
        """
        Analyze clinical and financial outcomes for a cohort
        
        Returns:
            - Readmission rates
            - Length of stay
            - Cost metrics
            - Utilization patterns
        """
        self.logger.info(f"Analyzing outcomes for {cohort_name} cohort")
        
        # Filter encounters and claims for cohort patients
        cohort_patient_ids = cohort_df['patient_id'].tolist()
        cohort_encounters = encounters_df[encounters_df['patient_id'].isin(cohort_patient_ids)]
        cohort_claims = claims_df[claims_df['patient_id'].isin(cohort_patient_ids)]
        
        result = {
            'cohort_name': cohort_name,
            'total_patients': int(len(cohort_df)),
            
            # Utilization
            'total_encounters': int(len(cohort_encounters)),
            'encounters_per_patient': float(len(cohort_encounters) / len(cohort_df)),
            'total_claims': int(len(cohort_claims)),
            'claims_per_patient': float(len(cohort_claims) / len(cohort_df)),
            
            # Readmissions
            'readmission_rate': float(cohort_encounters['is_readmission'].mean()) if 'is_readmission' in cohort_encounters.columns else None,
            'readmission_count': int(cohort_encounters['is_readmission'].sum()) if 'is_readmission' in cohort_encounters.columns else None,
            
            # Length of stay
            'avg_length_of_stay': float(cohort_encounters['length_of_stay'].mean()) if 'length_of_stay' in cohort_encounters.columns else None,
            'median_length_of_stay': float(cohort_encounters['length_of_stay'].median()) if 'length_of_stay' in cohort_encounters.columns else None,
            
            # Costs
            'total_claim_amount': float(cohort_claims['claim_amount'].sum()) if 'claim_amount' in cohort_claims.columns else None,
            'total_paid_amount': float(cohort_claims['paid_amount'].sum()) if 'paid_amount' in cohort_claims.columns else None,
            'avg_claim_per_patient': float(cohort_claims.groupby('patient_id')['claim_amount'].sum().mean()) if 'claim_amount' in cohort_claims.columns else None,
            'avg_paid_per_patient': float(cohort_claims.groupby('patient_id')['paid_amount'].sum().mean()) if 'paid_amount' in cohort_claims.columns else None,
            
            # Encounter types
            'encounter_type_distribution': cohort_encounters['encounter_type'].value_counts().to_dict() if 'encounter_type' in cohort_encounters.columns else {},
            
            # Severity
            'severity_distribution': cohort_encounters['severity_level'].value_counts().to_dict() if 'severity_level' in cohort_encounters.columns else {},
        }
        
        return result
    
    def compare_cohorts(self, cohort1_df: pd.DataFrame, cohort2_df: pd.DataFrame,
                       encounters_df: pd.DataFrame, claims_df: pd.DataFrame,
                       cohort1_name: str, cohort2_name: str) -> Dict[str, Any]:
        """
        Compare two cohorts across multiple dimensions
        
        Returns:
            - Demographic differences
            - Outcome differences
            - Statistical significance
        """
        self.logger.info(f"Comparing {cohort1_name} vs {cohort2_name}")
        
        # Get outcomes for both cohorts
        outcomes1 = self.cohort_outcomes(cohort1_df, encounters_df, claims_df, cohort1_name)
        outcomes2 = self.cohort_outcomes(cohort2_df, encounters_df, claims_df, cohort2_name)
        
        # Calculate differences
        comparison = {
            'cohort1_name': cohort1_name,
            'cohort2_name': cohort2_name,
            
            # Size comparison
            'cohort1_size': outcomes1['total_patients'],
            'cohort2_size': outcomes2['total_patients'],
            
            # Age comparison
            'age_difference': float(cohort1_df['age'].mean() - cohort2_df['age'].mean()),
            
            # Utilization comparison
            'encounters_per_patient_diff': outcomes1['encounters_per_patient'] - outcomes2['encounters_per_patient'],
            'encounters_per_patient_pct_diff': ((outcomes1['encounters_per_patient'] / outcomes2['encounters_per_patient']) - 1) * 100 if outcomes2['encounters_per_patient'] > 0 else None,
            
            # Readmission comparison
            'readmission_rate_diff': (outcomes1['readmission_rate'] - outcomes2['readmission_rate']) if outcomes1['readmission_rate'] and outcomes2['readmission_rate'] else None,
            
            # Cost comparison
            'avg_cost_per_patient_diff': (outcomes1['avg_claim_per_patient'] - outcomes2['avg_claim_per_patient']) if outcomes1['avg_claim_per_patient'] and outcomes2['avg_claim_per_patient'] else None,
            'avg_cost_per_patient_pct_diff': ((outcomes1['avg_claim_per_patient'] / outcomes2['avg_claim_per_patient']) - 1) * 100 if outcomes2['avg_claim_per_patient'] and outcomes2['avg_claim_per_patient'] > 0 else None,
            
            # LOS comparison
            'los_diff': (outcomes1['avg_length_of_stay'] - outcomes2['avg_length_of_stay']) if outcomes1['avg_length_of_stay'] and outcomes2['avg_length_of_stay'] else None,
        }
        
        return comparison
    
    def comorbidity_patterns(self, patients_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze comorbidity patterns across all patients
        
        Returns:
            - Most common comorbidities
            - Comorbidity combinations
            - Correlation between conditions
        """
        self.logger.info("Analyzing comorbidity patterns")
        
        # Extract all comorbidities
        all_comorbidities = []
        for comorbidities_str in patients_df['comorbidities'].dropna():
            if comorbidities_str:
                all_comorbidities.extend(comorbidities_str.split('|'))
        
        # Count frequencies
        comorbidity_counts = pd.Series(all_comorbidities).value_counts()
        
        # Find common combinations
        combinations = {}
        for comorbidities_str in patients_df['comorbidities'].dropna():
            if comorbidities_str and '|' in comorbidities_str:
                combo = tuple(sorted(comorbidities_str.split('|')))
                if len(combo) >= 2:
                    combinations[combo] = combinations.get(combo, 0) + 1
        
        # Sort combinations by frequency
        top_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        result = {
            'total_patients': int(len(patients_df)),
            'patients_with_comorbidities': int((patients_df['comorbidity_count'] > 0).sum()) if 'comorbidity_count' in patients_df.columns else None,
            'avg_comorbidities_per_patient': float(patients_df['comorbidity_count'].mean()) if 'comorbidity_count' in patients_df.columns else None,
            
            # Most common individual comorbidities
            'top_comorbidities': {k: int(v) for k, v in comorbidity_counts.head(10).items()},
            
            # Most common combinations
            'top_combinations': [
                {
                    'conditions': list(combo),
                    'count': int(count),
                    'percentage': float(count / len(patients_df) * 100)
                }
                for combo, count in top_combinations
            ],
        }
        
        return result
    
    def _calculate_age_groups(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate age group distribution"""
        age_groups = {
            '18-30': int(((df['age'] >= 18) & (df['age'] < 30)).sum()),
            '30-45': int(((df['age'] >= 30) & (df['age'] < 45)).sum()),
            '45-60': int(((df['age'] >= 45) & (df['age'] < 60)).sum()),
            '60-75': int(((df['age'] >= 60) & (df['age'] < 75)).sum()),
            '75+': int((df['age'] >= 75).sum()),
        }
        return age_groups
    
    def _analyze_comorbidities(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze comorbidity distribution"""
        if 'comorbidities' not in df.columns:
            return {}
        
        all_comorbidities = []
        for comorbidities_str in df['comorbidities'].dropna():
            if comorbidities_str:
                all_comorbidities.extend(comorbidities_str.split('|'))
        
        return pd.Series(all_comorbidities).value_counts().head(10).to_dict()
