"""Data generator configuration"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class GeneratorConfig:
    """Configuration for data generation"""
    
    # Data volumes
    num_patients: int = 10000
    num_providers: int = 500
    num_facilities: int = 50
    num_encounters: int = 50000
    num_claims: int = 100000
    
    # Date ranges
    start_date: datetime = datetime(2020, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    
    # Output paths
    output_dir: str = "data/landing_zone"
    
    # Data characteristics - Realistic rates
    readmission_rate: float = 0.15
    high_cost_patient_rate: float = 0.20
    fraud_rate: float = 0.02
    hai_rate: float = 0.05
    
    # Cohort prevalence rates (realistic US population)
    diabetes_prevalence: float = 0.11  # 11% of adults
    hypertension_prevalence: float = 0.47  # 47% of adults
    copd_prevalence: float = 0.06  # 6% of adults
    heart_disease_prevalence: float = 0.06  # 6% of adults
    obesity_prevalence: float = 0.42  # 42% of adults
    
    # Comorbidity correlation (patients with multiple conditions)
    comorbidity_correlation: float = 0.65  # 65% correlation
    
    # Seasonal variation for encounters
    seasonal_variation: bool = True
    winter_surge_factor: float = 1.3  # 30% more encounters in winter
    
    # Cost variation by severity
    cost_multiplier_severe: float = 3.0
    cost_multiplier_critical: float = 5.0
    
    # Random seed for reproducibility
    random_seed: int = 42
