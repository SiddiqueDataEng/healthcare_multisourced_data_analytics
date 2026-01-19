"""External reporting data generator (CDC, CMS)"""

import random
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class ReportingGenerator(BaseGenerator):
    """Generates external reporting data"""
    
    # CMS Quality Measures
    CMS_MEASURES = [
        "30-Day Readmission Rate",
        "Mortality Rate",
        "Safety Indicator",
        "Timely and Effective Care",
        "Use of Medical Imaging"
    ]
    
    # CDC NHSN HAI Types
    HAI_TYPES = [
        "CLABSI",  # Central Line-Associated Bloodstream Infection
        "CAUTI",   # Catheter-Associated Urinary Tract Infection
        "SSI",     # Surgical Site Infection
        "MRSA",    # Methicillin-resistant Staphylococcus aureus
        "C. difficile"
    ]
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_cms_measures(self, facility_ids: List[str]) -> pd.DataFrame:
        """Generate CMS quality measures"""
        measures = []
        
        for facility_id in facility_ids:
            for measure in self.CMS_MEASURES:
                report_date = self.config.start_date + timedelta(
                    days=random.randint(0, (self.config.end_date - self.config.start_date).days)
                )
                
                # Generate realistic measure values
                if "Readmission" in measure:
                    measure_value = round(random.uniform(0.10, 0.25), 4)
                    benchmark_value = 0.15
                elif "Mortality" in measure:
                    measure_value = round(random.uniform(0.01, 0.05), 4)
                    benchmark_value = 0.03
                else:
                    measure_value = round(random.uniform(0.70, 0.95), 4)
                    benchmark_value = 0.85
                
                entry = {
                    "report_id": f"CMS{random.randint(100000, 999999)}",
                    "facility_id": facility_id,
                    "measure_code": measure.replace(" ", "_").upper(),
                    "measure_name": measure,
                    "report_date": report_date.date(),
                    "measure_value": measure_value,
                    "benchmark_value": benchmark_value,
                    "percentile_rank": random.randint(10, 90),
                    "performance_status": "Above" if measure_value > benchmark_value else "Below"
                }
                measures.append(entry)
        
        return pd.DataFrame(measures)
    
    def generate_hai_data(self, facility_ids: List[str]) -> pd.DataFrame:
        """Generate CDC NHSN HAI data"""
        hai_data = []
        
        for facility_id in facility_ids:
            for hai_type in self.HAI_TYPES:
                report_date = self.config.start_date + timedelta(
                    days=random.randint(0, (self.config.end_date - self.config.start_date).days)
                )
                
                # Generate realistic HAI rates (per 1000 patient days)
                if hai_type == "CLABSI":
                    rate = round(random.uniform(0.5, 2.0), 2)
                elif hai_type == "CAUTI":
                    rate = round(random.uniform(1.0, 3.0), 2)
                elif hai_type == "SSI":
                    rate = round(random.uniform(0.5, 1.5), 2)
                else:
                    rate = round(random.uniform(0.1, 1.0), 2)
                
                entry = {
                    "report_id": f"HAI{random.randint(100000, 999999)}",
                    "facility_id": facility_id,
                    "hai_type": hai_type,
                    "report_date": report_date.date(),
                    "infection_count": random.randint(0, 10),
                    "patient_days": random.randint(5000, 20000),
                    "infection_rate": rate,
                    "national_benchmark": round(random.uniform(0.5, 2.0), 2),
                    "status": "Below Benchmark" if rate < 1.5 else "Above Benchmark"
                }
                hai_data.append(entry)
        
        return pd.DataFrame(hai_data)
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        entity_name = "CMS measures" if "measure_code" in df.columns else "HAI records"
        super().to_csv(df, filepath, entity_name)
