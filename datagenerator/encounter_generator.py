"""Clinical encounter data generator with seasonal patterns"""

import random
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class EncounterGenerator(BaseGenerator):
    """Generates clinical encounter data"""
    
    ENCOUNTER_TYPES = ["Inpatient", "Outpatient", "Emergency", "Urgent Care", "Telehealth"]
    
    # ICD-10 diagnosis codes (realistic sample)
    DIAGNOSES = [
        "I10", "E11", "J44.9", "I50.9", "E78.5",  # Common chronic conditions
        "M79.3", "F41.1", "N18.3", "C34.90", "E66.9",  # More conditions
        "I21.9", "J18.9", "K21.9", "F32.9", "M54.5"
    ]
    
    # CPT procedure codes (realistic sample)
    PROCEDURES = [
        "99213", "99214", "99215",  # Office visits
        "93000", "71046", "80053",  # Diagnostics
        "99291", "99292",  # Critical care
        "27447", "43239", "33935"   # Surgeries
    ]
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_encounters(self, patient_ids: List[str], provider_ids: List[str], 
                           facility_ids: List[str]) -> pd.DataFrame:
        """Generate encounter data"""
        encounters = []
        
        for i in range(self.config.num_encounters):
            encounter_date = self.config.start_date + timedelta(
                days=random.randint(0, (self.config.end_date - self.config.start_date).days)
            )
            
            encounter_type = random.choice(self.ENCOUNTER_TYPES)
            
            # Length of stay based on encounter type
            if encounter_type == "Inpatient":
                los = random.randint(1, 14)
            elif encounter_type == "Emergency":
                los = random.randint(0, 3)
            else:
                los = 0
            
            discharge_date = encounter_date + timedelta(days=los) if los > 0 else encounter_date
            
            # Determine if readmission
            is_readmission = random.random() < self.config.readmission_rate
            
            encounter = {
                "encounter_id": f"ENC{i+1:08d}",
                "patient_id": random.choice(patient_ids),
                "provider_id": random.choice(provider_ids),
                "facility_id": random.choice(facility_ids),
                "encounter_type": encounter_type,
                "encounter_date": encounter_date.date(),
                "discharge_date": discharge_date.date(),
                "length_of_stay": los,
                "primary_diagnosis": random.choice(self.DIAGNOSES),
                "secondary_diagnoses": "|".join(random.sample(self.DIAGNOSES, k=random.randint(0, 3))),
                "primary_procedure": random.choice(self.PROCEDURES) if random.random() > 0.3 else None,
                "secondary_procedures": "|".join(random.sample(self.PROCEDURES, k=random.randint(0, 2))),
                "is_readmission": is_readmission,
                "readmission_days": random.randint(1, 30) if is_readmission else None,
                "discharge_disposition": random.choice(["Home", "Skilled Nursing", "Rehab", "Expired"]),
                "severity_level": random.choice(["Minor", "Moderate", "Severe", "Critical"])
            }
            encounters.append(encounter)
        
        return pd.DataFrame(encounters)
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        super().to_csv(df, filepath, "encounters")
