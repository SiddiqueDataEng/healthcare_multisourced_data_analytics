"""Disease registry data generator"""

import random
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class RegistryGenerator(BaseGenerator):
    """Generates disease registry data"""
    
    SURGICAL_PROCEDURES = [
        "Coronary Artery Bypass", "Hip Replacement", "Knee Replacement",
        "Appendectomy", "Cholecystectomy", "Hysterectomy", "Prostatectomy"
    ]
    
    COMPLICATIONS = [
        "Infection", "Bleeding", "DVT", "Pneumonia", "Sepsis",
        "Cardiac Arrhythmia", "Acute Kidney Injury", "Stroke"
    ]
    
    OUTCOMES = ["Excellent", "Good", "Fair", "Poor", "Expired"]
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_registry(self, patient_ids: List[str], provider_ids: List[str]) -> pd.DataFrame:
        """Generate disease registry data"""
        registry = []
        
        # Generate registry entries for subset of patients (surgical cases)
        num_registry_entries = int(self.config.num_patients * 0.1)  # 10% have registry entries
        
        for i in range(num_registry_entries):
            procedure_date = self.config.start_date + timedelta(
                days=random.randint(0, (self.config.end_date - self.config.start_date).days)
            )
            
            # Complication likelihood increases with age
            has_complication = random.random() < 0.15
            
            entry = {
                "registry_id": f"REG{i+1:08d}",
                "patient_id": random.choice(patient_ids),
                "provider_id": random.choice(provider_ids),
                "procedure_name": random.choice(self.SURGICAL_PROCEDURES),
                "procedure_date": procedure_date.date(),
                "has_complication": has_complication,
                "complication_type": random.choice(self.COMPLICATIONS) if has_complication else None,
                "complication_date": (procedure_date + timedelta(days=random.randint(1, 30))).date() if has_complication else None,
                "outcome": random.choice(self.OUTCOMES),
                "risk_score": round(random.uniform(0.1, 0.9), 2),
                "mortality_flag": random.random() < 0.02,
                "readmission_flag": random.random() < self.config.readmission_rate,
                "days_to_readmission": random.randint(1, 30) if random.random() < self.config.readmission_rate else None
            }
            registry.append(entry)
        
        return pd.DataFrame(registry)
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        super().to_csv(df, filepath, "registry records")
