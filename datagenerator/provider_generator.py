"""Provider and facility data generator"""

import random
from typing import List, Dict
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class ProviderGenerator(BaseGenerator):
    """Generates provider and facility data"""
    
    SPECIALTIES = [
        "Internal Medicine", "Cardiology", "Orthopedics", "Neurology", "Oncology",
        "Pediatrics", "Psychiatry", "Surgery", "Emergency Medicine", "Radiology",
        "Pathology", "Anesthesiology", "Obstetrics", "Urology", "Gastroenterology"
    ]
    
    FACILITY_TYPES = ["Hospital", "Clinic", "Urgent Care", "Ambulatory Surgery Center"]
    
    STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_providers(self) -> pd.DataFrame:
        """Generate provider master data"""
        providers = []
        
        for i in range(self.config.num_providers):
            provider = {
                "provider_id": f"PROV{i+1:06d}",
                "npi": f"{random.randint(1000000000, 9999999999)}",
                "first_name": random.choice(["John", "Jane", "Michael", "Sarah", "David", "Emily"]),
                "last_name": random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"]),
                "specialty": random.choice(self.SPECIALTIES),
                "facility_id": f"FAC{random.randint(1, self.config.num_facilities):04d}",
                "years_experience": random.randint(2, 40),
                "board_certified": random.random() > 0.1,
                "active": random.random() > 0.05
            }
            providers.append(provider)
        
        return pd.DataFrame(providers)
    
    def generate_facilities(self) -> pd.DataFrame:
        """Generate facility master data"""
        facilities = []
        
        for i in range(self.config.num_facilities):
            facility = {
                "facility_id": f"FAC{i+1:04d}",
                "facility_name": f"{random.choice(['St.', 'Memorial', 'Regional', 'County'])} {random.choice(['Hospital', 'Medical Center', 'Health System'])}",
                "facility_type": random.choice(self.FACILITY_TYPES),
                "state": random.choice(self.STATES),
                "beds": random.randint(50, 500) if random.random() > 0.5 else random.randint(10, 50),
                "teaching_hospital": random.random() > 0.7,
                "trauma_center": random.random() > 0.8,
                "active": True
            }
            facilities.append(facility)
        
        return pd.DataFrame(facilities)
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        entity_name = "providers" if "provider_id" in df.columns else "facilities"
        super().to_csv(df, filepath, entity_name)
