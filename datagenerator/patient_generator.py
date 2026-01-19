"""Patient data generator with cohort support"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Set
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class PatientGenerator(BaseGenerator):
    """Generates realistic patient data"""
    
    FIRST_NAMES = [
        "James", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda",
        "David", "Barbara", "Richard", "Elizabeth", "Joseph", "Susan", "Thomas", "Jessica",
        "Charles", "Sarah", "Christopher", "Karen", "Daniel", "Nancy", "Matthew", "Lisa"
    ]
    
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"
    ]
    
    GENDERS = ["M", "F"]
    
    RACES = ["White", "Black", "Hispanic", "Asian", "Native American", "Other"]
    
    COMORBIDITIES = [
        "Hypertension", "Diabetes", "COPD", "Heart Disease", "Obesity",
        "Asthma", "Depression", "Arthritis", "Kidney Disease", "Cancer"
    ]
    
    # Comorbidity correlations (which conditions often occur together)
    COMORBIDITY_CORRELATIONS = {
        "Diabetes": ["Hypertension", "Obesity", "Kidney Disease", "Heart Disease"],
        "Hypertension": ["Diabetes", "Heart Disease", "Kidney Disease", "Obesity"],
        "COPD": ["Heart Disease", "Asthma"],
        "Heart Disease": ["Hypertension", "Diabetes", "Obesity"],
        "Obesity": ["Diabetes", "Hypertension", "Heart Disease", "Asthma"],
        "Kidney Disease": ["Diabetes", "Hypertension"]
    }
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_patients(self) -> pd.DataFrame:
        """Generate patient master data with realistic cohorts and comorbidities"""
        patients = []
        
        for i in range(self.config.num_patients):
            # Calculate age with realistic distribution (skewed toward older adults)
            age = random.gauss(55, 20)
            age = max(18, min(100, int(age)))
            
            # Generate comorbidities with realistic correlations
            comorbidities = self._generate_correlated_comorbidities(age)
            
            # Determine cohort memberships
            cohorts = self._determine_cohorts(comorbidities, age)
            
            # Calculate risk score based on age and comorbidities
            risk_score = self._calculate_patient_risk_score(age, comorbidities)
            
            # High-cost patients are those with high risk scores
            is_high_cost = risk_score > 0.6 or random.random() < self.config.high_cost_patient_rate
            
            patient = {
                "patient_id": f"PAT{i+1:08d}",
                "first_name": random.choice(self.FIRST_NAMES),
                "last_name": random.choice(self.LAST_NAMES),
                "date_of_birth": (datetime.now() - timedelta(days=age*365)).date(),
                "age": age,
                "gender": random.choice(self.GENDERS),
                "race": random.choice(self.RACES),
                "comorbidities": "|".join(comorbidities) if comorbidities else "",
                "comorbidity_count": len(comorbidities),
                "cohorts": "|".join(cohorts) if cohorts else "",
                "risk_score": round(risk_score, 3),
                "is_high_cost": is_high_cost,
                "is_diabetic": "Diabetes" in comorbidities,
                "is_hypertensive": "Hypertension" in comorbidities,
                "has_chronic_disease": len(comorbidities) >= 2,
                "enrollment_date": self.config.start_date.date(),
                "active": random.random() > 0.05  # 95% active patients
            }
            patients.append(patient)
        
        df = pd.DataFrame(patients)
        self.logger.info(f"Generated {len(df)} patients with cohorts:")
        self.logger.info(f"  - Diabetic: {df['is_diabetic'].sum()} ({df['is_diabetic'].mean():.1%})")
        self.logger.info(f"  - Hypertensive: {df['is_hypertensive'].sum()} ({df['is_hypertensive'].mean():.1%})")
        self.logger.info(f"  - Chronic disease: {df['has_chronic_disease'].sum()} ({df['has_chronic_disease'].mean():.1%})")
        self.logger.info(f"  - High cost: {df['is_high_cost'].sum()} ({df['is_high_cost'].mean():.1%})")
        
        return df
    
    def _generate_correlated_comorbidities(self, age: int) -> List[str]:
        """Generate comorbidities with realistic age-based prevalence and correlations"""
        comorbidities: Set[str] = set()
        
        # Age-based base probability
        base_prob = min(age / 100, 0.8)  # Increases with age, max 80%
        
        # Check for each major condition based on prevalence
        if random.random() < self.config.diabetes_prevalence * (1 + base_prob):
            comorbidities.add("Diabetes")
        
        if random.random() < self.config.hypertension_prevalence * (1 + base_prob):
            comorbidities.add("Hypertension")
        
        if random.random() < self.config.copd_prevalence * (1 + base_prob):
            comorbidities.add("COPD")
        
        if random.random() < self.config.heart_disease_prevalence * (1 + base_prob):
            comorbidities.add("Heart Disease")
        
        if random.random() < self.config.obesity_prevalence:
            comorbidities.add("Obesity")
        
        # Add correlated conditions
        for condition in list(comorbidities):
            if condition in self.COMORBIDITY_CORRELATIONS:
                for correlated in self.COMORBIDITY_CORRELATIONS[condition]:
                    if random.random() < self.config.comorbidity_correlation:
                        comorbidities.add(correlated)
        
        # Add other random conditions
        other_conditions = [c for c in self.COMORBIDITIES if c not in comorbidities]
        num_additional = random.randint(0, min(2, len(other_conditions)))
        comorbidities.update(random.sample(other_conditions, num_additional))
        
        return sorted(list(comorbidities))
    
    def _determine_cohorts(self, comorbidities: List[str], age: int) -> List[str]:
        """Determine which cohorts the patient belongs to"""
        cohorts = []
        
        if "Diabetes" in comorbidities:
            cohorts.append("Diabetic")
        
        if "Hypertension" in comorbidities:
            cohorts.append("Hypertensive")
        
        if len(comorbidities) >= 2:
            cohorts.append("Chronic Disease Management")
        
        if len(comorbidities) >= 3:
            cohorts.append("High Comorbidity")
        
        if age >= 65:
            cohorts.append("Medicare")
        
        if "Heart Disease" in comorbidities or "COPD" in comorbidities:
            cohorts.append("Cardiopulmonary")
        
        return cohorts
    
    def _calculate_patient_risk_score(self, age: int, comorbidities: List[str]) -> float:
        """Calculate patient risk score (0-1) based on age and comorbidities"""
        # Age component (0-0.4)
        age_score = min(age / 100, 0.4)
        
        # Comorbidity component (0-0.6)
        comorbidity_score = min(len(comorbidities) * 0.1, 0.6)
        
        # High-risk conditions add extra weight
        high_risk_conditions = ["Heart Disease", "Cancer", "Kidney Disease", "COPD"]
        high_risk_bonus = sum(0.1 for c in comorbidities if c in high_risk_conditions)
        
        total_score = min(age_score + comorbidity_score + high_risk_bonus, 1.0)
        return total_score
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        super().to_csv(df, filepath, "patients")
