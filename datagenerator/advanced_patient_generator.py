"""
Advanced Patient Generator - Production-grade realistic patient data
Includes demographics, clinical data, social determinants, insurance
"""

import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple
import pandas as pd
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class AdvancedPatientGenerator(BaseGenerator):
    """Generates highly realistic patient data with clinical and social context"""
    
    # Expanded name lists
    FIRST_NAMES_MALE = [
        "James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph",
        "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
        "Donald", "Steven", "Andrew", "Paul", "Joshua", "Kenneth", "Kevin", "Brian"
    ]
    
    FIRST_NAMES_FEMALE = [
        "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
        "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
        "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa"
    ]
    
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker"
    ]
    
    GENDERS = ["M", "F"]
    
    RACES = {
        "White": 0.60,
        "Black": 0.13,
        "Hispanic": 0.18,
        "Asian": 0.06,
        "Native American": 0.01,
        "Other": 0.02
    }
    
    STATES = {
        "CA": 0.12, "TX": 0.09, "FL": 0.06, "NY": 0.06, "PA": 0.04,
        "IL": 0.04, "OH": 0.04, "GA": 0.03, "NC": 0.03, "MI": 0.03,
        "NJ": 0.03, "VA": 0.03, "WA": 0.02, "AZ": 0.02, "MA": 0.02,
        "TN": 0.02, "IN": 0.02, "MO": 0.02, "MD": 0.02, "WI": 0.02
    }
    
    INSURANCE_TYPES = {
        "Commercial": 0.50,
        "Medicare": 0.20,
        "Medicaid": 0.15,
        "Self-Pay": 0.10,
        "Other": 0.05
    }
    
    MARITAL_STATUS = {
        "Married": 0.50,
        "Single": 0.30,
        "Divorced": 0.12,
        "Widowed": 0.08
    }
    
    EDUCATION_LEVELS = {
        "High School": 0.30,
        "Some College": 0.25,
        "Bachelor's": 0.25,
        "Graduate": 0.15,
        "Less than HS": 0.05
    }
    
    EMPLOYMENT_STATUS = {
        "Employed": 0.60,
        "Retired": 0.20,
        "Unemployed": 0.10,
        "Disabled": 0.10
    }
    
    # Clinical data
    BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    ALLERGIES = [
        "Penicillin", "Sulfa drugs", "Aspirin", "Ibuprofen", "Codeine",
        "Latex", "Peanuts", "Shellfish", "Eggs", "Milk", "None"
    ]
    
    SMOKING_STATUS = {
        "Never": 0.60,
        "Former": 0.25,
        "Current": 0.15
    }
    
    COMORBIDITIES = [
        "Hypertension", "Diabetes Type 2", "Diabetes Type 1", "COPD", 
        "Heart Disease", "Obesity", "Asthma", "Depression", "Anxiety",
        "Arthritis", "Kidney Disease", "Cancer", "Stroke", "Osteoporosis"
    ]
    
    COMORBIDITY_CORRELATIONS = {
        "Diabetes Type 2": ["Hypertension", "Obesity", "Kidney Disease", "Heart Disease"],
        "Hypertension": ["Diabetes Type 2", "Heart Disease", "Kidney Disease", "Obesity", "Stroke"],
        "COPD": ["Heart Disease", "Asthma"],
        "Heart Disease": ["Hypertension", "Diabetes Type 2", "Obesity", "Stroke"],
        "Obesity": ["Diabetes Type 2", "Hypertension", "Heart Disease", "Asthma"],
        "Kidney Disease": ["Diabetes Type 2", "Hypertension"]
    }
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def generate_patients(self) -> pd.DataFrame:
        """Generate comprehensive patient data"""
        patients = []
        
        for i in range(self.config.num_patients):
            patient = self._generate_single_patient(i)
            patients.append(patient)
        
        df = pd.DataFrame(patients)
        self._log_generation_summary(df)
        
        return df
    
    def _generate_single_patient(self, index: int) -> Dict:
        """Generate a single patient with all attributes"""
        
        # Basic demographics
        gender = random.choice(self.GENDERS)
        age = self._generate_realistic_age()
        dob = datetime.now() - timedelta(days=age*365.25)
        
        # Name
        first_name = random.choice(self.FIRST_NAMES_MALE if gender == "M" else self.FIRST_NAMES_FEMALE)
        last_name = random.choice(self.LAST_NAMES)
        middle_initial = random.choice(string.ascii_uppercase)
        
        # Contact information
        ssn = self._generate_ssn()
        phone = self._generate_phone()
        email = self._generate_email(first_name, last_name)
        
        # Address
        address = self._generate_address()
        
        # Race/Ethnicity
        race = self.weighted_choice(list(self.RACES.keys()), list(self.RACES.values()))
        
        # Insurance
        insurance_type = self.weighted_choice(
            list(self.INSURANCE_TYPES.keys()), 
            list(self.INSURANCE_TYPES.values())
        )
        insurance_id = self._generate_insurance_id(insurance_type)
        
        # Social determinants
        marital_status = self.weighted_choice(
            list(self.MARITAL_STATUS.keys()),
            list(self.MARITAL_STATUS.values())
        )
        education = self.weighted_choice(
            list(self.EDUCATION_LEVELS.keys()),
            list(self.EDUCATION_LEVELS.values())
        )
        employment = self.weighted_choice(
            list(self.EMPLOYMENT_STATUS.keys()),
            list(self.EMPLOYMENT_STATUS.values())
        )
        income = self._generate_income(education, employment)
        
        # Clinical data
        blood_type = random.choice(self.BLOOD_TYPES)
        height_cm = random.gauss(170 if gender == "M" else 160, 10)
        weight_kg = self._generate_weight(age, height_cm)
        bmi = weight_kg / ((height_cm/100) ** 2)
        
        smoking_status = self.weighted_choice(
            list(self.SMOKING_STATUS.keys()),
            list(self.SMOKING_STATUS.values())
        )
        
        # Allergies
        num_allergies = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
        allergies = random.sample(self.ALLERGIES, num_allergies) if num_allergies > 0 else ["None"]
        
        # Comorbidities
        comorbidities = self._generate_correlated_comorbidities(age, bmi, smoking_status)
        
        # Cohorts
        cohorts = self._determine_cohorts(comorbidities, age, insurance_type)
        
        # Risk score
        risk_score = self._calculate_comprehensive_risk_score(
            age, comorbidities, bmi, smoking_status
        )
        
        # High-cost flag
        is_high_cost = risk_score > 0.6 or random.random() < self.config.high_cost_patient_rate
        
        # Primary care provider assignment
        pcp_id = f"PROV{random.randint(1, self.config.num_providers):06d}"
        
        return {
            # Identifiers
            "patient_id": f"PAT{index+1:08d}",
            "ssn": ssn,
            "mrn": f"MRN{index+1:010d}",
            
            # Demographics
            "first_name": first_name,
            "middle_initial": middle_initial,
            "last_name": last_name,
            "date_of_birth": dob.date(),
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": "Hispanic" if race == "Hispanic" else "Non-Hispanic",
            
            # Contact
            "phone": phone,
            "email": email,
            "address_line1": address["line1"],
            "city": address["city"],
            "state": address["state"],
            "zip_code": address["zip"],
            
            # Insurance
            "insurance_type": insurance_type,
            "insurance_id": insurance_id,
            "insurance_group": f"GRP{random.randint(1000, 9999)}",
            
            # Social determinants
            "marital_status": marital_status,
            "education_level": education,
            "employment_status": employment,
            "annual_income": income,
            
            # Clinical
            "blood_type": blood_type,
            "height_cm": round(height_cm, 1),
            "weight_kg": round(weight_kg, 1),
            "bmi": round(bmi, 1),
            "smoking_status": smoking_status,
            "allergies": "|".join(allergies),
            
            # Comorbidities
            "comorbidities": "|".join(comorbidities) if comorbidities else "",
            "comorbidity_count": len(comorbidities),
            
            # Cohorts
            "cohorts": "|".join(cohorts) if cohorts else "",
            "is_diabetic": any("Diabetes" in c for c in comorbidities),
            "is_hypertensive": "Hypertension" in comorbidities,
            "has_chronic_disease": len(comorbidities) >= 2,
            
            # Risk
            "risk_score": round(risk_score, 3),
            "is_high_cost": is_high_cost,
            "is_high_risk": risk_score > 0.7,
            
            # Care management
            "primary_care_provider": pcp_id,
            "care_manager_assigned": risk_score > 0.7,
            
            # Status
            "enrollment_date": self.config.start_date.date(),
            "active": random.random() > 0.05,
            "deceased": False if age < 80 else random.random() < 0.05
        }
    
    def _generate_realistic_age(self) -> int:
        """Generate age with realistic distribution"""
        # Bimodal distribution: young adults and elderly
        if random.random() < 0.3:
            # Young adults (18-40)
            age = random.gauss(30, 8)
        else:
            # Middle-aged to elderly (40-90)
            age = random.gauss(60, 15)
        
        return max(18, min(100, int(age)))
    
    def _generate_ssn(self) -> str:
        """Generate realistic SSN format"""
        area = random.randint(1, 899)
        group = random.randint(1, 99)
        serial = random.randint(1, 9999)
        return f"{area:03d}-{group:02d}-{serial:04d}"
    
    def _generate_phone(self) -> str:
        """Generate realistic phone number"""
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(1000, 9999)
        return f"({area_code}) {exchange}-{number}"
    
    def _generate_email(self, first_name: str, last_name: str) -> str:
        """Generate realistic email"""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        separators = [".", "_", ""]
        
        separator = random.choice(separators)
        domain = random.choice(domains)
        
        # Sometimes add numbers
        suffix = str(random.randint(1, 99)) if random.random() < 0.3 else ""
        
        email = f"{first_name.lower()}{separator}{last_name.lower()}{suffix}@{domain}"
        return email
    
    def _generate_address(self) -> Dict:
        """Generate realistic address"""
        street_num = random.randint(1, 9999)
        street_names = ["Main", "Oak", "Maple", "Cedar", "Elm", "Washington", "Park", "Lake"]
        street_types = ["St", "Ave", "Rd", "Blvd", "Dr", "Ln", "Way", "Ct"]
        
        state = self.weighted_choice(list(self.STATES.keys()), list(self.STATES.values()))
        
        cities = {
            "CA": ["Los Angeles", "San Francisco", "San Diego", "Sacramento"],
            "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
            "FL": ["Miami", "Tampa", "Orlando", "Jacksonville"],
            "NY": ["New York", "Buffalo", "Rochester", "Albany"]
        }
        
        city = random.choice(cities.get(state, ["Springfield", "Franklin", "Clinton"]))
        
        return {
            "line1": f"{street_num} {random.choice(street_names)} {random.choice(street_types)}",
            "city": city,
            "state": state,
            "zip": f"{random.randint(10000, 99999)}"
        }
    
    def _generate_insurance_id(self, insurance_type: str) -> str:
        """Generate insurance ID based on type"""
        if insurance_type == "Medicare":
            return f"{random.randint(1, 9)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.randint(1000000, 9999999)}"
        elif insurance_type == "Medicaid":
            return f"MCD{random.randint(10000000, 99999999)}"
        else:
            return f"{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.randint(100000000, 999999999)}"
    
    def _generate_income(self, education: str, employment: str) -> int:
        """Generate income based on education and employment"""
        if employment == "Unemployed":
            return 0
        elif employment == "Retired":
            return random.randint(20000, 60000)
        elif employment == "Disabled":
            return random.randint(15000, 35000)
        else:
            # Employed - based on education
            income_ranges = {
                "Less than HS": (20000, 35000),
                "High School": (30000, 50000),
                "Some College": (35000, 60000),
                "Bachelor's": (50000, 90000),
                "Graduate": (70000, 150000)
            }
            min_income, max_income = income_ranges.get(education, (30000, 60000))
            return random.randint(min_income, max_income)
    
    def _generate_weight(self, age: int, height_cm: float) -> float:
        """Generate weight with realistic BMI distribution"""
        # Target BMI distribution (slightly overweight population)
        target_bmi = random.gauss(27, 5)  # Mean BMI 27 (slightly overweight)
        target_bmi = max(18, min(45, target_bmi))
        
        weight = target_bmi * ((height_cm/100) ** 2)
        return weight
    
    def _generate_correlated_comorbidities(self, age: int, bmi: float, 
                                          smoking_status: str) -> List[str]:
        """Generate comorbidities with realistic correlations"""
        comorbidities: Set[str] = set()
        
        # Age-based probability
        age_factor = min(age / 100, 0.8)
        
        # BMI-based conditions
        if bmi > 30:
            if random.random() < 0.4:
                comorbidities.add("Obesity")
            if random.random() < 0.3 * (1 + age_factor):
                comorbidities.add("Diabetes Type 2")
        
        # Smoking-related conditions
        if smoking_status == "Current":
            if random.random() < 0.3:
                comorbidities.add("COPD")
            if random.random() < 0.2:
                comorbidities.add("Heart Disease")
        
        # Age-related conditions
        if age > 60:
            if random.random() < 0.5:
                comorbidities.add("Hypertension")
            if random.random() < 0.2:
                comorbidities.add("Arthritis")
        
        # Add correlated conditions
        for condition in list(comorbidities):
            if condition in self.COMORBIDITY_CORRELATIONS:
                for correlated in self.COMORBIDITY_CORRELATIONS[condition]:
                    if random.random() < self.config.comorbidity_correlation:
                        comorbidities.add(correlated)
        
        return sorted(list(comorbidities))
    
    def _determine_cohorts(self, comorbidities: List[str], age: int, 
                          insurance_type: str) -> List[str]:
        """Determine patient cohorts"""
        cohorts = []
        
        if any("Diabetes" in c for c in comorbidities):
            cohorts.append("Diabetic")
        
        if "Hypertension" in comorbidities:
            cohorts.append("Hypertensive")
        
        if len(comorbidities) >= 2:
            cohorts.append("Chronic Disease Management")
        
        if len(comorbidities) >= 3:
            cohorts.append("High Comorbidity")
        
        if age >= 65 or insurance_type == "Medicare":
            cohorts.append("Medicare")
        
        if "Heart Disease" in comorbidities or "COPD" in comorbidities:
            cohorts.append("Cardiopulmonary")
        
        if insurance_type == "Medicaid":
            cohorts.append("Medicaid")
        
        return cohorts
    
    def _calculate_comprehensive_risk_score(self, age: int, comorbidities: List[str],
                                           bmi: float, smoking_status: str) -> float:
        """Calculate comprehensive risk score"""
        score = 0.0
        
        # Age component (0-0.3)
        score += min(age / 100, 0.3)
        
        # Comorbidity component (0-0.4)
        score += min(len(comorbidities) * 0.08, 0.4)
        
        # BMI component (0-0.1)
        if bmi > 35:
            score += 0.1
        elif bmi > 30:
            score += 0.05
        
        # Smoking component (0-0.1)
        if smoking_status == "Current":
            score += 0.1
        elif smoking_status == "Former":
            score += 0.05
        
        # High-risk conditions (0-0.1)
        high_risk = ["Heart Disease", "Cancer", "Kidney Disease", "COPD", "Stroke"]
        score += sum(0.02 for c in comorbidities if c in high_risk)
        
        return min(score, 1.0)
    
    def _log_generation_summary(self, df: pd.DataFrame):
        """Log generation summary"""
        self.logger.info(f"Generated {len(df)} patients with advanced attributes:")
        self.logger.info(f"  - Diabetic: {df['is_diabetic'].sum()} ({df['is_diabetic'].mean():.1%})")
        self.logger.info(f"  - Hypertensive: {df['is_hypertensive'].sum()} ({df['is_hypertensive'].mean():.1%})")
        self.logger.info(f"  - High risk: {df['is_high_risk'].sum()} ({df['is_high_risk'].mean():.1%})")
        self.logger.info(f"  - Medicare: {(df['insurance_type']=='Medicare').sum()}")
        self.logger.info(f"  - Medicaid: {(df['insurance_type']=='Medicaid').sum()}")
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV"""
        super().to_csv(df, filepath, "advanced patients")
