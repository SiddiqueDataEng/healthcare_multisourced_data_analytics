"""Claims data generator with realistic US healthcare billing patterns"""

import random
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
from datagenerator.config import GeneratorConfig
from datagenerator.base_generator import BaseGenerator


class ClaimsGenerator(BaseGenerator):
    """Generates comprehensive US healthcare claims data"""
    
    # US Healthcare Payers (realistic distribution)
    PAYERS = {
        "Medicare": 0.35,           # 35% - Government insurance for 65+
        "Medicaid": 0.20,           # 20% - Government insurance for low-income
        "Blue Cross Blue Shield": 0.15,  # 15% - Major commercial insurer
        "Aetna": 0.08,              # 8% - Commercial insurer
        "Cigna": 0.07,              # 7% - Commercial insurer
        "UnitedHealthcare": 0.10,   # 10% - Largest commercial insurer
        "Humana": 0.03,             # 3% - Commercial insurer
        "Self-Pay": 0.02            # 2% - Uninsured patients
    }
    
    # Claim adjudication statuses (realistic rates)
    ADJUDICATION_STATUSES = {
        "Paid": 0.82,               # 82% - Paid claims
        "Denied": 0.08,             # 8% - Denied claims
        "Pending": 0.05,            # 5% - Under review
        "Partial Payment": 0.04,    # 4% - Partially paid
        "Appealed": 0.01            # 1% - Under appeal
    }
    
    # Denial reasons (realistic US healthcare)
    DENIAL_REASONS = [
        "Prior Authorization Required",
        "Medical Necessity Not Established", 
        "Out of Network Provider",
        "Duplicate Claim",
        "Incorrect Coding",
        "Missing Documentation",
        "Timely Filing Limit Exceeded",
        "Non-Covered Service",
        "Coordination of Benefits Required",
        "Patient Not Eligible"
    ]
    
    # US Healthcare Service Lines
    SERVICE_LINES = [
        "Emergency Medicine", "Internal Medicine", "Surgery", "Cardiology",
        "Orthopedics", "Radiology", "Laboratory", "Pharmacy", "Oncology",
        "Neurology", "Gastroenterology", "Pulmonology", "Endocrinology"
    ]
    
    # Place of Service Codes (US Standard)
    PLACE_OF_SERVICE = {
        "11": "Office",
        "21": "Inpatient Hospital", 
        "22": "Outpatient Hospital",
        "23": "Emergency Room",
        "31": "Skilled Nursing Facility",
        "32": "Nursing Facility",
        "81": "Independent Laboratory",
        "99": "Other"
    }
    
    # Realistic procedure costs by CPT code (US market rates)
    PROCEDURE_COSTS = {
        # Office Visits
        "99213": (120, 180),        # Office visit, established patient
        "99214": (180, 280),        # Office visit, detailed
        "99215": (250, 400),        # Office visit, comprehensive
        "99201": (80, 120),         # New patient, straightforward
        "99202": (120, 180),        # New patient, low complexity
        "99203": (180, 280),        # New patient, moderate complexity
        
        # Emergency Department
        "99281": (150, 250),        # ED visit, straightforward
        "99282": (250, 400),        # ED visit, low complexity
        "99283": (400, 650),        # ED visit, moderate complexity
        "99284": (650, 1200),       # ED visit, high complexity
        "99285": (1200, 2500),      # ED visit, high complexity
        
        # Diagnostic Tests
        "93000": (45, 85),          # EKG
        "71046": (180, 350),        # Chest X-ray
        "80053": (85, 150),         # Comprehensive metabolic panel
        "85025": (25, 45),          # Complete blood count
        "80061": (65, 120),         # Lipid panel
        
        # Procedures
        "99291": (450, 850),        # Critical care, first hour
        "99292": (350, 650),        # Critical care, additional 30 min
        "27447": (15000, 35000),    # Total knee replacement
        "43239": (2500, 6500),      # Upper endoscopy
        "33935": (25000, 85000),    # Coronary artery bypass
        "47562": (8000, 18000),     # Laparoscopic cholecystectomy
        
        # Imaging
        "70553": (800, 1500),       # MRI brain
        "72148": (900, 1800),       # MRI lumbar spine
        "74177": (1200, 2200),      # CT abdomen/pelvis with contrast
        "76700": (200, 400),        # Abdominal ultrasound
        
        # Surgery
        "19301": (3500, 8500),      # Mastectomy
        "44970": (4500, 12000),     # Laparoscopic appendectomy
        "63030": (8000, 20000),     # Lumbar laminectomy
    }
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
    
    def _select_weighted_choice(self, choices_dict: Dict[str, float]) -> str:
        """Select item based on weighted probabilities"""
        choices = list(choices_dict.keys())
        weights = list(choices_dict.values())
        return np.random.choice(choices, p=weights)
    
    def _calculate_payment_amounts(self, claim_amount: float, payer: str, status: str) -> Dict[str, float]:
        """Calculate realistic payment amounts based on payer and status"""
        
        # Payer-specific reimbursement rates (realistic US rates)
        reimbursement_rates = {
            "Medicare": 0.78,           # Medicare pays ~78% of charges
            "Medicaid": 0.65,           # Medicaid pays ~65% of charges  
            "Blue Cross Blue Shield": 0.85,
            "Aetna": 0.82,
            "Cigna": 0.80,
            "UnitedHealthcare": 0.83,
            "Humana": 0.79,
            "Self-Pay": 0.45            # Self-pay often negotiated down
        }
        
        base_rate = reimbursement_rates.get(payer, 0.80)
        
        if status == "Paid":
            allowed_amount = claim_amount * base_rate * random.uniform(0.95, 1.05)
            paid_amount = allowed_amount * random.uniform(0.98, 1.0)
            patient_responsibility = claim_amount - paid_amount
            
        elif status == "Denied":
            allowed_amount = 0
            paid_amount = 0
            patient_responsibility = claim_amount
            
        elif status == "Partial Payment":
            allowed_amount = claim_amount * base_rate * random.uniform(0.95, 1.05)
            paid_amount = allowed_amount * random.uniform(0.60, 0.85)
            patient_responsibility = claim_amount - paid_amount
            
        elif status == "Pending":
            allowed_amount = claim_amount * base_rate
            paid_amount = 0  # Not yet paid
            patient_responsibility = 0  # TBD
            
        else:  # Appealed
            allowed_amount = claim_amount * base_rate * random.uniform(0.70, 0.90)
            paid_amount = 0  # Under review
            patient_responsibility = 0  # TBD
        
        return {
            "allowed_amount": round(allowed_amount, 2),
            "paid_amount": round(paid_amount, 2),
            "patient_responsibility": round(max(0, patient_responsibility), 2),
            "deductible_amount": round(patient_responsibility * random.uniform(0.3, 0.7), 2) if patient_responsibility > 0 else 0,
            "copay_amount": round(random.uniform(10, 50), 2) if status == "Paid" else 0,
            "coinsurance_amount": round(patient_responsibility * random.uniform(0.1, 0.3), 2) if patient_responsibility > 0 else 0
        }
    
    def _calculate_days_to_payment(self, status: str, payer: str) -> int:
        """Calculate realistic days to payment"""
        if status in ["Denied", "Pending", "Appealed"]:
            return 0
        
        # Realistic payment timelines by payer
        payment_days = {
            "Medicare": (14, 30),       # Medicare: 14-30 days
            "Medicaid": (30, 60),       # Medicaid: 30-60 days
            "Blue Cross Blue Shield": (21, 45),
            "Aetna": (18, 35),
            "Cigna": (20, 40),
            "UnitedHealthcare": (15, 35),
            "Humana": (25, 45),
            "Self-Pay": (60, 180)       # Self-pay: much longer
        }
        
        min_days, max_days = payment_days.get(payer, (20, 40))
        return random.randint(min_days, max_days)
    
    def generate_claims(self, patient_ids: List[str], provider_ids: List[str]) -> pd.DataFrame:
        """Generate comprehensive US healthcare claims data"""
        claims = []
        
        for i in range(self.config.num_claims):
            # Basic claim information
            claim_date = self.config.start_date + timedelta(
                days=random.randint(0, (self.config.end_date - self.config.start_date).days)
            )
            service_date = claim_date - timedelta(days=random.randint(0, 30))
            
            # Select payer and procedure
            payer = self._select_weighted_choice(self.PAYERS)
            procedure = random.choice(list(self.PROCEDURE_COSTS.keys()))
            service_line = random.choice(self.SERVICE_LINES)
            place_of_service_code = random.choice(list(self.PLACE_OF_SERVICE.keys()))
            place_of_service = self.PLACE_OF_SERVICE[place_of_service_code]
            
            # Calculate claim amount with realistic variation
            min_cost, max_cost = self.PROCEDURE_COSTS[procedure]
            base_amount = random.uniform(min_cost, max_cost)
            
            # Add geographic and facility variation
            geographic_modifier = random.uniform(0.85, 1.25)  # Geographic cost variation
            claim_amount = base_amount * geographic_modifier
            
            # Determine if fraudulent (inflate costs)
            is_fraudulent = random.random() < self.config.fraud_rate
            if is_fraudulent:
                claim_amount *= random.uniform(1.8, 4.0)
            
            # Adjudication status
            adjudication_status = self._select_weighted_choice(self.ADJUDICATION_STATUSES)
            
            # Calculate payment amounts
            payment_info = self._calculate_payment_amounts(claim_amount, payer, adjudication_status)
            
            # Days to payment
            days_to_payment = self._calculate_days_to_payment(adjudication_status, payer)
            
            # Denial reason if applicable
            denial_reason = None
            if adjudication_status == "Denied":
                denial_reason = random.choice(self.DENIAL_REASONS)
            
            # Revenue Code (for hospital billing)
            revenue_code = random.choice(["0450", "0360", "0370", "0250", "0300", "0710"])
            
            # Create comprehensive claim record
            claim = {
                # Basic Information
                "claim_id": f"CLM{i+1:08d}",
                "patient_id": random.choice(patient_ids),
                "provider_id": random.choice(provider_ids),
                "claim_date": claim_date.date(),
                "service_date": service_date.date(),
                
                # Payer Information
                "payer_name": payer,
                "payer_id": f"PAY{random.randint(10000, 99999)}",
                "member_id": f"MBR{random.randint(100000000, 999999999)}",
                
                # Service Information
                "procedure_code": procedure,
                "service_line": service_line,
                "place_of_service_code": place_of_service_code,
                "place_of_service": place_of_service,
                "revenue_code": revenue_code,
                "units": random.randint(1, 3),
                
                # Financial Information
                "claim_amount": round(claim_amount, 2),
                "allowed_amount": payment_info["allowed_amount"],
                "paid_amount": payment_info["paid_amount"],
                "patient_responsibility": payment_info["patient_responsibility"],
                "deductible_amount": payment_info["deductible_amount"],
                "copay_amount": payment_info["copay_amount"],
                "coinsurance_amount": payment_info["coinsurance_amount"],
                
                # Processing Information
                "adjudication_status": adjudication_status,
                "days_to_payment": days_to_payment,
                "denial_reason": denial_reason,
                "claim_line_number": random.randint(1, 5),
                
                # Quality Flags
                "is_fraudulent": is_fraudulent,
                "is_emergency": place_of_service == "Emergency Room",
                "is_inpatient": place_of_service == "Inpatient Hospital",
                
                # Additional US Healthcare Fields
                "drg_code": f"DRG{random.randint(100, 999)}" if place_of_service == "Inpatient Hospital" else None,
                "modifier_1": random.choice(["25", "26", "TC", "59", ""]),
                "ndc_code": f"NDC{random.randint(10000000000, 99999999999)}" if service_line == "Pharmacy" else None,
            }
            claims.append(claim)
        
        return pd.DataFrame(claims)
    
    def to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV with comprehensive billing data"""
        super().to_csv(df, filepath, "claims")
