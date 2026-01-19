"""
Data Validator - Comprehensive validation rules and checks
Validates data quality, business rules, and referential integrity
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_dataset(self, df: pd.DataFrame, data_type: str, 
                        reference_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            data_type: Type of data (patients, encounters, claims, etc.)
            reference_data: Dictionary of reference dataframes for FK validation
        
        Returns:
            Validation report with errors and warnings
        """
        logger.info(f"Validating {data_type} dataset ({len(df):,} rows)")
        
        self.errors = []
        self.warnings = []
        
        # 1. Schema validation
        self._validate_schema(df, data_type)
        
        # 2. Data type validation
        self._validate_data_types(df, data_type)
        
        # 3. Required fields validation
        self._validate_required_fields(df, data_type)
        
        # 4. Format validation
        self._validate_formats(df, data_type)
        
        # 5. Range validation
        self._validate_ranges(df, data_type)
        
        # 6. Business rules validation
        self._validate_business_rules(df, data_type)
        
        # 7. Referential integrity (if reference data provided)
        if reference_data:
            self._validate_referential_integrity(df, data_type, reference_data)
        
        # 8. Statistical validation
        self._validate_statistical_properties(df, data_type)
        
        # Generate report
        validation_report = {
            'data_type': data_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'is_valid': len(self.errors) == 0,
            'error_details': self.errors,
            'warning_details': self.warnings,
            'validation_score': self._calculate_validation_score()
        }
        
        self.validation_results[data_type] = validation_report
        
        logger.info(f"Validation complete: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        return validation_report
    
    def _validate_schema(self, df: pd.DataFrame, data_type: str):
        """Validate expected schema"""
        expected_schemas = {
            'patients': ['patient_id', 'first_name', 'last_name', 'date_of_birth', 'gender'],
            'encounters': ['encounter_id', 'patient_id', 'encounter_date', 'encounter_type'],
            'claims': ['claim_id', 'patient_id', 'claim_date', 'claim_amount'],
            'providers': ['provider_id', 'first_name', 'last_name', 'specialty'],
        }
        
        expected_cols = expected_schemas.get(data_type, [])
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            self.errors.append({
                'type': 'schema',
                'severity': 'error',
                'message': f"Missing required columns: {missing_cols}"
            })
    
    def _validate_data_types(self, df: pd.DataFrame, data_type: str):
        """Validate data types"""
        for col in df.columns:
            # Date columns should be datetime
            if 'date' in col.lower():
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    self.warnings.append({
                        'type': 'data_type',
                        'column': col,
                        'message': f"{col} should be datetime, found {df[col].dtype}"
                    })
            
            # Amount/cost columns should be numeric
            elif any(x in col.lower() for x in ['amount', 'cost', 'price']):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.errors.append({
                        'type': 'data_type',
                        'column': col,
                        'message': f"{col} should be numeric, found {df[col].dtype}"
                    })
    
    def _validate_required_fields(self, df: pd.DataFrame, data_type: str):
        """Validate required fields are not null"""
        required_fields = {
            'patients': ['patient_id', 'first_name', 'last_name', 'date_of_birth'],
            'encounters': ['encounter_id', 'patient_id', 'encounter_date'],
            'claims': ['claim_id', 'patient_id', 'claim_date', 'claim_amount'],
        }
        
        required = required_fields.get(data_type, [])
        
        for col in required:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.errors.append({
                        'type': 'required_field',
                        'column': col,
                        'message': f"{col} has {null_count} null values ({null_count/len(df)*100:.1f}%)"
                    })
    
    def _validate_formats(self, df: pd.DataFrame, data_type: str):
        """Validate data formats"""
        
        if data_type == 'patients':
            # Validate SSN format
            if 'ssn' in df.columns:
                invalid_ssn = df[df['ssn'].notna() & ~df['ssn'].str.match(r'^\d{3}-\d{2}-\d{4}$')]
                if len(invalid_ssn) > 0:
                    self.warnings.append({
                        'type': 'format',
                        'column': 'ssn',
                        'message': f"{len(invalid_ssn)} SSNs have invalid format (expected XXX-XX-XXXX)"
                    })
            
            # Validate email format
            if 'email' in df.columns:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_email = df[df['email'].notna() & ~df['email'].str.match(email_pattern)]
                if len(invalid_email) > 0:
                    self.warnings.append({
                        'type': 'format',
                        'column': 'email',
                        'message': f"{len(invalid_email)} emails have invalid format"
                    })
            
            # Validate phone format
            if 'phone' in df.columns:
                phone_pattern = r'^\(\d{3}\) \d{3}-\d{4}$'
                invalid_phone = df[df['phone'].notna() & ~df['phone'].str.match(phone_pattern)]
                if len(invalid_phone) > 0:
                    self.warnings.append({
                        'type': 'format',
                        'column': 'phone',
                        'message': f"{len(invalid_phone)} phone numbers have invalid format"
                    })
            
            # Validate gender values
            if 'gender' in df.columns:
                valid_genders = ['M', 'F', 'Male', 'Female']
                invalid_gender = df[df['gender'].notna() & ~df['gender'].isin(valid_genders)]
                if len(invalid_gender) > 0:
                    self.errors.append({
                        'type': 'format',
                        'column': 'gender',
                        'message': f"{len(invalid_gender)} invalid gender values"
                    })
    
    def _validate_ranges(self, df: pd.DataFrame, data_type: str):
        """Validate numeric ranges"""
        
        if data_type == 'patients':
            # Age range
            if 'age' in df.columns:
                invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
                if len(invalid_age) > 0:
                    self.errors.append({
                        'type': 'range',
                        'column': 'age',
                        'message': f"{len(invalid_age)} ages outside valid range (0-120)"
                    })
            
            # BMI range
            if 'bmi' in df.columns:
                invalid_bmi = df[(df['bmi'] < 10) | (df['bmi'] > 80)]
                if len(invalid_bmi) > 0:
                    self.warnings.append({
                        'type': 'range',
                        'column': 'bmi',
                        'message': f"{len(invalid_bmi)} BMI values outside typical range (10-80)"
                    })
            
            # Height range
            if 'height_cm' in df.columns:
                invalid_height = df[(df['height_cm'] < 100) | (df['height_cm'] > 250)]
                if len(invalid_height) > 0:
                    self.warnings.append({
                        'type': 'range',
                        'column': 'height_cm',
                        'message': f"{len(invalid_height)} heights outside typical range (100-250 cm)"
                    })
            
            # Weight range
            if 'weight_kg' in df.columns:
                invalid_weight = df[(df['weight_kg'] < 20) | (df['weight_kg'] > 300)]
                if len(invalid_weight) > 0:
                    self.warnings.append({
                        'type': 'range',
                        'column': 'weight_kg',
                        'message': f"{len(invalid_weight)} weights outside typical range (20-300 kg)"
                    })
        
        elif data_type == 'claims':
            # Claim amounts should be positive
            amount_cols = [col for col in df.columns if 'amount' in col.lower()]
            for col in amount_cols:
                negative_amounts = df[df[col] < 0]
                if len(negative_amounts) > 0:
                    self.errors.append({
                        'type': 'range',
                        'column': col,
                        'message': f"{len(negative_amounts)} negative amounts found"
                    })

    
    def _validate_business_rules(self, df: pd.DataFrame, data_type: str):
        """Validate business rules"""
        
        if data_type == 'patients':
            # Medicare patients should typically be 65+
            if 'insurance_type' in df.columns and 'age' in df.columns:
                medicare_under_65 = df[(df['insurance_type'] == 'Medicare') & (df['age'] < 65)]
                if len(medicare_under_65) > 0:
                    # This is a warning, not error (disabled can have Medicare)
                    self.warnings.append({
                        'type': 'business_rule',
                        'message': f"{len(medicare_under_65)} Medicare patients under 65 (may be disabled)"
                    })
            
            # BMI should match height/weight
            if all(col in df.columns for col in ['bmi', 'height_cm', 'weight_kg']):
                df['_calculated_bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
                bmi_mismatch = df[abs(df['bmi'] - df['_calculated_bmi']) > 1]
                if len(bmi_mismatch) > 0:
                    self.warnings.append({
                        'type': 'business_rule',
                        'message': f"{len(bmi_mismatch)} BMI values don't match height/weight"
                    })
                df = df.drop(columns=['_calculated_bmi'])
            
            # Date of birth should be in the past
            if 'date_of_birth' in df.columns:
                future_dob = df[df['date_of_birth'] > pd.Timestamp.now()]
                if len(future_dob) > 0:
                    self.errors.append({
                        'type': 'business_rule',
                        'column': 'date_of_birth',
                        'message': f"{len(future_dob)} future dates of birth found"
                    })
        
        elif data_type == 'encounters':
            # Discharge date should be after admission date
            if 'admission_date' in df.columns and 'discharge_date' in df.columns:
                invalid_dates = df[df['discharge_date'] < df['admission_date']]
                if len(invalid_dates) > 0:
                    self.errors.append({
                        'type': 'business_rule',
                        'message': f"{len(invalid_dates)} encounters with discharge before admission"
                    })
            
            # Length of stay should match dates
            if all(col in df.columns for col in ['admission_date', 'discharge_date', 'length_of_stay']):
                df['_calculated_los'] = (df['discharge_date'] - df['admission_date']).dt.days
                los_mismatch = df[abs(df['length_of_stay'] - df['_calculated_los']) > 1]
                if len(los_mismatch) > 0:
                    self.warnings.append({
                        'type': 'business_rule',
                        'message': f"{len(los_mismatch)} length of stay doesn't match dates"
                    })
                df = df.drop(columns=['_calculated_los'])
        
        elif data_type == 'claims':
            # Paid amount should not exceed billed amount
            if 'billed_amount' in df.columns and 'paid_amount' in df.columns:
                overpaid = df[df['paid_amount'] > df['billed_amount']]
                if len(overpaid) > 0:
                    self.errors.append({
                        'type': 'business_rule',
                        'message': f"{len(overpaid)} claims with paid > billed amount"
                    })
            
            # Patient responsibility should be reasonable
            if all(col in df.columns for col in ['billed_amount', 'paid_amount', 'patient_responsibility']):
                df['_expected_patient'] = df['billed_amount'] - df['paid_amount']
                patient_mismatch = df[abs(df['patient_responsibility'] - df['_expected_patient']) > 1]
                if len(patient_mismatch) > 0:
                    self.warnings.append({
                        'type': 'business_rule',
                        'message': f"{len(patient_mismatch)} claims with patient responsibility mismatch"
                    })
                df = df.drop(columns=['_expected_patient'])
    
    def _validate_referential_integrity(self, df: pd.DataFrame, data_type: str, 
                                       reference_data: Dict[str, pd.DataFrame]):
        """Validate foreign key relationships"""
        
        # Define FK relationships
        fk_relationships = {
            'encounters': {'patient_id': 'patients', 'provider_id': 'providers'},
            'claims': {'patient_id': 'patients', 'encounter_id': 'encounters'},
            'registry': {'patient_id': 'patients'}
        }
        
        relationships = fk_relationships.get(data_type, {})
        
        for fk_col, ref_table in relationships.items():
            if fk_col in df.columns and ref_table in reference_data:
                ref_df = reference_data[ref_table]
                ref_pk = f"{ref_table[:-1]}_id" if ref_table.endswith('s') else f"{ref_table}_id"
                
                if ref_pk in ref_df.columns:
                    # Find orphaned records
                    orphaned = df[~df[fk_col].isin(ref_df[ref_pk])]
                    if len(orphaned) > 0:
                        self.errors.append({
                            'type': 'referential_integrity',
                            'column': fk_col,
                            'message': f"{len(orphaned)} orphaned records (FK not in {ref_table})"
                        })
    
    def _validate_statistical_properties(self, df: pd.DataFrame, data_type: str):
        """Validate statistical properties"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if '_id' in col.lower():
                continue
            
            # Check for constant columns
            if df[col].nunique() == 1:
                self.warnings.append({
                    'type': 'statistical',
                    'column': col,
                    'message': f"{col} has only one unique value"
                })
            
            # Check for extreme skewness
            if len(df[col].dropna()) > 0:
                skewness = df[col].skew()
                if abs(skewness) > 5:
                    self.warnings.append({
                        'type': 'statistical',
                        'column': col,
                        'message': f"{col} has extreme skewness ({skewness:.2f})"
                    })
    
    def _calculate_validation_score(self) -> float:
        """Calculate overall validation score (0-100)"""
        # Start with 100, deduct points for errors and warnings
        score = 100.0
        score -= len(self.errors) * 10  # 10 points per error
        score -= len(self.warnings) * 2  # 2 points per warning
        
        return max(0.0, score)
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get validation summary as DataFrame"""
        summary_data = []
        
        for data_type, results in self.validation_results.items():
            summary_data.append({
                'Data Type': data_type,
                'Total Rows': results['total_rows'],
                'Errors': results['errors'],
                'Warnings': results['warnings'],
                'Valid': 'Yes' if results['is_valid'] else 'No',
                'Score': f"{results['validation_score']:.1f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def export_validation_report(self, filepath: str):
        """Export detailed validation report"""
        with open(filepath, 'w') as f:
            f.write("Data Validation Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for data_type, results in self.validation_results.items():
                f.write(f"\n{data_type.upper()}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total Rows: {results['total_rows']:,}\n")
                f.write(f"Total Columns: {results['total_columns']}\n")
                f.write(f"Validation Score: {results['validation_score']:.1f}/100\n")
                f.write(f"Status: {'VALID' if results['is_valid'] else 'INVALID'}\n\n")
                
                if results['errors']:
                    f.write(f"ERRORS ({len(results['errors'])}):\n")
                    for i, error in enumerate(results['errors'], 1):
                        f.write(f"  {i}. [{error['type']}] {error['message']}\n")
                    f.write("\n")
                
                if results['warnings']:
                    f.write(f"WARNINGS ({len(results['warnings'])}):\n")
                    for i, warning in enumerate(results['warnings'], 1):
                        f.write(f"  {i}. [{warning['type']}] {warning['message']}\n")
                    f.write("\n")
        
        logger.info(f"Validation report exported to {filepath}")
    
    def get_validation_results(self) -> Dict:
        """Get all validation results"""
        return self.validation_results
