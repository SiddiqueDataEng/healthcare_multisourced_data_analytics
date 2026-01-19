"""
Advanced Data Cleaner - Production-grade data cleaning with multiple methods
Includes outlier detection, standardization, validation, and referential integrity
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedDataCleaner:
    """Advanced data cleaning with multiple detection methods"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.cleaning_report = {}
        self.transformations = []
    
    def _default_config(self) -> Dict:
        """Default cleaning configuration"""
        return {
            'outlier_method': 'iqr',  # iqr, zscore, isolation_forest, all
            'iqr_multiplier': 3.0,
            'zscore_threshold': 3.0,
            'isolation_contamination': 0.05,
            'remove_duplicates': True,
            'standardize_formats': True,
            'validate_integrity': True,
            'handle_missing': 'drop',  # drop, fill, flag
            'missing_threshold': 0.5  # Drop columns with >50% missing
        }
    
    def clean_comprehensive(self, df: pd.DataFrame, data_type: str = "generic") -> pd.DataFrame:
        """
        Comprehensive cleaning pipeline
        
        Args:
            df: Input dataframe
            data_type: Type of data for context-aware cleaning
        
        Returns:
            Cleaned dataframe with audit trail
        """
        logger.info(f"Starting advanced cleaning for {data_type} ({len(df):,} rows)")
        initial_rows = len(df)
        
        # Track transformations
        self.transformations = []
        
        # 1. Remove duplicates
        df = self._remove_duplicates_advanced(df, data_type)
        
        # 2. Handle missing data
        df = self._handle_missing_data(df)
        
        # 3. Standardize data types
        df = self._standardize_data_types(df)
        
        # 4. Detect and handle outliers
        df = self._detect_outliers_multi_method(df, data_type)
        
        # 5. Standardize formats
        if self.config['standardize_formats']:
            df = self._standardize_all_formats(df, data_type)
        
        # 6. Validate referential integrity
        if self.config['validate_integrity']:
            df = self._validate_referential_integrity(df, data_type)
        
        # 7. Apply business rules
        df = self._apply_business_rules(df, data_type)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        # Generate report
        self.cleaning_report[data_type] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'removed_rows': removed_rows,
            'removal_rate': f"{(removed_rows/initial_rows*100):.2f}%",
            'transformations': len(self.transformations),
            'quality_improvement': self._calculate_quality_improvement(df)
        }
        
        logger.info(f"Cleaning complete: {removed_rows:,} rows removed ({removed_rows/initial_rows*100:.1f}%)")
        logger.info(f"Applied {len(self.transformations)} transformations")
        
        return df

    
    def _remove_duplicates_advanced(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Advanced duplicate detection and removal"""
        initial_count = len(df)
        
        # Identify primary key columns
        pk_cols = self._identify_primary_keys(df)
        
        if pk_cols:
            # Remove exact duplicates on PK
            df = df.drop_duplicates(subset=pk_cols, keep='first')
            exact_dupes = initial_count - len(df)
            
            if exact_dupes > 0:
                self.transformations.append(f"Removed {exact_dupes} exact duplicates on {pk_cols}")
                logger.info(f"Removed {exact_dupes} exact duplicates")
        
        # Check for near-duplicates (fuzzy matching on names, etc.)
        if data_type == 'patients':
            df = self._remove_near_duplicate_patients(df)
        
        return df
    
    def _remove_near_duplicate_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect near-duplicate patients (similar names, DOB)"""
        if 'first_name' not in df.columns or 'last_name' not in df.columns:
            return df
        
        initial_count = len(df)
        
        # Create composite key for fuzzy matching
        df['_name_key'] = (
            df['first_name'].str.lower().str.strip() + '_' +
            df['last_name'].str.lower().str.strip()
        )
        
        if 'date_of_birth' in df.columns:
            df['_name_key'] = df['_name_key'] + '_' + df['date_of_birth'].astype(str)
        
        # Remove duplicates on fuzzy key
        df = df.drop_duplicates(subset=['_name_key'], keep='first')
        df = df.drop(columns=['_name_key'])
        
        near_dupes = initial_count - len(df)
        if near_dupes > 0:
            self.transformations.append(f"Removed {near_dupes} near-duplicate patients")
            logger.info(f"Removed {near_dupes} near-duplicate patients")
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data based on configuration"""
        initial_count = len(df)
        
        # Drop columns with excessive missing data
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > self.config['missing_threshold']].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.transformations.append(f"Dropped {len(cols_to_drop)} columns with >{self.config['missing_threshold']*100}% missing")
            logger.info(f"Dropped columns: {cols_to_drop}")
        
        # Handle remaining missing data
        if self.config['handle_missing'] == 'drop':
            # Drop rows with any missing critical values
            critical_cols = [col for col in df.columns if '_id' in col.lower()]
            df = df.dropna(subset=critical_cols, how='any')
        
        elif self.config['handle_missing'] == 'fill':
            # Fill numeric with median, categorical with mode
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        removed = initial_count - len(df)
        if removed > 0:
            self.transformations.append(f"Removed {removed} rows with missing critical data")
        
        return df
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types with error handling"""
        for col in df.columns:
            # Date columns
            if 'date' in col.lower() or col.lower().endswith('_dt'):
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        self.transformations.append(f"Converted {col} to datetime")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to datetime: {e}")
            
            # Numeric columns
            elif any(x in col.lower() for x in ['amount', 'cost', 'price', 'rate', 'score']):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        self.transformations.append(f"Converted {col} to numeric")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to numeric: {e}")
            
            # Boolean columns
            elif any(x in col.lower() for x in ['is_', 'has_', 'flag']):
                if df[col].dtype != bool:
                    try:
                        df[col] = df[col].astype(bool)
                        self.transformations.append(f"Converted {col} to boolean")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to boolean: {e}")
        
        return df
    
    def _detect_outliers_multi_method(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Detect outliers using multiple methods"""
        method = self.config['outlier_method']
        
        if method == 'all':
            # Use ensemble approach - flag if detected by multiple methods
            df = self._detect_outliers_ensemble(df)
        elif method == 'iqr':
            df = self._detect_outliers_iqr(df)
        elif method == 'zscore':
            df = self._detect_outliers_zscore(df)
        elif method == 'isolation_forest':
            df = self._detect_outliers_isolation_forest(df)
        
        return df
    
    def _detect_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR method for outlier detection"""
        initial_count = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if '_id' in col.lower() or col.lower() == 'id':
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['iqr_multiplier'] * IQR
            upper_bound = Q3 + self.config['iqr_multiplier'] * IQR
            
            outliers_before = len(df)
            df = df[(df[col] >= lower_bound) | (df[col].isna())]
            df = df[(df[col] <= upper_bound) | (df[col].isna())]
            outliers_removed = outliers_before - len(df)
            
            if outliers_removed > 0:
                self.transformations.append(f"IQR: Removed {outliers_removed} outliers from {col}")
        
        total_removed = initial_count - len(df)
        if total_removed > 0:
            logger.info(f"IQR method removed {total_removed} outlier rows")
        
        return df
    
    def _detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score method for outlier detection"""
        initial_count = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if '_id' in col.lower() or col.lower() == 'id':
                continue
            
            if df[col].std() == 0:
                continue
            
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = self.config['zscore_threshold']
            
            # Create mask for valid rows
            mask = pd.Series(True, index=df.index)
            mask.loc[df[col].notna()] = z_scores < threshold
            
            outliers_removed = (~mask).sum()
            df = df[mask]
            
            if outliers_removed > 0:
                self.transformations.append(f"Z-score: Removed {outliers_removed} outliers from {col}")
        
        total_removed = initial_count - len(df)
        if total_removed > 0:
            logger.info(f"Z-score method removed {total_removed} outlier rows")
        
        return df

    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolation Forest for multivariate outlier detection"""
        initial_count = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if '_id' not in col.lower()]
        
        if len(numeric_cols) < 2:
            logger.info("Not enough numeric columns for Isolation Forest")
            return df
        
        # Prepare data
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config['isolation_contamination'],
            random_state=42
        )
        predictions = iso_forest.fit_predict(X)
        
        # Keep only inliers (prediction == 1)
        df = df[predictions == 1]
        
        outliers_removed = initial_count - len(df)
        if outliers_removed > 0:
            self.transformations.append(f"Isolation Forest: Removed {outliers_removed} multivariate outliers")
            logger.info(f"Isolation Forest removed {outliers_removed} outlier rows")
        
        return df
    
    def _detect_outliers_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensemble outlier detection - flag if detected by 2+ methods"""
        initial_count = len(df)
        
        # Create outlier flags for each method
        df['_outlier_iqr'] = False
        df['_outlier_zscore'] = False
        df['_outlier_iso'] = False
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if '_id' not in col.lower() and not col.startswith('_outlier')]
        
        # IQR flags
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df.loc[(df[col] < lower) | (df[col] > upper), '_outlier_iqr'] = True
        
        # Z-score flags
        for col in numeric_cols:
            if df[col].std() > 0:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                df.loc[z_scores > 3, '_outlier_zscore'] = True
        
        # Isolation Forest flags
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            predictions = iso_forest.fit_predict(X)
            df['_outlier_iso'] = predictions == -1
        
        # Flag rows detected by 2+ methods
        df['_outlier_count'] = (
            df['_outlier_iqr'].astype(int) +
            df['_outlier_zscore'].astype(int) +
            df['_outlier_iso'].astype(int)
        )
        
        # Remove rows flagged by 2+ methods
        df = df[df['_outlier_count'] < 2]
        
        # Clean up temporary columns
        df = df.drop(columns=['_outlier_iqr', '_outlier_zscore', '_outlier_iso', '_outlier_count'])
        
        outliers_removed = initial_count - len(df)
        if outliers_removed > 0:
            self.transformations.append(f"Ensemble: Removed {outliers_removed} outliers (detected by 2+ methods)")
            logger.info(f"Ensemble method removed {outliers_removed} outlier rows")
        
        return df
    
    def _standardize_all_formats(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Comprehensive format standardization"""
        
        # Standardize IDs (uppercase, trim)
        id_cols = [col for col in df.columns if '_id' in col.lower() or col.lower() == 'id']
        for col in id_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.upper().str.strip()
        
        # Standardize string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in id_cols:
                df[col] = df[col].str.strip()
        
        # Data type specific standardization
        if data_type == 'patients':
            df = self._standardize_patient_data(df)
        elif data_type == 'encounters':
            df = self._standardize_encounter_data(df)
        elif data_type == 'claims':
            df = self._standardize_claims_data(df)
        
        return df
    
    def _standardize_patient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize patient-specific data"""
        
        # Gender (M/F uppercase)
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.upper().str.strip()
            df['gender'] = df['gender'].replace({'MALE': 'M', 'FEMALE': 'F'})
        
        # Names (Title case)
        for col in ['first_name', 'last_name']:
            if col in df.columns:
                df[col] = df[col].str.title().str.strip()
        
        # Race/Ethnicity (Title case)
        if 'race' in df.columns:
            df['race'] = df['race'].str.title()
        
        # Phone numbers (standardize format)
        if 'phone' in df.columns:
            df['phone'] = df['phone'].apply(self._standardize_phone)
        
        # Email (lowercase)
        if 'email' in df.columns:
            df['email'] = df['email'].str.lower().str.strip()
        
        # SSN (format XXX-XX-XXXX)
        if 'ssn' in df.columns:
            df['ssn'] = df['ssn'].apply(self._standardize_ssn)
        
        # State codes (uppercase)
        if 'state' in df.columns:
            df['state'] = df['state'].str.upper().str.strip()
        
        # Zip codes (5 digits)
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
        
        return df
    
    def _standardize_encounter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize encounter-specific data"""
        
        # Encounter type (uppercase)
        if 'encounter_type' in df.columns:
            df['encounter_type'] = df['encounter_type'].str.upper()
        
        # Status (Title case)
        if 'status' in df.columns:
            df['status'] = df['status'].str.title()
        
        return df
    
    def _standardize_claims_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize claims-specific data"""
        
        # Ensure amounts are positive
        amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'cost' in col.lower()]
        for col in amount_cols:
            df[col] = df[col].abs()
        
        # Claim status (uppercase)
        if 'claim_status' in df.columns:
            df['claim_status'] = df['claim_status'].str.upper()
        
        return df
    
    def _standardize_phone(self, phone: str) -> str:
        """Standardize phone number format"""
        if pd.isna(phone):
            return phone
        
        # Remove all non-digits
        digits = ''.join(filter(str.isdigit, str(phone)))
        
        # Format as (XXX) XXX-XXXX
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        
        return phone
    
    def _standardize_ssn(self, ssn: str) -> str:
        """Standardize SSN format"""
        if pd.isna(ssn):
            return ssn
        
        # Remove all non-digits
        digits = ''.join(filter(str.isdigit, str(ssn)))
        
        # Format as XXX-XX-XXXX
        if len(digits) == 9:
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
        
        return ssn

    
    def _validate_referential_integrity(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate referential integrity (will be enhanced with cross-table validation)"""
        initial_count = len(df)
        
        # Check for null foreign keys
        fk_cols = [col for col in df.columns if col.endswith('_id') and col != f"{data_type}_id"]
        
        for col in fk_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df = df[df[col].notna()]
                self.transformations.append(f"Removed {null_count} rows with null {col}")
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Referential integrity check removed {removed} rows")
        
        return df
    
    def _apply_business_rules(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply business rule validation"""
        initial_count = len(df)
        
        if data_type == 'patients':
            df = self._validate_patient_rules(df)
        elif data_type == 'encounters':
            df = self._validate_encounter_rules(df)
        elif data_type == 'claims':
            df = self._validate_claims_rules(df)
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Business rules removed {removed} invalid rows")
        
        return df
    
    def _validate_patient_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate patient business rules"""
        initial_count = len(df)
        
        # Age must be 0-120
        if 'age' in df.columns:
            df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        
        # BMI must be 10-80
        if 'bmi' in df.columns:
            df = df[(df['bmi'] >= 10) & (df['bmi'] <= 80)]
        
        # Medicare patients should be 65+ (with exceptions)
        if 'insurance_type' in df.columns and 'age' in df.columns:
            # Allow Medicare for disabled (under 65)
            medicare_under_65 = (df['insurance_type'] == 'Medicare') & (df['age'] < 65)
            if 'employment_status' in df.columns:
                # Keep if disabled
                invalid_medicare = medicare_under_65 & (df['employment_status'] != 'Disabled')
                df = df[~invalid_medicare]
            else:
                # Remove all Medicare under 65 if no employment status
                df = df[~medicare_under_65]
        
        # Height must be reasonable (100-250 cm)
        if 'height_cm' in df.columns:
            df = df[(df['height_cm'] >= 100) & (df['height_cm'] <= 250)]
        
        # Weight must be reasonable (20-300 kg)
        if 'weight_kg' in df.columns:
            df = df[(df['weight_kg'] >= 20) & (df['weight_kg'] <= 300)]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.transformations.append(f"Patient rules: Removed {removed} invalid patients")
        
        return df
    
    def _validate_encounter_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate encounter business rules"""
        initial_count = len(df)
        
        # Encounter date must be reasonable (2000-present)
        if 'encounter_date' in df.columns:
            df = df[df['encounter_date'] >= pd.Timestamp('2000-01-01')]
            df = df[df['encounter_date'] <= pd.Timestamp.now()]
        
        # Length of stay must be positive
        if 'length_of_stay' in df.columns:
            df = df[df['length_of_stay'] >= 0]
        
        # Discharge date must be after admission date
        if 'admission_date' in df.columns and 'discharge_date' in df.columns:
            df = df[df['discharge_date'] >= df['admission_date']]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.transformations.append(f"Encounter rules: Removed {removed} invalid encounters")
        
        return df
    
    def _validate_claims_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate claims business rules"""
        initial_count = len(df)
        
        # Claim amounts must be positive
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        for col in amount_cols:
            df = df[df[col] >= 0]
        
        # Claim date must be reasonable
        if 'claim_date' in df.columns:
            df = df[df['claim_date'] >= pd.Timestamp('2000-01-01')]
            df = df[df['claim_date'] <= pd.Timestamp.now()]
        
        # Paid amount should not exceed billed amount
        if 'billed_amount' in df.columns and 'paid_amount' in df.columns:
            df = df[df['paid_amount'] <= df['billed_amount']]
        
        # Patient responsibility should not be negative
        if 'patient_responsibility' in df.columns:
            df = df[df['patient_responsibility'] >= 0]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.transformations.append(f"Claims rules: Removed {removed} invalid claims")
        
        return df
    
    def _identify_primary_keys(self, df: pd.DataFrame) -> List[str]:
        """Identify primary key columns"""
        pk_candidates = []
        
        for col in df.columns:
            if '_id' in col.lower() or col.lower() in ['id', 'key']:
                if df[col].nunique() == len(df):
                    pk_candidates.append(col)
        
        return pk_candidates[:1] if pk_candidates else []
    
    def _calculate_quality_improvement(self, df: pd.DataFrame) -> str:
        """Calculate quality improvement metrics"""
        # Simple quality score based on completeness
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        return f"{completeness:.1f}%"
    
    def get_cleaning_report(self) -> Dict:
        """Get detailed cleaning report"""
        return self.cleaning_report
    
    def get_transformations(self) -> List[str]:
        """Get list of all transformations applied"""
        return self.transformations
    
    def export_audit_trail(self, filepath: str) -> None:
        """Export audit trail to file"""
        with open(filepath, 'w') as f:
            f.write("Data Cleaning Audit Trail\n")
            f.write("=" * 50 + "\n\n")
            
            for data_type, report in self.cleaning_report.items():
                f.write(f"\n{data_type.upper()}\n")
                f.write("-" * 50 + "\n")
                for key, value in report.items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\n\nTransformations Applied:\n")
            f.write("-" * 50 + "\n")
            for i, transform in enumerate(self.transformations, 1):
                f.write(f"{i}. {transform}\n")
        
        logger.info(f"Audit trail exported to {filepath}")
