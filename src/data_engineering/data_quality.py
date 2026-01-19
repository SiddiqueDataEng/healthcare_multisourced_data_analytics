"""
Data Quality Checker - Validates data quality and generates quality reports
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Checks and reports on data quality"""
    
    def __init__(self):
        self.quality_report = {}
    
    def check_data_quality(self, df: pd.DataFrame, data_type: str = 'generic') -> Dict:
        """
        Comprehensive data quality check
        
        Args:
            df: Input dataframe
            data_type: Type of data for context-aware checks
        
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"Checking data quality for {data_type}")
        
        quality_metrics = {
            'data_type': data_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'completeness': self._check_completeness(df),
            'uniqueness': self._check_uniqueness(df),
            'validity': self._check_validity(df),
            'consistency': self._check_consistency(df),
            'accuracy': self._check_accuracy(df, data_type),
            'overall_score': 0
        }
        
        # Calculate overall score
        scores = [
            quality_metrics['completeness']['score'],
            quality_metrics['uniqueness']['score'],
            quality_metrics['validity']['score'],
            quality_metrics['consistency']['score'],
            quality_metrics['accuracy']['score']
        ]
        quality_metrics['overall_score'] = np.mean(scores)
        
        self.quality_report[data_type] = quality_metrics
        
        logger.info(f"Data quality score: {quality_metrics['overall_score']:.1f}%")
        
        return quality_metrics
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness (missing values)"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_pct = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        missing_by_column = df.isnull().sum()
        critical_missing = missing_by_column[missing_by_column > 0]
        
        return {
            'score': completeness_pct,
            'total_missing': missing_cells,
            'missing_percentage': 100 - completeness_pct,
            'columns_with_missing': len(critical_missing),
            'details': critical_missing.to_dict()
        }
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records"""
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        duplicates = total_rows - unique_rows
        uniqueness_pct = (unique_rows / total_rows * 100) if total_rows > 0 else 0
        
        return {
            'score': uniqueness_pct,
            'total_rows': total_rows,
            'unique_rows': unique_rows,
            'duplicate_rows': duplicates,
            'duplicate_percentage': 100 - uniqueness_pct
        }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict:
        """Check data validity (data types, ranges)"""
        validity_issues = []
        
        for col in df.columns:
            # Check for unexpected data types
            if 'date' in col.lower() and not pd.api.types.is_datetime64_any_dtype(df[col]):
                validity_issues.append(f"{col}: Expected datetime, got {df[col].dtype}")
            
            # Check for negative amounts
            if 'amount' in col.lower() or 'cost' in col.lower():
                if (df[col] < 0).any():
                    negative_count = (df[col] < 0).sum()
                    validity_issues.append(f"{col}: {negative_count} negative values found")
            
            # Check for invalid IDs
            if '_id' in col.lower():
                if df[col].isnull().any():
                    null_count = df[col].isnull().sum()
                    validity_issues.append(f"{col}: {null_count} null IDs found")
        
        validity_pct = max(0, 100 - (len(validity_issues) * 10))
        
        return {
            'score': validity_pct,
            'issues_found': len(validity_issues),
            'issues': validity_issues
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check data consistency"""
        consistency_issues = []
        
        # Check for inconsistent date ranges
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if df[col].min() < pd.Timestamp('2000-01-01'):
                consistency_issues.append(f"{col}: Date before 2000 found")
            if df[col].max() > pd.Timestamp.now():
                consistency_issues.append(f"{col}: Future date found")
        
        # Check for inconsistent numeric ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'age' in col.lower():
                if (df[col] < 0).any() or (df[col] > 150).any():
                    consistency_issues.append(f"{col}: Age outside valid range (0-150)")
        
        consistency_pct = max(0, 100 - (len(consistency_issues) * 10))
        
        return {
            'score': consistency_pct,
            'issues_found': len(consistency_issues),
            'issues': consistency_issues
        }
    
    def _check_accuracy(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Check data accuracy based on business rules"""
        accuracy_issues = []
        
        if data_type == 'patients':
            # Check age distribution
            if 'age' in df.columns:
                mean_age = df['age'].mean()
                if mean_age < 20 or mean_age > 80:
                    accuracy_issues.append(f"Unusual mean age: {mean_age:.1f}")
        
        elif data_type == 'encounters':
            # Check encounter date is reasonable
            if 'encounter_date' in df.columns:
                recent_pct = (df['encounter_date'] >= pd.Timestamp.now() - pd.Timedelta(days=365)).sum() / len(df) * 100
                if recent_pct < 50:
                    accuracy_issues.append(f"Only {recent_pct:.1f}% encounters in last year")
        
        elif data_type == 'claims':
            # Check claim amounts are reasonable
            if 'claim_amount' in df.columns:
                mean_claim = df['claim_amount'].mean()
                if mean_claim < 100 or mean_claim > 1000000:
                    accuracy_issues.append(f"Unusual mean claim amount: ${mean_claim:,.0f}")
        
        accuracy_pct = max(0, 100 - (len(accuracy_issues) * 10))
        
        return {
            'score': accuracy_pct,
            'issues_found': len(accuracy_issues),
            'issues': accuracy_issues
        }
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate quality report for all checked datasets"""
        report_data = []
        
        for data_type, metrics in self.quality_report.items():
            report_data.append({
                'Data Type': data_type,
                'Total Rows': metrics['total_rows'],
                'Completeness': f"{metrics['completeness']['score']:.1f}%",
                'Uniqueness': f"{metrics['uniqueness']['score']:.1f}%",
                'Validity': f"{metrics['validity']['score']:.1f}%",
                'Consistency': f"{metrics['consistency']['score']:.1f}%",
                'Accuracy': f"{metrics['accuracy']['score']:.1f}%",
                'Overall Score': f"{metrics['overall_score']:.1f}%"
            })
        
        return pd.DataFrame(report_data)
    
    def get_quality_report(self) -> Dict:
        """Get quality report"""
        return self.quality_report
