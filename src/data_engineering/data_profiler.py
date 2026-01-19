"""
Data Profiler - Comprehensive data profiling and distribution analysis
Analyzes data patterns, distributions, cardinality, and quality metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy import stats
from collections import Counter

logger = logging.getLogger(__name__)


class DataProfiler:
    """Comprehensive data profiling"""
    
    def __init__(self):
        self.profile_results = {}
    
    def profile_dataset(self, df: pd.DataFrame, data_type: str = "generic") -> Dict:
        """
        Comprehensive dataset profiling
        
        Args:
            df: DataFrame to profile
            data_type: Type of data for context
        
        Returns:
            Comprehensive profile report
        """
        logger.info(f"Profiling {data_type} dataset ({len(df):,} rows, {len(df.columns)} columns)")
        
        profile = {
            'data_type': data_type,
            'overview': self._profile_overview(df),
            'columns': self._profile_columns(df),
            'numeric_analysis': self._profile_numeric_columns(df),
            'categorical_analysis': self._profile_categorical_columns(df),
            'date_analysis': self._profile_date_columns(df),
            'correlations': self._analyze_correlations(df),
            'data_quality': self._analyze_data_quality(df),
            'patterns': self._detect_patterns(df, data_type)
        }
        
        self.profile_results[data_type] = profile
        
        logger.info(f"Profiling complete for {data_type}")
        
        return profile
    
    def _profile_overview(self, df: pd.DataFrame) -> Dict:
        """Generate dataset overview"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': len(df) - len(df.drop_duplicates()),
            'duplicate_percentage': (len(df) - len(df.drop_duplicates())) / len(df) * 100,
            'total_missing_cells': df.isnull().sum().sum(),
            'missing_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'column_types': {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns)
            }
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict:
        """Profile each column"""
        column_profiles = {}
        
        for col in df.columns:
            column_profiles[col] = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().sum() / len(df) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': df[col].nunique() / len(df) * 100,
                'memory_usage_kb': df[col].memory_usage(deep=True) / 1024
            }
            
            # Add type-specific metrics
            if pd.api.types.is_numeric_dtype(df[col]):
                column_profiles[col].update(self._profile_numeric_column(df[col]))
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_profiles[col].update(self._profile_date_column(df[col]))
            else:
                column_profiles[col].update(self._profile_categorical_column(df[col]))
        
        return column_profiles
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict:
        """Profile numeric column"""
        if series.notna().sum() == 0:
            return {}
        
        return {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75)),
            'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'zeros_count': int((series == 0).sum()),
            'zeros_percentage': float((series == 0).sum() / len(series) * 100),
            'negative_count': int((series < 0).sum()),
            'negative_percentage': float((series < 0).sum() / len(series) * 100)
        }
    
    def _profile_categorical_column(self, series: pd.Series) -> Dict:
        """Profile categorical column"""
        value_counts = series.value_counts()
        
        profile = {
            'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'most_frequent_percentage': float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
            'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
            'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
        }
        
        # Add top 10 values
        if len(value_counts) > 0:
            profile['top_10_values'] = {
                str(k): int(v) for k, v in value_counts.head(10).items()
            }
        
        # Calculate entropy (measure of randomness)
        if len(value_counts) > 0:
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            profile['entropy'] = float(entropy)
        
        return profile
    
    def _profile_date_column(self, series: pd.Series) -> Dict:
        """Profile date column"""
        if series.notna().sum() == 0:
            return {}
        
        return {
            'min_date': str(series.min()),
            'max_date': str(series.max()),
            'date_range_days': int((series.max() - series.min()).days),
            'most_common_year': int(series.dt.year.mode()[0]) if not series.dt.year.mode().empty else None,
            'most_common_month': int(series.dt.month.mode()[0]) if not series.dt.month.mode().empty else None,
            'most_common_day_of_week': int(series.dt.dayofweek.mode()[0]) if not series.dt.dayofweek.mode().empty else None
        }
    
    def _profile_numeric_columns(self, df: pd.DataFrame) -> Dict:
        """Comprehensive numeric column analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        analysis = {
            'total_numeric_columns': len(numeric_cols),
            'columns': {}
        }
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                # Distribution analysis
                analysis['columns'][col] = {
                    'distribution_type': self._detect_distribution(df[col]),
                    'outliers_iqr': self._count_outliers_iqr(df[col]),
                    'outliers_zscore': self._count_outliers_zscore(df[col]),
                    'is_normally_distributed': self._test_normality(df[col])
                }
        
        return analysis
    
    def _profile_categorical_columns(self, df: pd.DataFrame) -> Dict:
        """Comprehensive categorical column analysis"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return {}
        
        analysis = {
            'total_categorical_columns': len(categorical_cols),
            'high_cardinality_columns': [],
            'low_cardinality_columns': [],
            'columns': {}
        }
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            cardinality_ratio = cardinality / len(df)
            
            if cardinality_ratio > 0.9:
                analysis['high_cardinality_columns'].append(col)
            elif cardinality_ratio < 0.05:
                analysis['low_cardinality_columns'].append(col)
            
            analysis['columns'][col] = {
                'cardinality': cardinality,
                'cardinality_ratio': cardinality_ratio,
                'is_potential_id': cardinality == len(df) or '_id' in col.lower()
            }
        
        return analysis
    
    def _profile_date_columns(self, df: pd.DataFrame) -> Dict:
        """Comprehensive date column analysis"""
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_cols) == 0:
            return {}
        
        analysis = {
            'total_date_columns': len(date_cols),
            'columns': {}
        }
        
        for col in date_cols:
            if df[col].notna().sum() > 0:
                # Temporal patterns
                analysis['columns'][col] = {
                    'temporal_coverage_days': (df[col].max() - df[col].min()).days,
                    'year_distribution': df[col].dt.year.value_counts().to_dict(),
                    'month_distribution': df[col].dt.month.value_counts().to_dict(),
                    'day_of_week_distribution': df[col].dt.dayofweek.value_counts().to_dict(),
                    'has_weekend_data': df[col].dt.dayofweek.isin([5, 6]).any(),
                    'has_future_dates': (df[col] > pd.Timestamp.now()).any()
                }
        
        return analysis
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if '_id' not in col.lower()]
        
        if len(numeric_cols) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'strong_correlations': strong_correlations,
            'correlation_matrix_shape': corr_matrix.shape
        }

    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze overall data quality"""
        return {
            'completeness_score': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'uniqueness_score': (len(df.drop_duplicates()) / len(df)) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'columns_with_high_missing': df.columns[df.isnull().sum() / len(df) > 0.5].tolist(),
            'constant_columns': df.columns[df.nunique() == 1].tolist()
        }
    
    def _detect_patterns(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Detect data patterns specific to healthcare data"""
        patterns = {}
        
        if data_type == 'patients':
            patterns['age_distribution'] = self._analyze_age_distribution(df)
            patterns['gender_distribution'] = self._analyze_gender_distribution(df)
            patterns['insurance_distribution'] = self._analyze_insurance_distribution(df)
        
        elif data_type == 'encounters':
            patterns['encounter_type_distribution'] = self._analyze_encounter_types(df)
            patterns['temporal_patterns'] = self._analyze_temporal_patterns(df)
        
        elif data_type == 'claims':
            patterns['claim_amount_distribution'] = self._analyze_claim_amounts(df)
            patterns['claim_status_distribution'] = self._analyze_claim_status(df)
        
        return patterns
    
    def _analyze_age_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze age distribution"""
        if 'age' not in df.columns:
            return {}
        
        return {
            'mean_age': float(df['age'].mean()),
            'median_age': float(df['age'].median()),
            'age_groups': {
                '0-17': int((df['age'] < 18).sum()),
                '18-39': int(((df['age'] >= 18) & (df['age'] < 40)).sum()),
                '40-64': int(((df['age'] >= 40) & (df['age'] < 65)).sum()),
                '65+': int((df['age'] >= 65).sum())
            }
        }
    
    def _analyze_gender_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze gender distribution"""
        if 'gender' not in df.columns:
            return {}
        
        return df['gender'].value_counts().to_dict()
    
    def _analyze_insurance_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze insurance distribution"""
        if 'insurance_type' not in df.columns:
            return {}
        
        return df['insurance_type'].value_counts().to_dict()
    
    def _analyze_encounter_types(self, df: pd.DataFrame) -> Dict:
        """Analyze encounter types"""
        if 'encounter_type' not in df.columns:
            return {}
        
        return df['encounter_type'].value_counts().to_dict()
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in encounters"""
        date_col = None
        for col in ['encounter_date', 'admission_date', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return {}
        
        df_temp = df[df[date_col].notna()].copy()
        
        return {
            'by_year': df_temp[date_col].dt.year.value_counts().to_dict(),
            'by_month': df_temp[date_col].dt.month.value_counts().to_dict(),
            'by_day_of_week': df_temp[date_col].dt.dayofweek.value_counts().to_dict(),
            'weekend_percentage': float((df_temp[date_col].dt.dayofweek.isin([5, 6])).sum() / len(df_temp) * 100)
        }
    
    def _analyze_claim_amounts(self, df: pd.DataFrame) -> Dict:
        """Analyze claim amounts"""
        amount_col = None
        for col in ['claim_amount', 'billed_amount', 'amount']:
            if col in df.columns:
                amount_col = col
                break
        
        if not amount_col:
            return {}
        
        amounts = df[amount_col].dropna()
        
        return {
            'mean': float(amounts.mean()),
            'median': float(amounts.median()),
            'total': float(amounts.sum()),
            'ranges': {
                '0-100': int((amounts < 100).sum()),
                '100-1000': int(((amounts >= 100) & (amounts < 1000)).sum()),
                '1000-10000': int(((amounts >= 1000) & (amounts < 10000)).sum()),
                '10000+': int((amounts >= 10000).sum())
            }
        }
    
    def _analyze_claim_status(self, df: pd.DataFrame) -> Dict:
        """Analyze claim status"""
        if 'claim_status' not in df.columns:
            return {}
        
        return df['claim_status'].value_counts().to_dict()
    
    def _detect_distribution(self, series: pd.Series) -> str:
        """Detect distribution type"""
        if series.notna().sum() < 10:
            return "insufficient_data"
        
        # Test for normal distribution
        if self._test_normality(series):
            return "normal"
        
        # Test for uniform distribution
        if abs(series.skew()) < 0.5 and abs(series.kurtosis()) < 1:
            return "uniform"
        
        # Test for exponential (right-skewed)
        if series.skew() > 1:
            return "right_skewed"
        
        # Test for left-skewed
        if series.skew() < -1:
            return "left_skewed"
        
        return "unknown"
    
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if data is normally distributed"""
        if series.notna().sum() < 20:
            return False
        
        try:
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(series.dropna().sample(min(5000, len(series.dropna()))))
            return p_value > 0.05
        except:
            return False
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        return int(((series < lower_bound) | (series > upper_bound)).sum())
    
    def _count_outliers_zscore(self, series: pd.Series) -> int:
        """Count outliers using Z-score method"""
        if series.std() == 0:
            return 0
        
        z_scores = np.abs(stats.zscore(series.dropna()))
        return int((z_scores > 3).sum())
    
    def generate_profile_report(self, data_type: str = None) -> str:
        """Generate human-readable profile report"""
        if data_type:
            profiles = {data_type: self.profile_results[data_type]}
        else:
            profiles = self.profile_results
        
        report = []
        report.append("=" * 70)
        report.append("DATA PROFILING REPORT")
        report.append("=" * 70)
        report.append("")
        
        for dtype, profile in profiles.items():
            report.append(f"\n{dtype.upper()}")
            report.append("-" * 70)
            
            # Overview
            overview = profile['overview']
            report.append(f"Total Rows: {overview['total_rows']:,}")
            report.append(f"Total Columns: {overview['total_columns']}")
            report.append(f"Memory Usage: {overview['memory_usage_mb']:.2f} MB")
            report.append(f"Duplicate Rows: {overview['duplicate_rows']:,} ({overview['duplicate_percentage']:.2f}%)")
            report.append(f"Missing Cells: {overview['total_missing_cells']:,} ({overview['missing_percentage']:.2f}%)")
            report.append("")
            
            # Data Quality
            quality = profile['data_quality']
            report.append(f"Data Quality:")
            report.append(f"  Completeness: {quality['completeness_score']:.2f}%")
            report.append(f"  Uniqueness: {quality['uniqueness_score']:.2f}%")
            if quality['constant_columns']:
                report.append(f"  Constant Columns: {', '.join(quality['constant_columns'])}")
            report.append("")
            
            # Numeric Analysis
            if profile['numeric_analysis']:
                report.append(f"Numeric Columns: {profile['numeric_analysis']['total_numeric_columns']}")
                report.append("")
            
            # Categorical Analysis
            if profile['categorical_analysis']:
                report.append(f"Categorical Columns: {profile['categorical_analysis']['total_categorical_columns']}")
                if profile['categorical_analysis']['high_cardinality_columns']:
                    report.append(f"  High Cardinality: {', '.join(profile['categorical_analysis']['high_cardinality_columns'])}")
                report.append("")
            
            # Correlations
            if profile['correlations'] and profile['correlations'].get('strong_correlations'):
                report.append(f"Strong Correlations:")
                for corr in profile['correlations']['strong_correlations'][:5]:
                    report.append(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f}")
                report.append("")
        
        return "\n".join(report)
    
    def export_profile_report(self, filepath: str, data_type: str = None):
        """Export profile report to file"""
        report = self.generate_profile_report(data_type)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Profile report exported to {filepath}")
    
    def get_profile_results(self) -> Dict:
        """Get all profile results"""
        return self.profile_results
