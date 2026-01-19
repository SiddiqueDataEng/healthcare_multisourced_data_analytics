"""
Comprehensive Statistical Analysis for Healthcare Data
Includes descriptive statistics, hypothesis testing, correlation analysis, and distribution analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for healthcare data"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def descriptive_statistics(self, df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics
        
        Returns:
            - Mean, median, mode
            - Standard deviation, variance
            - Min, max, range
            - Quartiles, IQR
            - Skewness, kurtosis
        """
        self.logger.info("Calculating descriptive statistics")
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            results[col] = {
                # Central tendency
                'mean': float(data.mean()),
                'median': float(data.median()),
                'mode': float(data.mode()[0]) if len(data.mode()) > 0 else None,
                
                # Dispersion
                'std': float(data.std()),
                'variance': float(data.var()),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min()),
                
                # Quartiles
                'q1': float(data.quantile(0.25)),
                'q2': float(data.quantile(0.50)),
                'q3': float(data.quantile(0.75)),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                
                # Shape
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                
                # Count
                'count': int(len(data)),
                'missing': int(df[col].isna().sum()),
                'missing_pct': float(df[col].isna().mean() * 100)
            }
        
        self.logger.info(f"Calculated descriptive statistics for {len(results)} columns")
        return results
    
    def hypothesis_testing(self, df: pd.DataFrame, group_col: str, value_col: str, 
                          test_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform hypothesis testing between groups
        
        Args:
            group_col: Column defining groups (e.g., 'is_diabetic')
            value_col: Column with values to compare (e.g., 'cost')
            test_type: 'ttest', 'mannwhitney', or 'auto'
        
        Returns:
            Test statistic, p-value, interpretation
        """
        self.logger.info(f"Performing hypothesis test: {group_col} vs {value_col}")
        
        groups = df[group_col].unique()
        
        if len(groups) != 2:
            return {'error': f'Expected 2 groups, found {len(groups)}'}
        
        group1 = df[df[group_col] == groups[0]][value_col].dropna()
        group2 = df[df[group_col] == groups[1]][value_col].dropna()
        
        # Auto-select test based on normality
        if test_type == 'auto':
            # Shapiro-Wilk test for normality
            _, p1 = stats.shapiro(group1.sample(min(5000, len(group1))))
            _, p2 = stats.shapiro(group2.sample(min(5000, len(group2))))
            
            # If both groups are normal, use t-test; otherwise Mann-Whitney
            test_type = 'ttest' if (p1 > 0.05 and p2 > 0.05) else 'mannwhitney'
        
        # Perform test
        if test_type == 'ttest':
            statistic, pvalue = ttest_ind(group1, group2)
            test_name = "Independent t-test"
        else:
            statistic, pvalue = mannwhitneyu(group1, group2)
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d for t-test)
        effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        
        result = {
            'test_name': test_name,
            'test_type': test_type,
            'group1': str(groups[0]),
            'group2': str(groups[1]),
            'group1_mean': float(group1.mean()),
            'group2_mean': float(group2.mean()),
            'group1_median': float(group1.median()),
            'group2_median': float(group2.median()),
            'group1_n': int(len(group1)),
            'group2_n': int(len(group2)),
            'statistic': float(statistic),
            'pvalue': float(pvalue),
            'effect_size': float(effect_size),
            'significant': pvalue < 0.05,
            'interpretation': self._interpret_pvalue(pvalue, effect_size)
        }
        
        self.logger.info(f"Test result: p={pvalue:.4f}, significant={result['significant']}")
        return result
    
    def correlation_analysis(self, df: pd.DataFrame, cols: List[str] = None, 
                            method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation matrix and identify significant correlations
        
        Args:
            cols: Columns to analyze (default: all numeric)
            method: 'pearson' or 'spearman'
        
        Returns:
            Correlation matrix, p-values, significant pairs
        """
        self.logger.info(f"Performing {method} correlation analysis")
        
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[cols].corr(method=method)
        
        # Calculate p-values
        pvalue_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), 
                                     index=cols, columns=cols)
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    data1 = df[col1].dropna()
                    data2 = df[col2].dropna()
                    
                    # Align data
                    common_idx = data1.index.intersection(data2.index)
                    data1 = data1.loc[common_idx]
                    data2 = data2.loc[common_idx]
                    
                    if method == 'pearson':
                        _, pval = pearsonr(data1, data2)
                    else:
                        _, pval = spearmanr(data1, data2)
                    
                    pvalue_matrix.loc[col1, col2] = pval
                    pvalue_matrix.loc[col2, col1] = pval
        
        # Find significant correlations
        significant_pairs = []
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    pval = pvalue_matrix.loc[col1, col2]
                    
                    if pval < 0.05 and abs(corr) > 0.3:  # Significant and moderate correlation
                        significant_pairs.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': float(corr),
                            'pvalue': float(pval),
                            'strength': self._interpret_correlation(corr)
                        })
        
        # Sort by absolute correlation
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        result = {
            'method': method,
            'correlation_matrix': corr_matrix.to_dict(),
            'pvalue_matrix': pvalue_matrix.to_dict(),
            'significant_pairs': significant_pairs,
            'num_significant': len(significant_pairs)
        }
        
        self.logger.info(f"Found {len(significant_pairs)} significant correlations")
        return result
    
    def distribution_analysis(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Analyze distribution of a variable
        
        Returns:
            - Normality test
            - Distribution parameters
            - Histogram data
            - Best-fit distribution
        """
        self.logger.info(f"Analyzing distribution of {col}")
        
        data = df[col].dropna()
        
        # Normality test
        if len(data) > 5000:
            data_sample = data.sample(5000)
        else:
            data_sample = data
        
        shapiro_stat, shapiro_p = stats.shapiro(data_sample)
        
        # Kolmogorov-Smirnov test against normal distribution
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Histogram data
        hist, bin_edges = np.histogram(data, bins=30)
        
        result = {
            'column': col,
            'n': int(len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
            
            # Normality tests
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_pvalue': float(shapiro_p),
            'is_normal_shapiro': shapiro_p > 0.05,
            
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_p),
            'is_normal_ks': ks_p > 0.05,
            
            # Histogram
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            
            'interpretation': self._interpret_distribution(shapiro_p, data.skew(), data.kurtosis())
        }
        
        self.logger.info(f"Distribution analysis complete: normal={result['is_normal_shapiro']}")
        return result
    
    def chi_square_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence for categorical variables
        
        Args:
            col1, col2: Categorical columns to test
        
        Returns:
            Chi-square statistic, p-value, contingency table
        """
        self.logger.info(f"Performing chi-square test: {col1} vs {col2}")
        
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        # Perform chi-square test
        chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
        
        result = {
            'test_name': 'Chi-square test of independence',
            'variable1': col1,
            'variable2': col2,
            'chi2_statistic': float(chi2),
            'pvalue': float(pvalue),
            'degrees_of_freedom': int(dof),
            'significant': pvalue < 0.05,
            'contingency_table': contingency_table.to_dict(),
            'interpretation': f"{'Significant' if pvalue < 0.05 else 'No significant'} association between {col1} and {col2}"
        }
        
        self.logger.info(f"Chi-square test result: p={pvalue:.4f}")
        return result
    
    def _interpret_pvalue(self, pvalue: float, effect_size: float) -> str:
        """Interpret p-value and effect size"""
        if pvalue < 0.001:
            sig = "highly significant (p < 0.001)"
        elif pvalue < 0.01:
            sig = "very significant (p < 0.01)"
        elif pvalue < 0.05:
            sig = "significant (p < 0.05)"
        else:
            sig = "not significant (p >= 0.05)"
        
        if abs(effect_size) < 0.2:
            effect = "small effect size"
        elif abs(effect_size) < 0.5:
            effect = "medium effect size"
        else:
            effect = "large effect size"
        
        return f"{sig}, {effect}"
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def _interpret_distribution(self, shapiro_p: float, skewness: float, kurtosis: float) -> str:
        """Interpret distribution characteristics"""
        parts = []
        
        if shapiro_p > 0.05:
            parts.append("approximately normal")
        else:
            parts.append("non-normal")
        
        if abs(skewness) < 0.5:
            parts.append("symmetric")
        elif skewness > 0.5:
            parts.append("right-skewed")
        else:
            parts.append("left-skewed")
        
        if abs(kurtosis) < 0.5:
            parts.append("mesokurtic (normal tails)")
        elif kurtosis > 0.5:
            parts.append("leptokurtic (heavy tails)")
        else:
            parts.append("platykurtic (light tails)")
        
        return ", ".join(parts)
