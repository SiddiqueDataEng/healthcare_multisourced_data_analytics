"""
Analytics Orchestrator - Runs all analytics and ML pipelines
Coordinates statistical analysis, cohort analysis, time series, and ML models
"""

import logging
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from src.analytics.statistical_analysis import StatisticalAnalyzer
from src.analytics.cohort_analysis import CohortAnalyzer
from src.analytics.time_series_analysis import TimeSeriesAnalyzer
from src.ml.ml_pipeline import MLPipeline

logger = logging.getLogger(__name__)


class AnalyticsOrchestrator:
    """Orchestrates all analytics and ML workflows"""
    
    def __init__(self, data_dir: str = "data/curated", output_dir: str = "data/curated/analytics"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.stats_analyzer = StatisticalAnalyzer()
        self.cohort_analyzer = CohortAnalyzer()
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.ml_pipeline = MLPipeline()
    
    def run_complete_analytics(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame,
                               claims_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete analytics pipeline
        
        Includes:
            1. Descriptive statistics
            2. Hypothesis testing
            3. Correlation analysis
            4. Cohort analysis
            5. Time series analysis
            6. ML model training
        
        Returns:
            Comprehensive analytics results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE HEALTHCARE ANALYTICS")
        self.logger.info("=" * 80)
        
        results = {}
        
        # 1. DESCRIPTIVE STATISTICS
        self.logger.info("\n[1/7] Running Descriptive Statistics...")
        results['descriptive_stats'] = self._run_descriptive_statistics(
            patients_df, encounters_df, claims_df
        )
        
        # 2. HYPOTHESIS TESTING
        self.logger.info("\n[2/7] Running Hypothesis Testing...")
        results['hypothesis_tests'] = self._run_hypothesis_tests(
            patients_df, encounters_df, claims_df
        )
        
        # 3. CORRELATION ANALYSIS
        self.logger.info("\n[3/7] Running Correlation Analysis...")
        results['correlations'] = self._run_correlation_analysis(
            patients_df, encounters_df, claims_df
        )
        
        # 4. DISTRIBUTION ANALYSIS
        self.logger.info("\n[4/7] Running Distribution Analysis...")
        results['distributions'] = self._run_distribution_analysis(
            patients_df, encounters_df, claims_df
        )
        
        # 5. COHORT ANALYSIS
        self.logger.info("\n[5/7] Running Cohort Analysis...")
        results['cohort_analysis'] = self._run_cohort_analysis(
            patients_df, encounters_df, claims_df
        )
        
        # 6. TIME SERIES ANALYSIS
        self.logger.info("\n[6/7] Running Time Series Analysis...")
        results['time_series'] = self._run_time_series_analysis(
            encounters_df, claims_df
        )
        
        # 7. MACHINE LEARNING
        self.logger.info("\n[7/7] Running Machine Learning Models...")
        results['machine_learning'] = self._run_ml_models(
            patients_df, encounters_df, claims_df
        )
        
        # Save results
        self._save_results(results)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANALYTICS COMPLETE")
        self.logger.info("=" * 80)
        
        return results
    
    def _run_descriptive_statistics(self, patients_df: pd.DataFrame,
                                    encounters_df: pd.DataFrame,
                                    claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Run descriptive statistics on all datasets"""
        
        results = {}
        
        # Patient statistics
        patient_numeric_cols = ['age', 'comorbidity_count', 'risk_score']
        patient_numeric_cols = [c for c in patient_numeric_cols if c in patients_df.columns]
        results['patients'] = self.stats_analyzer.descriptive_statistics(
            patients_df, patient_numeric_cols
        )
        
        # Encounter statistics
        encounter_numeric_cols = ['length_of_stay']
        encounter_numeric_cols = [c for c in encounter_numeric_cols if c in encounters_df.columns]
        results['encounters'] = self.stats_analyzer.descriptive_statistics(
            encounters_df, encounter_numeric_cols
        )
        
        # Claims statistics
        claim_numeric_cols = ['claim_amount', 'paid_amount', 'allowed_amount']
        claim_numeric_cols = [c for c in claim_numeric_cols if c in claims_df.columns]
        results['claims'] = self.stats_analyzer.descriptive_statistics(
            claims_df, claim_numeric_cols
        )
        
        self.logger.info("Descriptive statistics complete")
        return results
    
    def _run_hypothesis_tests(self, patients_df: pd.DataFrame,
                             encounters_df: pd.DataFrame,
                             claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Run hypothesis tests comparing groups"""
        
        results = {}
        
        # Merge data for testing
        test_df = encounters_df.merge(
            patients_df[['patient_id', 'is_diabetic', 'is_hypertensive', 'is_high_cost']],
            on='patient_id',
            how='left'
        )
        
        # Test 1: Diabetic vs non-diabetic length of stay
        if 'is_diabetic' in test_df.columns and 'length_of_stay' in test_df.columns:
            results['diabetic_los'] = self.stats_analyzer.hypothesis_testing(
                test_df, 'is_diabetic', 'length_of_stay'
            )
        
        # Test 2: High-cost vs normal-cost readmission rates
        if 'is_high_cost' in test_df.columns and 'is_readmission' in test_df.columns:
            test_df['is_readmission_int'] = test_df['is_readmission'].astype(int)
            results['high_cost_readmission'] = self.stats_analyzer.hypothesis_testing(
                test_df, 'is_high_cost', 'is_readmission_int'
            )
        
        # Test 3: Hypertensive vs non-hypertensive costs
        cost_df = claims_df.merge(
            patients_df[['patient_id', 'is_hypertensive']],
            on='patient_id',
            how='left'
        )
        if 'is_hypertensive' in cost_df.columns and 'claim_amount' in cost_df.columns:
            results['hypertensive_cost'] = self.stats_analyzer.hypothesis_testing(
                cost_df, 'is_hypertensive', 'claim_amount'
            )
        
        self.logger.info(f"Completed {len(results)} hypothesis tests")
        return results
    
    def _run_correlation_analysis(self, patients_df: pd.DataFrame,
                                  encounters_df: pd.DataFrame,
                                  claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Run correlation analysis"""
        
        results = {}
        
        # Patient correlations
        patient_cols = ['age', 'comorbidity_count', 'risk_score']
        patient_cols = [c for c in patient_cols if c in patients_df.columns]
        if len(patient_cols) >= 2:
            results['patients'] = self.stats_analyzer.correlation_analysis(
                patients_df, patient_cols, method='pearson'
            )
        
        # Encounter correlations
        encounter_cols = ['length_of_stay']
        if 'length_of_stay' in encounters_df.columns:
            # Merge with patient data
            enc_analysis = encounters_df.merge(
                patients_df[['patient_id', 'age', 'comorbidity_count']],
                on='patient_id',
                how='left'
            )
            encounter_cols = ['length_of_stay', 'age', 'comorbidity_count']
            results['encounters'] = self.stats_analyzer.correlation_analysis(
                enc_analysis, encounter_cols, method='pearson'
            )
        
        self.logger.info("Correlation analysis complete")
        return results
    
    def _run_distribution_analysis(self, patients_df: pd.DataFrame,
                                   encounters_df: pd.DataFrame,
                                   claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of key variables"""
        
        results = {}
        
        # Age distribution
        if 'age' in patients_df.columns:
            results['age'] = self.stats_analyzer.distribution_analysis(patients_df, 'age')
        
        # Length of stay distribution
        if 'length_of_stay' in encounters_df.columns:
            results['length_of_stay'] = self.stats_analyzer.distribution_analysis(
                encounters_df, 'length_of_stay'
            )
        
        # Cost distribution
        if 'claim_amount' in claims_df.columns:
            results['claim_amount'] = self.stats_analyzer.distribution_analysis(
                claims_df, 'claim_amount'
            )
        
        self.logger.info(f"Analyzed {len(results)} distributions")
        return results
    
    def _run_cohort_analysis(self, patients_df: pd.DataFrame,
                            encounters_df: pd.DataFrame,
                            claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive cohort analysis"""
        
        results = {}
        
        # Identify cohorts
        cohorts = self.cohort_analyzer.identify_cohorts(patients_df)
        results['cohort_sizes'] = {name: len(df) for name, df in cohorts.items()}
        
        # Analyze each cohort
        results['cohort_demographics'] = {}
        results['cohort_outcomes'] = {}
        
        for cohort_name, cohort_df in cohorts.items():
            if len(cohort_df) > 0:
                # Demographics
                results['cohort_demographics'][cohort_name] = \
                    self.cohort_analyzer.cohort_demographics(cohort_df, cohort_name)
                
                # Outcomes
                results['cohort_outcomes'][cohort_name] = \
                    self.cohort_analyzer.cohort_outcomes(
                        cohort_df, encounters_df, claims_df, cohort_name
                    )
        
        # Comorbidity patterns
        results['comorbidity_patterns'] = self.cohort_analyzer.comorbidity_patterns(patients_df)
        
        # Compare key cohorts
        if 'diabetic' in cohorts and 'hypertensive' in cohorts:
            results['diabetic_vs_hypertensive'] = self.cohort_analyzer.compare_cohorts(
                cohorts['diabetic'], cohorts['hypertensive'],
                encounters_df, claims_df,
                'Diabetic', 'Hypertensive'
            )
        
        self.logger.info(f"Analyzed {len(cohorts)} cohorts")
        return results
    
    def _run_time_series_analysis(self, encounters_df: pd.DataFrame,
                                  claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Run time series analysis and forecasting"""
        
        results = {}
        
        # Encounter trends
        results['encounter_trends'] = self.ts_analyzer.encounter_trends(encounters_df)
        
        # Seasonal patterns
        results['seasonal_patterns'] = self.ts_analyzer.seasonal_patterns(encounters_df)
        
        # Cost trends
        results['cost_trends'] = self.ts_analyzer.cost_trends(claims_df)
        
        # Disease prevalence trends
        if 'primary_diagnosis' in encounters_df.columns:
            results['disease_trends'] = self.ts_analyzer.disease_prevalence_trends(encounters_df)
        
        # Forecasting
        results['encounter_forecast'] = self.ts_analyzer.forecast_encounters(encounters_df, periods=30)
        results['cost_forecast'] = self.ts_analyzer.forecast_costs(claims_df, periods=12)
        
        self.logger.info("Time series analysis complete")
        return results
    
    def _run_ml_models(self, patients_df: pd.DataFrame,
                      encounters_df: pd.DataFrame,
                      claims_df: pd.DataFrame) -> Dict[str, Any]:
        """Train and evaluate ML models"""
        
        results = {}
        
        # Readmission prediction
        self.logger.info("Training readmission prediction models...")
        readmission_features = self.ml_pipeline.engineer_readmission_features(
            patients_df, encounters_df, claims_df
        )
        results['readmission_prediction'] = self.ml_pipeline.train_readmission_models(
            readmission_features
        )
        
        # Cost prediction
        self.logger.info("Training cost prediction models...")
        cost_features = self.ml_pipeline.engineer_cost_features(
            patients_df, encounters_df, claims_df
        )
        results['cost_prediction'] = self.ml_pipeline.train_cost_prediction_models(
            cost_features
        )
        
        self.logger.info("ML model training complete")
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analytics results to JSON"""
        import json
        
        output_file = self.output_dir / 'analytics_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (pd.DataFrame, pd.Series)):
                return convert_types(obj.to_dict())
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary of analytics results"""
        
        report = []
        report.append("=" * 80)
        report.append("HEALTHCARE ANALYTICS SUMMARY REPORT")
        report.append("=" * 80)
        
        # Cohort Analysis Summary
        if 'cohort_analysis' in results:
            report.append("\n## COHORT ANALYSIS")
            report.append("-" * 80)
            cohort_sizes = results['cohort_analysis'].get('cohort_sizes', {})
            for cohort, size in cohort_sizes.items():
                report.append(f"  {cohort.title()}: {size:,} patients")
        
        # Hypothesis Testing Summary
        if 'hypothesis_tests' in results:
            report.append("\n## HYPOTHESIS TESTING")
            report.append("-" * 80)
            for test_name, test_result in results['hypothesis_tests'].items():
                if 'interpretation' in test_result:
                    report.append(f"  {test_name}: {test_result['interpretation']}")
        
        # ML Model Summary
        if 'machine_learning' in results:
            report.append("\n## MACHINE LEARNING MODELS")
            report.append("-" * 80)
            
            if 'readmission_prediction' in results['machine_learning']:
                ml_results = results['machine_learning']['readmission_prediction']
                report.append(f"  Readmission Prediction:")
                report.append(f"    Best Model: {ml_results.get('best_model', 'N/A')}")
                report.append(f"    ROC-AUC: {ml_results.get('best_roc_auc', 0):.3f}")
            
            if 'cost_prediction' in results['machine_learning']:
                cost_results = results['machine_learning']['cost_prediction']
                report.append(f"  Cost Prediction:")
                report.append(f"    Best Model: {cost_results.get('best_model', 'N/A')}")
                report.append(f"    R² Score: {cost_results.get('best_r2_score', 0):.3f}")
        
        # Time Series Summary
        if 'time_series' in results:
            report.append("\n## TIME SERIES ANALYSIS")
            report.append("-" * 80)
            
            if 'encounter_trends' in results['time_series']:
                trends = results['time_series']['encounter_trends']
                report.append(f"  Encounter Trend: {trends.get('trend_direction', 'N/A')}")
                report.append(f"  Monthly Growth Rate: {trends.get('monthly_growth_rate_pct', 0):.2f}%")
            
            if 'encounter_forecast' in results['time_series']:
                forecast = results['time_series']['encounter_forecast']
                report.append(f"  30-Day Forecast: {forecast.get('forecast_total', 0):.0f} encounters")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
