"""
Data Engineering Orchestrator - Integrates all data engineering components
Provides end-to-end data cleaning, validation, and profiling pipeline
"""

import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.data_engineering.advanced_data_cleaner import AdvancedDataCleaner
from src.data_engineering.data_validator import DataValidator
from src.data_engineering.data_profiler import DataProfiler
from src.data_engineering.data_cleaner import DataCleaner
from src.data_engineering.data_quality import DataQualityChecker

logger = logging.getLogger(__name__)


class DataEngineeringOrchestrator:
    """Orchestrates complete data engineering pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.advanced_cleaner = AdvancedDataCleaner(self.config.get('cleaning'))
        self.validator = DataValidator()
        self.profiler = DataProfiler()
        self.quality_checker = DataQualityChecker()
        
        self.pipeline_results = {}
    
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            'cleaning': {
                'outlier_method': 'iqr',
                'iqr_multiplier': 3.0,
                'remove_duplicates': True,
                'standardize_formats': True,
                'validate_integrity': True
            },
            'validation': {
                'strict_mode': False,
                'export_reports': True
            },
            'profiling': {
                'detailed': True,
                'export_reports': True
            },
            'output_dir': 'data/curated/data_engineering'
        }
    
    def run_complete_pipeline(self, datasets: Dict[str, pd.DataFrame], 
                             output_dir: Optional[str] = None) -> Dict:
        """
        Run complete data engineering pipeline
        
        Args:
            datasets: Dictionary of {data_type: dataframe}
            output_dir: Output directory for results
        
        Returns:
            Pipeline results with cleaned data and reports
        """
        logger.info("=" * 70)
        logger.info("STARTING DATA ENGINEERING PIPELINE")
        logger.info("=" * 70)
        
        output_dir = output_dir or self.config['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'datasets_processed': len(datasets),
            'cleaned_data': {},
            'validation_results': {},
            'profile_results': {},
            'quality_results': {},
            'summary': {}
        }
        
        # Process each dataset
        for data_type, df in datasets.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {data_type}")
            logger.info(f"{'='*70}")
            
            try:
                # 1. Initial profiling
                logger.info(f"[1/5] Initial profiling...")
                initial_profile = self.profiler.profile_dataset(df, f"{data_type}_initial")
                
                # 2. Initial quality check
                logger.info(f"[2/5] Initial quality check...")
                initial_quality = self.quality_checker.check_data_quality(df, f"{data_type}_initial")
                
                # 3. Advanced cleaning
                logger.info(f"[3/5] Advanced cleaning...")
                cleaned_df = self.advanced_cleaner.clean_comprehensive(df, data_type)
                
                # 4. Validation
                logger.info(f"[4/5] Validation...")
                validation_results = self.validator.validate_dataset(cleaned_df, data_type)
                
                # 5. Final profiling and quality check
                logger.info(f"[5/5] Final profiling and quality check...")
                final_profile = self.profiler.profile_dataset(cleaned_df, f"{data_type}_final")
                final_quality = self.quality_checker.check_data_quality(cleaned_df, f"{data_type}_final")
                
                # Store results
                results['cleaned_data'][data_type] = cleaned_df
                results['validation_results'][data_type] = validation_results
                results['profile_results'][data_type] = {
                    'initial': initial_profile,
                    'final': final_profile
                }
                results['quality_results'][data_type] = {
                    'initial': initial_quality,
                    'final': final_quality
                }
                
                # Generate summary
                results['summary'][data_type] = self._generate_dataset_summary(
                    df, cleaned_df, initial_quality, final_quality, validation_results
                )
                
                # Save cleaned data
                output_file = Path(output_dir) / f"{data_type}_cleaned.csv"
                cleaned_df.to_csv(output_file, index=False)
                logger.info(f"Saved cleaned data: {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {data_type}: {str(e)}")
                results['summary'][data_type] = {'error': str(e)}
        
        # Export reports
        if self.config['validation']['export_reports']:
            self._export_all_reports(results, output_dir)
        
        # Save pipeline results
        self.pipeline_results = results
        
        logger.info("\n" + "=" * 70)
        logger.info("DATA ENGINEERING PIPELINE COMPLETE")
        logger.info("=" * 70)
        self._print_pipeline_summary(results)
        
        return results
    
    def run_cleaning_only(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Run cleaning only (no validation or profiling)"""
        logger.info(f"Running cleaning for {data_type}")
        return self.advanced_cleaner.clean_comprehensive(df, data_type)
    
    def run_validation_only(self, df: pd.DataFrame, data_type: str, 
                           reference_data: Optional[Dict] = None) -> Dict:
        """Run validation only"""
        logger.info(f"Running validation for {data_type}")
        return self.validator.validate_dataset(df, data_type, reference_data)
    
    def run_profiling_only(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Run profiling only"""
        logger.info(f"Running profiling for {data_type}")
        return self.profiler.profile_dataset(df, data_type)
    
    def _generate_dataset_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                 initial_quality: Dict, final_quality: Dict,
                                 validation: Dict) -> Dict:
        """Generate summary for a dataset"""
        return {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(original_df) - len(cleaned_df),
            'removal_rate': f"{((len(original_df) - len(cleaned_df)) / len(original_df) * 100):.2f}%",
            'initial_quality_score': f"{initial_quality['overall_score']:.1f}%",
            'final_quality_score': f"{final_quality['overall_score']:.1f}%",
            'quality_improvement': f"{(final_quality['overall_score'] - initial_quality['overall_score']):.1f}%",
            'validation_score': f"{validation['validation_score']:.1f}",
            'validation_errors': validation['errors'],
            'validation_warnings': validation['warnings'],
            'is_valid': validation['is_valid']
        }
    
    def _export_all_reports(self, results: Dict, output_dir: str):
        """Export all reports to files"""
        logger.info("\nExporting reports...")
        
        # Export validation reports
        validation_file = Path(output_dir) / "validation_report.txt"
        self.validator.export_validation_report(str(validation_file))
        
        # Export profile reports
        for data_type in results['profile_results'].keys():
            profile_file = Path(output_dir) / f"profile_{data_type}.txt"
            self.profiler.export_profile_report(str(profile_file), f"{data_type}_final")
        
        # Export cleaning audit trail
        audit_file = Path(output_dir) / "cleaning_audit_trail.txt"
        self.advanced_cleaner.export_audit_trail(str(audit_file))
        
        # Export summary JSON
        summary_file = Path(output_dir) / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            # Convert DataFrames to summaries for JSON serialization
            json_results = {
                'timestamp': results['timestamp'],
                'datasets_processed': results['datasets_processed'],
                'summary': results['summary'],
                'validation_results': {
                    k: {
                        'errors': v['errors'],
                        'warnings': v['warnings'],
                        'is_valid': v['is_valid'],
                        'validation_score': v['validation_score']
                    }
                    for k, v in results['validation_results'].items()
                }
            }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Reports exported to {output_dir}")
    
    def _print_pipeline_summary(self, results: Dict):
        """Print pipeline summary"""
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        
        for data_type, summary in results['summary'].items():
            if 'error' in summary:
                print(f"\n{data_type.upper()}: ERROR - {summary['error']}")
                continue
            
            print(f"\n{data_type.upper()}:")
            print(f"  Original Rows: {summary['original_rows']:,}")
            print(f"  Cleaned Rows: {summary['cleaned_rows']:,}")
            print(f"  Rows Removed: {summary['rows_removed']:,} ({summary['removal_rate']})")
            print(f"  Quality Improvement: {summary['initial_quality_score']} → {summary['final_quality_score']} ({summary['quality_improvement']})")
            print(f"  Validation: {'✓ PASS' if summary['is_valid'] else '✗ FAIL'} (Score: {summary['validation_score']})")
            print(f"  Errors: {summary['validation_errors']}, Warnings: {summary['validation_warnings']}")
        
        print("\n" + "=" * 70)
    
    def get_pipeline_results(self) -> Dict:
        """Get pipeline results"""
        return self.pipeline_results
    
    def get_cleaned_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Get cleaned data for a specific type"""
        return self.pipeline_results.get('cleaned_data', {}).get(data_type)
    
    def get_validation_results(self, data_type: str) -> Optional[Dict]:
        """Get validation results for a specific type"""
        return self.pipeline_results.get('validation_results', {}).get(data_type)
    
    def get_quality_improvement(self, data_type: str) -> Optional[Dict]:
        """Get quality improvement metrics"""
        quality_results = self.pipeline_results.get('quality_results', {}).get(data_type)
        if not quality_results:
            return None
        
        initial = quality_results['initial']['overall_score']
        final = quality_results['final']['overall_score']
        
        return {
            'initial_score': initial,
            'final_score': final,
            'improvement': final - initial,
            'improvement_percentage': ((final - initial) / initial * 100) if initial > 0 else 0
        }
