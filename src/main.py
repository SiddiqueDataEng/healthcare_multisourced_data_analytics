"""
Healthcare Analytics Platform - Main Entry Point
Orchestrates complete workflow: data generation → data engineering → ML → analytics
"""

import logging
import sys
from pathlib import Path
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_data(config, use_advanced_patients=False):
    """Generate synthetic healthcare data"""
    try:
        logger.info("=" * 70)
        logger.info("PHASE 1: GENERATING SYNTHETIC HEALTHCARE DATA")
        logger.info("=" * 70)
        
        from datagenerator.main import DataGenerator
        from datagenerator.config import GeneratorConfig
        
        gen_config = GeneratorConfig(
            num_patients=10000,
            num_providers=500,
            num_facilities=50,
            num_encounters=50000,
            num_claims=100000,
            output_dir=str(config.LANDING_ZONE)
        )
        
        generator = DataGenerator(gen_config)
        
        # Use advanced patient generator if requested
        if use_advanced_patients:
            logger.info("Using advanced patient generator with 50+ attributes...")
            from datagenerator.advanced_patient_generator import AdvancedPatientGenerator
            
            adv_patient_gen = AdvancedPatientGenerator(gen_config)
            patients_df = adv_patient_gen.generate_patients()
            adv_patient_gen.to_csv(patients_df, f"{gen_config.output_dir}/patients.csv")
            
            # Generate other data types normally
            generator.generate_providers()
            generator.generate_encounters()
            generator.generate_claims()
            generator.generate_registry()
            generator.generate_reporting()
        else:
            generator.generate_all()
        
        logger.info("✅ Data generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data generation failed: {str(e)}", exc_info=True)
        return False


def run_data_engineering(config, use_advanced=True):
    """Run complete data engineering pipeline"""
    try:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: DATA ENGINEERING PIPELINE")
        logger.info("=" * 70)
        
        from src.data_engineering.data_loader import DataLoader
        
        loader = DataLoader(str(config.LANDING_ZONE))
        pipeline_data = loader.run_complete_pipeline()
        
        # Run advanced data engineering if enabled
        if use_advanced:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 2B: ADVANCED DATA ENGINEERING")
            logger.info("=" * 70)
            
            from src.data_engineering.data_engineering_orchestrator import DataEngineeringOrchestrator
            
            # Prepare datasets for advanced cleaning
            datasets = {
                'patients': pipeline_data['imputed_data']['patients'],
                'encounters': pipeline_data['imputed_data']['encounters'],
                'claims': pipeline_data['imputed_data']['claims']
            }
            
            # Run advanced pipeline
            orchestrator = DataEngineeringOrchestrator()
            advanced_results = orchestrator.run_complete_pipeline(
                datasets,
                output_dir=str(config.CURATED_ZONE / 'data_engineering')
            )
            
            # Update pipeline data with cleaned data
            pipeline_data['advanced_cleaned'] = advanced_results['cleaned_data']
            pipeline_data['advanced_validation'] = advanced_results['validation_results']
            pipeline_data['advanced_quality'] = advanced_results['quality_results']
            
            # Use advanced cleaned data for downstream processing
            pipeline_data['imputed_data']['patients'] = advanced_results['cleaned_data']['patients']
            pipeline_data['imputed_data']['encounters'] = advanced_results['cleaned_data']['encounters']
            pipeline_data['imputed_data']['claims'] = advanced_results['cleaned_data']['claims']
            
            logger.info("✅ Advanced data engineering completed successfully")
        
        logger.info("✅ Data engineering pipeline completed successfully")
        return pipeline_data
    except Exception as e:
        logger.error(f"❌ Data engineering failed: {str(e)}", exc_info=True)
        return None


def run_ml_pipeline(pipeline_data):
    """Train all ML models with comprehensive feature engineering"""
    try:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: MACHINE LEARNING PIPELINE")
        logger.info("=" * 70)
        
        from src.ml.ml_pipeline import MLPipeline
        
        # Get cleaned data
        patients_df = pipeline_data['imputed_data']['patients']
        encounters_df = pipeline_data['imputed_data']['encounters']
        claims_df = pipeline_data['imputed_data']['claims']
        
        # Initialize ML pipeline
        ml_pipeline = MLPipeline(model_dir='models')
        
        # Train readmission prediction models
        logger.info("\nTraining readmission prediction models...")
        readmission_features = ml_pipeline.engineer_readmission_features(
            patients_df, encounters_df, claims_df
        )
        readmission_results = ml_pipeline.train_readmission_models(readmission_features)
        
        # Train cost prediction models
        logger.info("\nTraining cost prediction models...")
        cost_features = ml_pipeline.engineer_cost_features(
            patients_df, encounters_df, claims_df
        )
        cost_results = ml_pipeline.train_cost_prediction_models(cost_features)
        
        metrics = {
            'readmission': readmission_results,
            'cost': cost_results
        }
        
        logger.info("✅ ML pipeline completed successfully")
        return metrics
    except Exception as e:
        logger.error(f"❌ ML pipeline failed: {str(e)}", exc_info=True)
        return None


def calculate_analytics(config, pipeline_data):
    """Calculate comprehensive analytics including statistical analysis, cohorts, and time series"""
    try:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: COMPREHENSIVE ANALYTICS")
        logger.info("=" * 70)
        
        from src.analytics.analytics_orchestrator import AnalyticsOrchestrator
        
        # Get cleaned data
        patients_df = pipeline_data['imputed_data']['patients']
        encounters_df = pipeline_data['imputed_data']['encounters']
        claims_df = pipeline_data['imputed_data']['claims']
        
        # Run comprehensive analytics
        orchestrator = AnalyticsOrchestrator(
            data_dir=str(config.CURATED_ZONE),
            output_dir=str(config.CURATED_ZONE / 'analytics')
        )
        
        analytics_results = orchestrator.run_complete_analytics(
            patients_df, encounters_df, claims_df
        )
        
        # Generate and display summary report
        summary_report = orchestrator.generate_summary_report(analytics_results)
        logger.info("\n" + summary_report)
        
        logger.info("✅ Comprehensive analytics completed successfully")
        return analytics_results
    except Exception as e:
        logger.error(f"❌ Analytics failed: {str(e)}", exc_info=True)
        return None


def generate_reports(config, pipeline_data, ml_metrics, analytics):
    """Generate comprehensive summary reports"""
    try:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: GENERATING REPORTS")
        logger.info("=" * 70)
        
        # Create reports directory
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        # Generate summary report
        report_file = reports_dir / 'pipeline_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HEALTHCARE ANALYTICS PLATFORM - COMPREHENSIVE SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = pipeline_data.get('raw_data', {})
            for data_type, df in summary.items():
                f.write(f"{data_type}: {len(df):,} records\n")
            f.write("\n")
            
            # Quality summary
            f.write("DATA QUALITY SUMMARY\n")
            f.write("-" * 80 + "\n")
            quality_report = pipeline_data.get('quality_report', {})
            for data_type, metrics in quality_report.items():
                f.write(f"{data_type}: {metrics['overall_score']:.1f}%\n")
            f.write("\n")
            
            # Cohort Analysis
            if analytics and 'cohort_analysis' in analytics:
                f.write("COHORT ANALYSIS\n")
                f.write("-" * 80 + "\n")
                cohort_sizes = analytics['cohort_analysis'].get('cohort_sizes', {})
                for cohort, size in cohort_sizes.items():
                    f.write(f"{cohort.title()}: {size:,} patients\n")
                f.write("\n")
            
            # Hypothesis Testing Results
            if analytics and 'hypothesis_tests' in analytics:
                f.write("HYPOTHESIS TESTING RESULTS\n")
                f.write("-" * 80 + "\n")
                for test_name, test_result in analytics['hypothesis_tests'].items():
                    if 'interpretation' in test_result:
                        f.write(f"{test_name}:\n")
                        f.write(f"  {test_result['interpretation']}\n")
                f.write("\n")
            
            # Time Series Analysis
            if analytics and 'time_series' in analytics:
                f.write("TIME SERIES ANALYSIS\n")
                f.write("-" * 80 + "\n")
                
                if 'encounter_trends' in analytics['time_series']:
                    trends = analytics['time_series']['encounter_trends']
                    f.write(f"Encounter Trend: {trends.get('trend_direction', 'N/A')}\n")
                    f.write(f"Monthly Growth Rate: {trends.get('monthly_growth_rate_pct', 0):.2f}%\n")
                
                if 'encounter_forecast' in analytics['time_series']:
                    forecast = analytics['time_series']['encounter_forecast']
                    f.write(f"30-Day Forecast: {forecast.get('forecast_total', 0):.0f} encounters\n")
                
                if 'cost_forecast' in analytics['time_series']:
                    cost_forecast = analytics['time_series']['cost_forecast']
                    f.write(f"12-Month Cost Forecast: ${cost_forecast.get('forecast_total', 0):,.0f}\n")
                f.write("\n")
            
            # ML Model Performance
            if ml_metrics:
                f.write("MACHINE LEARNING MODEL PERFORMANCE\n")
                f.write("-" * 80 + "\n")
                
                if 'readmission' in ml_metrics:
                    m = ml_metrics['readmission']
                    f.write(f"Readmission Prediction:\n")
                    f.write(f"  Best Model: {m.get('best_model', 'N/A')}\n")
                    f.write(f"  ROC-AUC: {m.get('best_roc_auc', 0):.3f}\n")
                    
                    # Feature importance
                    if 'models' in m and m['best_model'] in m['models']:
                        best_model_results = m['models'][m['best_model']]
                        if 'feature_importance' in best_model_results:
                            f.write(f"  Top Features:\n")
                            for feat in best_model_results['feature_importance'][:5]:
                                f.write(f"    - {feat['feature']}: {feat['importance']:.3f}\n")
                    f.write("\n")
                
                if 'cost' in ml_metrics:
                    m = ml_metrics['cost']
                    f.write(f"Cost Prediction:\n")
                    f.write(f"  Best Model: {m.get('best_model', 'N/A')}\n")
                    f.write(f"  R² Score: {m.get('best_r2_score', 0):.3f}\n")
                    f.write(f"  RMSE: ${m['models'][m['best_model']]['rmse']:,.2f}\n")
                    f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated successfully\n")
            f.write("For detailed analytics, see: data/curated/analytics/analytics_results.json\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"✅ Report generated: {report_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}", exc_info=True)
        return False


def main():
    """Main application entry point"""
    try:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Healthcare Analytics Platform')
        parser.add_argument('--advanced-patients', action='store_true', 
                          help='Use advanced patient generator (50+ attributes)')
        parser.add_argument('--advanced-cleaning', action='store_true', default=True,
                          help='Use advanced data cleaning pipeline (default: True)')
        parser.add_argument('--skip-generation', action='store_true',
                          help='Skip data generation phase')
        args = parser.parse_args()
        
        logger.info("=" * 70)
        logger.info("HEALTHCARE ANALYTICS PLATFORM - STARTING")
        logger.info("=" * 70)
        
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration loaded")
        logger.info(f"Landing Zone: {config.LANDING_ZONE}")
        logger.info(f"Curated Zone: {config.CURATED_ZONE}")
        
        if args.advanced_patients:
            logger.info("✓ Advanced patient generation enabled")
        if args.advanced_cleaning:
            logger.info("✓ Advanced data cleaning enabled")
        
        # Phase 1: Generate synthetic data
        if not args.skip_generation:
            if not generate_data(config, use_advanced_patients=args.advanced_patients):
                logger.warning("⚠️  Data generation had issues, attempting to continue...")
        else:
            logger.info("⏭️  Skipping data generation phase")
        
        # Phase 2: Run data engineering pipeline
        pipeline_data = run_data_engineering(config, use_advanced=args.advanced_cleaning)
        if pipeline_data is None:
            logger.error("❌ Data engineering failed, cannot continue")
            sys.exit(1)
        
        # Phase 3: Train ML models
        ml_metrics = run_ml_pipeline(pipeline_data)
        
        # Phase 4: Calculate analytics
        analytics = calculate_analytics(config, pipeline_data)
        
        # Phase 5: Generate reports
        generate_reports(config, pipeline_data, ml_metrics, analytics)
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("HEALTHCARE ANALYTICS PLATFORM - COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\n✅ All phases completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. View reports in 'reports/' directory")
        logger.info("  2. Launch dashboard: launch_dashboard.bat")
        logger.info("  3. Explore trained models in 'models/' directory")
        if args.advanced_cleaning:
            logger.info("  4. View advanced data engineering reports in 'data/curated/data_engineering/'")
        logger.info("\n" + "=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
