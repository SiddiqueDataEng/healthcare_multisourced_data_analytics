"""
Data pipeline orchestrator - manages ETL workflows
"""

import logging
from typing import Dict, Any
from pyspark.sql import SparkSession
from src.config import Config
from src.data_pipeline.extractors import DataExtractor
from src.data_pipeline.transformers import DataTransformer
from src.data_pipeline.loaders import DataLoader
from src.data_pipeline.validators import DataValidator

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Orchestrates end-to-end data pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.spark = self._initialize_spark()
        self.extractor = DataExtractor(config, self.spark)
        self.transformer = DataTransformer(config, self.spark)
        self.loader = DataLoader(config, self.spark)
        self.validator = DataValidator(config, self.spark)
    
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session"""
        return SparkSession.builder \
            .master(self.config.SPARK_MASTER) \
            .appName("HealthcareAnalyticsPlatform") \
            .config("spark.driver.memory", self.config.SPARK_MEMORY) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute complete data pipeline"""
        try:
            logger.info("Starting data pipeline execution")
            
            # Extract data from sources
            logger.info("Phase 1: Extracting data from sources")
            ehr_data = self.extractor.extract_ehr_data()
            claims_data = self.extractor.extract_claims_data()
            registry_data = self.extractor.extract_registry_data()
            
            # Validate raw data
            logger.info("Phase 2: Validating raw data")
            self.validator.validate_ehr_data(ehr_data)
            self.validator.validate_claims_data(claims_data)
            
            # Transform data
            logger.info("Phase 3: Transforming data")
            fact_encounters = self.transformer.create_fact_encounters(ehr_data)
            fact_claims = self.transformer.create_fact_claims(claims_data)
            dim_patient = self.transformer.create_dim_patient(ehr_data)
            dim_provider = self.transformer.create_dim_provider(ehr_data)
            dim_diagnosis = self.transformer.create_dim_diagnosis(ehr_data)
            
            # Load to warehouse
            logger.info("Phase 4: Loading to data warehouse")
            self.loader.load_fact_table("fact_encounters", fact_encounters)
            self.loader.load_fact_table("fact_claims", fact_claims)
            self.loader.load_dimension_table("dim_patient", dim_patient)
            self.loader.load_dimension_table("dim_provider", dim_provider)
            self.loader.load_dimension_table("dim_diagnosis", dim_diagnosis)
            
            logger.info("Data pipeline completed successfully")
            return {"status": "success", "records_processed": 0}
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
