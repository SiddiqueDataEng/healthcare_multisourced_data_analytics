"""
Data extractors for various healthcare data sources
"""

import logging
from typing import Optional
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from src.config import Config

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extracts data from various healthcare sources"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
    
    def _read_csv(self, filepath: str) -> DataFrame:
        """Read CSV file from landing zone"""
        try:
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(filepath)
            return df
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {str(e)}")
            return None
    
    def extract_ehr_data(self) -> DataFrame:
        """Extract EHR data from landing zone (encounters + patients)"""
        logger.info("Extracting EHR data")
        try:
            # Read encounters data
            encounters_path = str(self.config.LANDING_ZONE / "encounters.csv")
            encounters_df = self._read_csv(encounters_path)
            
            if encounters_df is None:
                logger.warning("No encounters data found, creating empty schema")
                ehr_schema = """
                    patient_id STRING,
                    encounter_id STRING,
                    encounter_date DATE,
                    diagnosis_code STRING,
                    procedure_code STRING,
                    provider_id STRING,
                    facility_id STRING
                """
                return self.spark.createDataFrame([], ehr_schema)
            
            # Select relevant columns for EHR
            ehr_data = encounters_df.select(
                "patient_id",
                "encounter_id",
                "encounter_date",
                "primary_diagnosis",
                "primary_procedure",
                "provider_id",
                "facility_id"
            ).withColumnRenamed("primary_diagnosis", "diagnosis_code") \
             .withColumnRenamed("primary_procedure", "procedure_code")
            
            logger.info(f"EHR data extracted: {ehr_data.count()} records")
            return ehr_data
        except Exception as e:
            logger.error(f"Failed to extract EHR data: {str(e)}")
            raise
    
    def extract_claims_data(self) -> DataFrame:
        """Extract claims data from landing zone"""
        logger.info("Extracting claims data")
        try:
            claims_path = str(self.config.LANDING_ZONE / "claims.csv")
            claims_df = self._read_csv(claims_path)
            
            if claims_df is None:
                logger.warning("No claims data found, creating empty schema")
                claims_schema = """
                    claim_id STRING,
                    patient_id STRING,
                    provider_id STRING,
                    claim_date DATE,
                    service_date DATE,
                    claim_amount DOUBLE,
                    allowed_amount DOUBLE,
                    paid_amount DOUBLE,
                    procedure_code STRING,
                    adjudication_status STRING
                """
                return self.spark.createDataFrame([], claims_schema)
            
            logger.info(f"Claims data extracted: {claims_df.count()} records")
            return claims_df
        except Exception as e:
            logger.error(f"Failed to extract claims data: {str(e)}")
            raise
    
    def extract_registry_data(self) -> DataFrame:
        """Extract disease registry data"""
        logger.info("Extracting registry data")
        try:
            registry_path = str(self.config.LANDING_ZONE / "registry.csv")
            registry_df = self._read_csv(registry_path)
            
            if registry_df is None:
                logger.warning("No registry data found, creating empty schema")
                registry_schema = """
                    registry_id STRING,
                    patient_id STRING,
                    procedure_date DATE,
                    has_complication BOOLEAN,
                    outcome STRING,
                    risk_score DOUBLE
                """
                return self.spark.createDataFrame([], registry_schema)
            
            logger.info(f"Registry data extracted: {registry_df.count()} records")
            return registry_df
        except Exception as e:
            logger.error(f"Failed to extract registry data: {str(e)}")
            raise
    
    def extract_external_reporting_data(self) -> DataFrame:
        """Extract external reporting data (CDC, CMS, etc.)"""
        logger.info("Extracting external reporting data")
        try:
            cms_path = str(self.config.LANDING_ZONE / "cms_measures.csv")
            cms_df = self._read_csv(cms_path)
            
            if cms_df is None:
                logger.warning("No CMS data found, creating empty schema")
                reporting_schema = """
                    report_id STRING,
                    facility_id STRING,
                    report_date DATE,
                    measure_code STRING,
                    measure_value DOUBLE,
                    benchmark_value DOUBLE
                """
                return self.spark.createDataFrame([], reporting_schema)
            
            logger.info(f"External reporting data extracted: {cms_df.count()} records")
            return cms_df
        except Exception as e:
            logger.error(f"Failed to extract external reporting data: {str(e)}")
            raise
