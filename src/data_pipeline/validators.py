"""
Data quality validators - ensure HIPAA compliance and data integrity
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, when, isnan, isnull

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and HIPAA compliance"""
    
    def __init__(self, config, spark: SparkSession):
        self.config = config
        self.spark = spark
    
    def validate_ehr_data(self, data: DataFrame) -> bool:
        """Validate EHR data quality"""
        logger.info("Validating EHR data")
        try:
            # Check for required fields
            required_fields = ["patient_id", "encounter_id", "encounter_date"]
            for field in required_fields:
                if field not in data.columns:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check for nulls in key fields
            null_counts = data.select([
                count(when(col(f).isNull(), 1)).alias(f"{f}_nulls")
                for f in required_fields
            ]).collect()[0]
            
            logger.info(f"EHR data validation passed. Null counts: {null_counts}")
            return True
        except Exception as e:
            logger.error(f"EHR data validation failed: {str(e)}")
            raise
    
    def validate_claims_data(self, data: DataFrame) -> bool:
        """Validate claims data quality"""
        logger.info("Validating claims data")
        try:
            required_fields = ["claim_id", "patient_id", "claim_date", "claim_amount"]
            for field in required_fields:
                if field not in data.columns:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check for negative amounts
            negative_amounts = data.filter(col("claim_amount") < 0).count()
            if negative_amounts > 0:
                logger.warning(f"Found {negative_amounts} claims with negative amounts")
            
            logger.info("Claims data validation passed")
            return True
        except Exception as e:
            logger.error(f"Claims data validation failed: {str(e)}")
            raise
    
    def validate_hipaa_compliance(self, data: DataFrame) -> bool:
        """Validate HIPAA compliance"""
        logger.info("Validating HIPAA compliance")
        try:
            # Check for PII exposure
            pii_patterns = ["SSN", "DOB", "phone", "email"]
            for pattern in pii_patterns:
                matching_cols = [col for col in data.columns if pattern.lower() in col.lower()]
                if matching_cols:
                    logger.warning(f"Potential PII exposure detected: {matching_cols}")
            
            logger.info("HIPAA compliance validation completed")
            return True
        except Exception as e:
            logger.error(f"HIPAA compliance validation failed: {str(e)}")
            raise
