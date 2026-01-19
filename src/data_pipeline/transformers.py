"""
Data transformers - convert raw data to analytics-ready schemas
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, row_number, current_timestamp
from pyspark.sql.window import Window
from src.config import Config

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms raw data into dimensional and fact tables"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
    
    def create_fact_encounters(self, ehr_data: DataFrame) -> DataFrame:
        """Create fact_encounters table"""
        logger.info("Creating fact_encounters table")
        try:
            fact_encounters = ehr_data.select(
                col("encounter_id"),
                col("patient_id"),
                col("provider_id"),
                col("facility_id"),
                col("encounter_date"),
                col("diagnosis_code"),
                col("procedure_code"),
                current_timestamp().alias("load_date")
            ).distinct()
            
            logger.info(f"Fact encounters created: {fact_encounters.count()} records")
            return fact_encounters
        except Exception as e:
            logger.error(f"Failed to create fact_encounters: {str(e)}")
            raise
    
    def create_fact_claims(self, claims_data: DataFrame) -> DataFrame:
        """Create fact_claims table"""
        logger.info("Creating fact_claims table")
        try:
            fact_claims = claims_data.select(
                col("claim_id"),
                col("patient_id"),
                col("provider_id"),
                col("claim_date"),
                col("service_date"),
                col("claim_amount"),
                col("allowed_amount"),
                col("paid_amount"),
                col("procedure_code"),
                col("diagnosis_code"),
                col("adjudication_status"),
                current_timestamp().alias("load_date")
            ).distinct()
            
            logger.info(f"Fact claims created: {fact_claims.count()} records")
            return fact_claims
        except Exception as e:
            logger.error(f"Failed to create fact_claims: {str(e)}")
            raise
    
    def create_dim_patient(self, ehr_data: DataFrame) -> DataFrame:
        """Create dim_patient dimension table with SCD Type 2"""
        logger.info("Creating dim_patient table")
        try:
            # Implement Slowly Changing Dimension Type 2
            window_spec = Window.partitionBy("patient_id").orderBy(col("encounter_date").desc())
            
            dim_patient = ehr_data.select("patient_id").distinct() \
                .withColumn("patient_key", row_number().over(window_spec)) \
                .withColumn("effective_date", current_timestamp()) \
                .withColumn("end_date", col("effective_date")) \
                .withColumn("is_current", col("is_current").cast("boolean"))
            
            logger.info(f"Dim patient created: {dim_patient.count()} records")
            return dim_patient
        except Exception as e:
            logger.error(f"Failed to create dim_patient: {str(e)}")
            raise
    
    def create_dim_provider(self, ehr_data: DataFrame) -> DataFrame:
        """Create dim_provider dimension table"""
        logger.info("Creating dim_provider table")
        try:
            dim_provider = ehr_data.select("provider_id").distinct() \
                .withColumn("provider_key", row_number().over(Window.orderBy("provider_id"))) \
                .withColumn("load_date", current_timestamp())
            
            logger.info(f"Dim provider created: {dim_provider.count()} records")
            return dim_provider
        except Exception as e:
            logger.error(f"Failed to create dim_provider: {str(e)}")
            raise
    
    def create_dim_diagnosis(self, ehr_data: DataFrame) -> DataFrame:
        """Create dim_diagnosis dimension table"""
        logger.info("Creating dim_diagnosis table")
        try:
            dim_diagnosis = ehr_data.select("diagnosis_code").distinct() \
                .withColumn("diagnosis_key", row_number().over(Window.orderBy("diagnosis_code"))) \
                .withColumn("load_date", current_timestamp())
            
            logger.info(f"Dim diagnosis created: {dim_diagnosis.count()} records")
            return dim_diagnosis
        except Exception as e:
            logger.error(f"Failed to create dim_diagnosis: {str(e)}")
            raise
    
    def create_dim_time(self) -> DataFrame:
        """Create dim_time dimension table"""
        logger.info("Creating dim_time table")
        try:
            # Create time dimension for 10 years
            dates = self.spark.sql("""
                SELECT 
                    date_format(date_add('2014-01-01', seq), 'yyyyMMdd') as date_key,
                    date_add('2014-01-01', seq) as calendar_date,
                    year(date_add('2014-01-01', seq)) as year,
                    month(date_add('2014-01-01', seq)) as month,
                    quarter(date_add('2014-01-01', seq)) as quarter,
                    dayofweek(date_add('2014-01-01', seq)) as day_of_week
                FROM (SELECT explode(sequence(0, 3650)) as seq)
            """)
            
            logger.info(f"Dim time created: {dates.count()} records")
            return dates
        except Exception as e:
            logger.error(f"Failed to create dim_time: {str(e)}")
            raise
