"""
Data loaders - persist transformed data to warehouse
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from src.config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads data to warehouse"""
    
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
    
    def load_fact_table(self, table_name: str, data: DataFrame) -> None:
        """Load fact table to warehouse"""
        logger.info(f"Loading fact table: {table_name}")
        try:
            # In production, this would write to Snowflake/BigQuery/Redshift
            # For now, save as parquet in curated zone
            output_path = self.config.CURATED_ZONE / table_name
            data.write.mode("overwrite").parquet(str(output_path))
            logger.info(f"Fact table {table_name} loaded: {data.count()} records")
        except Exception as e:
            logger.error(f"Failed to load fact table {table_name}: {str(e)}")
            raise
    
    def load_dimension_table(self, table_name: str, data: DataFrame) -> None:
        """Load dimension table to warehouse"""
        logger.info(f"Loading dimension table: {table_name}")
        try:
            output_path = self.config.CURATED_ZONE / table_name
            data.write.mode("overwrite").parquet(str(output_path))
            logger.info(f"Dimension table {table_name} loaded: {data.count()} records")
        except Exception as e:
            logger.error(f"Failed to load dimension table {table_name}: {str(e)}")
            raise
    
    def load_to_database(self, table_name: str, data: DataFrame, mode: str = "overwrite") -> None:
        """Load data to PostgreSQL database"""
        logger.info(f"Loading to database table: {table_name}")
        try:
            data.write \
                .format("jdbc") \
                .mode(mode) \
                .option("url", self.config.database_url) \
                .option("dbtable", table_name) \
                .option("user", self.config.DB_USER) \
                .option("password", self.config.DB_PASSWORD) \
                .save()
            logger.info(f"Database table {table_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load to database: {str(e)}")
            raise
