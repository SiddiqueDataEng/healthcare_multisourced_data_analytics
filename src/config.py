"""
Configuration management for Healthcare Analytics Platform
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Application configuration"""
    
    def __init__(self):
        # Paths
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.LANDING_ZONE = self.DATA_DIR / "landing_zone"
        self.CURATED_ZONE = self.DATA_DIR / "curated"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Database
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        self.DB_NAME = os.getenv("DB_NAME", "healthcare_analytics")
        self.DB_USER = os.getenv("DB_USER", "postgres")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
        
        # Spark
        self.SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
        self.SPARK_MEMORY = os.getenv("SPARK_MEMORY", "4g")
        
        # Airflow
        self.AIRFLOW_HOME = self.PROJECT_ROOT / "airflow"
        self.DAGS_FOLDER = self.AIRFLOW_HOME / "dags"
        
        # ML Models
        self.READMISSION_MODEL_PATH = self.MODELS_DIR / "readmission_model.pkl"
        self.COST_CLUSTERING_MODEL_PATH = self.MODELS_DIR / "cost_clustering_model.pkl"
        self.FRAUD_DETECTION_MODEL_PATH = self.MODELS_DIR / "fraud_detection_model.pkl"
        
        # Feature flags
        self.ENABLE_HIPAA_VALIDATION = True
        self.ENABLE_DATA_QUALITY_CHECKS = True
        self.ENABLE_ML_MODELS = True
        
        # Create necessary directories
        self.LANDING_ZONE.mkdir(parents=True, exist_ok=True)
        self.CURATED_ZONE.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.DAGS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    @property
    def config_path(self) -> Path:
        """Get config file path"""
        return self.PROJECT_ROOT / ".env"
    
    @property
    def database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Load environment variables
load_dotenv()
