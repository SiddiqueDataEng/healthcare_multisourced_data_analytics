"""Base generator class to eliminate code duplication"""

import random
import logging
from typing import Any
import pandas as pd
from datagenerator.config import GeneratorConfig

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Base class for all data generators"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        random.seed(config.random_seed)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def to_csv(self, df: pd.DataFrame, filepath: str, entity_name: str = "records") -> None:
        """Save DataFrame to CSV with consistent logging"""
        try:
            df.to_csv(filepath, index=False)
            self.logger.info(f"Generated {len(df):,} {entity_name} -> {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save {entity_name} to {filepath}: {str(e)}")
            raise
    
    def get_seasonal_factor(self, date) -> float:
        """Calculate seasonal factor for encounter volume (winter surge)"""
        if not self.config.seasonal_variation:
            return 1.0
        
        month = date.month
        # Winter months (Dec, Jan, Feb) have higher encounter rates
        if month in [12, 1, 2]:
            return self.config.winter_surge_factor
        # Spring/Fall moderate
        elif month in [3, 4, 5, 9, 10, 11]:
            return 1.1
        # Summer lower
        else:
            return 0.9
    
    def calculate_age_risk_factor(self, age: int) -> float:
        """Calculate risk factor based on age (0-1 scale)"""
        if age < 40:
            return 0.1
        elif age < 60:
            return 0.3
        elif age < 75:
            return 0.6
        else:
            return 0.9
    
    def weighted_choice(self, choices: list, weights: list) -> Any:
        """Make a weighted random choice"""
        return random.choices(choices, weights=weights, k=1)[0]
