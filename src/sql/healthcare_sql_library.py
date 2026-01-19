"""
Healthcare SQL Query Library
"""

import pandas as pd
import sqlite3
import os
from typing import Dict, List, Any

class HealthcareSQLLibrary:
    """Healthcare SQL queries library"""
    
    def __init__(self):
        self.queries = {
            "patient_age_distribution": {
                "sql": """
                SELECT 
                    CASE 
                        WHEN age BETWEEN 0 AND 17 THEN 'Pediatric (0-17)'
                        WHEN age BETWEEN 18 AND 34 THEN 'Young Adult (18-34)'
                        WHEN age BETWEEN 35 AND 54 THEN 'Middle Age (35-54)'
                        WHEN age BETWEEN 55 AND 74 THEN 'Older Adult (55-74)'
                        WHEN age >= 75 THEN 'Elderly (75+)'
                    END as age_group,
                    COUNT(*) as patient_count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM patients), 2) as percentage
                FROM patients
                GROUP BY age_group
                ORDER BY MIN(age);
                """,
                "description": "Patient age distribution analysis",
                "category": "Patient Demographics"
            },
            "encounter_summary": {
                "sql": """
                SELECT 
                    e.encounter_type,
                    COUNT(*) as encounter_count,
                    ROUND(AVG(c.claim_amount), 2) as avg_cost
                FROM encounters e
                LEFT JOIN claims c ON e.patient_id = c.patient_id
                GROUP BY e.encounter_type
                ORDER BY encounter_count DESC;
                """,
                "description": "Encounter type summary",
                "category": "Encounters"
            }
        }
        
        self.query_categories = {
            "Patient Demographics": ["patient_age_distribution"],
            "Encounters": ["encounter_summary"]
        }
        
        self.query_explanations = {
            "Patient Demographics": "Analyze patient population characteristics",
            "Encounters": "Analyze healthcare encounters and utilization"
        }
    
    def get_query_list(self) -> Dict[str, List[str]]:
        """Get list of available queries by category"""
        return self.query_categories
    
    def get_query_text(self, query_name: str) -> str:
        """Get SQL text for a query"""
        if query_name in self.queries:
            return self.queries[query_name].get("sql", "")
        return ""
    
    def explain_query(self, query_name: str) -> str:
        """Get explanation for a query"""
        if query_name in self.queries:
            return self.queries[query_name].get("description", "")
        return f"Query '{query_name}' not found"
    
    def execute_query(self, query_name: str, connection=None) -> pd.DataFrame:
        """Execute a query"""
        sql = self.get_query_text(query_name)
        if not sql:
            return pd.DataFrame()
        
        if connection is None:
            connection = self._create_database()
        
        try:
            return pd.read_sql_query(sql, connection)
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()
    
    def _create_database(self):
        """Create in-memory database with sample data"""
        conn = sqlite3.connect(':memory:')
        
        # Load data files if they exist
        data_files = {
            'patients': 'data/landing_zone/patients.csv',
            'encounters': 'data/landing_zone/encounters.csv',
            'claims': 'data/landing_zone/claims.csv'
        }
        
        for table_name, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df.to_sql(table_name, conn, index=False, if_exists='replace')
                except Exception as e:
                    print(f"Error loading {table_name}: {e}")
        
        return conn

# Global instance
sql_library = HealthcareSQLLibrary()