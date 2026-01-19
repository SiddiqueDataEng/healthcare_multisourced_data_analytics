"""
Healthcare Data Generator - Main Entry Point
Generates realistic healthcare data for the analytics platform
"""

import os
import logging
from pathlib import Path
from datagenerator.config import GeneratorConfig
from datagenerator.patient_generator import PatientGenerator
from datagenerator.provider_generator import ProviderGenerator
from datagenerator.encounter_generator import EncounterGenerator
from datagenerator.claims_generator import ClaimsGenerator
from datagenerator.registry_generator import RegistryGenerator
from datagenerator.reporting_generator import ReportingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Main data generator orchestrator"""
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")
    
    def generate_all(self):
        """Generate all healthcare data"""
        try:
            logger.info("=" * 60)
            logger.info("Healthcare Data Generator Starting")
            logger.info("=" * 60)
            
            # Generate master data
            logger.info("\n[1/6] Generating patient data...")
            patient_gen = PatientGenerator(self.config)
            patients_df = patient_gen.generate_patients()
            patient_gen.to_csv(patients_df, f"{self.config.output_dir}/patients.csv")
            patient_ids = patients_df["patient_id"].tolist()
            
            logger.info("\n[2/6] Generating provider and facility data...")
            provider_gen = ProviderGenerator(self.config)
            providers_df = provider_gen.generate_providers()
            provider_gen.to_csv(providers_df, f"{self.config.output_dir}/providers.csv")
            provider_ids = providers_df["provider_id"].tolist()
            
            facilities_df = provider_gen.generate_facilities()
            provider_gen.to_csv(facilities_df, f"{self.config.output_dir}/facilities.csv")
            facility_ids = facilities_df["facility_id"].tolist()
            
            # Generate transactional data
            logger.info("\n[3/6] Generating encounter data...")
            encounter_gen = EncounterGenerator(self.config)
            encounters_df = encounter_gen.generate_encounters(patient_ids, provider_ids, facility_ids)
            encounter_gen.to_csv(encounters_df, f"{self.config.output_dir}/encounters.csv")
            
            logger.info("\n[4/6] Generating claims data...")
            claims_gen = ClaimsGenerator(self.config)
            claims_df = claims_gen.generate_claims(patient_ids, provider_ids)
            claims_gen.to_csv(claims_df, f"{self.config.output_dir}/claims.csv")
            
            logger.info("\n[5/6] Generating disease registry data...")
            registry_gen = RegistryGenerator(self.config)
            registry_df = registry_gen.generate_registry(patient_ids, provider_ids)
            registry_gen.to_csv(registry_df, f"{self.config.output_dir}/registry.csv")
            
            logger.info("\n[6/6] Generating external reporting data...")
            reporting_gen = ReportingGenerator(self.config)
            
            cms_measures_df = reporting_gen.generate_cms_measures(facility_ids)
            reporting_gen.to_csv(cms_measures_df, f"{self.config.output_dir}/cms_measures.csv")
            
            hai_data_df = reporting_gen.generate_hai_data(facility_ids)
            reporting_gen.to_csv(hai_data_df, f"{self.config.output_dir}/hai_data.csv")
            
            # Print summary
            self._print_summary(
                patients_df, providers_df, facilities_df, encounters_df,
                claims_df, registry_df, cms_measures_df, hai_data_df
            )
            
            logger.info("\n" + "=" * 60)
            logger.info("Data generation completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Data generation failed: {str(e)}", exc_info=True)
            raise
    
    def _print_summary(self, *dataframes):
        """Print data generation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA GENERATION SUMMARY")
        logger.info("=" * 60)
        
        labels = [
            "Patients",
            "Providers",
            "Facilities",
            "Encounters",
            "Claims",
            "Registry Records",
            "CMS Measures",
            "HAI Records"
        ]
        
        for label, df in zip(labels, dataframes):
            logger.info(f"{label:.<40} {len(df):>10,} records")
        
        logger.info("=" * 60)


def main():
    """Main entry point"""
    # Create custom config if needed
    config = GeneratorConfig(
        num_patients=10000,
        num_providers=500,
        num_facilities=50,
        num_encounters=50000,
        num_claims=100000
    )
    
    generator = DataGenerator(config)
    generator.generate_all()


if __name__ == "__main__":
    main()
