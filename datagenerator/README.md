# Healthcare Data Generator

Generates realistic synthetic healthcare data for the Healthcare Analytics Platform. Creates EHR, claims, registry, and external reporting data with realistic distributions and relationships.

## Features

- **Patient Data**: Realistic demographics, comorbidities, and risk profiles
- **Provider Data**: Specialties, credentials, and facility assignments
- **Facility Data**: Hospital types, bed counts, and geographic distribution
- **Encounter Data**: Clinical encounters with diagnoses, procedures, and outcomes
- **Claims Data**: Medical and pharmacy claims with realistic costs and adjudication
- **Registry Data**: Disease registry entries with complications and outcomes
- **External Reporting**: CMS quality measures and CDC HAI data

## Data Characteristics

### Realistic Distributions
- Age distribution: Normal distribution centered around 55 years
- Comorbidities: Increase with age
- Readmission rates: ~15% (configurable)
- High-cost patients: ~20% (configurable)
- Fraud rate: ~2% (configurable)
- HAI rates: Realistic per 1000 patient days

### Data Relationships
- Patients linked to encounters and claims
- Providers assigned to facilities
- Encounters generate claims
- Registry entries for surgical procedures
- Quality measures tied to facilities

## Usage

### Quick Start

Generate data with default parameters:
```bash
python run_datagenerator.py
```

### Custom Parameters

Generate with custom volumes:
```bash
python run_datagenerator.py --patients 5000 --encounters 25000 --claims 50000
```

### Available Options

```
--patients N          Number of patients (default: 10000)
--providers N         Number of providers (default: 500)
--facilities N        Number of facilities (default: 50)
--encounters N        Number of encounters (default: 50000)
--claims N            Number of claims (default: 100000)
--output DIR          Output directory (default: data/landing_zone)
--seed N              Random seed for reproducibility (default: 42)
```

### Examples

Generate small dataset for testing:
```bash
python run_datagenerator.py --patients 1000 --encounters 5000 --claims 10000
```

Generate large dataset for performance testing:
```bash
python run_datagenerator.py --patients 50000 --encounters 250000 --claims 500000
```

Generate with specific seed for reproducibility:
```bash
python run_datagenerator.py --seed 12345
```

## Output Files

Generated CSV files in `data/landing_zone/`:

### Master Data
- `patients.csv` - Patient demographics and risk profiles
- `providers.csv` - Provider information and specialties
- `facilities.csv` - Facility information and characteristics

### Transactional Data
- `encounters.csv` - Clinical encounters with diagnoses and procedures
- `claims.csv` - Medical and pharmacy claims
- `registry.csv` - Disease registry entries

### Reporting Data
- `cms_measures.csv` - CMS quality measures
- `hai_data.csv` - CDC NHSN HAI data

## Data Schema

### patients.csv
```
patient_id, first_name, last_name, date_of_birth, age, gender, race,
comorbidities, is_high_cost, enrollment_date, active
```

### encounters.csv
```
encounter_id, patient_id, provider_id, facility_id, encounter_type,
encounter_date, discharge_date, length_of_stay, primary_diagnosis,
secondary_diagnoses, primary_procedure, secondary_procedures,
is_readmission, readmission_days, discharge_disposition, severity_level
```

### claims.csv
```
claim_id, patient_id, provider_id, claim_date, service_date,
procedure_code, claim_amount, allowed_amount, paid_amount,
adjudication_status, is_fraudulent, denial_reason, claim_line_number
```

### registry.csv
```
registry_id, patient_id, provider_id, procedure_name, procedure_date,
has_complication, complication_type, complication_date, outcome,
risk_score, mortality_flag, readmission_flag, days_to_readmission
```

### cms_measures.csv
```
report_id, facility_id, measure_code, measure_name, report_date,
measure_value, benchmark_value, percentile_rank, performance_status
```

### hai_data.csv
```
report_id, facility_id, hai_type, report_date, infection_count,
patient_days, infection_rate, national_benchmark, status
```

## Integration with Pipeline

The generated data is automatically integrated into the main pipeline:

1. **Data Generation**: `run_datagenerator.py` creates CSV files
2. **Data Extraction**: `src/data_pipeline/extractors.py` reads CSV files
3. **Data Transformation**: `src/data_pipeline/transformers.py` creates fact/dimension tables
4. **Data Loading**: `src/data_pipeline/loaders.py` persists to warehouse

## Configuration

Modify `datagenerator/config.py` to adjust default parameters:

```python
@dataclass
class GeneratorConfig:
    num_patients: int = 10000
    num_providers: int = 500
    num_facilities: int = 50
    num_encounters: int = 50000
    num_claims: int = 100000
    readmission_rate: float = 0.15
    high_cost_patient_rate: float = 0.20
    fraud_rate: float = 0.02
    random_seed: int = 42
```

## Performance

Typical generation times (on standard hardware):

| Dataset Size | Time |
|---|---|
| 1K patients, 5K encounters | < 1 second |
| 10K patients, 50K encounters | 5-10 seconds |
| 50K patients, 250K encounters | 30-60 seconds |
| 100K patients, 500K encounters | 2-3 minutes |

## Reproducibility

Use the `--seed` parameter to generate identical datasets:

```bash
# Generate dataset A
python run_datagenerator.py --seed 42

# Generate identical dataset B
python run_datagenerator.py --seed 42
```

## Customization

### Add Custom Data Generators

Create a new generator class:

```python
# datagenerator/custom_generator.py
class CustomGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    def generate(self) -> pd.DataFrame:
        # Your generation logic
        pass
```

### Modify Data Distributions

Edit generator classes to adjust distributions:

```python
# In patient_generator.py
age = random.gauss(55, 20)  # Mean 55, StdDev 20
```

## Troubleshooting

### Out of Memory
For large datasets, reduce volumes:
```bash
python run_datagenerator.py --patients 5000 --encounters 25000
```

### Slow Generation
Generation is I/O bound. Ensure output directory is on fast storage.

### Missing Data Files
Check that `data/landing_zone/` directory exists and is writable.

## License

Proprietary - Healthcare Analytics Platform
