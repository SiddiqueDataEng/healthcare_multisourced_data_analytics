# Healthcare Analytics & Intelligence Platform

Enterprise-scale healthcare data platform integrating EHR, Claims, Disease Registry, and External Reporting data to enable clinical insights, cost optimization, regulatory compliance, and predictive analytics.

## Project Overview

This platform supports:
- Clinical quality improvement
- Population health management
- Cost & utilization analysis
- Regulatory & public reporting
- Predictive modeling for risk and outcomes

## Architecture

```
Synthetic Data Generation (datagenerator/)
        ↓
Landing Zone (Raw Data Lake) - CSV files
        ↓
ETL / ELT (Spark + Airflow)
        ↓
Curated Data Warehouse (Parquet)
        ↓
Analytics + ML + BI
```

## Tech Stack

- **Cloud**: AWS / Azure / GCP
- **Storage**: S3 / ADLS / GCS
- **Processing**: Apache Spark (PySpark)
- **Orchestration**: Apache Airflow
- **Warehouse**: Snowflake / BigQuery / Redshift
- **ML**: scikit-learn, XGBoost
- **BI**: Power BI / Tableau / Looker
- **Data Generation**: Pandas (synthetic data)

## Quick Start

### Prerequisites
- Python 3.9+
- Windows OS (for batch script)

### Setup (Automated)

Run the setup script to create virtual environment, install dependencies, and generate data:

```bash
setup.bat
```

This script will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Validate the installation
4. Generate synthetic healthcare data

### Manual Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate.bat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate synthetic data:
```bash
python run_datagenerator.py
```

4. Run the application:
```bash
python src/main.py
```

## Data Generation

The platform includes a comprehensive synthetic data generator that creates realistic healthcare data.

### Quick Data Generation

Generate data with default parameters (5K patients, 25K encounters, 50K claims):
```bash
python run_datagenerator.py
```

### Custom Data Volumes

Generate with custom parameters:
```bash
python run_datagenerator.py --patients 10000 --encounters 50000 --claims 100000
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

### Generated Data Files

The generator creates CSV files in `data/landing_zone/`:

**Master Data:**
- `patients.csv` - 5,000 patients with demographics and comorbidities
- `providers.csv` - 500 healthcare providers with specialties
- `facilities.csv` - 50 healthcare facilities

**Transactional Data:**
- `encounters.csv` - 25,000 clinical encounters
- `claims.csv` - 50,000 medical claims
- `registry.csv` - 500 disease registry entries

**Reporting Data:**
- `cms_measures.csv` - CMS quality measures
- `hai_data.csv` - CDC NHSN HAI data

See `datagenerator/README.md` for detailed documentation.

## Configuration

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Update `.env` with your configuration:
```
DB_HOST=your_database_host
DB_PORT=5432
DB_NAME=healthcare_analytics
DB_USER=your_username
DB_PASSWORD=your_password
SPARK_MASTER=local[*]
SPARK_MEMORY=4g
```

## Running the Application

Activate the virtual environment:
```bash
venv\Scripts\activate.bat
```

Run the main application (generates data → runs pipeline → calculates metrics → trains models):
```bash
python src/main.py
```

## Project Structure

```
healthcare-analytics/
├── datagenerator/              # Synthetic data generation
│   ├── main.py                 # Data generator orchestrator
│   ├── config.py               # Generator configuration
│   ├── patient_generator.py    # Patient data generation
│   ├── provider_generator.py   # Provider/facility generation
│   ├── encounter_generator.py  # Clinical encounter generation
│   ├── claims_generator.py     # Claims data generation
│   ├── registry_generator.py   # Disease registry generation
│   ├── reporting_generator.py  # External reporting generation
│   └── README.md               # Data generator documentation
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration management
│   ├── data_pipeline/
│   │   ├── orchestrator.py     # Pipeline orchestration
│   │   ├── extractors.py       # Data extraction from CSV
│   │   ├── transformers.py     # Data transformation
│   │   ├── loaders.py          # Data loading
│   │   └── validators.py       # Data validation
│   ├── analytics/
│   │   └── quality_metrics.py  # Quality metrics calculation
│   └── ml/
│       └── models.py           # ML model management
├── tests/                      # Test suite
├── data/
│   ├── landing_zone/           # Raw data (CSV files)
│   └── curated/                # Processed data (Parquet)
├── models/                     # Trained ML models
├── airflow/                    # Airflow DAGs
├── requirements.txt            # Python dependencies
├── setup.bat                   # Automated setup script
├── run_datagenerator.py        # Standalone data generator
├── .env.example                # Environment template
└── README.md                   # This file
```

## Data Pipeline Flow

```
1. Data Generation (datagenerator/)
   ↓
2. Data Extraction (extractors.py)
   - Read CSV files from landing_zone
   - Parse EHR, claims, registry data
   ↓
3. Data Validation (validators.py)
   - Check data quality
   - Validate HIPAA compliance
   ↓
4. Data Transformation (transformers.py)
   - Create fact tables (encounters, claims)
   - Create dimension tables (patient, provider, diagnosis)
   - Implement SCD Type 2 for patient history
   ↓
5. Data Loading (loaders.py)
   - Save to Parquet in curated zone
   - Load to data warehouse (Snowflake/BigQuery/Redshift)
   ↓
6. Analytics & ML (quality_metrics.py, models.py)
   - Calculate quality metrics
   - Train predictive models
   - Generate insights
```

## Data Models

### Fact Tables
- `fact_encounters` - Clinical encounters with diagnoses and procedures
- `fact_claims` - Claims transactions with costs and adjudication
- `fact_procedures` - Procedure details and outcomes
- `fact_quality_measures` - Quality metrics by facility

### Dimension Tables
- `dim_patient` - Patient master (SCD Type 2 for history tracking)
- `dim_provider` - Provider master with specialties
- `dim_facility` - Facility master with characteristics
- `dim_diagnosis` - ICD-10 diagnosis codes
- `dim_time` - Time dimension for temporal analysis

## Analytics & BI

### Dashboards
- Readmission Rate Dashboard
- Cost per Encounter by Diagnosis
- Length of Stay Analysis
- HAI Trends (CDC-aligned)
- Provider Performance Scorecards
- Benchmarking vs Peer Hospitals

## Machine Learning Models

### Readmission Prediction
- Algorithm: Logistic Regression / XGBoost
- Features: Diagnosis history, procedure complexity, prior utilization, comorbidities
- Use Case: Identify high-risk patients for intervention

### High-Cost Patient Identification
- Algorithm: K-Means Clustering
- Features: Total claims cost, procedure complexity, comorbidity burden
- Use Case: Target cost reduction initiatives

### Length of Stay Forecasting
- Algorithm: Regression Models
- Features: Diagnosis, procedure type, patient demographics
- Use Case: Optimize bed management and staffing

### Fraud & Anomaly Detection
- Algorithm: Isolation Forest
- Features: Claims cost patterns, billing patterns, procedure frequency
- Use Case: Detect fraudulent claims and billing anomalies

## Regulatory Compliance

- HIPAA-compliant data handling
- Automated CMS Quality Measures reporting
- CDC NHSN HAI submissions
- State adverse event reporting
- Data validation & quality checks

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src
```

## Development

### Code Style
- Format code with Black: `black src/`
- Lint with Flake8: `flake8 src/`

### Logging
All modules use Python's logging module. Configure log level in `.env`:
```
LOG_LEVEL=INFO
```

## Performance Optimization

- Adaptive query execution in Spark
- Partitioned data storage
- Indexed dimension tables
- Optimized Spark job configuration
- Caching frequently accessed data

## Data Characteristics

The synthetic data generator creates realistic healthcare data with:

- **Patient Demographics**: Age distribution (mean 55, std 20), realistic gender/race distribution
- **Comorbidities**: Increase with age, realistic prevalence rates
- **Encounters**: Mix of inpatient, outpatient, emergency, and telehealth
- **Claims**: Realistic procedure costs with adjudication logic
- **Readmissions**: ~15% readmission rate (configurable)
- **High-Cost Patients**: ~20% of population (configurable)
- **Fraud**: ~2% fraudulent claims (configurable)
- **HAI Rates**: Realistic infection rates per 1000 patient days

## Business Impact

- 📉 Reduced readmission rates through predictive insights
- 💰 Identified key cost drivers across populations
- 📊 Enabled peer benchmarking and accreditation readiness
- ⚙️ Scaled analytics across millions of claims records
- 🏥 Improved clinical quality and operational efficiency

## Support

For issues or questions, refer to:
- `datagenerator/README.md` - Data generation documentation
- `src/` - Source code with inline documentation
- `tests/` - Test examples

## License

Proprietary - Healthcare Analytics Platform
