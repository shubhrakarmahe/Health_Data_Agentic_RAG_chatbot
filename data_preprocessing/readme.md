# Health Data Preprocessing & Storage Pipeline

A modular, production-ready Python pipeline designed to ingest, clean, transform, and store clinical and health-related datasets into SQLite. This project specifically handles common healthcare data challenges such as de-identification (PII masking), physiological feature engineering, and activity log aggregation.

## 🌟 Key Features

* **Automated Data Cleaning**: Handles missing values, type casting, and column name normalization (snake_case conversion).
* **Clinical Transformation**:
* Converts binary codes (0/1) to human-readable labels (`yes`/`no`).
* Maps stress level numerical codes to descriptive categories (`low`, `normal`, `high`).
* Converts encoded sex values to labels.


* **Feature Engineering**:
* **BMI Category**: Bins numeric BMI into 'Underweight', 'Normal', 'Overweight', etc.
* **Hemoglobin Categorization**: Sex-specific logic to determine 'low', 'normal', or 'high' levels.
* **Age Grouping**: Automatic binning into age brackets (e.g., 18-29, 30-39).


* **Security & Privacy**: Built-in de-identification for sensitive columns (e.g., `Patient_Number`) using deterministic masking.
* **Robust Storage**:
* SQLAlchemy-powered SQLite backend.
* Supports `append`, `replace`, and `upsert` (Insert or Replace) logic.
* Automatic index creation for optimized downstream queries.
* WAL (Write-Ahead Logging) mode enabled for improved SQLite performance.


* **Schema Validation**: Integration with `pandera` for strict data quality enforcement.

## 🏗 Project Structure

```text
├── pipeline.py          # Orchestration logic for the entire process
├── preprocess.py        # Transformation and feature engineering functions
├── db_store.py          # SQLAlchemy engine setup and database I/O operations
├── utils.py             # Logging, de-identification, and helper utilities
└── example_usage.py     # End-to-end execution script for Health Datasets

```

## 🚀 Quick Start

### 1. Requirements

* Python 3.8+
* Pandas, SQLAlchemy, Openpyxl, Pandera

### 2. Basic Execution

Prepare your datasets in the `dataset/` folder and configure your options in `example_usage.py`:

```python
from pipeline import process_and_store

# Define your DataFrames in a dictionary
dfs = {
    "health_dataset_1": df1,
    "health_dataset_2_agg": df2_agg
}

# Run the pipeline
process_and_store(
    dfs, 
    sqlite_path="health_data.sqlite", 
    table_options=table_options
)

```

## 🛠 Configuration Options

The pipeline is highly configurable via the `table_options` dictionary. Key configurations include:

* `pii_columns`: List of columns to be masked.
* `type_map`: Force specific dtypes (e.g., `{"Age": "int"}`).
* `feature_engineering_list`: Define custom output columns for BMI, Age, or Hemoglobin.
* `yesno_columns`: Convert 0/1 columns to 'yes'/'no' strings.
* `create_indexes`: Automatically generate SQLite indexes for specific columns.

## 📊 Transformations Detail

### Hemoglobin Categorization

The pipeline applies sex-aware ranges to categorize hemoglobin levels:

* **Male**: Normal between 14-18 g/dL.
* **Female**: Normal between 12-16 g/dL.

### Activity Aggregation

For longitudinal activity logs, the `aggregate_physical_activity` function calculates:

* Total, Max, Min, and Mean physical activity.
* Count of active days vs. missed days.

## 🔒 Privacy & De-identification

The `mask_patient_number` utility ensures that identifiers are protected. If a value is a digit string of 6+ characters, it preserves the first and last two digits (e.g., `12****78`). Otherwise, it generates a deterministic `ANON_` hash.

## 📝 Logging & Auditing

Every processed table includes an `_ingestion_time` column in ISO format for auditing purposes. Detailed logs are generated for row counts (in/out), processing time, and any missing required columns.
