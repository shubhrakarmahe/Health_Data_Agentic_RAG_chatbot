# example_usage.py
import pandas as pd
import openpyxl
from pipeline import process_and_store
from preprocess import aggregate_physical_activity

# Load the first dataset
df1 = pd.read_excel('dataset/Health Dataset 1.xlsm', engine='openpyxl')

# Load the second dataset
df2 = pd.read_excel('dataset/Health Dataset 2.xlsm', engine='openpyxl')

# build aggregated dataset for health_dataset_2
health_dataset_2_agg = aggregate_physical_activity(pd.read_excel('dataset/Health Dataset 2.xlsm', engine='openpyxl'),
                                                    patient_col='Patient_Number', activity_col='Physical_activity')

dfs = {
    "health_dataset_1": df1,
    "health_dataset_2": df2,
    "health_dataset_2_agg": health_dataset_2_agg
}

table_options = {
    "health_dataset_1": {
    "required_columns": ["Patient_Number"],
    "type_map": {
        "Patient_Number": "int",
        "Blood_Pressure_Abnormality": "object",
        "Level_of_Hemoglobin": "float",
        "Genetic_Pedigree_Coefficient": "float",
        "Age": "int",
        "BMI": "int",
        "Sex": "object",
        "Pregnancy": "object",
        "Smoking": "object",
        "salt_content_in_the_diet": "int",
        "alcohol_consumption_per_day": "float",
        "Level_of_Stress": "object",
        "Chronic_kidney_disease": "object",
        "Adrenal_and_thyroid_disorders": "object"
    },
    "pii_columns": ["Patient_Number"],
    "missing_strategy": "fill",
    "fill_map": {"Pregnancy": "data not available"},
    "lowercase_columns": True,
    "column_rename_list": {"Level_of_Hemoglobin": "level_of_hemoglobin_g_per_dl", 
                       "salt_content_in_the_diet": "salt_content_in_the_diet_mg_per_day",
                       "alcohol_consumption_per_day": "alcohol_consumption_ml_per_day"},
    "feature_engineering": True,
    "feature_engineering_list": [{"column": "BMI", "output": "bmi_category"},
                                {"column": "Level_of_Hemoglobin", "output": "hemoglobin_category"},
                                {"column": "Age", "output": "age_group"},
                                {"column": "Genetic_Pedigree_Coefficient", "output": "gpc_availability"}],
    "sex_column": ["Sex"],
    "stress_level_codes": ["Level_of_Stress"],
    "yesno_columns": ["Smoking","Pregnancy", "Blood_Pressure_Abnormality", "Chronic_kidney_disease", "Adrenal_and_thyroid_disorders"],
    "if_exists": "replace",
    "create_indexes": ["Patient_Number", "Age","Sex"]
}
,
    "health_dataset_2": {
        "required_columns": ["Patient_Number"],
        "type_map": {"Patient_Number": "int", "Day_Number": "int", "Physical_activity": "int"},
        "pii_columns": ["Patient_Number"],
        "missing_strategy": "fill",
        "lowercase_columns": True,
        "fill_map": {"Physical_activity": 0},
        "if_exists": "replace",
        "create_indexes": ["Patient_Number", "Day_Number"]
    }
,
    "health_dataset_2_agg": {
        "required_columns": ["Patient_Number"],
        "type_map": {"Patient_Number": "int", "Total_physical_Activity": "int",
                     "Max_physical_Activity": "int", "Min_physical_Activity": "int",
                     "Mean_physical_Activity": "float", "No_of_days_missed_physical_activity": "int"},
        "pii_columns": ["Patient_Number"],
        "lowercase_columns": True,
        "if_exists": "replace",
        "create_indexes": ["Patient_Number"]
    }
}

process_and_store(dfs, sqlite_path="health_data.sqlite", table_options=table_options)


