# pipeline.py
from typing import Dict
import pandas as pd
from preprocessing_pipeline.db_store import create_sqlite_engine, write_dataframe, upsert_sqlite
from preprocessing_pipeline.preprocess import normalize_column_names, validate_columns, cast_types, handle_missing, deidentify_dataframe, sample_and_sanitize, \
      convert_01_to_yesno, convert_sex_01_to_label, convert_stress_level_codes, feature_engineering, validate_schema_pandera
from preprocessing_pipeline.utils import simple_logger, now_iso
import time
logger = simple_logger()

def process_and_store(
    dfs: Dict[str, pd.DataFrame],
    sqlite_path: str,
    table_options: Dict[str, Dict] = None
):
    """
    Orchestrate preprocessing and storage for multiple DataFrames.
    - dfs: {"table_name": DataFrame}
    - table_options: per-table options like required_columns, type_map, pii_columns, if_exists, create_indexes, upsert_keys
    """
    engine = create_sqlite_engine(sqlite_path)
    table_options = table_options or {}

    for table_name, df in dfs.items():
        opts = table_options.get(table_name, {})
        logger.info("Processing table %s", table_name)

        start_ts = time.monotonic()
        rows_in = len(df)

        # 2. Validate required columns
        missing = validate_columns(df, opts.get("required_columns"))
        if missing:
            logger.warning("Skipping table %s due to missing columns: %s", table_name, missing)
            continue

        # Optional schema validation via pandera
        if opts.get("schema"):
            try:
                df = validate_schema_pandera(df, opts.get("schema"))
            except Exception as e:
                logger.error("Schema validation failed for %s: %s", table_name, e)
                continue

        # 3. Cast types
        df = cast_types(df, opts.get("type_map"))

        # 4. Handle missing values
        df = handle_missing(df, strategy=opts.get("missing_strategy", "keep"), fill_map=opts.get("fill_map"))

        # 5. Convert 0/1 to yes/no if specified
        if "yesno_columns" in opts:
            df = convert_01_to_yesno(df, columns=opts["yesno_columns"])
        
        # 6. Convert 0 to "male" and 1 to "female"
        if "sex_column" in opts:
            df = convert_sex_01_to_label(df, columns=opts["sex_column"]) 
        
        # 7. Convert stress level codes to labels
        if "stress_level_codes" in opts:
            df = convert_stress_level_codes(df, columns=opts["stress_level_codes"])

        # 8. Feature engineering
        if opts.get("feature_engineering"):
            fe_list = opts.get("feature_engineering_list")
            df = feature_engineering(df, feature_engineering_list=fe_list, sex_col=opts.get("sex_column"))

        # 5. De-identify
        df = deidentify_dataframe(df, opts.get("pii_columns"))

        # 6. Optional sampling metadata
        samples = sample_and_sanitize(df, n=10)
        logger.info("Sample values for %s: %s", table_name, {k: samples[k] for k in list(samples)[:3]})

        # 1. Normalize names (support per-table rename_map and lowercase option)
        df = normalize_column_names(df, rename_map=opts.get("column_rename_list"), lower=opts.get("lowercase_columns", True))


        print(df.head())
        print(df.info())
        print(df.isnull().sum())

        # Add ingestion timestamp for auditing
        df['_ingestion_time'] = now_iso()

        # 7. Write or upsert (dedupe for idempotency if upsert_keys provided)
        if opts.get("upsert_keys"):
            key_cols = opts["upsert_keys"]
            # drop duplicates keeping the last occurrence for idempotency
            if all(k in df.columns for k in key_cols):
                before = len(df)
                df = df.drop_duplicates(subset=key_cols, keep='last')
                logger.info("Deduplicated %s rows -> %s rows based on keys %s", before, len(df), key_cols)
            else:
                logger.warning("Upsert keys %s not found in dataframe columns; proceeding without deduplication", key_cols)

            upsert_sqlite(engine, df, table_name, key_columns=key_cols, chunk_size=opts.get("chunk_size", 500))
        else:
            write_dataframe(engine, df, table_name, if_exists=opts.get("if_exists", "replace"), chunk_size=opts.get("chunk_size", 500), create_indexes=opts.get("create_indexes"))

        elapsed = time.monotonic() - start_ts
        rows_out = len(df)
        logger.info("Processed table %s: rows_in=%d rows_out=%d time_s=%.2f", table_name, rows_in, rows_out, elapsed)

    logger.info("All tables processed and stored.")