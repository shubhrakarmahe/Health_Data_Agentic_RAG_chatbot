# preprocess.py
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import re
from utils import mask_patient_number, simple_logger
logger = simple_logger()

NUMERIC_FILL = {"int": 0, "float": 0.0}

def validate_columns(df: pd.DataFrame, required: Optional[List[str]] = None) -> List[str]:
    """
    Ensure required columns exist. Return missing list.
    """
    required = required or []
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
    return missing

def normalize_column_names(df: pd.DataFrame, rename_map: Optional[Dict[str, str]] = None, lower: bool = True) -> pd.DataFrame:
    """
    Normalize column names to snake_case and strip whitespace.

    Additional behavior:
    - If `rename_map` is provided it will be applied after normalization. Keys in
      `rename_map` are matched case-insensitively against normalized column names.
    - Non-alphanumeric characters (except underscore) are removed and consecutive
      underscores are collapsed.

    Args:
        df: Input dataframe.
        rename_map: Optional mapping from existing column name -> new name.
        lower: Whether to lowercase column names during normalization (default: True).

    Returns:
        A dataframe with normalized (and optionally renamed) column names.
    """
    df = df.copy()

    def _normalize_name(name: str) -> str:
        n = str(name).strip()
        if lower:
            n = n.lower()
        # replace spaces/hyphens with underscore
        n = re.sub(r"[ \-]+", "_", n)
        # remove non-alphanumeric/underscore characters
        n = re.sub(r"[^0-9a-zA-Z_]", "", n)
        # collapse multiple underscores
        n = re.sub(r"__+", "_", n)
        return n

    normalized = [_normalize_name(c) for c in df.columns]
    df.columns = normalized

    if rename_map:
        # Normalize the rename_map keys so match is done on normalized names
        norm_map = {}
        for k, v in rename_map.items():
            kn = _normalize_name(k)
            norm_map[kn] = v
        # Build dict for rename where the normalized column exists in norm_map
        to_rename = {col: norm_map[col] for col in df.columns if col in norm_map}
        if to_rename:
            df = df.rename(columns=to_rename)

    return df

def cast_types(df: pd.DataFrame, type_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Cast columns according to type_map: {"col": "int"|"float"|"str"|"datetime"}.
    Non-castable values become NaN.
    """
    df = df.copy()
    type_map = type_map or {}
    for col, t in type_map.items():
        if col not in df.columns:
            continue
        try:
            if t == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif t == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif t == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(str)
        except Exception as e:
            logger.warning("Failed to cast %s to %s: %s", col, t, e)
    return df

def handle_missing(df: pd.DataFrame, strategy: str = "drop", fill_map: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Missing value strategies:
    - drop: drop rows with any nulls
    - fill: fill with fill_map or defaults
    - keep: leave as-is
    """
    df = df.copy()
    if strategy == "drop":
        return df.dropna()
    if strategy == "fill":
        fill_map = fill_map or {}
        for col in df.columns:
            if col in fill_map:
                df[col] = df[col].fillna(fill_map[col])
            else:
                # default numeric fill
                if pd.api.types.is_integer_dtype(df[col].dtype):
                    df[col] = df[col].fillna(NUMERIC_FILL["int"])
                elif pd.api.types.is_float_dtype(df[col].dtype):
                    df[col] = df[col].fillna(NUMERIC_FILL["float"])
                else:
                    df[col] = df[col].fillna("UNKNOWN")
        return df
    return df

def deidentify_dataframe(df: pd.DataFrame, pii_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Mask PII columns in-place. Heuristic detection if pii_columns not provided.
    """
    df = df.copy()
    if pii_columns is None:
        pii_columns = [c for c in df.columns if any(k in c.lower() for k in ["patient", "id", "name", "email", "phone", "ssn"])]
    for col in pii_columns:
        if col not in df.columns:
            continue
        # apply masking heuristics
        df[col] = df[col].apply(mask_patient_number)
    return df

def sample_and_sanitize(df: pd.DataFrame, n: int = 10) -> Dict[str, str]:
    """
    Return sanitized sample values per column for metadata.
    """
    samples = {}
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique()[:n]
        # mask long strings and obvious PII tokens
        vals = [v if len(v) <= 100 else v[:100] + "..." for v in vals]
        samples[col] = ", ".join(vals)
    return samples


def convert_01_to_yesno(df: pd.DataFrame, columns: List[str], inplace: bool = False) -> pd.DataFrame:
    """
    Convert 0/1 values to 'no'/'yes' for the specified list of `columns`.

    Behavior:
    - Accepts numeric 0/1 and string '0'/'1' (and other convertible numeric strings).
    - Columns that are missing will be skipped and a warning logged.
    - Other values are left unchanged.

    Args:
        df: Input dataframe.
        columns: List of column names to convert.
        inplace: If True, modify `df` in-place and return it; otherwise operate on a copy.

    Returns:
        The dataframe with converted columns.
    """
    if not inplace:
        df = df.copy()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("convert_01_to_yesno: missing columns skipped: %s", missing)

    for col in columns:
        if col not in df.columns:
            continue
        # Work with a copy of the series to avoid SettingWithCopyWarning
        s = df[col].copy()
        # Try to convert values to numeric to make matching robust
        s_num = pd.to_numeric(s, errors='coerce')

        # Masks for exact 0 and 1 (numeric)
        mask0 = s_num == 0
        mask1 = s_num == 1

        # Assign 'no'/'yes' where masks are true
        if mask0.any():
            s.loc[mask0] = 'no'
        if mask1.any():
            s.loc[mask1] = 'yes'

        df[col] = s

    return df


def convert_sex_01_to_label(df: pd.DataFrame, columns: List[str], inplace: bool = False, zero_label: str = 'male', one_label: str = 'women') -> pd.DataFrame:
    """
    Convert sex encoded as 0/1 to labels for the specified `columns`.

    Behavior:
    - 0 -> 'male' (or `zero_label`), 1 -> 'women' (or `one_label`).
    - Accepts numeric 0/1 and string '0'/'1' (and other convertible numeric strings).
    - Missing columns are skipped and a warning logged.
    - Other values are left unchanged.

    Args:
        df: Input dataframe.
        columns: List of column names to convert.
        inplace: If True, modify `df` in-place; otherwise operate on a copy.
        zero_label: Label to use for 0.
        one_label: Label to use for 1.

    Returns:
        The dataframe with converted columns.
    """
    if not inplace:
        df = df.copy()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("convert_sex_01_to_label: missing columns skipped: %s", missing)

    for col in columns:
        if col not in df.columns:
            continue

        s = df[col].copy()
        s_num = pd.to_numeric(s, errors='coerce')

        mask0 = s_num == 0
        mask1 = s_num == 1

        if mask0.any():
            s.loc[mask0] = zero_label
        if mask1.any():
            s.loc[mask1] = one_label

        df[col] = s

    return df


def convert_stress_level_codes(df: pd.DataFrame, columns: List[str], inplace: bool = False, mapping: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """
    Convert stress level codes to labels for specified `columns`.

    Default mapping: {1: 'low', 2: 'normal', 3: 'high'}.

    Behavior:
    - Accepts numeric codes and numeric-string codes (e.g., '1'), coerces to integers.
    - Missing columns are skipped and a warning is logged.
    - Other values are left unchanged.

    Args:
        df: Input dataframe.
        columns: List of column names to convert.
        inplace: If True, modify `df` in-place; otherwise operate on a copy.
        mapping: Optional mapping of int->label to override defaults.

    Returns:
        The dataframe with converted columns.
    """
    default_mapping = {1: 'low', 2: 'normal', 3: 'high'}
    mapping = mapping or default_mapping

    if not inplace:
        df = df.copy()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("convert_stress_level_codes: missing columns skipped: %s", missing)

    for col in columns:
        if col not in df.columns:
            continue

        s = df[col].copy()
        # Coerce to numeric so we can match integer codes reliably
        s_num = pd.to_numeric(s, errors='coerce')

        for code, label in mapping.items():
            try:
                mask = s_num == int(code)
            except Exception:
                # if code cannot be converted to int, skip
                continue
            if mask.any():
                s.loc[mask] = label

        df[col] = s

    return df


# --- Feature engineering helpers ------------------------------------------------

def bmi_to_category(df: pd.DataFrame, bmi_col: str = 'BMI', new_col: str = 'BMI_Category',
                    bins: Optional[list] = None, labels: Optional[list] = None, inplace: bool = False) -> pd.DataFrame:
    """
    Convert BMI numeric values into categorical bins.
    """
    if not inplace:
        df = df.copy()

    bins = bins or [0, 18.5, 25, 30, 35, float('inf')]
    labels = labels or ['Underweight', 'Normal', 'Overweight', 'Obese_class_I', 'Obese_class_II+']

    if bmi_col not in df.columns:
        logger.warning("bmi_to_category: column '%s' not found", bmi_col)
        return df

    df[new_col] = pd.cut(pd.to_numeric(df[bmi_col], errors='coerce'), bins=bins, labels=labels, include_lowest=True)
    return df


def age_to_group(df: pd.DataFrame, age_col: str = 'Age', new_col: str = 'Age_Group',
                 bins: Optional[list] = None, labels: Optional[list] = None, inplace: bool = False) -> pd.DataFrame:
    """
    Bin age into groups.
    """
    if not inplace:
        df = df.copy()

    bins = bins or [0, 17, 29, 39, 49, 59, 69, 120]
    labels = labels or ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']

    if age_col not in df.columns:
        logger.warning("age_to_group: column '%s' not found", age_col)
        return df

    df[new_col] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'), bins=bins, labels=labels, include_lowest=True)
    return df


def hemoglobin_category(df: pd.DataFrame, hb_col: str = 'Level_of_Hemoglobin', sex_col: str = 'Sex',
                        new_col: str = 'Hemoglobin_Category',
                        male_normal: tuple = (14, 18), female_normal: tuple = (12, 16), inplace: bool = False) -> pd.DataFrame:
    """
    Create hemoglobin category ('low','normal','high') based on sex-specific ranges.

    Ranges are inclusive for 'normal'. Other values that cannot be interpreted are left unchanged.
    """
    if not inplace:
        df = df.copy()

    if hb_col not in df.columns or sex_col not in df.columns:
        logger.warning("hemoglobin_category: required columns not found: %s, %s", hb_col, sex_col)
        return df

    hb = pd.to_numeric(df[hb_col], errors='coerce')
    s = df[sex_col].astype(str).str.strip().str.lower()
    s_num = pd.to_numeric(df[sex_col], errors='coerce')

    # male mask: numeric 0 or string like 'male','m'
    male_mask = (s.isin(['male', 'm'])) | (s_num == 0)
    female_mask = (s.isin(['female', 'f'])) | (s_num == 1)

    # Initialize column if not present
    if new_col not in df.columns:
        df[new_col] = pd.NA

    # Male normal
    mn_low, mn_high = male_normal
    m_normal_mask = male_mask & hb.between(mn_low, mn_high)
    df.loc[m_normal_mask, new_col] = 'normal'
    m_low_mask = male_mask & hb < mn_low
    df.loc[m_low_mask, new_col] = 'low'
    m_high_mask = male_mask & hb > mn_high
    df.loc[m_high_mask, new_col] = 'high'

    # Female normal
    fn_low, fn_high = female_normal
    f_normal_mask = female_mask & hb.between(fn_low, fn_high)
    df.loc[f_normal_mask, new_col] = 'normal'
    f_low_mask = female_mask & hb < fn_low
    df.loc[f_low_mask, new_col] = 'low'
    f_high_mask = female_mask & hb > fn_high
    df.loc[f_high_mask, new_col] = 'high'

    return df


def add_gpc_available(df: pd.DataFrame, gpc_col: str = 'Genetic_Pedigree_Coefficient', new_col: str = 'GPC_available', inplace: bool = False) -> pd.DataFrame:
    """
    Create 'GPC_available' indicating whether Genetic_Pedigree_Coefficient is present.
    """
    if not inplace:
        df = df.copy()

    if gpc_col not in df.columns:
        logger.warning("add_gpc_available: column '%s' not found", gpc_col)
        return df

    df[new_col] = df[gpc_col].notna().map({True: 'yes', False: 'no'})
    return df


def feature_engineering(df: pd.DataFrame, inplace: bool = False,
                        bmi_col: Any = 'BMI', age_col: Any = 'Age', hb_col: Any = 'Level_of_Hemoglobin',
                        sex_col: Any = 'Sex', gpc_col: Any = 'Genetic_Pedigree_Coefficient',
                        feature_engineering_list: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
    """
    Apply a sequence of feature engineering transformations to the dataframe.

    This function accepts either explicit column arguments (single name or list) or a
    `feature_engineering_list` which is a list of dicts describing the features to
    create. Each dict may contain:
      - 'column' (required): source column name
      - 'output' (optional): destination column name
      - 'transform' (optional): one of 'bmi','age','hemoglobin','gpc' to override

    When `feature_engineering_list` is provided it takes precedence over the
    individual column arguments.
    """
    if not inplace:
        df = df.copy()

    def _ensure_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # If a structured feature list is provided, use it.
    if feature_engineering_list:
        for entry in feature_engineering_list:
            if not isinstance(entry, dict) or 'column' not in entry:
                logger.warning("feature_engineering: invalid entry skipped: %s", entry)
                continue
            col = entry['column']
            out = entry.get('output')
            transform = (entry.get('transform') or '').lower()

            # Determine transform type if not provided
            if not transform:
                cn = str(col).lower()
                if 'bmi' in cn:
                    transform = 'bmi'
                elif 'age' in cn:
                    transform = 'age'
                elif 'hemoglobin' in cn or 'hb' in cn:
                    transform = 'hemoglobin'
                elif 'genetic' in cn or 'gpc' in cn:
                    transform = 'gpc'
                else:
                    logger.warning("feature_engineering: cannot infer transform for column '%s', skipping", col)
                    continue

            # Call the appropriate transform
            if transform == 'bmi':
                new_col = out or (f"{col}_Category")
                df = bmi_to_category(df, bmi_col=col, new_col=new_col, inplace=True)
            elif transform == 'age':
                new_col = out or (f"{col}_Group")
                df = age_to_group(df, age_col=col, new_col=new_col, inplace=True)
            elif transform == 'hemoglobin':
                new_col = out or (f"{col}_Category")
                sex_for = entry.get('sex_column') or sex_col
                # If a list of sex columns was provided, pick the first one
                if isinstance(sex_for, (list, tuple)):
                    if len(sex_for) == 0:
                        logger.warning("feature_engineering: empty sex_column list for hemoglobin; using default 'Sex'")
                        sex_for = 'Sex'
                    else:
                        sex_for = sex_for[0]
                df = hemoglobin_category(df, hb_col=col, sex_col=sex_for, new_col=new_col, inplace=True)
            elif transform == 'gpc':
                new_col = out or (f"{col}_available")
                df = add_gpc_available(df, gpc_col=col, new_col=new_col, inplace=True)
            else:
                logger.warning("feature_engineering: unknown transform '%s' for column '%s'", transform, col)

        return df

    # Fallback: older API using column args (can be str or list)
    bmi_cols = _ensure_list(bmi_col)
    age_cols = _ensure_list(age_col)
    hb_cols = _ensure_list(hb_col)
    sex_cols = _ensure_list(sex_col)
    gpc_cols = _ensure_list(gpc_col)

    for col in bmi_cols:
        new_col = f"{col}_Category" if col != 'BMI' else 'BMI_Category'
        df = bmi_to_category(df, bmi_col=col, new_col=new_col, inplace=True)

    for col in age_cols:
        new_col = f"{col}_Group" if col != 'Age' else 'Age_Group'
        df = age_to_group(df, age_col=col, new_col=new_col, inplace=True)

    if hb_cols:
        if not sex_cols:
            logger.warning("feature_engineering: no sex column provided; hemoglobin categorization may be inaccurate")
        for i, hb in enumerate(hb_cols):
            if sex_cols:
                sex_for_this = sex_cols[i] if i < len(sex_cols) else sex_cols[-1]
            else:
                sex_for_this = 'Sex'
            new_col = f"{hb}_Category" if hb != 'Level_of_Hemoglobin' else 'Hemoglobin_Category'
            df = hemoglobin_category(df, hb_col=hb, sex_col=sex_for_this, new_col=new_col, inplace=True)

    for col in gpc_cols:
        new_col = f"{col}_available" if col != 'Genetic_Pedigree_Coefficient' else 'GPC_available'
        df = add_gpc_available(df, gpc_col=col, new_col=new_col, inplace=True)

    return df

# -----------------------------------------------------------------------------
def aggregate_physical_activity(df: pd.DataFrame, patient_col: str = 'Patient_Number', activity_col: str = 'Physical_activity') -> pd.DataFrame:
    """
    Aggregate physical activity metrics by patient.

    Returns a dataframe with columns:
      - No_of_days_physical_activity: count of rows where activity != 0 and not NA
      - Total_physical_activity: sum of activity (zeros included in sum)
      - Max_physical_activity: max (ignores NaN)
      - Min_physical_activity: min (ignores NaN)
      - Mean_physical_activity: mean (ignores NaN)
      - No_of_days_missed_physical_activity: count of rows where activity is 0 or NaN

    """
    if patient_col not in df.columns:
        raise KeyError(f"patient_col '{patient_col}' not found in dataframe")
    if activity_col not in df.columns:
        raise KeyError(f"activity_col '{activity_col}' not found in dataframe")

    tmp = df[[patient_col, activity_col]].copy()
    tmp[activity_col] = pd.to_numeric(tmp[activity_col], errors='coerce')

    def non_zero_count(series: pd.Series) -> int:
        # Count True values of the boolean mask and return as int
        return int((series.notna() & (series != 0)).sum())

    def missed_count(series: pd.Series) -> int:
        # Count days where activity is missing or zero
        return int((series.isna() | (series == 0)).sum())

    agg = tmp.groupby(patient_col).agg(
        No_of_days_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=non_zero_count),
        Total_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=lambda s: float(s.fillna(0).sum())),
        Max_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=lambda s: float(s.max()) if s.notna().any() else float('nan')),
        Min_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=lambda s: float(s.min()) if s.notna().any() else float('nan')),
        Mean_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=lambda s: float(s.mean()) if s.notna().any() else float('nan')),
        No_of_days_missed_physical_activity=pd.NamedAgg(column=activity_col, aggfunc=missed_count),
    ).reset_index()

    return agg


def validate_schema_pandera(df: pd.DataFrame, schema_def: Any) -> pd.DataFrame:
    """
    Validate DataFrame using a pandera schema or a simple dict definition.

    schema_def may be:
      - a pandera.DataFrameSchema object
      - a dict mapping column -> dtype string (e.g., {'Age': 'int', 'BMI':'float'})

    Returns the validated DataFrame (or raises pandera.errors.SchemaError).
    """
    try:
        import pandera as pa
        from pandera import Column
    except Exception as e:
        raise ImportError("Please install pandera to use schema validation (pip install pandera)") from e

    if schema_def is None:
        return df

    if hasattr(schema_def, 'validate'):
        # Assume it's already a pandera DataFrameSchema
        return schema_def.validate(df)

    # Build a simple schema from dict
    if isinstance(schema_def, dict):
        cols = {}
        for col, dtype in schema_def.items():
            # map simple dtype strings to pandas dtypes
            dtype_str = str(dtype).lower()
            if dtype_str in ('int', 'integer'):
                cols[col] = Column(pa.Int, nullable=True)
            elif dtype_str in ('float', 'double'):
                cols[col] = Column(pa.Float, nullable=True)
            elif dtype_str in ('str', 'string', 'object'):
                cols[col] = Column(pa.String, nullable=True)
            elif dtype_str in ('bool', 'boolean'):
                cols[col] = Column(pa.Bool, nullable=True)
            else:
                cols[col] = Column(pa.Object, nullable=True)

        schema = pa.DataFrameSchema(cols)
        return schema.validate(df)

    raise ValueError("schema_def must be a pandera DataFrameSchema or a dict mapping columns to types")
