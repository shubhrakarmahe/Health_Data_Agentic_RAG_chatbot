# db_store.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import Integer, Float, Text, DateTime, Boolean
from typing import Dict, Any, List, Optional
import pandas as pd
from utils import simple_logger
logger = simple_logger()

def create_sqlite_engine(path: str, enable_wal: bool = True) -> Engine:
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    if enable_wal:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.execute(text("PRAGMA synchronous=NORMAL;"))
            conn.execute(text("PRAGMA temp_store=MEMORY;"))
    return engine

def pandas_to_sqlalchemy_types(df: pd.DataFrame) -> Dict[str, Any]:
    mapping = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            mapping[col] = Integer()
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = Float()
        elif pd.api.types.is_bool_dtype(dtype):
            mapping[col] = Boolean()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = DateTime()
        else:
            mapping[col] = Text()
    return mapping

def write_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append",
    chunk_size: int = 500,
    create_indexes: Optional[List[str]] = None
) -> None:
    """
    Write DataFrame to SQLite with batching and index creation.
    """
    dtype_map = pandas_to_sqlalchemy_types(df)
    with engine.begin() as conn:
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False, chunksize=chunk_size, dtype=dtype_map, method="multi")
        if create_indexes:
            for col in create_indexes:
                safe_col = col.replace('"', '""')
                idx_name = f"idx_{table_name}_{safe_col}"
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{safe_col}");'))
    logger.info("Wrote table %s rows=%d", table_name, len(df))

def upsert_sqlite(
    engine: Engine,
    df: pd.DataFrame,
    table_name: str,
    key_columns: List[str],
    chunk_size: int = 500
) -> None:
    """
    Emulate upsert in SQLite:
    1. write df to a temp table
    2. run INSERT OR REPLACE into target table using a primary key or unique index
    Note: target table must have a UNIQUE constraint on key_columns for correct behavior.
    """
    temp_table = f"{table_name}_tmp"
    dtype_map = pandas_to_sqlalchemy_types(df)
    with engine.begin() as conn:
        # write temp
        df.to_sql(name=temp_table, con=conn, if_exists="replace", index=False, chunksize=chunk_size, dtype=dtype_map, method="multi")
        # build column lists
        cols = list(df.columns)
        col_list = ", ".join([f'"{c}"' for c in cols])
        select_list = ", ".join([f'"{c}"' for c in cols])
        # SQLite upsert via INSERT OR REPLACE requires the target table to have a primary key or unique index
        insert_sql = f'INSERT OR REPLACE INTO "{table_name}" ({col_list}) SELECT {select_list} FROM "{temp_table}";'
        conn.execute(text(insert_sql))
        conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table}";'))
    logger.info("Upserted into %s rows=%d", table_name, len(df))
