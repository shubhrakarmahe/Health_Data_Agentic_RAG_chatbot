# utils.py
import hashlib
import logging
import datetime
from typing import Any, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()

def deterministic_id(namespace: str, table: str, name: str) -> str:
    base = f"{namespace}.{table}.{name}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{base}.{h}"

def mask_patient_number(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val)
    if s.isdigit() and len(s) >= 6:
        return s[:2] + "****" + s[-2:]
    # fallback deterministic hash
    return "ANON_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def simple_logger():
    return logging.getLogger("preprocess_pipeline")