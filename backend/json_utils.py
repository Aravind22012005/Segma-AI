"""Recursively convert numpy/pandas types into plain JSON-serialisable Python types."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    return obj
