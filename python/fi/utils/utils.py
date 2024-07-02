

import pandas as pd

from fi.utils.constants import MAX_FUTURE_YEARS_FROM_CURRENT_TIME, MAX_PAST_YEARS_FROM_CURRENT_TIME

def is_timestamp_in_range(now: int, ts: int):
    max_time = now + (MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    min_time = now - (MAX_PAST_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    return min_time <= ts <= max_time

def convert_element(value):
    """converts scalar or array to python native"""
    val = getattr(value, "tolist", lambda: value)()
    # Check if it's a list since elements from pd indices are converted to a scalar
    # whereas pd series/dataframe elements are converted to list of 1 with the native value
    if isinstance(val, list):
        val = val[0] if val else None
    if pd.isna(val):
        return None
    return val

