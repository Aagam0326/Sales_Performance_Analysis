"""Loading, validation, and cleaning utilities for Superstore data."""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

DATE_COLUMNS = ["order_date", "ship_date"]
NUMERIC_COLUMNS = ["sales", "quantity", "discount", "profit", "postal_code"]


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert source headers to lower-case snake_case without mutating input."""
    result = frame.copy()
    result.columns = [re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", c.lower())).strip("_")
                      for c in result.columns]
    return result


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle CSV while accommodating common Windows encodings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place the Kaggle CSV there.")
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error  # pragma: no cover


def clean_superstore(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Clean Superstore records and return a frame plus transparent audit data.

    Exact duplicate rows are removed. Monetary/discount values are retained even
    when extreme because they are legitimate commercial observations; the audit
    records IQR flags rather than silently deleting business data.
    """
    raw = standardize_columns(load_raw_data(path))
    audit = {"rows_raw": len(raw), "duplicates_removed": int(raw.duplicated().sum())}
    frame = raw.drop_duplicates().copy()
    for column in DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["postal_code"] = frame.get("postal_code", pd.Series(index=frame.index)).astype("Int64")
    required = ["order_id", "order_date", "customer_id", "sales", "profit"]
    missing_required = frame[required].isna().any(axis=1)
    audit["rows_dropped_missing_required"] = int(missing_required.sum())
    frame = frame.loc[~missing_required].copy()
    frame["order_month"] = frame["order_date"].dt.to_period("M").dt.to_timestamp()
    frame["order_quarter"] = frame["order_date"].dt.to_period("Q").astype(str)
    frame["profit_margin"] = frame["profit"].div(frame["sales"].replace(0, pd.NA))
    audit["nulls_after_cleaning"] = frame.isna().sum().to_dict()
    audit["outlier_flags_iqr"] = {}
    for column in ("sales", "profit"):
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        audit["outlier_flags_iqr"][column] = int(((frame[column] < q1 - 1.5 * iqr) |
                                                    (frame[column] > q3 + 1.5 * iqr)).sum())
    audit["rows_clean"] = len(frame)
    return frame, audit


def save_cleaned_data(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Persist derived data only; raw source files are never modified."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
