"""I/O utility helpers shared across projects.

These functions focus on safe reading/writing of large flat files
and standardising how chunked CSV ingestion is handled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd


def read_csv_chunked(
    path: Path | str,
    usecols: Optional[Iterable[int | str]] = None,
    chunksize: int = 2_000_000,
    low_memory: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """Read a large CSV in chunks and concatenate to a DataFrame.

    Parameters
    ----------
    path : Path | str
        Path to the CSV file.
    usecols : Optional[Iterable[int | str]]
        Columns to read (passed to ``pandas.read_csv``).
    chunksize : int
        Number of rows per chunk. Defaults to 2 million.
    low_memory : bool
        Forwarded to ``pandas.read_csv``. Defaults to False to avoid dtype
        guessing issues.
    **read_kwargs
        Additional keyword arguments forwarded to ``pandas.read_csv``.
    """
    chunks: Iterator[pd.DataFrame] = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=low_memory,
        **read_kwargs,
    )
    return pd.concat(chunks, ignore_index=True)


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to parquet using an atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def read_excel_cols(
    path: Path | str,
    sheet_name: str,
    usecols: Optional[str | Iterable[int]] = None,
    header: int = 0,
    **read_kwargs,
) -> pd.DataFrame:
    """Read selected columns from an Excel workbook."""
    return pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols=usecols,
        header=header,
        **read_kwargs,
    )
