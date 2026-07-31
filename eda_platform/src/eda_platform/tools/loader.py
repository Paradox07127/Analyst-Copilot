from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eda_platform.core.ids import hash_file
from eda_platform.schemas.datasets import DatasetRecord

# Ordered by specificity. ``gb18030`` is a superset of GBK/GB2312 (common for
# Chinese CSVs); ``latin-1`` never fails to decode and is the final fallback.
_ENCODING_CANDIDATES = (
    "utf-8-sig",
    "utf-8",
    "utf-32",
    "utf-16",
    "gb18030",
    "latin-1",
)
_SAMPLE_BYTES = 65_536
# Measured on a 208 MiB / 2.6M-row CSV: at 10k rows the peak stays flat as the
# file grows (~95 MiB), at 50k it tracks file size (150 MiB at 55 MiB, 300 MiB at
# 208 MiB) because pandas' chunked C parser keeps growing. Wall time is unchanged.
CSV_CHUNK_ROWS = 10_000
_IDENTIFIER_NAME_TOKENS = frozenset(
    {
        "uuid",
        "guid",
        "code",
        "zip",
        "zipcode",
        "postal",
        "phone",
        "mobile",
        "ssn",
        "passport",
        "account",
        "sku",
    }
)
_LEADING_ZERO_INTEGER = re.compile(r"^[+-]?0\d+$")


@dataclass(frozen=True)
class LoadedDataset:
    record: DatasetRecord
    frame: pd.DataFrame


def sniff_encoding(sample: bytes) -> str:
    for encoding in _ENCODING_CANDIDATES:
        try:
            sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding
    return "latin-1"


def sniff_delimiter(text_sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_csv(
    path: Path | str,
    *,
    dataset_id: str | None = None,
    content_hash: str | None = None,
    cancel_check: Callable[[], object] | None = None,
) -> LoadedDataset:
    source = Path(path)
    encoding, delimiter = _sniff(source)
    actual_content_hash = content_hash or hash_file(
        source,
        cancel_check=cancel_check,
    )

    if cancel_check is None:
        frame = _read_frame(source, encoding=encoding, delimiter=delimiter)
    else:
        def collect(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
            collected: list[pd.DataFrame] = []
            for chunk in chunks:
                cancel_check()
                collected.append(chunk)
            return (
                pd.concat(collected, ignore_index=True)
                if collected
                else pd.DataFrame()
            )

        frame = stream_csv_chunks(source, collect)

    record = DatasetRecord(
        dataset_id=dataset_id or f"ds_{actual_content_hash}",
        name=source.name,
        path=source,
        content_hash=actual_content_hash,
        encoding=encoding,
        delimiter=delimiter,
    )
    return LoadedDataset(record=record, frame=frame)


def read_csv_columns(path: Path | str) -> list[str]:
    """Column names from the header alone, so a caller can validate a column
    selection before committing to a read."""
    source = Path(path)
    encoding, delimiter = _sniff(source)
    try:
        header = pd.read_csv(source, encoding=encoding, sep=delimiter, nrows=0)
    except (UnicodeDecodeError, pd.errors.ParserError):
        with source.open("r", encoding=encoding, errors="replace", newline="") as handle:
            header = pd.read_csv(handle, sep=None, engine="python", nrows=0)
    return [str(column) for column in header.columns]


def stream_csv_chunks[T](
    path: Path | str,
    consume: Callable[[Iterator[pd.DataFrame]], T],
    *,
    usecols: Sequence[str] | None = None,
    chunksize: int | None = None,
) -> T:
    """Hand ``consume`` row chunks instead of one whole DataFrame, so a caller can
    shape an arbitrarily large CSV in bounded memory.

    ``consume`` must build its own state and may be called twice: a parser failure
    restarts it against the lenient reader, matching ``_read_frame``'s fallback.
    """
    source = Path(path)
    encoding, delimiter = _sniff(source)
    rows = chunksize or CSV_CHUNK_ROWS
    columns = None if usecols is None else list(usecols)
    try:
        return consume(
            _chunks(source, encoding=encoding, delimiter=delimiter, usecols=columns, rows=rows)
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        return consume(_lenient_chunks(source, encoding=encoding, usecols=columns, rows=rows))


def _sniff(source: Path) -> tuple[str, str]:
    # Read only the sniffing window. ``Path.read_bytes()[:N]`` first allocates
    # the complete file, which makes large CSVs pay an avoidable full-file read
    # and a matching transient memory spike before pandas even starts.
    with source.open("rb") as handle:
        sample = handle.read(_SAMPLE_BYTES)
    encoding = sniff_encoding(sample)
    return encoding, sniff_delimiter(sample.decode(encoding, errors="replace"))


def _chunks(
    source: Path, *, encoding: str, delimiter: str, usecols: list[str] | None, rows: int
) -> Iterator[pd.DataFrame]:
    # Closed explicitly: stream_csv_chunks abandons this reader mid-iteration when
    # it falls back, and a consumer may stop early.
    with pd.read_csv(
        source,
        encoding=encoding,
        sep=delimiter,
        usecols=usecols,
        chunksize=rows,
        dtype=cast(
            Any,
            _lexical_string_dtypes(source, encoding=encoding, delimiter=delimiter),
        ),
        dtype_backend="numpy_nullable",
    ) as reader:
        yield from reader


def _lenient_chunks(
    source: Path, *, encoding: str, usecols: list[str] | None, rows: int
) -> Iterator[pd.DataFrame]:
    with (
        source.open("r", encoding=encoding, errors="replace", newline="") as handle,
        pd.read_csv(
            handle,
            sep=None,
            engine="python",
            usecols=usecols,
            chunksize=rows,
            dtype_backend="numpy_nullable",
        ) as reader,
    ):
        yield from reader


def _read_frame(source: Path, *, encoding: str, delimiter: str) -> pd.DataFrame:
    try:
        return pd.read_csv(
            source,
            encoding=encoding,
            sep=delimiter,
            dtype=cast(
                Any,
                _lexical_string_dtypes(source, encoding=encoding, delimiter=delimiter),
            ),
            dtype_backend="numpy_nullable",
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        # Last resort: decode leniently and let pandas infer the separator,
        # without materialising a second complete copy of the source bytes.
        with source.open("r", encoding=encoding, errors="replace", newline="") as handle:
            return pd.read_csv(handle, sep=None, engine="python", dtype_backend="numpy_nullable")


def _lexical_string_dtypes(
    source: Path,
    *,
    encoding: str,
    delimiter: str,
) -> dict[str, str] | None:
    """Preserve lexically meaningful identifiers and leading-zero integers.

    A generic ``*_id`` is deliberately not enough to force string dtype:
    numeric surrogate keys are commonly used as sequence fields and existing
    consumers rely on that numeric role. Values with leading zeroes are still
    preserved regardless of the column name.
    """
    sample_values: dict[str, list[str]] = {}
    try:
        with source.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader)
            sample_values = {str(column): [] for column in header}
            for row_number, row in enumerate(reader):
                if row_number >= 1_000:
                    break
                for index, column in enumerate(header):
                    if index < len(row):
                        sample_values[str(column)].append(row[index])
    except (UnicodeDecodeError, OSError, csv.Error, StopIteration):
        return None
    string_columns: dict[str, str] = {}
    for column, raw_values in sample_values.items():
        tokens = {
            token
            for token in re.split(r"[^0-9a-z]+", column.strip().lower())
            if token
        }
        preserves_lexeme = bool(tokens & _IDENTIFIER_NAME_TOKENS) or bool(
            any(
                _LEADING_ZERO_INTEGER.fullmatch(value.strip())
                for value in raw_values
            )
        )
        if preserves_lexeme:
            string_columns[column] = "string"
    return string_columns or None
