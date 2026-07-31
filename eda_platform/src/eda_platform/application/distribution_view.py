"""Per-column mini distribution statistics returned by the API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
import pandas as pd

from eda_platform.tools.frame_stats import distribution_kind, histogram_bin_count

DIST_SAMPLE_CAP = 10_000
DIST_SAMPLE_SEED = 0
DIST_NUMERIC_BINS = 10
DIST_TOP_K = 5


def numeric_distribution(
    series: pd.Series, bins: int = DIST_NUMERIC_BINS
) -> dict[str, Any] | None:
    """Histogram summary of a numeric column; None when it has no finite values."""
    numeric = cast("pd.Series", pd.to_numeric(series, errors="coerce")).dropna().astype(float)
    numeric = cast("pd.Series", numeric[np.isfinite(numeric)])
    if numeric.empty:
        return None
    lo, hi = float(numeric.min()), float(numeric.max())
    if lo == hi:
        return {
            "kind": "numeric",
            "counts": [int(numeric.size)],
            "bin_edges": [lo, hi],
            "min": lo,
            "max": hi,
        }
    counts, edges = np.histogram(numeric.to_numpy(), bins=bins)
    return {
        "kind": "numeric",
        "counts": [int(count) for count in counts],
        "bin_edges": [float(edge) for edge in edges],
        "min": lo,
        "max": hi,
    }


def categorical_distribution(series: pd.Series, top_k: int = DIST_TOP_K) -> dict[str, Any] | None:
    """Top-k value counts plus an "other" bucket; None when the column is all null."""
    values = series.dropna().astype(str)
    if values.empty:
        return None
    counts = values.value_counts()
    lengths = values.str.len()
    return {
        "kind": "categorical",
        "top": [(str(label), int(count)) for label, count in counts.head(top_k).items()],
        "other_count": int(counts.iloc[top_k:].sum()),
        "unique_count": int(counts.size),
        "len_min": int(lengths.min()),
        "len_max": int(lengths.max()),
    }


def column_distributions(
    frame: pd.DataFrame,
    *,
    sample_cap: int = DIST_SAMPLE_CAP,
    bins: int = DIST_NUMERIC_BINS,
    top_k: int = DIST_TOP_K,
) -> list[dict[str, Any]]:
    """Per-column mini distribution stats for the Table preview header strip."""
    if len(frame) > sample_cap:
        frame = frame.sample(n=sample_cap, random_state=0)
    distributions: list[dict[str, Any]] = []
    # Positional access keeps duplicate column labels from collapsing into a frame.
    for position in range(frame.shape[1]):
        series = frame.iloc[:, position]
        dist: dict[str, Any] | None
        # Flags and small integer level sets are categories in a numeric dtype;
        # binning them into ranges renders empty gaps between the real values.
        shape = distribution_kind(series)
        if (
            shape == "continuous"
            and not pd.api.types.is_bool_dtype(series)
            and pd.api.types.is_numeric_dtype(series)
        ):
            dist = numeric_distribution(series, bins=histogram_bin_count(series))
        else:
            dist = categorical_distribution(series, top_k=top_k)
        if dist is None:
            dist = {"kind": "empty"}
        dist["name"] = str(frame.columns[position])
        dist["dtype"] = str(series.dtype)
        dist["missing_percent"] = float(series.isna().mean() * 100) if len(series) else 100.0
        distributions.append(dist)
    return distributions


def reservoir_sample(
    chunks: Iterable[pd.DataFrame],
    *,
    cap: int = DIST_SAMPLE_CAP,
    seed: int = DIST_SAMPLE_SEED,
    cancel_check: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, int]:
    """A uniform random ``cap``-row sample of the concatenated chunks, plus the
    total row count, in O(cap + chunk) memory.

    Keeping the rows with the smallest i.i.d. uniform keys makes every ``cap``-sized
    subset equally likely — the same guarantee ``DataFrame.sample`` gives, over a
    different set of rows.
    """
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    keys = np.empty(0, dtype=np.float64)
    total = 0
    for chunk in chunks:
        if cancel_check is not None:
            cancel_check()
        total += len(chunk)
        chunk_keys = rng.random(len(chunk))
        if reservoir is None:
            reservoir, keys = chunk, chunk_keys
        else:
            reservoir = pd.concat([reservoir, chunk], ignore_index=True)
            keys = np.concatenate([keys, chunk_keys])
        if len(reservoir) > cap:
            # Sorted back into arrival order so the sample still reads like the file.
            kept = np.sort(np.argpartition(keys, cap)[:cap])
            reservoir = reservoir.iloc[kept].reset_index(drop=True)
            keys = keys[kept]
    if reservoir is None:
        return pd.DataFrame(), 0
    return reservoir, total
