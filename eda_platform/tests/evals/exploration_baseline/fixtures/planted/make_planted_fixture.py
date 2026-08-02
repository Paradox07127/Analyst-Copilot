"""Regenerates planted_retail.csv deterministically (stdlib only, fixed LCG).

Planted signals (see ground_truth.json): revenue uptrend over time, North>South
revenue, satisfaction missing mostly on phone channel, 2025-04-15 revenue spike.
Deliberately absent: any age trend, region difference in units, missing revenue,
units outliers — those back fixtures/negative/absent_patterns.json.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

SEED = 20260801
START = date(2025, 1, 1)
DAYS = 181  # 2025-01-01 .. 2025-06-30
REGIONS = {"North": 12.0, "South": 6.5}
CHANNELS = ("online", "store", "phone")
CATEGORIES = ("Electronics", "Books", "Home")
SPIKE_DAY = date(2025, 4, 15)


def _lcg(seed: int):
    state = seed
    while True:
        state = (state * 1103515245 + 12345) % 2**31
        yield state / 2**31


def build_rows() -> list[dict[str, object]]:
    rand = _lcg(SEED)
    rows: list[dict[str, object]] = []
    row_index = 0
    for day_index in range(DAYS):
        day = START + timedelta(days=day_index)
        for region, price in REGIONS.items():
          for _order in range(3):
            units = 1 + int(next(rand) * 19)  # 1..20, region-independent
            trend = 1 + 0.008 * day_index
            revenue = units * price * trend * (0.9 + 0.2 * next(rand))
            if day == SPIKE_DAY:
                revenue *= 6
            channel = CHANNELS[row_index % len(CHANNELS)]
            missing_p = 0.7 if channel == "phone" else 0.03
            satisfaction = "" if next(rand) < missing_p else 1 + int(next(rand) * 4.999)
            rows.append(
                {
                    "order_date": day.isoformat(),
                    "region": region,
                    "category": CATEGORIES[row_index % len(CATEGORIES)],
                    "channel": channel,
                    "units": units,
                    "revenue": round(revenue, 2),
                    "customer_age": 18 + int(next(rand) * 50),
                    "satisfaction": satisfaction,
                }
            )
            row_index += 1
    return rows


def main() -> None:
    rows = build_rows()
    out = Path(__file__).parent / "planted_retail.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
