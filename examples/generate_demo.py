#!/usr/bin/env python3
"""Generate deterministic synthetic longitudinal classification data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    generator = np.random.default_rng(arguments.seed)
    rows = []
    for year_offset, year in enumerate(range(2014, 2026)):
        for sample in range(40):
            site = "north" if sample % 3 else "south"
            instrument_shift = year_offset * 0.08
            latent = generator.normal(instrument_shift, 1.0) + (0.35 if site == "south" else 0)
            probability = 1.0 / (1.0 + np.exp(-latent))
            outcome = int(generator.random() < probability)
            rows.append(
                {
                    "sample_date": f"{year}-{sample % 12 + 1:02d}-15",
                    "outcome": outcome,
                    "signal": latent + generator.normal(0, 0.3),
                    "noise": generator.normal(0, 1),
                    "site": site,
                    "entity_id": f"sample-{year}-{sample:03d}",
                }
            )
    frame = pd.DataFrame(rows)
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.destination, index=False, lineterminator="\n")
    print(arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
