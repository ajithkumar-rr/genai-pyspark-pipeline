"""Generate synthetic e-commerce data and save it as local Parquet files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Sequence

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and export datasets, reporting timings and sizes.

    Optional CLI arguments override default counts, seed, and output directory.
    Existing named Parquet files are overwritten. Return zero on success or
    one on generation/export failure, logging the traceback for debugging.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    args = parser.parse_args(argv)
    config.configure_logging()
    started = perf_counter()

    try:
        # Check the Parquet engine before spending time generating records.
        import pyarrow  # noqa: F401

        args.output_dir.mkdir(parents=True, exist_ok=True)
        generator = SyntheticDataGenerator(
            num_customers=args.customers,
            num_products=args.products,
            num_orders=args.orders,
            seed=args.seed,
        )
        generation_started = perf_counter()
        datasets = generator.generate_all()
        print(f"Generation time: {perf_counter() - generation_started:.2f} seconds")

        for name, dataframe in datasets.items():
            destination = args.output_dir / f"{name}.parquet"
            dataframe.to_parquet(
                destination, engine="pyarrow", compression="snappy", index=False,
            )
            size_bytes = destination.stat().st_size
            print(
                f"{destination}: {len(dataframe):,} rows, "
                f"{size_bytes:,} bytes ({size_bytes / 1024**2:.2f} MiB)"
            )
            LOGGER.info("Saved %s", destination)

        print(f"Total generation and export time: {perf_counter() - started:.2f} seconds")
        return 0
    except ImportError:
        LOGGER.exception("Parquet dependencies are missing; run python -m pip install -r requirements.txt")
    except (ValueError, OSError):
        LOGGER.exception("Invalid settings or failed to write Parquet files")
    except Exception:
        LOGGER.exception("Unexpected failure while generating or exporting data")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
