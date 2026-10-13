"""Run SalesAnalytics on raw Parquet data and export small summaries as CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Sequence

from src import config
from src.spark_analytics import SalesAnalytics, _write_result

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Load orders/products, print summaries, and save CSV results.

    Return zero on success or one on input, Spark, export, or cleanup failure.
    Default paths come from config and work regardless of the current directory.
    Existing summary CSV files are overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    config.configure_logging()
    analytics: SalesAnalytics | None = None
    started = perf_counter()
    exit_code = 0
    try:
        if args.top_n < 1:
            raise ValueError("--top-n must be positive.")
        for name in ("orders", "products"):
            path = args.raw_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Generate Parquet inputs with python main.py first.")
        analytics = SalesAnalytics()
        orders = analytics.load_parquet(args.raw_dir / "orders.parquet")
        products = analytics.load_parquet(args.raw_dir / "products.parquet")
        results = {
            "top_customers_by_revenue": analytics.top_customers_by_revenue(orders, products, args.top_n),
            "sales_by_category": analytics.sales_by_category(orders, products),
            "monthly_trends": analytics.monthly_trends(orders, products),
        }
        for name, dataframe in results.items():
            print(f"\n{name.replace('_', ' ').title()}")
            dataframe.show(20, truncate=False)
            _write_result(dataframe, args.output_dir / f"{name}.csv")
        print(f"\nAnalytics completed in {perf_counter() - started:.2f} seconds.")
        print(f"Results saved to {args.output_dir.resolve()}")
    except Exception:
        LOGGER.exception("Analytics failed")
        exit_code = 1
    finally:
        if analytics is not None:
            try:
                analytics.stop()
            except Exception:
                LOGGER.exception("Spark cleanup failed")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
