"""Compare Pandas and PySpark on the same Parquet customer-revenue workload.

Run: python compare_pandas_pyspark.py
Spark stages persist and materialize their results to prevent lazy plan timing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, functions as F

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)
ORDER_COLUMNS = ["customer_id", "product_id", "quantity"]
PRODUCT_COLUMNS = ["product_id", "price"]


def benchmark_pandas(raw_dir: Path) -> tuple[dict[str, float], pd.DataFrame, int]:
    """Time eager load, join/revenue, and aggregation/top-ten stages.

    Compute money as integer cents in both engines to avoid floating-point
    reduction differences. Prices are interpreted as two-decimal currency.
    All order rows are included regardless of status.
    """
    times: dict[str, float] = {}
    started = perf_counter()
    orders = pd.read_parquet(raw_dir / "orders.parquet", columns=ORDER_COLUMNS)
    products = pd.read_parquet(raw_dir / "products.parquet", columns=PRODUCT_COLUMNS)
    times["Load Parquet"] = perf_counter() - started
    row_count = len(orders)

    started = perf_counter()
    joined = orders.merge(products, on="product_id", how="inner")
    joined["revenue_cents"] = (
        np.rint(joined["price"].to_numpy() * 100).astype(np.int64)
        * joined["quantity"].to_numpy(dtype=np.int64)
    )
    times["Join + calculate revenue"] = perf_counter() - started

    started = perf_counter()
    top = (
        joined.groupby("customer_id", as_index=False)["revenue_cents"].sum()
        .sort_values(["revenue_cents", "customer_id"], ascending=[False, True])
        .head(10).reset_index(drop=True)
    )
    times["Group + top 10"] = perf_counter() - started
    return times, top, row_count


def benchmark_spark(
    analytics: SalesAnalytics, raw_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame, int]:
    """Time Spark stages through actions and return the collected top ten.

    Inputs and joined results are cached with MEMORY_AND_DISK and counted
    within their own timed stages. This includes cache materialization costs.
    Final timing includes collect, transferring only ten results to Python.
    Spark startup and unpersist cleanup are excluded from the stage totals.
    """
    times: dict[str, float] = {}
    cached: list[DataFrame] = []
    try:
        started = perf_counter()
        orders = analytics.load_parquet(raw_dir / "orders.parquet").select(*ORDER_COLUMNS)
        products = analytics.load_parquet(raw_dir / "products.parquet").select(*PRODUCT_COLUMNS)
        orders = orders.persist(StorageLevel.MEMORY_AND_DISK)
        products = products.persist(StorageLevel.MEMORY_AND_DISK)
        cached.extend([orders, products])
        row_count = orders.count()
        products.count()
        times["Load Parquet"] = perf_counter() - started

        started = perf_counter()
        joined = orders.join(products, "product_id", "inner").withColumn(
            "revenue_cents",
            (F.col("price").cast("decimal(18,2)") * F.lit(100)).cast("long")
            * F.col("quantity").cast("long"),
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(joined)
        joined.count()
        times["Join + calculate revenue"] = perf_counter() - started

        started = perf_counter()
        result = (
            joined.groupBy("customer_id")
            .agg(F.sum("revenue_cents").alias("revenue_cents"))
            .orderBy(F.desc("revenue_cents"), F.asc("customer_id"))
            .limit(10).collect()
        )
        times["Group + top 10"] = perf_counter() - started
        top = pd.DataFrame([row.asDict() for row in result], columns=["customer_id", "revenue_cents"])
        return times, top, row_count
    finally:
        for dataframe in reversed(cached):
            dataframe.unpersist(blocking=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run both engines, verify identical top tens, and print a timing table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    config.configure_logging()
    analytics: SalesAnalytics | None = None
    exit_code = 0
    try:
        for name in ("orders", "products"):
            path = args.raw_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; generate inputs using python main.py.")
        pandas_times, pandas_top, pandas_rows = benchmark_pandas(args.raw_dir)
        LOGGER.info("Pandas benchmark finished: %s orders", f"{pandas_rows:,}")
        if pandas_rows != args.expected_rows:
            raise ValueError(f"Expected {args.expected_rows:,} orders, found {pandas_rows:,}. Use --expected-rows to override.")
        started = perf_counter()
        analytics = SalesAnalytics(app_name="PandasVsPySpark")
        startup = perf_counter() - started
        spark_times, spark_top, spark_rows = benchmark_spark(analytics, args.raw_dir)
        if pandas_rows != spark_rows:
            raise AssertionError("Engines loaded different order counts.")
        pd.testing.assert_frame_equal(pandas_top, spark_top, check_dtype=False)
        records = []
        for stage in pandas_times:
            pandas_seconds, spark_seconds = pandas_times[stage], spark_times[stage]
            records.append({"Operation": stage, "Pandas (s)": pandas_seconds,
                            "PySpark (s)": spark_seconds,
                            "Spark/Pandas time ratio": spark_seconds / pandas_seconds})
        pandas_total, spark_total = sum(pandas_times.values()), sum(spark_times.values())
        records.append({"Operation": "Total measured stages", "Pandas (s)": pandas_total,
                        "PySpark (s)": spark_total, "Spark/Pandas time ratio": spark_total / pandas_total})
        print(f"\nOrders: {pandas_rows:,}. Top-ten results match exactly.")
        print(pd.DataFrame(records).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"\nSpark startup (excluded above): {startup:.4f} seconds")
        print(f"Spark measured stages + startup: {spark_total + startup:.4f} seconds")
        display = pandas_top.copy()
        display["total_revenue"] = display["revenue_cents"] / 100
        print("\nTop 10 customers:")
        print(display[["customer_id", "total_revenue"]].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
        print("\nRatio > 1 means Spark took longer. Both engines include all orders.")
        print("Single run, Pandas first; OS caching and Spark startup/JIT affect results.")
        print("Spark stage times include persistence/materialization; the final stage includes collect.")
    except KeyboardInterrupt:
        LOGGER.warning("Comparison cancelled with Ctrl+C while starting or running Spark.")
        exit_code = 130
    except Exception:
        LOGGER.exception("Pandas/PySpark comparison failed")
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
