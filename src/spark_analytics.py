"""Run batch business analytics over the generated CSV files with PySpark."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    coalesce,
    date_format,
    desc,
    lit,
    sum as spark_sum,
    to_date,
)

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    configure_logging,
    ensure_project_directories,
)

LOGGER = logging.getLogger(__name__)
REQUIRED_INPUTS: tuple[str, ...] = ("customers.csv", "products.csv", "orders.csv")


class SalesAnalytics:
    """Analyze generated order lines using quantity times current product price.

    Products must have unique product IDs and non-null nonnegative prices;
    orders must have valid product/customer IDs, dates, and positive quantities.
    Inner joins include matching products only. Native Parquet orders have no
    status; all are included. If status exists, only completed orders count.
    Product prices are a snapshot, so this is not historical booked revenue.
    """

    def __init__(self, app_name: str = "SalesAnalytics", master: str = "local[*]") -> None:
        """Create the configured local Spark session; stop it when finished."""
        if not (master == "local" or master.startswith("local[")):
            raise ValueError("SalesAnalytics requires a local Spark master.")
        self.app_name = app_name
        self.master = master
        self.spark = self.create_spark_session()

    def create_spark_session(self) -> SparkSession:
        """Create/reuse Spark with 4 GB heaps, AQE, and Kryo serialization.

        These settings must be applied before the first SparkContext starts.
        An already running context cannot have its JVM heap resized. In local
        mode work runs inside the driver JVM; executor memory is also configured
        for consistency. Use spark-submit --driver-memory 4g when launching
        through spark-submit. The heap limit excludes Python and native memory.
        """
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        LOGGER.info("Starting Spark Java gateway (first startup can take tens of seconds)...")
        spark = (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        LOGGER.info("Spark session ready: %s", spark.sparkContext.master)
        return spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or dataset directory, including Spark-supported URIs.

        Spark errors propagate to the caller with a logged traceback. Reading
        is lazy; corrupt data may fail only when an action is executed.
        """
        LOGGER.info("Loading Parquet: %s", path)
        try:
            return self.spark.read.parquet(str(path))
        except Exception:
            LOGGER.exception("Could not load Parquet from %s", path)
            raise

    def _revenue_lines(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join minimal columns and compute decimal revenue without float sums."""
        for frame, required in (
            (orders_df, {"customer_id", "product_id", "quantity", "order_date"}),
            (products_df, {"product_id", "price", "category"}),
        ):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        if "status" in orders_df.columns:
            orders_df = orders_df.filter(F.col("status") == "completed")
        orders = orders_df.select("customer_id", "product_id", "order_date", "quantity")
        products = products_df.select("product_id", "category", "price")
        return orders.join(products, "product_id", "inner").withColumn(
            "line_revenue",
            F.col("quantity").cast("decimal(10,0)") * F.col("price").cast("decimal(18,2)"),
        )

    def top_customers_by_revenue(
        self, orders_df: DataFrame, products_df: DataFrame, n: int = 10,
    ) -> DataFrame:
        """Return up to n customers ranked by total_spend, with deterministic ties."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        LOGGER.info("Calculating top %s customers", n)
        return (
            self._revenue_lines(orders_df, products_df)
            .groupBy("customer_id")
            .agg(F.sum("line_revenue").alias("total_spend"))
            .orderBy(F.desc("total_spend"), F.asc("customer_id"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return total_revenue and units_sold for each category with sales."""
        LOGGER.info("Calculating category sales")
        return (
            self._revenue_lines(orders_df, products_df)
            .groupBy("category")
            .agg(F.sum("line_revenue").alias("total_revenue"),
                 F.sum(F.col("quantity").cast("long")).alias("units_sold"))
            .orderBy(F.desc("total_revenue"), F.asc("category"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue, prior revenue, and month-over-month growth.

        Include zero-sales months between the first and last observed months.
        Growth = 100 * (current - previous) / previous. The first month and
        months following zero revenue have null growth (undefined percentage).
        Empty inputs produce an empty result. month is a first-of-month date.
        """
        LOGGER.info("Calculating monthly revenue growth")
        monthly = (
            self._revenue_lines(orders_df, products_df)
            .withColumn("month", F.trunc(F.to_date("order_date"), "month"))
            .groupBy("month")
            .agg(F.sum("line_revenue").alias("total_revenue"))
        )
        bounds = monthly.agg(F.min("month").alias("start"), F.max("month").alias("end"))
        calendar = bounds.select(F.explode(F.sequence(
            "start", "end", F.expr("INTERVAL 1 MONTH"))).alias("month"))
        complete = calendar.join(monthly, "month", "left").withColumn(
            "total_revenue", F.coalesce(F.col("total_revenue"), F.lit(0)),
        )
        previous = complete.withColumn(
            "previous_month_revenue", F.lag("total_revenue").over(Window.orderBy("month")),
        )
        return previous.withColumn(
            "revenue_growth_pct",
            F.when(F.col("previous_month_revenue") > 0, F.round(
                (F.col("total_revenue") - F.col("previous_month_revenue"))
                / F.col("previous_month_revenue") * 100, 2)),
        ).orderBy("month")

    def stop(self) -> None:
        """Stop the session and its shared SparkContext after materializing results."""
        self.spark.stop()
        LOGGER.info("Spark session stopped")


def create_spark_session(app_name: str = "EcommercePipelineAnalytics") -> SparkSession:
    """Create a local Spark session for pipeline analysis.

    Args:
        app_name: Name displayed in the Spark UI and logs.

    Returns:
        A Spark session configured to use all available local cores.
    """
    return SalesAnalytics(app_name=app_name).spark


def _write_result(dataframe: DataFrame, destination: Path) -> Path:
    """Write one small analytics DataFrame as a headered CSV file.

    Aggregate result tables are intentionally streamed through the Python
    driver so local output works on Windows without Hadoop ``winutils.exe``.

    Args:
        dataframe: Result DataFrame to persist.
        destination: CSV file path for the analytics result.

    Returns:
        The output CSV file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(dataframe.columns)
        for row in dataframe.toLocalIterator():
            writer.writerow([row[column_name] for column_name in dataframe.columns])
    LOGGER.info("Wrote analytics result to %s", destination)
    return destination


def run_analytics(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Analyze source CSV files and persist business insight datasets.

    The output includes overall KPIs, monthly sales, revenue by product
    category, top customers, and order counts by status.

    Args:
        raw_data_dir: Directory containing the three generated source CSVs.
        output_dir: Directory for the analytics CSV files.

    Returns:
        A mapping from analytics result name to its CSV file.

    Raises:
        FileNotFoundError: If one or more required input CSV files are absent.
    """
    missing_files: list[Path] = [
        raw_data_dir / filename
        for filename in REQUIRED_INPUTS
        if not (raw_data_dir / filename).is_file()
    ]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required input CSV file(s): {missing_list}")

    output_dir.mkdir(parents=True, exist_ok=True)
    spark = create_spark_session()
    try:
        customers = (
            spark.read.option("header", True)
            .schema("customer_id STRING, first_name STRING, last_name STRING, email STRING, city STRING, country STRING, signup_date DATE")
            .option("mode", "FAILFAST")
            .csv(str(raw_data_dir / "customers.csv"))
        )
        products = (
            spark.read.option("header", True)
            .schema("product_id STRING, product_name STRING, category STRING, price DECIMAL(18,2), stock_quantity INT")
            .option("mode", "FAILFAST")
            .csv(str(raw_data_dir / "products.csv"))
            .withColumn("price", col("price").cast("decimal(18,2)"))
        )
        orders = (
            spark.read.option("header", True)
            .schema("order_id STRING, customer_id STRING, product_id STRING, order_date DATE, quantity INT, total_amount DECIMAL(18,2), status STRING")
            .option("mode", "FAILFAST")
            .csv(str(raw_data_dir / "orders.csv"))
            .withColumn("order_date", to_date(col("order_date")))
            .withColumn("quantity", col("quantity").cast("integer"))
            .withColumn("total_amount", col("total_amount").cast("decimal(18,2)"))
        )

        all_orders = orders
        orders = orders.filter(col("status") == "completed")
        LOGGER.info("Revenue summaries include completed orders only")

        kpis = (
            orders.agg(
                count("order_id").alias("order_count"),
                coalesce(spark_sum("total_amount"), lit(0)).alias("total_revenue"),
                coalesce(avg("total_amount"), lit(0)).alias("average_order_value"),
                countDistinct("customer_id").alias("active_customers"),
            )
            .withColumn("customer_count", lit(customers.count()))
            .withColumn("product_count", lit(products.count()))
        )
        monthly_sales = (
            orders.withColumn("order_month", date_format(col("order_date"), "yyyy-MM"))
            .groupBy("order_month")
            .agg(
                count("order_id").alias("order_count"),
                spark_sum("total_amount").alias("revenue"),
            )
            .orderBy("order_month")
        )
        revenue_by_category = (
            orders.join(products, on="product_id", how="inner")
            .groupBy("category")
            .agg(
                count("order_id").alias("order_count"),
                spark_sum("total_amount").alias("revenue"),
            )
            .orderBy(desc("revenue"))
        )
        top_customers = (
            orders.join(customers, on="customer_id", how="inner")
            .groupBy("customer_id", "first_name", "last_name")
            .agg(
                count("order_id").alias("order_count"),
                spark_sum("total_amount").alias("customer_revenue"),
            )
            .orderBy(desc("customer_revenue"))
            .limit(10)
        )
        orders_by_status = (
            all_orders.groupBy("status")
            .agg(count("order_id").alias("order_count"))
            .orderBy(desc("order_count"))
        )

        results: dict[str, DataFrame] = {
            "kpis": kpis,
            "monthly_sales": monthly_sales,
            "revenue_by_category": revenue_by_category,
            "top_customers": top_customers,
            "orders_by_status": orders_by_status,
        }
        return {
            result_name: _write_result(dataframe, output_dir / f"{result_name}.csv")
            for result_name, dataframe in results.items()
        }
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options and execute the PySpark analytics pipeline.

    Args:
        argv: Optional command-line arguments, mainly useful for testing.

    Returns:
        Process exit status; zero indicates success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args(argv)

    configure_logging()
    ensure_project_directories()
    run_analytics(raw_data_dir=args.raw_dir, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
