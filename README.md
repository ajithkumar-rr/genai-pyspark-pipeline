# E-commerce Data Pipeline

Generate linked fake customer, product, and order CSV files with Faker, then analyze them using PySpark. All functions have type hints and docstrings; command-line jobs log their progress.

## Project structure

```text
genai-pyspark-pipeline/
|-- src/
|   |-- __init__.py              # Empty Python package marker
|   |-- config.py                # Paths and logging settings
|   |-- data_generator.py        # Seeded synthetic CSV generation
|   `-- spark_analytics.py       # Spark business summaries
|-- data/
|   |-- raw/
|   |   |-- .gitkeep
|   |   |-- customers.csv        # Generated
|   |   |-- products.csv         # Generated
|   |   `-- orders.csv           # Generated
|   `-- processed/
|       |-- .gitkeep
|       |-- kpis.csv
|       |-- monthly_sales.csv
|       |-- revenue_by_category.csv
|       |-- top_customers.csv
|       `-- orders_by_status.csv
|-- tests/
|   `-- test_data_generator.py
|-- notebooks/
|   `-- README.md
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- LICENSE
```

## Setup

Use Python 3.10+ and Java 17, 21, or 25. Set JAVA_HOME to your JDK or ensure java is available on PATH. See the official [Spark requirements](https://spark.apache.org/docs/4.2.0/index.html).

Run these commands from this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data_generator
.\.venv\Scripts\python.exe -m src.spark_analytics
.\.venv\Scripts\python.exe -m pytest --basetemp=.test-tmp
```

On macOS/Linux use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. No activation is required. To use the existing workspace environment, substitute `venv` for `.venv`.

## Generate data

To generate the requested DataFrame schemas as Snappy-compressed Parquet files:

```powershell
.\.venv\Scripts\python.exe main.py
```

This saves `customers.parquet`, `products.parquet`, and `orders.parquet` in
`data/raw/`, prints generation/total time and file sizes, and returns exit code
1 on failure. Use `--customers`, `--products`, `--orders`, `--seed`, or
`--output-dir` to override defaults. The existing Spark job reads the legacy
CSV inputs; these Parquet files use the class's native fields.

Defaults: 100,000 customers, 10,000 products, 1,000,000 orders, seed 42. SyntheticDataGenerator returns pandas DataFrames with normal ages (mean 35, standard deviation 12, clipped to 18--90), Pareto customer activity, and tqdm progress bars. The top 20% of customers receive 80% of order probability; sampled counts vary. Prices range from 10--500, ratings from 1--5, and quantities from 1--10. Use --schema native for the requested class fields; the default CLI preserves the Spark CSV schemas.

```powershell
.\.venv\Scripts\python.exe -m src.data_generator --customers 200 --products 30 --orders 1200 --seed 23
```

For custom paths:

```powershell
.\.venv\Scripts\python.exe -m src.data_generator --output-dir data/raw
.\.venv\Scripts\python.exe -m src.spark_analytics --raw-dir data/raw --output-dir data/processed
```

The generator uses valid foreign keys, unique IDs, Faker email addresses, positive quantities, and exact decimal order totals. Orders never predate signup. Its fixed default end date is 2026-01-01, so the same seed produces the same CSVs across days with the same dependency versions. The Python API accepts an `as_of_date` argument to change this. Each order contains one product; prices represent a single fictional currency, without tax or discounts. Statuses are synthetic and are not a fulfillment simulation. Commands overwrite their output files.

## Business insights

Revenue means the total amount of completed orders. Processing, shipped, and cancelled orders are excluded from revenue summaries.

| Output | Contents |
| --- | --- |
| kpis.csv | Completed order count, revenue, average order value, distinct purchasing customers, total customer and product counts |
| monthly_sales.csv | Completed order count and revenue by month |
| revenue_by_category.csv | Completed order count and revenue by category |
| top_customers.csv | Ten customers with the highest completed-order revenue |
| orders_by_status.csv | Counts of all orders by status |

Explicit Spark schemas preserve dates and decimal money, including header-only inputs. CSV parse failures stop analysis. Empty completed-order sets produce zero revenue and average order value. Spark performs all aggregations; Python streams the small result tables to individual CSVs using `toLocalIterator`, avoiding a Hadoop writer dependency on Windows. This local teaching project keeps generated records in memory; large workloads should use partitioned generation and distributed output.

## Tests and notebooks

## File-format benchmark

```powershell
.\.venv\Scripts\python.exe benchmark_formats.py
```

Defaults to 500,000 rows and exports CSV, streamed XLSX, Parquet with both
pyarrow and fastparquet, ORC, and Feather into `data/benchmarks/`. It validates
readback values, prints write/read wall times, traced allocation peaks, process
CPU times, file sizes in decimal MB, estimated Wh, and percentage savings vs
CSV. Results are saved to `data/benchmarks/comparison.csv`. For a quick check,
pass `--rows 1000`. Existing benchmark outputs are overwritten.

Energy uses `(write_CPU_s + read_CPU_s) * 65 / 3600`; override with
`--tdp-watts`. This is a CPU TDP proxy, not measured hardware energy.
tracemalloc does not represent total process RAM or all native Arrow buffers.
Reads can benefit from OS caching; timings include tracing overhead. The codec
and XLSX streaming method are listed in the output. A failure is reported as an
error, and the script returns a nonzero exit code if any format fails.

Tests cover row counts, foreign keys, exact money, signup/order chronology, reproducibility, and invalid counts. They do not need Java. Analytics requires Java and is run separately with the command above.

Start JupyterLab with `.\.venv\Scripts\python.exe -m jupyterlab` and save exploratory notebooks under `notebooks/`.


## SalesAnalytics with Parquet inputs

```python
from src.config import RAW_DATA_DIR, configure_logging
from src.spark_analytics import SalesAnalytics

configure_logging()
analytics = SalesAnalytics()
try:
    orders = analytics.load_parquet(RAW_DATA_DIR / "orders.parquet")
    products = analytics.load_parquet(RAW_DATA_DIR / "products.parquet")
    analytics.top_customers_by_revenue(orders, products, n=10).show()
    analytics.sales_by_category(orders, products).show()
    analytics.monthly_trends(orders, products).show()
finally:
    analytics.stop()
```

Revenue is quantity times the current product price, summed as decimals. All
native generated orders count; if a status column exists, only completed orders
count. Product IDs must be unique, order references valid, quantities positive,
and dates and prices non-null. Inner joins include only matching products.
Monthly trends fill missing months with zero revenue. Growth is null for the
first month or after a zero-revenue month.

Start a fresh process to apply 4 GB driver/executor settings, AQE, and Kryo.
Local tasks run in the driver JVM; 4 GB is its heap limit rather than a cap on
all Python/native memory. An existing SparkContext cannot be resized. With
spark-submit, also pass `--driver-memory 4g` at launch.

Run the Parquet analytics entry point from PowerShell:

```powershell
.\venv\Scripts\python.exe run_analytics.py
```

It prints summaries and writes `top_customers_by_revenue.csv`,
`sales_by_category.csv`, and `monthly_trends.csv` to `data/processed/`.
Pass `--raw-dir`, `--output-dir`, or `--top-n` to customize the run.

## Pandas versus PySpark

```powershell
.\venv\Scripts\python.exe compare_pandas_pyspark.py
```

Compares loading, joining/calculating revenue, and grouping/finding the top ten
on the same million-order Parquet inputs. It verifies identical results and
prints per-stage wall times, totals, and Spark/Pandas time ratios. Spark stages
persist and materialize their data so lazy evaluation is included in timing.
Spark startup is reported separately. Both engines use integer cents and include
all orders. Inputs must contain valid unique product IDs, two-decimal prices,
and integer quantities. This is a single-run local benchmark; OS caching and
Spark warmup affect the result. Pass --raw-dir or --expected-rows for other inputs.
