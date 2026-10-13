"""Compare file size, I/O time, Python memory, CPU time, and estimated energy.

Run: python benchmark_formats.py --rows 500000
Reads run immediately after writes and may benefit from the OS page cache.
tracemalloc reports traced allocations, not total RSS/native Arrow memory.
Energy is a fixed-TDP proxy, not a measurement of hardware power consumption.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable, TypeVar

import fastparquet
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.orc as orc
import pyarrow.parquet as parquet
from openpyxl import Workbook

from src import config

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
MB = 1_000_000  # Decimal megabytes, not MiB.


@dataclass(frozen=True)
class Measurement:
    """Wall time, process CPU time, and traced allocation peak for one operation."""

    wall_seconds: float
    cpu_seconds: float
    peak_mb: float


def measure(operation: Callable[[], T]) -> tuple[T, Measurement]:
    """Measure an operation with fresh tracing; exclude preceding garbage collection.

    The pre-existing source DataFrame is outside the traced allocation window.
    CPU time includes this process's threads but excludes child processes.
    Tracing overhead is included in the reported wall and CPU times.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("Run the benchmark without an existing tracemalloc session.")
    gc.collect()
    tracemalloc.start()
    wall_start = perf_counter()
    cpu_start = process_time()
    try:
        result = operation()
        cpu_seconds = process_time() - cpu_start
        wall_seconds = perf_counter() - wall_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        return result, Measurement(wall_seconds, cpu_seconds, peak_bytes / MB)
    finally:
        tracemalloc.stop()


def create_dataframe(rows: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Create a seeded six-column dataset with numeric, string, and date fields."""
    if rows < 1:
        raise ValueError("rows must be positive.")
    rng = np.random.default_rng(seed)
    ids = np.arange(1, rows + 1, dtype=np.int64)
    return pd.DataFrame({
        "id": ids,
        "name": np.array([f"Customer {value}" for value in ids], dtype=object),
        "email": np.array([f"customer{value}@example.com" for value in ids], dtype=object),
        "amount": rng.integers(1000, 50001, rows) / 100,
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D"),
        "category": rng.choice(["Electronics", "Clothing", "Home", "Sports", "Books"], rows).astype(object),
    })


def write_xlsx(frame: pd.DataFrame, path: Path) -> None:
    """Stream DataFrame rows through openpyxl to limit XLSX write allocations.

    One sheet has a 1,048,576-row limit including its header. Dates are exported
    as ISO text and parsed on read, matching the CSV date representation.
    """
    if len(frame) > 1_048_575:
        raise ValueError("XLSX supports at most 1,048,575 data rows per sheet.")
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("data")
    sheet.append(list(frame.columns))
    for identifier, name, email, amount, timestamp, category in frame.itertuples(index=False, name=None):
        sheet.append((identifier, name, email, amount, timestamp.strftime("%Y-%m-%d"), category))
    workbook.save(path)
    workbook.close()


def format_operations(
    frame: pd.DataFrame, output_dir: Path,
) -> list[tuple[str, str, Path, Callable[[], None], Callable[[], pd.DataFrame]]]:
    """Define explicit codecs and readers; compare both requested Parquet engines.

    Arrow conversion belongs to the measured operations. CSV is uncompressed;
    XLSX uses ZIP; Parquet and ORC use Snappy; Feather uses LZ4.
    """
    csv_path = output_dir / "sample.csv"
    xlsx_path = output_dir / "sample.xlsx"
    parquet_path = output_dir / "sample_pyarrow.parquet"
    fast_path = output_dir / "sample_fastparquet.parquet"
    orc_path = output_dir / "sample.orc"
    feather_path = output_dir / "sample.feather"
    return [
        ("CSV", "none", csv_path,
         lambda: frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d"),
         lambda: pd.read_csv(csv_path, parse_dates=["date"])),
        ("XLSX", "ZIP/streamed", xlsx_path,
         lambda: write_xlsx(frame, xlsx_path),
         lambda: pd.read_excel(xlsx_path, engine="openpyxl", parse_dates=["date"])),
        ("Parquet/pyarrow", "Snappy", parquet_path,
         lambda: parquet.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path, compression="snappy"),
         lambda: parquet.read_table(parquet_path).to_pandas()),
        ("Parquet/fastparquet", "Snappy", fast_path,
         lambda: fastparquet.write(str(fast_path), frame, compression="SNAPPY", write_index=False),
         lambda: fastparquet.ParquetFile(str(fast_path)).to_pandas()),
        ("ORC", "Snappy", orc_path,
         lambda: write_orc(frame, orc_path),
         lambda: orc.ORCFile(orc_path).read().to_pandas()),
        ("Feather", "LZ4", feather_path,
         lambda: feather.write_feather(frame, feather_path, compression="lz4"),
         lambda: feather.read_feather(feather_path)),
    ]


def write_orc(frame: pd.DataFrame, path: Path) -> None:
    """Write calendar dates as Arrow date32, avoiding timestamp timezone lookup.

    The generated date column contains midnight calendar dates, so conversion
    to ORC's DATE logical type preserves all values on Windows and Unix.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    position = table.schema.get_field_index("date")
    table = table.set_column(position, "date", table.column(position).cast(pa.date32()))
    orc.write_table(table, path, compression="snappy")


def validate_round_trip(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Check every value after reading, allowing serialization dtype differences."""
    actual = actual.copy()
    actual["date"] = pd.to_datetime(actual["date"]).astype("datetime64[ns]")
    expected = expected.copy()
    expected["date"] = expected["date"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-10)


def run_benchmark(
    rows: int = 500_000,
    output_dir: Path = config.DATA_DIR / "benchmarks",
    tdp_watts: float = 65.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Write/read each format, validate values, print metrics, and save reports.

    Peak memory is reported separately for write/read and as their maximum,
    never their sum. CPU and energy totals combine write and read operations.
    Savings are 100 * (CSV - format) / CSV; negative values mean an increase.
    Per-format failures are logged and recorded rather than misreported as zero.
    Existing sample and report files in output_dir are overwritten.
    """
    if not math.isfinite(tdp_watts) or tdp_watts <= 0:
        raise ValueError("tdp_watts must be positive and finite.")
    frame = create_dataframe(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name, codec, path, writer, reader in format_operations(frame, output_dir):
        LOGGER.info("Benchmarking %s (%s), %s rows", name, codec, rows)
        try:
            _, write = measure(writer)
            restored, read = measure(reader)
            validate_round_trip(frame, restored)
            del restored
            cpu_total = write.cpu_seconds + read.cpu_seconds
            records.append({
                "format": name, "compression": codec,
                "size_MB": path.stat().st_size / MB,
                "write_s": write.wall_seconds, "read_s": read.wall_seconds,
                "write_peak_MB": write.peak_mb, "read_peak_MB": read.peak_mb,
                "peak_MB": max(write.peak_mb, read.peak_mb),
                "write_CPU_s": write.cpu_seconds, "read_CPU_s": read.cpu_seconds,
                "CPU_s": cpu_total, "energy_Wh": cpu_total * tdp_watts / 3600,
                "status": "OK",
            })
        except Exception as error:
            LOGGER.exception("Benchmark failed for %s", name)
            records.append({"format": name, "compression": codec,
                            "status": f"ERROR: {type(error).__name__}: {error}"})
    results = pd.DataFrame(records)
    if "size_MB" not in results:
        raise RuntimeError("All format benchmarks failed; consult the logged exceptions.")
    metrics = ["size_MB", "write_s", "read_s", "peak_MB", "CPU_s", "energy_Wh"]
    baseline = results.loc[results["format"] == "CSV"].iloc[0]
    for metric in metrics:
        value = baseline.get(metric, float("nan"))
        results[f"{metric}_saving_pct"] = (
            100 * (value - results[metric]) / value if pd.notna(value) and value > 0 else np.nan
        )
    print(f"\n{rows:,} rows | decimal MB | CPU TDP proxy: {tdp_watts:g} W")
    print(results[["format", "compression", "size_MB", "write_s", "read_s",
                   "write_peak_MB", "read_peak_MB", "peak_MB", "write_CPU_s",
                   "read_CPU_s", "CPU_s", "energy_Wh", "status"]].to_string(
                       index=False, float_format=lambda value: f"{value:.4f}", na_rep="N/A"))
    print("\nSavings vs CSV (%; negative = more usage/slower):")
    print(results[["format"] + [f"{metric}_saving_pct" for metric in metrics]].to_string(
        index=False, float_format=lambda value: f"{value:.2f}", na_rep="N/A"))
    print("\nMemory: tracemalloc allocation peaks; excludes the existing input and may omit native buffers.")
    print("Energy: CPU_s * TDP_W / 3600; an estimate, excluding disk, memory, and idle-system energy.")
    print("Reads may be cached. Single-run timings include tracing overhead and codec/engine differences.")
    results.to_csv(output_dir / "comparison.csv", index=False)
    if baseline["status"] != "OK":
        LOGGER.warning("CSV baseline failed; savings cannot be calculated.")
    return results


def main() -> int:
    """Parse command-line settings and return failure if any format fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--output-dir", type=Path, default=config.DATA_DIR / "benchmarks")
    parser.add_argument("--tdp-watts", type=float, default=65.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config.configure_logging()
    try:
        results = run_benchmark(args.rows, args.output_dir, args.tdp_watts, args.seed)
        return 0 if results["status"].eq("OK").all() else 1
    except Exception:
        LOGGER.exception("Unable to run file-format benchmark")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
