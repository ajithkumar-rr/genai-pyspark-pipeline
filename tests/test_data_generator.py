"""Tests for synthetic e-commerce data generation."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from src.data_generator import generate_data


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a generated CSV file into a list of row dictionaries.

    Args:
        path: CSV file to read.

    Returns:
        CSV records with values represented as strings.
    """
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_generate_data_writes_requested_records(tmp_path: Path) -> None:
    """Verify generated CSV files contain the requested number of records."""
    paths = generate_data(
        num_customers=4,
        num_products=3,
        num_orders=12,
        output_dir=tmp_path,
        seed=7,
    )

    assert set(paths) == {"customers", "products", "orders"}
    assert all(path.is_file() for path in paths.values())
    assert len(_read_csv(paths["customers"])) == 4
    assert len(_read_csv(paths["products"])) == 3
    assert len(_read_csv(paths["orders"])) == 12


def test_orders_reference_existing_customers_and_products(tmp_path: Path) -> None:
    """Ensure generated order foreign keys point to generated records."""
    paths = generate_data(5, 4, 20, output_dir=tmp_path, seed=17)
    customers = _read_csv(paths["customers"])
    products = _read_csv(paths["products"])
    orders = _read_csv(paths["orders"])
    customer_ids = {row["customer_id"] for row in customers}
    product_ids = {row["product_id"] for row in products}

    assert all(row["customer_id"] in customer_ids for row in orders)
    assert all(row["product_id"] in product_ids for row in orders)
    assert all(float(row["total_amount"]) > 0 for row in orders)
    signup_dates = {row["customer_id"]: row["signup_date"] for row in customers}
    prices = {row["product_id"]: Decimal(row["price"]) for row in products}
    assert all(row["order_date"] >= signup_dates[row["customer_id"]] for row in orders)
    assert all(Decimal(row["total_amount"]) == prices[row["product_id"]] * int(row["quantity"]) for row in orders)


def test_seed_is_reproducible(tmp_path: Path) -> None:
    """Ensure identical seeds and dates produce byte-identical CSV datasets."""
    first = generate_data(5, 4, 20, output_dir=tmp_path / "first", seed=17)
    second = generate_data(5, 4, 20, output_dir=tmp_path / "second", seed=17)
    assert all(first[name].read_bytes() == second[name].read_bytes() for name in first)


def test_generate_data_rejects_orders_without_dimensions(tmp_path: Path) -> None:
    """Reject orders when customer or product lookup data is unavailable."""
    with pytest.raises(ValueError, match="at least one customer"):
        generate_data(0, 2, 1, output_dir=tmp_path)


def test_generate_data_rejects_negative_counts(tmp_path: Path) -> None:
    """Reject invalid negative record counts before writing files."""
    with pytest.raises(ValueError, match="cannot be negative"):
        generate_data(-1, 2, 0, output_dir=tmp_path)
