"""Generate reproducible e-commerce DataFrames and pipeline-compatible CSVs."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Create linked datasets with normal ages and Pareto customer activity.

    Defaults generate 100,000 customers, 10,000 products, and 1,000,000 orders.
    With identical parameters and dependency versions, a seed reproduces data.
    Ages default to mean 35 and standard deviation 12, clipped to 18--90.
    Dates span start_date through end_date; orders cannot predate registration.
    City and country are independent synthetic fields, not geographic pairs.
    """

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int | None = 42,
        age_mean: float = 35.0,
        age_std: float = 12.0,
        start_date: date = date(2021, 1, 1),
        end_date: date = date(2026, 1, 1),
        show_progress: bool = True,
    ) -> None:
        """Configure counts, randomness, age distribution, dates, and progress.

        Raises ValueError for invalid counts, dates, or distribution parameters.
        """
        for count in (num_customers, num_products, num_orders):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Record counts must be integers and cannot be negative.")
        if num_orders and (not num_customers or not num_products):
            raise ValueError("Orders require at least one customer and one product.")
        if not np.isfinite(age_mean) or not np.isfinite(age_std) or age_std < 0:
            raise ValueError("Age parameters must be finite; age_std cannot be negative.")
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.age_mean = age_mean
        self.age_std = age_std
        self.start_date = start_date
        self.end_date = end_date
        self.show_progress = show_progress
        self.rng = np.random.default_rng(seed)
        self.fake = Faker("en_US")
        self.fake.seed_instance(seed)

    def generate_customers(self) -> pd.DataFrame:
        """Return customers with Faker names/emails and clipped normal ages."""
        LOGGER.info("Generating %s customers", self.num_customers)
        records = [
            (index + 1, self.fake.name(), self.fake.email(),
             self.fake.city(), self.fake.country())
            for index in tqdm(range(self.num_customers), desc="Customers",
                              disable=not self.show_progress)
        ]
        frame = pd.DataFrame(records, columns=["customer_id", "name", "email", "city", "country"])
        frame["customer_id"] = frame["customer_id"].astype("int64")
        frame["age"] = np.clip(np.rint(self.rng.normal(
            self.age_mean, self.age_std, self.num_customers)), 18, 90).astype("int16")
        days = (self.end_date - self.start_date).days
        frame["registration_date"] = pd.Timestamp(self.start_date) + pd.to_timedelta(
            self.rng.integers(0, days + 1, self.num_customers), unit="D")
        return frame[["customer_id", "name", "email", "age", "city", "country", "registration_date"]]

    def generate_products(self) -> pd.DataFrame:
        """Return products with uniform prices, stock levels, and ratings."""
        LOGGER.info("Generating %s products", self.num_products)
        names = [
            f"{self.fake.catch_phrase()} {index + 1}"
            for index in tqdm(range(self.num_products), desc="Products",
                              disable=not self.show_progress)
        ]
        return pd.DataFrame({
            "product_id": np.arange(1, self.num_products + 1, dtype=np.int64),
            "name": names,
            "category": self.rng.choice(CATEGORIES, self.num_products),
            "price": self.rng.integers(1000, 50001, self.num_products) / 100,
            "stock": self.rng.integers(0, 1001, self.num_products),
            "rating": np.round(self.rng.uniform(1, 5, self.num_products), 2),
        })

    def _customer_probabilities(self, count: int) -> np.ndarray:
        """Return Pareto weights calibrated to an 80%/20% expected order split.

        The most active ceil(20% * count) customers receive 80% probability.
        Actual sampled counts fluctuate. Pareto draws vary activity within
        each group. With one customer all orders belong to that customer.
        """
        if count == 1:
            return np.ones(1)
        # Classical Pareto alpha whose population top quintile holds 80%.
        alpha = 1 / (1 - np.log(0.8) / np.log(0.2))
        weights = self.rng.pareto(alpha, count) + 1
        top_count = max(1, int(np.ceil(0.2 * count)))
        top = np.argsort(weights)[-top_count:]
        top_mask = np.zeros(count, dtype=bool)
        top_mask[top] = True
        weights[top_mask] *= 0.8 / weights[top_mask].sum()
        weights[~top_mask] *= 0.2 / weights[~top_mask].sum()
        return weights / weights.sum()

    def generate_orders(
        self, customers: pd.DataFrame, products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return orders referencing supplied dimensions, in vectorized chunks.

        Customer IDs and product IDs must be unique and non-null. Registration
        dates must be valid and no later than end_date. Quantities are 1--10.
        Stock is a snapshot and is not depleted during synthetic generation.
        """
        for frame, key in ((customers, "customer_id"), (products, "product_id")):
            if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
                raise ValueError(f"{key} must exist and contain unique non-null IDs.")
        if "registration_date" not in customers:
            raise ValueError("Customers require registration_date.")
        if self.num_orders and (customers.empty or products.empty):
            raise ValueError("Orders require at least one customer and one product.")
        registrations = pd.to_datetime(customers["registration_date"], errors="raise")
        if registrations.isna().any() or (registrations > pd.Timestamp(self.end_date)).any():
            raise ValueError("Registration dates must be valid and no later than end_date.")
        LOGGER.info("Generating %s orders using Pareto customer activity", self.num_orders)
        if not self.num_orders:
            return pd.DataFrame({
                "order_id": pd.Series(dtype="int64"),
                "customer_id": pd.Series(dtype=customers["customer_id"].dtype),
                "product_id": pd.Series(dtype=products["product_id"].dtype),
                "quantity": pd.Series(dtype="int16"),
                "order_date": pd.Series(dtype="datetime64[ns]"),
            })
        probabilities = self._customer_probabilities(len(customers))
        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        registration_days = registrations.to_numpy(dtype="datetime64[D]")
        end_day = np.datetime64(self.end_date, "D")
        chunks: list[pd.DataFrame] = []
        with tqdm(total=self.num_orders, desc="Orders", disable=not self.show_progress) as progress:
            for offset in range(0, self.num_orders, 100_000):
                size = min(100_000, self.num_orders - offset)
                customer_indices = self.rng.choice(len(customers), size=size, p=probabilities)
                starts = registration_days[customer_indices]
                spans = (end_day - starts).astype(np.int64) + 1
                day_offsets = self.rng.integers(0, spans)
                chunks.append(pd.DataFrame({
                    "order_id": np.arange(offset + 1, offset + size + 1, dtype=np.int64),
                    "customer_id": customer_ids[customer_indices],
                    "product_id": self.rng.choice(product_ids, size=size),
                    "quantity": self.rng.integers(1, 11, size=size, dtype=np.int16),
                    "order_date": starts + day_offsets.astype("timedelta64[D]"),
                }))
                progress.update(size)
        return pd.concat(chunks, ignore_index=True)

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Return customers, products, and linked orders in a named mapping."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers, products)
        LOGGER.info("Generated %s customers, %s products, %s orders",
                    len(customers), len(products), len(orders))
        return {"customers": customers, "products": products, "orders": orders}


def generate_data(
    num_customers: int = 100_000,
    num_products: int = 10_000,
    num_orders: int = 1_000_000,
    output_dir: Path = RAW_DATA_DIR,
    seed: int | None = 42,
    as_of_date: date = date(2026, 1, 1),
) -> dict[str, Path]:
    """Write legacy CSV schemas so the existing Spark analytics keeps working.

    The class API returns the requested DataFrames. This compatibility helper
    maps names and dates to the existing pipeline schema and adds exact order
    amounts and synthetic statuses. Returns a mapping of dataset names to paths.
    """
    generator = SyntheticDataGenerator(num_customers, num_products, num_orders,
                                       seed=seed, end_date=as_of_date)
    frames = generator.generate_all()
    customers = frames["customers"].copy()
    names = customers["name"].str.split(" ", n=1, expand=True)
    customers["first_name"] = names[0] if len(customers) else pd.Series(dtype="str")
    customers["last_name"] = names[1] if len(customers) else pd.Series(dtype="str")
    customers = customers.rename(columns={"registration_date": "signup_date"})
    customers = customers[["customer_id", "first_name", "last_name", "email", "city", "country", "signup_date"]]
    products = frames["products"].rename(columns={"name": "product_name", "stock": "stock_quantity"})
    products = products[["product_id", "product_name", "category", "price", "stock_quantity"]]
    orders = frames["orders"].copy()
    cents = np.rint(products.set_index("product_id")["price"] * 100).astype("int64")
    totals = orders["product_id"].map(cents).to_numpy(dtype="int64") * orders["quantity"].to_numpy(dtype="int64")
    orders["total_amount"] = [f"{int(value) // 100}.{int(value) % 100:02d}" for value in totals]
    orders["status"] = generator.rng.choice(
        ["completed", "processing", "shipped", "cancelled"], size=len(orders), p=[0.5, 0.2, 0.2, 0.1])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in {"customers": customers, "products": products, "orders": orders}.items():
        paths[name] = output_dir / f"{name}.csv"
        frame.to_csv(paths[name], index=False, date_format="%Y-%m-%d")
        LOGGER.info("Saved %s", paths[name])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Generate CSVs from CLI options; --schema selects class or legacy fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--schema", choices=["pipeline", "native"], default="pipeline")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.schema == "pipeline":
        generate_data(args.customers, args.products, args.orders, args.output_dir, args.seed)
    else:
        frames = SyntheticDataGenerator(args.customers, args.products, args.orders, args.seed).generate_all()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(args.output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
