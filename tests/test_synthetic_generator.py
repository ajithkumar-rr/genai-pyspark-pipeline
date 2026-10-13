"""Validate class schemas, distributions, links, chronology, and empty data."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import SyntheticDataGenerator


def test_class_generation() -> None:
    """Check realistic ranges and relationships across all three DataFrames."""
    generator = SyntheticDataGenerator(100, 20, 10_000, show_progress=False)
    frames = generator.generate_all()
    customers, products, orders = (frames[key] for key in ("customers", "products", "orders"))
    assert list(customers.columns) == ["customer_id", "name", "email", "age", "city", "country", "registration_date"]
    assert list(products.columns) == ["product_id", "name", "category", "price", "stock", "rating"]
    assert list(orders.columns) == ["order_id", "customer_id", "product_id", "quantity", "order_date"]
    assert customers.age.between(18, 90).all()
    assert products.price.between(10, 500).all()
    assert products.rating.between(1, 5).all()
    assert orders.quantity.between(1, 10).all()
    assert orders.customer_id.isin(customers.customer_id).all()
    assert orders.product_id.isin(products.product_id).all()
    joined = orders.merge(customers, on="customer_id")
    assert (joined.order_date >= joined.registration_date).all()
    for key, frame in frames.items():
        other = SyntheticDataGenerator(100, 20, 10_000, show_progress=False).generate_all()[key]
        pd.testing.assert_frame_equal(frame, other)


def test_pareto_expected_split() -> None:
    """Verify top-quintile probability and observed million-order concentration."""
    generator = SyntheticDataGenerator(show_progress=False)
    probabilities = generator._customer_probabilities(100_000)
    top = np.argsort(probabilities)[-20_000:]
    assert probabilities[top].sum() == pytest.approx(0.8)
    counts = generator.rng.multinomial(1_000_000, probabilities)
    assert 0.795 < counts[top].sum() / counts.sum() < 0.805


def test_empty_and_single_customer() -> None:
    """Check empty datasets and the one-customer Pareto edge case."""
    frames = SyntheticDataGenerator(0, 0, 0, show_progress=False).generate_all()
    assert all(frame.empty for frame in frames.values())
    frames = SyntheticDataGenerator(1, 1, 10, show_progress=False).generate_all()
    assert frames["orders"].customer_id.eq(1).all()
