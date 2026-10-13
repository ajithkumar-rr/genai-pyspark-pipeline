"""Check Spark revenue arithmetic, missing months, zero baselines, and settings."""

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.spark_analytics import SalesAnalytics


def test_sales_analytics(tmp_path: Path) -> None:
    """Validate analytics against a hand-calculated four-month Parquet fixture."""
    pd.DataFrame({"product_id": [1, 2], "category": ["Books", "Sports"],
                  "price": [10.0, 20.0]}).to_parquet(tmp_path / "products.parquet", index=False)
    pd.DataFrame({
        "customer_id": [1, 2, 1, 2], "product_id": [1, 2, 2, 1],
        "quantity": [2, 1, 2, 8],
        "order_date": pd.to_datetime(["2025-01-10", "2025-01-20", "2025-03-10", "2025-04-10"]),
    }).to_parquet(tmp_path / "orders.parquet", index=False)
    analytics = SalesAnalytics(master="local[2]")
    try:
        orders = analytics.load_parquet(tmp_path / "orders.parquet")
        products = analytics.load_parquet(tmp_path / "products.parquet")
        top = analytics.top_customers_by_revenue(orders, products).collect()
        assert [(row.customer_id, row.total_spend) for row in top] == [(2, Decimal("100")), (1, Decimal("60"))]
        categories = {row.category: row for row in analytics.sales_by_category(orders, products).collect()}
        assert categories["Books"].total_revenue == Decimal("100")
        assert categories["Books"].units_sold == 10
        assert categories["Sports"].total_revenue == Decimal("60")
        trends = analytics.monthly_trends(orders, products).collect()
        assert [row.total_revenue for row in trends] == [40, 0, 40, 80]
        assert [row.revenue_growth_pct for row in trends] == [None, -100, None, 100]
        empty = orders.limit(0)
        assert analytics.monthly_trends(empty, products).count() == 0
        assert analytics.top_customers_by_revenue(empty, products).count() == 0
        with pytest.raises(ValueError, match="positive integer"):
            analytics.top_customers_by_revenue(orders, products, n=0)
        with pytest.raises(ValueError, match="Missing required"):
            analytics.sales_by_category(orders.drop("quantity"), products)
        assert analytics.spark.conf.get("spark.sql.adaptive.enabled") == "true"
        conf = analytics.spark.sparkContext.getConf()
        assert conf.get("spark.driver.memory") == "4g"
        assert conf.get("spark.serializer") == "org.apache.spark.serializer.KryoSerializer"
    finally:
        analytics.stop()
