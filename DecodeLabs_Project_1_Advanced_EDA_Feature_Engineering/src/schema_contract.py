"""Optional Pandera runtime contract inspired by the DecodeLabs brief."""
from __future__ import annotations

try:
    import pandera.pandas as pa
    from pandera import Check, Column, DataFrameSchema
except ImportError as exc:
    raise ImportError("Install optional dependency with: pip install pandera") from exc

ORDER_SCHEMA = DataFrameSchema(
    {
        "OrderID": Column(str, nullable=False, unique=True),
        "Date": Column(pa.DateTime, nullable=False),
        "CustomerID": Column(str, nullable=False),
        "Product": Column(str, nullable=False),
        "Quantity": Column(int, Check.in_range(1, 5), nullable=False),
        "UnitPrice": Column(float, Check.gt(0), coerce=True, nullable=False),
        "ShippingAddress": Column(str, nullable=False),
        "PaymentMethod": Column(str, nullable=False),
        "OrderStatus": Column(str, nullable=False),
        "TrackingNumber": Column(str, nullable=False, unique=True),
        "ItemsInCart": Column(int, Check.gt(0), nullable=False),
        "CouponCode": Column(str, nullable=True),
        "ReferralSource": Column(str, nullable=False),
        "TotalPrice": Column(float, Check.gt(0), coerce=True, nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_with_pandera(df):
    return ORDER_SCHEMA.validate(df, lazy=True)
