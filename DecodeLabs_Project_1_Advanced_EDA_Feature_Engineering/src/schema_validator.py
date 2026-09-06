"""
schema_validator.py
----------------------
A lightweight, dependency-free "data contract" validator inspired by
the PHASE 3 brief (runtime structural contracts, e.g. Pandera). It
treats the cleaned dataset as a critical interface: every column has
an expected dtype and, where relevant, a statistical boundary.

Validation is "lazy" (like Pandera's lazy=True): it does not stop at
the first failure. It collects every failure into a single report so
the whole dataset can be diagnosed in one pass.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnRule:
    name: str
    dtype_check: str  # "numeric", "datetime", "categorical", "string"
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list | None = None
    nullable: bool = False


@dataclass
class ValidationResult:
    passed: bool
    failures: list = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.failures)


def validate_schema(df: pd.DataFrame, rules: list[ColumnRule]) -> ValidationResult:
    """Validate a dataframe against a list of ColumnRule contracts."""
    failures = []

    for rule in rules:
        if rule.name not in df.columns:
            failures.append({"column": rule.name, "check": "presence", "detail": "Column missing from dataframe."})
            continue

        col = df[rule.name]

        # Nullability
        if not rule.nullable and col.isna().any():
            failures.append({
                "column": rule.name, "check": "nullability",
                "detail": f"{int(col.isna().sum())} unexpected null value(s) found.",
            })

        # Dtype
        if rule.dtype_check == "numeric" and not pd.api.types.is_numeric_dtype(col):
            failures.append({"column": rule.name, "check": "dtype", "detail": "Expected numeric dtype."})
        elif rule.dtype_check == "datetime" and not pd.api.types.is_datetime64_any_dtype(col):
            failures.append({"column": rule.name, "check": "dtype", "detail": "Expected datetime dtype."})

        # Range
        if rule.dtype_check == "numeric":
            if rule.min_value is not None and (col.dropna() < rule.min_value).any():
                failures.append({"column": rule.name, "check": "min_bound",
                                  "detail": f"Value(s) below minimum {rule.min_value}."})
            if rule.max_value is not None and (col.dropna() > rule.max_value).any():
                failures.append({"column": rule.name, "check": "max_bound",
                                  "detail": f"Value(s) above maximum {rule.max_value}."})

        # Allowed categorical values
        if rule.allowed_values is not None:
            bad_values = set(col.dropna().unique()) - set(rule.allowed_values)
            if bad_values:
                failures.append({"column": rule.name, "check": "allowed_values",
                                  "detail": f"Unexpected value(s): {sorted(bad_values)}"})

    return ValidationResult(passed=(len(failures) == 0), failures=failures)
