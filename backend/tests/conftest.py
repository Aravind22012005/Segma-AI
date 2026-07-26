"""Shared fixtures for backend tests.

Fixtures build small, deterministic in-memory DataFrames shaped like the
Unified Customer View produced by backend.features.build_unified_customer_view.
The real CSVs under data/ are never read or written here -- everything is
constructed in-memory so the exact threshold math in segmentation_tool.py
can be verified precisely.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def tiering_view() -> pd.DataFrame:
    """A 24-row Unified-Customer-View slice for rule_based_tiering(), with
    AvgMonthlyBalance, MonthlyTxnFrequency and RecencyDays constructed so
    that:

      - row 0 has an extreme MonthlyTxnFrequency spike but the LOWEST
        AvgMonthlyBalance in the set. Its composite score alone clears the
        75th-percentile Priority bar (frequency dominates the z-score), but
        its balance sits below the median -- this is the AND-not-OR edge
        case: it must NOT be tagged Priority.
      - rows 1-4 have MonthlyTxnFrequency at/below the 15th-percentile
        threshold -> Dormant.
      - rows 19-23 have both high balance and high frequency -> Priority.
      - everyone else is Regular.
      - RecencyDays is a constant 10 for every row, so recency never
        contributes to the Dormant flag here (that's covered separately).
    """
    n = 24
    balance = np.linspace(2000, 50000, n)
    freq = np.linspace(10, 33, n)
    freq[0] = 300.0  # frequency spike on the lowest-balance customer
    recency = np.full(n, 10.0)

    return pd.DataFrame(
        {
            "CustomerID": [f"CUST{i:03d}" for i in range(n)],
            "AvgMonthlyBalance": balance,
            "MonthlyTxnFrequency": freq,
            "RecencyDays": recency,
        }
    )


@pytest.fixture
def zero_txn_row() -> dict:
    """A row shaped like build_unified_customer_view's output for a customer
    with zero transactions in the window: txn-derived columns (here,
    MonthlyTxnFrequency) are filled with 0 and RecencyDays is filled with
    9999, while AvgMonthlyBalance stays populated since it comes from
    customers.csv, not the transaction aggregation.
    """
    return {
        "CustomerID": "CUST_ZERO_TXN",
        "AvgMonthlyBalance": 15000.0,
        "MonthlyTxnFrequency": 0.0,
        "RecencyDays": 9999.0,
    }


@pytest.fixture
def clustering_view() -> pd.DataFrame:
    """A 30-row numeric-only DataFrame with three well-separated groups,
    shaped like a Unified-Customer-View slice, for ml_clustering()."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=[20000, 20, 600], scale=[500, 1, 5], size=(10, 3))
    group_b = rng.normal(loc=[60000, 40, 700], scale=[500, 1, 5], size=(10, 3))
    group_c = rng.normal(loc=[100000, 5, 500], scale=[500, 1, 5], size=(10, 3))
    data = np.vstack([group_a, group_b, group_c])

    df = pd.DataFrame(data, columns=["AnnualIncome", "MonthlyTxnFrequency", "CreditScore"])
    df.insert(0, "CustomerID", [f"CUST{i:03d}" for i in range(len(df))])
    return df


@pytest.fixture(scope="module")
def app_client():
    """A TestClient wrapping the real FastAPI app (backend.main.app), used for
    end-to-end tests that go through the actual /api/chat pipeline (Planner ->
    Executor -> response) instead of calling tool functions directly.

    Uses the `with` context-manager form deliberately: FastAPI/Starlette only
    fires @app.on_event("startup") handlers (here, loading customers.csv /
    transactions.csv / products.csv and building the unified view into
    backend.main._STATE) when TestClient is used as a context manager --
    `TestClient(app)` on its own does NOT trigger startup, and /api/chat would
    fail with a KeyError on _STATE["view"] without it.

    Scoped to the test module (not function): booting the app re-reads the
    CSVs and rebuilds the unified view every time, which is the expensive
    part. Session isolation in the real app is per session_id (see
    backend.main.SESSIONS), not per TestClient instance, so a single shared
    client across a module is safe as long as each test uses its own fresh
    session_id (see the `session_id` fixture below) -- that's what actually
    proves auto-run works with no prior manual segmentation, not spinning up
    a new process per test.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def session_id() -> str:
    """A fresh, unique session_id for each test, so tests against `app_client`
    prove the executor auto-runs segmentation with no prior state in that
    session -- not reusing a session another test already segmented."""
    return str(uuid.uuid4())
