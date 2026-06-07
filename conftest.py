"""Pytest configuration.

Having this file at the project root puts the root on ``sys.path`` so ``src``
imports resolve, and sets safe defaults for the API tests (mock data, no
scheduler, no auth, throwaway snapshot path).
"""
import os
import tempfile

import pytest

os.environ.setdefault("USE_MOCK_DATA", "true")
os.environ.setdefault("MOCK_N_USERS", "8")
os.environ.setdefault("ENABLE_SCHEDULER", "false")
os.environ.setdefault("BOOTSTRAP_ON_START", "true")
os.environ.setdefault("RECOMMENDER_API_KEY", "")
os.environ.setdefault(
    "INDEX_SNAPSHOT_PATH",
    os.path.join(tempfile.gettempdir(), "wasla_test_snapshot.pkl"),
)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    snapshot = os.environ["INDEX_SNAPSHOT_PATH"]
    if os.path.exists(snapshot):
        os.remove(snapshot)

    from src.api.app import app

    with TestClient(app) as test_client:
        yield test_client
