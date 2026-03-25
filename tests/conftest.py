"""Shared fixtures and Streamlit mock for testing src/app.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock streamlit before importing src.app (app.py runs st.* at module level)
# ---------------------------------------------------------------------------
mock_st = MagicMock()


# Make cache decorators pass-through
def _passthrough_decorator(func=None, **kwargs):
    if func is not None:
        return func
    return lambda f: f


mock_st.cache_data = _passthrough_decorator
mock_st.cache_resource = _passthrough_decorator


def _make_ctx():
    """Create a MagicMock that works as a context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


# columns() returns N context-manager mocks based on the argument
def _mock_columns(spec, **kwargs):
    n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [_make_ctx() for _ in range(n)]


mock_st.columns = _mock_columns
mock_st.tabs.return_value = [_make_ctx() for _ in range(4)]

# Widgets must return concrete values (used in comparisons at module level)
mock_st.number_input.return_value = 0
mock_st.selectbox.return_value = "United Kingdom"
mock_st.radio.return_value = "Look up existing customer by ID"
mock_st.slider.return_value = 5
mock_st.button.return_value = False

sys.modules["streamlit"] = mock_st

# Add repo root to sys.path so `import src.app` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Now safe to import
import src.app as app_module  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def app():
    return app_module


@pytest.fixture(scope="session")
def classify_segment():
    return app_module.classify_segment


@pytest.fixture(scope="session")
def clv_data():
    import pandas as pd

    return pd.read_csv(REPO_ROOT / "data" / "processed" / "clv_final.csv")


@pytest.fixture(scope="session")
def model_and_encoders():
    import joblib

    clf = joblib.load(REPO_ROOT / "models" / "purchase_propensity_model.pkl")
    encoders = joblib.load(REPO_ROOT / "models" / "label_encoders.pkl")
    return clf, encoders
