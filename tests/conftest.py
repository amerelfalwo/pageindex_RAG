import os
import sys
import tempfile
import pytest

# Ensure app is discoverable on python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Strictly isolate test execution from any live database and default directories BEFORE imports
_SESSION_TMP = tempfile.mkdtemp(prefix="atlas_test_isolation_")
_SESSION_DATA = os.path.join(_SESSION_TMP, "data")
_SESSION_UPLOADS = os.path.join(_SESSION_TMP, "uploads")
_SESSION_STATIC = os.path.join(_SESSION_TMP, "static")
os.makedirs(_SESSION_DATA, exist_ok=True)
os.makedirs(_SESSION_UPLOADS, exist_ok=True)
os.makedirs(_SESSION_STATIC, exist_ok=True)

os.environ["DATABASE_URL"] = ""
os.environ["POSTGRES_URL"] = ""
os.environ["POSTGRESQL_URL"] = ""
os.environ["ATLAS_DATA_DIR"] = _SESSION_DATA
os.environ["ATLAS_UPLOAD_DIR"] = _SESSION_UPLOADS


@pytest.fixture(autouse=True)
def setup_isolated_test_env(tmp_path, monkeypatch):
    """Enforces completely isolated local directories and SQLite database for each test."""
    data_dir = str(tmp_path / "data")
    upload_dir = str(tmp_path / "uploads")
    static_dir = str(tmp_path / "static")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("POSTGRES_URL", "")
    monkeypatch.setenv("POSTGRESQL_URL", "")
    monkeypatch.setenv("ATLAS_DATA_DIR", data_dir)
    monkeypatch.setenv("ATLAS_UPLOAD_DIR", upload_dir)

    yield
