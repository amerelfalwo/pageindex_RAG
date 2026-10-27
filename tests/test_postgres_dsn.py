import pytest
from psycopg2.extensions import parse_dsn
from core.postgres_db import normalize_postgres_dsn


def test_normalize_postgres_dsn_standard():
    dsn = "postgresql://user:secret123@db.supabase.co:5432/postgres"
    norm = normalize_postgres_dsn(dsn)
    parsed = parse_dsn(norm)
    assert parsed["user"] == "user"
    assert parsed["password"] == "secret123"
    assert parsed["host"] == "db.supabase.co"
    assert parsed["port"] == "5432"
    assert parsed["dbname"] == "postgres"


def test_normalize_postgres_dsn_with_at_symbol_in_password():
    """
    Prevents the libpq error:
    invalid integer value "<host>:<port>" for connection option "port"
    when password contains an unencoded '@' character.
    """
    dsn = "postgresql://postgres.myproject:my@secret@atlas123asdqwat-1.pooler.supabase.com:5432/postgres"
    norm = normalize_postgres_dsn(dsn)
    parsed = parse_dsn(norm)
    assert parsed["user"] == "postgres.myproject"
    assert parsed["password"] == "my@secret"
    assert parsed["host"] == "atlas123asdqwat-1.pooler.supabase.com"
    assert parsed["port"] == "5432"
    assert parsed["dbname"] == "postgres"


def test_normalize_postgres_dsn_with_special_characters_in_password():
    dsn = "postgresql://postgres.ref:p@ss:w0rd!#%@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    norm = normalize_postgres_dsn(dsn)
    parsed = parse_dsn(norm)
    assert parsed["user"] == "postgres.ref"
    assert parsed["password"] == "p@ss:w0rd!#%"
    assert parsed["host"] == "aws-0-eu-central-1.pooler.supabase.com"
    assert parsed["port"] == "6543"
    assert parsed["dbname"] == "postgres"
    assert parsed["sslmode"] == "require"


def test_normalize_postgres_dsn_handles_postgres_scheme_and_quotes():
    dsn = " 'postgres://postgres:pass123@localhost:5432/test_db' "
    norm = normalize_postgres_dsn(dsn)
    assert norm.startswith("postgresql://")
    parsed = parse_dsn(norm)
    assert parsed["user"] == "postgres"
    assert parsed["password"] == "pass123"
    assert parsed["host"] == "localhost"
    assert parsed["port"] == "5432"
    assert parsed["dbname"] == "test_db"
