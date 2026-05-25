"""
db_connector.py
---------------
SQL Server / Azure SQL connection module for the Finance Cleaning Pipeline.
Supports two modes controlled by config.csv:
  - 'demo'      : Uses a local SQLite file (no credentials needed)
  - 'sqlserver' : Uses pyodbc + SQLAlchemy to connect to SQL Server / Azure SQL

Credential values in config.csv can be written as ${ENV_VAR} placeholders.
At runtime those are resolved from environment variables (or a .env file).

Usage:
    from db_connector import get_engine
    engine = get_engine(config)
"""

import logging
import os
import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env-var placeholder resolver
# ---------------------------------------------------------------------------

def _resolve(value: str) -> str:
    """
    Replace ${VAR_NAME} placeholders with their environment variable values.
    Falls back to the placeholder string if the variable is not set.
    """
    def replacer(match):
        var_name = match.group(1)
        resolved = os.environ.get(var_name)
        if resolved is None:
            logger.warning(f"[Config] Environment variable '{var_name}' is not set.")
            return match.group(0)  # leave placeholder as-is
        return resolved

    return re.sub(r"\$\{(\w+)\}", replacer, value)


def resolve_config(config: dict) -> dict:
    """Return a new config dict with all ${VAR} placeholders expanded."""
    return {k: _resolve(v) for k, v in config.items()}


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def get_engine(config: dict) -> Engine:
    """
    Build and return a SQLAlchemy engine based on config settings.

    Parameters
    ----------
    config : dict
        Parsed key-value pairs from config.csv (placeholders already resolved)

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    # Load .env file if present (no hard dependency on python-dotenv)
    _load_dotenv()

    config = resolve_config(config)
    mode = config.get("db_mode", "demo").strip().lower()

    if mode == "demo":
        db_path = "finance_demo.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        logger.info(f"[DB] Demo mode — using SQLite at '{db_path}'")
        return engine

    elif mode == "sqlserver":
        server   = config["db_server"]
        database = config["db_name"]
        user     = config["db_user"]
        password = config["db_password"]
        driver   = config.get("db_driver", "ODBC Driver 18 for SQL Server")

        from urllib.parse import quote_plus
        conn_str = (
            f"mssql+pyodbc://{user}:{quote_plus(password)}@{server}/{database}"
            f"?driver={quote_plus(driver)}&Encrypt=yes&TrustServerCertificate=no"
        )
        engine = create_engine(conn_str, echo=False, fast_executemany=True)
        logger.info(f"[DB] SQL Server mode — connecting to {server}/{database}")

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("[DB] Connection verified successfully.")
        return engine

    else:
        raise ValueError(
            f"Unknown db_mode='{mode}' in config.csv. "
            "Valid options are 'demo' or 'sqlserver'."
        )


def table_exists(engine: Engine, schema: str, table: str) -> bool:
    """Return True if schema.table exists in the database."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    try:
        tables = inspector.get_table_names(schema=schema)
    except Exception:
        tables = inspector.get_table_names()
    return table in tables


# ---------------------------------------------------------------------------
# Minimal .env loader (no external dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.
    Skips comments (#) and blank lines. Does not override existing env vars.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    logger.info(f"[DB] Loaded environment from '{path}'")
