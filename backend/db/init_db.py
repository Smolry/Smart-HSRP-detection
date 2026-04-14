import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DB_NAME = "smart_hsrp"
DB_USER = "postgres"
DB_PASSWORD = "aniket8087160135"
DB_HOST = "localhost"
DB_PORT = 5432

SCHEMA_PATH = "backend/db/schema.sql"   # path to your dumped schema

# ─────────────────────────────────────────────
# CREATE DATABASE IF NOT EXISTS
# ─────────────────────────────────────────────

def create_database():
    conn = psycopg2.connect(
        dbname="postgres",  # connect to default DB
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (DB_NAME,)
    )

    exists = cur.fetchone()
    if not exists:
        print(f"Creating database '{DB_NAME}'...")
        cur.execute(f'CREATE DATABASE "{DB_NAME}"')
    else:
        print(f"Database '{DB_NAME}' already exists.")

    cur.close()
    conn.close()


# ─────────────────────────────────────────────
# APPLY SCHEMA
# ─────────────────────────────────────────────

def apply_schema():
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cur = conn.cursor()

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    print("Applying schema...")
    cur.execute(schema_sql)

    conn.commit()
    cur.close()
    conn.close()

    print("Schema applied successfully.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    create_database()
    apply_schema()
    print("Database setup complete.")
