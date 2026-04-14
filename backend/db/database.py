import psycopg2
from psycopg2.extras import RealDictCursor
from config.settings import settings


def get_db():
    conn = psycopg2.connect(
        dsn=settings.DATABASE_URL,
        cursor_factory=RealDictCursor,
    )
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()
