import os
import logging
import asyncpg

from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")


async def init_connection(connection):
    await register_vector(connection)
    logger.info("pgvector registered successfully")


async def create_database_pool():

    if not DATABASE_URL:
        logger.error("DATABASE_URL is missing in .env")
        raise ValueError("DATABASE_URL is not configured")

    try:
        logger.info("Connecting to PostgreSQL database...")

        pool = await asyncpg.create_pool(
            DATABASE_URL,
            init=init_connection
        )

        logger.info("PostgreSQL connection pool created successfully")

        return pool

    except Exception:
        logger.exception("Failed to connect to PostgreSQL database")
        raise

    