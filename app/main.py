from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import chat
from app.ai.rag.pipeline import rag_pipeline
from app.ai.rag.vector_store import vector_store
from app.db.database import create_database_pool
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):

    db_pool = None

    try:
        # 1. Connect to PostgreSQL
        logger.info("Connecting to PostgreSQL...")

        db_pool = await create_database_pool()
        app.state.db_pool = db_pool

        # 2. Assign connection pool to VectorStore
        vector_store.set_pool(db_pool)

        logger.info("PostgreSQL connected successfully")
        logger.info("Vector store connected successfully")

        # 3. Run document ingestion
        logger.info("App startup: data ingestion is running...")

        await rag_pipeline.data_ingestion(UPLOADS_DIR)

        logger.info("Data ingestion process finished")

        yield

    except Exception:
        logger.exception("Application startup or runtime failed")
        raise

    finally:
        # 4. Close PostgreSQL connection
        if db_pool is not None:
            vector_store.pool = None
            await db_pool.close()
            logger.info("PostgreSQL connection pool closed")

        logger.info("App shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def landing_page():
    return {"message": "Hi this is your backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(chat.router)