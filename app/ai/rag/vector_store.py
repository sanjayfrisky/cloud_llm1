import logging
from uuid import UUID
from datetime import datetime

import asyncpg

from app.utils.ids import generate_uuid


logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768


class VectorStore:

    def __init__(self):
        self.pool = None

    def set_pool(self, pool: asyncpg.Pool):
        """Set the PostgreSQL connection pool created by FastAPI."""
        self.pool = pool
        logger.info("Vector store connected to PostgreSQL")

    def _get_pool(self):
        if self.pool is None:
            raise RuntimeError(
                "Vector store is not connected to PostgreSQL"
            )
        return self.pool

    @staticmethod
    def _parse_datetime(value):
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    async def get_document_by_hash(
        self,
        file_hash: str,
        embedding_model_name: str | None = None
    ) -> dict | None:

        pool = self._get_pool()

        async with pool.acquire() as connection:

            if embedding_model_name:
                row = await connection.fetchrow(
                    """
                    SELECT *
                    FROM documents
                    WHERE file_hash = $1
                    AND embedding_model = $2
                    LIMIT 1
                    """,
                    file_hash,
                    embedding_model_name
                )
            else:
                row = await connection.fetchrow(
                    """
                    SELECT *
                    FROM documents
                    WHERE file_hash = $1
                    LIMIT 1
                    """,
                    file_hash
                )

            if row:
                result = dict(row)
                result["document_id"] = str(
                    result["document_id"]
                )
                return result

            return None

    async def get_document_by_id(
        self,
        document_id: str
    ) -> dict | None:

        pool = self._get_pool()

        async with pool.acquire() as connection:

            row = await connection.fetchrow(
                """
                SELECT *
                FROM documents
                WHERE document_id = $1
                """,
                UUID(document_id)
            )

            if row:
                result = dict(row)
                result["document_id"] = str(
                    result["document_id"]
                )
                return result

            return None

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]],
        embedding_model_name: str | None = None
    ) -> None:

        pool = self._get_pool()

        # Validate embedding model consistency
        document_model = document_record["embedding_model"]

        if (
            embedding_model_name
            and document_model != embedding_model_name
        ):
            raise ValueError(
                "Embedding model name mismatch"
            )

        # Validate chunk and embedding counts
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal"
            )

        # Validate embedding dimensions
        for vector in embeddings:
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"received {len(vector)}"
                )

        document_id = UUID(
            str(document_record["document_id"])
        )

        async with pool.acquire() as connection:

            async with connection.transaction():

                # 1. Insert document metadata
                await connection.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        filename,
                        file_hash,
                        file_size_bytes,
                        uploaded_at,
                        ingested_at,
                        ingestion_status,
                        chunk_count,
                        chunking_version,
                        embedding_model
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    document_id,
                    document_record["filename"],
                    document_record["file_hash"],
                    int(document_record["file_size_bytes"]),
                    self._parse_datetime(
                        document_record["uploaded_at"]
                    ),
                    self._parse_datetime(
                        document_record["ingested_at"]
                    ),
                    document_record["ingestion_status"],
                    int(document_record["chunk_count"]),
                    document_record["chunking_version"],
                    embedding_model_name or document_model
                )

                # 2. Prepare chunk records
                chunk_records = []

                for chunk, vector in zip(chunks, embeddings):

                    chunk_id = UUID(
                        str(chunk.get("id") or generate_uuid())
                    )

                    chunk_records.append((
                        chunk_id,
                        document_id,
                        chunk["text"],
                        vector,
                        document_record["filename"],
                        int(chunk["chunk_index"]),
                        int(chunk["page_number"])
                    ))

                # 3. Insert chunks and embeddings
                if chunk_records:

                    await connection.executemany(
                        """
                        INSERT INTO document_chunks (
                            id,
                            document_id,
                            content,
                            embedding,
                            filename,
                            chunk_index,
                            page_number
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7)
                        """,
                        chunk_records
                    )

        logger.info(
            "Stored document '%s' with %d chunks in PostgreSQL",
            document_record["filename"],
            len(chunks)
        )

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:

        pool = self._get_pool()

        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected query embedding dimension "
                f"{EMBEDDING_DIMENSION}, received "
                f"{len(query_embedding)}"
            )

        if n_results < 1:
            raise ValueError("n_results must be greater than zero")

        async with pool.acquire() as connection:

            if document_id:

                rows = await connection.fetch(
                    """
                    SELECT
                        id,
                        document_id,
                        content,
                        filename,
                        chunk_index,
                        page_number,
                        embedding <=> $1 AS distance
                    FROM document_chunks
                    WHERE document_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                    """,
                    query_embedding,
                    UUID(document_id),
                    n_results
                )

            else:

                rows = await connection.fetch(
                    """
                    SELECT
                        id,
                        document_id,
                        content,
                        filename,
                        chunk_index,
                        page_number,
                        embedding <=> $1 AS distance
                    FROM document_chunks
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    query_embedding,
                    n_results
                )

            results = []

            for row in rows:

                results.append({
                    "id": str(row["id"]),
                    "document": row["content"],
                    "metadata": {
                        "document_id": str(row["document_id"]),
                        "filename": row["filename"],
                        "chunk_index": row["chunk_index"],
                        "page_number": row["page_number"]
                    },
                    "distance": float(row["distance"])
                })

            logger.info(
                "Retrieved %d similar chunks from PostgreSQL",
                len(results)
            )

            return results


vector_store = VectorStore()