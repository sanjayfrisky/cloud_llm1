import logging

from google import genai
from app.core.config import settings


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


class Embedding:

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )
        self.model_name = model_name

    async def embed_chunks(
        self,
        chunks: list[str],
        model_name: str | None = None
    ) -> list[list[float]]:

        if not chunks:
            return []

        selected_model = model_name or self.model_name

        response = await self.client.aio.models.embed_content(
            model=selected_model,
            contents=chunks,
            config={
                "output_dimensionality": EMBEDDING_DIMENSION
            }
        )

        embeddings = [
            item.values for item in response.embeddings
        ]

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count mismatch")

        for vector in embeddings:
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"received {len(vector)}"
                )

        logger.info(
            "Generated %d embeddings using %s",
            len(embeddings),
            selected_model
        )

        return embeddings

    async def embed_query(
        self,
        prompt: str,
        model_name: str | None = None
    ) -> list[float]:

        if not prompt.strip():
            raise ValueError("Query cannot be empty")

        selected_model = model_name or self.model_name

        response = await self.client.aio.models.embed_content(
            model=selected_model,
            contents=prompt,
            config={
                "output_dimensionality": EMBEDDING_DIMENSION
            }
        )

        vector = response.embeddings[0].values

        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"received {len(vector)}"
            )

        return vector


embedding = Embedding()