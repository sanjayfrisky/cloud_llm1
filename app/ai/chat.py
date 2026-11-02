import asyncio
import logging
import random
from collections.abc import AsyncIterator
from app.ai.rag.pipeline import rag_pipeline
from app.ai.models import models
from app.ai.prompts.gemini_prompt import RAG_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "I couldn't find anything about that in the Virat Kohli document."


class ChatService:

    @staticmethod
    async def retry_with_jitter(func, *args, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0, **kwargs):
        """
        Executes an async callable with Exponential Backoff and Full Jitter.

        Formula:
            backoff = min(max_delay, base_delay * (2 ** attempt))
            sleep_time = random.uniform(0, backoff)
        """
        for attempt in range(1, max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as error:
                if attempt >= max_retries:
                    logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, error)
                    raise
                backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
                sleep_time = random.uniform(0, backoff)
                logger.info("Attempt %d/%d failed: %s — retrying in %.2fs", attempt, max_retries, error, sleep_time)
                await asyncio.sleep(sleep_time)

    @staticmethod
    async def _open_stream(model_func, prompt: str) -> tuple[str, AsyncIterator[str]]:
        """Open a model stream and return the first token + remaining stream.

        Raises RuntimeError if the model yields nothing or an empty first token.
        """
        stream = model_func(prompt)
        if asyncio.iscoroutine(stream):
            stream = await stream
        try:
            first_token = await anext(stream)
        except Exception as e:
            raise RuntimeError(f"No output from model: {e}")
        if not first_token or not first_token.strip():
            raise RuntimeError("Model produced an empty first token")
        return first_token, stream

    @staticmethod
    async def chat(user_prompt: str) -> AsyncIterator[str]:
        """
        Retrieves relevant context, then streams the raw answer text
        from the first model in the fallback chain that responds.
        """

        # 1. Retrieve relevant chunks from the vector store
        retrieval = await rag_pipeline.retrieve(user_prompt)
        if not retrieval["results"]:
            yield NO_CONTEXT_MESSAGE
            return

        # 2. Build the RAG prompt
        context, sources = rag_pipeline.build_context(retrieval["results"])
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_prompt)

        # 3. Try each model in the fallback chain
        fallback_chain = models.get_fallback_chain()

        for model_name, model_func in fallback_chain:
            logger.info("Trying provider: %s", model_name)

            try:
                first_token, stream = await ChatService.retry_with_jitter(
                    ChatService._open_stream, model_func, prompt
                )
            except Exception as error:
                logger.warning("%s failed: %s — moving to next provider", model_name, error)
                continue

            # Stream the answer text directly
            yield first_token
            try:
                async for token in stream:
                    yield token
            except Exception as error:
                logger.error("%s failed mid-stream: %s", model_name, error)
                return

            logger.info("Answered by %s", model_name)
            return

        # All models failed
        yield NO_CONTEXT_MESSAGE


chat_service = ChatService()
