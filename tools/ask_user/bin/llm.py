"""
LLM utilities for the ask_user tool.

Uses OpenAI client with optional LiteLLM proxy for multi-provider routing.
"""

import logging
import os

from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Module-level client (initialized lazily)
_client = None


def get_openai_client() -> OpenAI:
    """
    Get an OpenAI client, optionally configured for LiteLLM proxy.

    Environment variables:
        OPENAI_API_KEY or LLM_API_KEY: Your API key
        OPENAI_BASE_URL or LLM_BASE_URL: Proxy URL for multi-provider routing
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or LLM_API_KEY environment variable is required. "
            "Set with: export OPENAI_API_KEY='your-key' or create a .env file"
        )

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    # Note: base_url can be None for direct OpenAI API usage

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,  # None = use default OpenAI API
        max_retries=0,  # Retries handled by tenacity for better logging
    )
    return _client


@retry(
    stop=stop_after_attempt(1),  # No retries - fail fast for debugging
    wait=wait_exponential(multiplier=1, exp_base=2, max=5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_completion(model: str, messages: list, **kwargs) -> str:
    """
    LLM completion with automatic retries.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o" for LiteLLM proxy)
        messages: List of message dicts with "role" and "content"
        **kwargs: Additional arguments passed to chat.completions.create

    Returns:
        Response text from the model
    """
    client = get_openai_client()

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content
