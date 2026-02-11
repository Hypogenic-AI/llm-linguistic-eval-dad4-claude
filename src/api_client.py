"""API client for calling LLM models with retry logic."""
import time
import logging
import openai
from src.config import MODELS, MAX_TOKENS, TEMPERATURE, MAX_RETRIES, RETRY_BASE_DELAY

logger = logging.getLogger(__name__)


def get_client(model_name: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client for the given model."""
    cfg = MODELS[model_name]
    kwargs = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return openai.OpenAI(**kwargs)


def call_model(model_name: str, prompt: str, client: openai.OpenAI = None) -> str:
    """Call a model with retry logic. Returns the response text."""
    if client is None:
        client = get_client(model_name)
    cfg = MODELS[model_name]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(f"Rate limit/timeout for {model_name}, retrying in {delay}s: {e}")
            time.sleep(delay)
        except openai.APIError as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(f"API error for {model_name}, retrying in {delay}s: {e}")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error for {model_name}: {e}")
            return f"ERROR: {e}"

    logger.error(f"Max retries exceeded for {model_name}")
    return "ERROR: max retries exceeded"
