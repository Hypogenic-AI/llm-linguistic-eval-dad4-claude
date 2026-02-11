"""Configuration for multilingual LLM evaluation experiments."""
import os

SEED = 42
SAMPLES_PER_LANGUAGE = 50  # number of items sampled per language per task

# API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

# Models
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "base_url": None,
        "api_key": OPENAI_API_KEY,
    },
    "claude-sonnet-4": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_KEY,
    },
}

# Languages for each task
MGSM_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "ja": "Japanese",
    "sw": "Swahili",
    "bn": "Bengali",
}

BELEBELE_LANGUAGES = {
    "eng": "English",
    "zho_Hans": "Chinese",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "rus_Cyrl": "Russian",
    "jpn_Jpan": "Japanese",
    "swh_Latn": "Swahili",
    "hin_Deva": "Hindi",
}

# Language resource levels (for analysis)
LANGUAGE_RESOURCE_LEVEL = {
    "English": "high",
    "Chinese": "high",
    "German": "high",
    "French": "high",
    "Russian": "medium",
    "Japanese": "medium",
    "Swahili": "low",
    "Bengali": "low",
    "Hindi": "medium",
}

# Prompting strategies
STRATEGIES = ["direct", "self_translate", "english_cot"]

# API settings
MAX_TOKENS = 512
TEMPERATURE = 0
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # seconds

# Paths
DATA_DIR = "datasets"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
