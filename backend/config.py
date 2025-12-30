"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "ollama/gemma3:27b",
    "ollama/gpt-oss:120b",
    "ollama/olmo-3:32b",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "ollama/gpt-oss:120b"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ollama API endpoint (OpenAI compatible)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.1.123:11434/v1/chat/completions")

# Model used for generating conversation titles
# TITLE_GENERATION_MODEL = "google/gemini-2.5-flash"
TITLE_GENERATION_MODEL = "ollama/gpt-oss:120b"

# Request timeout in seconds (default 120s) - Increase for large local models
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "300.0"))

# Data directory for conversation storage
DATA_DIR = "data/conversations"
