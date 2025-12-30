"""Unified LLM client for OpenRouter and Ollama."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OLLAMA_API_URL, MODEL_TIMEOUT


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = MODEL_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter or Ollama.

    Args:
        model: Model identifier (e.g., "openai/gpt-5.1" or "ollama/llama3")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    is_ollama = model.startswith("ollama/")
    
    if is_ollama:
        # Route to Ollama
        api_url = OLLAMA_API_URL
        api_key = "ollama"  # Dummy key for Ollama
        # Strip "ollama/" prefix for the actual API call
        actual_model = model.replace("ollama/", "", 1)
        headers = {
            "Content-Type": "application/json",
        }
    else:
        # Route to OpenRouter
        api_url = OPENROUTER_API_URL
        api_key = OPENROUTER_API_KEY
        actual_model = model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    payload = {
        "model": actual_model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        import traceback
        error_type = type(e).__name__
        print(f"Error querying model {model} ({actual_model}): {error_type}: {e}")
        # print(traceback.format_exc()) # Uncomment for full trace
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
