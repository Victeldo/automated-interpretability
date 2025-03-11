import asyncio
import contextlib
import os
import random
import traceback
import logging
from asyncio import Semaphore
from functools import wraps
from typing import Any, Callable, Optional

import httpx
import orjson

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_api_error(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        response = err.response
        try:
            error_data = response.json().get("error", {})
        except Exception:
            error_data = {}
        error_message = error_data.get("message", "")
        
        if response.status_code in [400, 404, 415]:
            if error_data.get("type") == "idempotency_error":
                logger.warning(f"Retrying after idempotency error: {error_message} ({response.url})")
                return True
            return False  # Invalid request, do not retry
        else:
            logger.warning(f"Retrying after API error: {error_message} ({response.url})")
            return True

    elif isinstance(err, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError)):
        logger.warning(f"Retrying after connection/timeout/read error... ({err.request.url})")
        return True
    
    logger.error(f"Retrying after an unexpected error: {repr(err)}")
    traceback.print_tb(err.__traceback__)
    return True

def exponential_backoff(
    retry_on: Callable[[Exception], bool] = lambda err: True
) -> Callable[[Callable], Callable]:
    """
    Exponential backoff retry decorator with jitter.
    """
    init_delay_s = 1.0
    max_delay_s = 10.0
    max_tries = 200
    backoff_multiplier = 2.0
    jitter = 0.2

    def decorate(f: Callable) -> Callable:
        assert asyncio.iscoroutinefunction(f)

        @wraps(f)
        async def f_retry(*args: Any, **kwargs: Any) -> Any:
            delay_s = init_delay_s
            for i in range(max_tries):
                try:
                    return await f(*args, **kwargs)
                except Exception as err:
                    if not retry_on(err) or i == max_tries - 1:
                        raise
                    jittered_delay = random.uniform(delay_s * (1 - jitter), delay_s * (1 + jitter))
                    await asyncio.sleep(jittered_delay)
                    delay_s = min(delay_s * backoff_multiplier, max_delay_s)

        return f_retry

    return decorate

API_KEY = os.getenv("ANTHROPIC_API_KEY")
assert API_KEY, "Please set the ANTHROPIC_API_KEY environment variable"
API_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
}
BASE_API_URL = "https://api.anthropic.com/v1/messages"

class ApiClient:
    """Performs inference using the Anthropic API."""
    
    def __init__(
        self,
        model_name: str,
        max_concurrent: Optional[int] = None,
        cache: bool = False,
    ):
        self.model_name = model_name
        self._concurrency_check = Semaphore(max_concurrent) if max_concurrent else None
        self._cache = {} if cache else None

    @exponential_backoff(retry_on=is_api_error)
    async def make_request(
        self, timeout_seconds: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        import copy
        request_data = copy.deepcopy(kwargs)

        # Convert "prompt" to Anthropic messages format
        if "prompt" in request_data and "messages" not in request_data:
            prompt = request_data.pop("prompt")
            request_data["messages"] = [{"role": "user", "content": prompt}]

        request_data["model"] = self.model_name

        # Extract system messages and set them as top-level "system"
        system_messages = [m for m in request_data.get("messages", []) if m.get("role") == "system"]
        if system_messages:
            request_data["system"] = "\n".join(m.get("content", "") for m in system_messages)
            request_data["messages"] = [m for m in request_data["messages"] if m.get("role") != "system"]
        
        # Ensure correct message formatting
        for msg in request_data.get("messages", []):
            if isinstance(msg.get("content"), str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
        
        # Set max_tokens if missing
        request_data.setdefault("max_tokens", 256)

        # Remove unsupported OpenAI parameters
        openai_params = ["logprobs", "logit_bias", "presence_penalty", "frequency_penalty", "best_of", "echo", "n", "stop", "stream"]
        for param in openai_params:
            if param in request_data:
                # logger.warning(f"Removing unsupported OpenAI parameter '{param}' for Anthropic API")
                request_data.pop(param, None)
        
        # Rename stop to stop_sequences
        if "stop" in request_data:
            request_data["stop_sequences"] = request_data.pop("stop")
        
        if self._cache is not None:
            key = orjson.dumps(request_data)
            if key in self._cache:
                return self._cache[key]


        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check:
                await stack.enter_async_context(self._concurrency_check)
            http_client = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout_seconds))
            response = await http_client.post(BASE_API_URL, headers=API_HTTP_HEADERS, json=request_data)

        # response.raise_for_status()
        response_json = response.json()
        
        # Transform response into OpenAI-like format
        # Important: preserve the exact response format including special delimiters
        response_text = "".join(block.get("text", "") for block in response_json.get("content", []) if block.get("type") == "text")
        
        # Create a proper message field for HARMONY_V4 format
        message = {"role": "assistant", "content": response_text}
        
        transformed_response = {
            "choices": [{
                "text": response_text,
                "message": message,
                "logprobs": None,
                "index": 0,
                "finish_reason": response_json.get("stop_reason", "stop"),
                # Store raw response for debugging
                "raw_response": response_json
            }],
            "usage": response_json.get("usage", {})
        }

        if self._cache is not None:
            self._cache[key] = transformed_response
        
        return transformed_response

if __name__ == "__main__":
    async def main() -> None:
        client = ApiClient(model_name="claude-3-opus-20240229", max_concurrent=1)
        response = await client.make_request(prompt="Why did the chicken cross the road?", max_tokens=9, temperature=0.7)
        print(response)
    asyncio.run(main())