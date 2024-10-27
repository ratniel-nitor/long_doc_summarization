import os

import time
import json
import instructor
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from typing import Any, Sequence
from datetime import datetime, timezone
from pydantic import ValidationError, Field
from llama_index.llms.gemini import Gemini as BaseGemini
from google.api_core.exceptions import ResourceExhausted
from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from llama_index.llms.cerebras import Cerebras as LlamaIndexCerebras
from llama_index.llms.groq import Groq as LlamaIndexGroq

load_dotenv()

# Configure logging
logger.add("model_loaders.log", rotation="10 MB", level="DEBUG")


class ModelInitializer:
    @staticmethod
    def initialize_groq(
        model_name: str = "llama-3.1-70b-versatile",
        use_instructor: bool = False,
        use_llamaindex: bool = False,
    ) -> Any:
        """
        Initialize the Groq model (implementation not provided).

        Args:
            model_name (str): The name of the model to initialize.
            use_instructor (bool): Whether to use the instructor with the model.
            use_llamaindex (bool): Whether to use LlamaIndex with the model.

        Returns:
            Any: The initialized model or instructor-wrapped model.
        """
        from groq import Groq

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        logger.info(f"Initialized Groq client")

        if use_instructor:
            groq_client = instructor.from_groq(groq_client, mode=instructor.Mode.TOOLS)
            logger.info("Using instructor with Groq")

        if use_llamaindex:
            from llama_index.llms.groq import Groq as LlamaIndexGroq

            logger.info("Using LlamaIndex with Groq")
            groq_client = RateLimitedGroq(
                model=model_name, api_key=os.getenv("GROQ_API_KEY")
            )
            return groq_client

        return groq_client

    @staticmethod
    def initialize_gemini(
        model_name: str = "models/gemini-1.5-flash-002",
        use_instructor: bool = False,
        use_llamaindex: bool = False,
    ) -> Any:
        """
        Initialize the Gemini model with optional instructor and LlamaIndex integration.

        Args:
            model_name (str): The name of the model to initialize.
            use_instructor (bool): Whether to use the instructor with the model.
            use_llamaindex (bool): Whether to use LlamaIndex with the model.

        Returns:
            Any: The initialized model or instructor-wrapped model.
        """
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # custom rate-limited Gemini instance
        client = RateLimitedGemini(api_key=google_api_key, model=model_name)
        logger.info(f"Initialized rate-limited Gemini model: {model_name}")

        if use_instructor:
            logger.info("Using instructor with Gemini")
            return instructor.from_gemini(
                client=client._model,  # fetch the generative-ai model
                mode=instructor.Mode.GEMINI_JSON,
            )

        if use_llamaindex:
            logger.info("Using LlamaIndex with Gemini")
            return client

        return client._model

    @staticmethod
    def initialize_cerebras(
        model_name: str = "llama3.1-70b",
        use_instructor: bool = False,
        use_llamaindex: bool = False,
    ) -> Any:
        """
        Initialize the Cerebras model with optional LlamaIndex integration.

        Args:
            model_name (str): The name of the model to initialize.
            use_instructor (bool): Whether to use the instructor with the model (not implemented).
            use_llamaindex (bool): Whether to use LlamaIndex with the model.

        Returns:
            Any: The initialized model or LlamaIndex-wrapped model.

        Raises:
            ValueError: If CEREBRAS_API_KEY is not found in environment variables.
        """
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
        if not cerebras_api_key:
            logger.error("CEREBRAS_API_KEY not found in environment variables")
            raise ValueError("CEREBRAS_API_KEY not found in environment variables")

        if use_llamaindex:
            try:
                client = RateLimitedCerebras(  # Changed from LlamaIndexCerebras to RateLimitedCerebras
                    model=model_name, api_key=cerebras_api_key
                )
                logger.info(
                    f"Initialized rate-limited Cerebras model with LlamaIndex: {model_name}"
                )
                return client
            except Exception as e:
                logger.exception(f"Error initializing Cerebras model: {str(e)}")
                raise

        # Default to raw Cerebras client if LlamaIndex is not used
        from cerebras.cloud.sdk import Cerebras

        client = Cerebras(api_key=cerebras_api_key)
        logger.info(f"Initialized Cerebras client with model: {model_name}")
        return client


class RateLimiterStore:
    """Store rate limiter data for Gemini models for daily resets."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.store_dir = Path("rate_limiter_data")
        self.store_file = self.store_dir / f"{model_name.replace('/', '_')}_limits.json"
        self._initialize_store()

    def _initialize_store(self):
        """Create storage directory and file if they don't exist"""
        self.store_dir.mkdir(exist_ok=True)
        if not self.store_file.exists():
            self._save_data(
                {
                    "last_reset_timestamp": time.time(),
                    "daily_request_count": 0,
                    "daily_token_count": 0,
                }
            )

    def _save_data(self, data: dict):
        """Save data to JSON file"""
        with open(self.store_file, "w") as f:
            json.dump(data, f)

    def _load_data(self) -> dict:
        """Load data from JSON file"""
        try:
            with open(self.store_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load rate limiter data for {self.model_name}")
            return {
                "last_reset_timestamp": time.time(),
                "daily_request_count": 0,
                "daily_token_count": 0,
            }

    def get_counts(self) -> tuple[int, int]:
        """Get current counts and reset if needed"""
        data = self._load_data()
        last_reset = datetime.fromtimestamp(
            data["last_reset_timestamp"], tz=timezone.utc
        )
        now = datetime.now(timezone.utc)

        # Reset if last reset was on a different day
        if last_reset.date() < now.date():
            data.update(
                {
                    "last_reset_timestamp": now.timestamp(),
                    "daily_request_count": 0,
                    "daily_token_count": 0,
                }
            )
            self._save_data(data)
            logger.info(f"Reset daily counts for {self.model_name}")

        return data["daily_request_count"], data["daily_token_count"]

    def update_counts(self, requests: int = 0, tokens: int = 0):
        """Update the counts"""
        data = self._load_data()
        data["daily_request_count"] += requests
        data["daily_token_count"] += tokens
        self._save_data(data)


class RateLimiter:
    """Handle rate limits for Gemini models."""

    MODEL_LIMITS = {
        # Gemini Flash models
        "models/gemini-1.5-flash": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},
        "models/gemini-1.5-flash-002": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},
        "models/gemini-1.5-flash-8b": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},

        # Gemini Pro models
        "models/gemini-1.5-pro": {"rpm": 2, "tpm": 32_000, "rpd": 50},
        "models/gemini-1.5-pro-002": {"rpm": 2, "tpm": 32_000, "rpd": 50},
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.limits = self.MODEL_LIMITS.get(
            model_name, self.MODEL_LIMITS["models/gemini-1.5-flash"]
        )
        self.rpm_interval = 60 / self.limits["rpm"]
        self.last_request_time = 0
        self.store = RateLimiterStore(model_name)
        self.request_count_since_last_log = 0
        self.token_count_since_last_log = 0
        self.LOG_FREQUENCY = 10
        logger.info(
            f"Initialized RateLimiter for {model_name} with limits: {self.limits}"
        )

    def wait(self):
        """Wait for the appropriate time to comply with RPM and RPD limits."""
        self._enforce_rpm_limit()
        self._enforce_rpd_limit()

    def update_token_count(self, token_count: int):
        """Update the daily token count and check against the TPM limit."""
        self.token_count_since_last_log += token_count
        self.store.update_counts(tokens=token_count)

        _, daily_token_count = self.store.get_counts()

        if self.request_count_since_last_log >= self.LOG_FREQUENCY:
            logger.info(
                f"Token count update (last {self.LOG_FREQUENCY} requests): "
                f"+{self.token_count_since_last_log} tokens. "
                f"Total: {daily_token_count}/{self.limits['tpm']}"
            )
            self.token_count_since_last_log = 0
            self.request_count_since_last_log = 0

        if daily_token_count > self.limits["tpm"]:
            logger.error(
                f"Token limit exceeded for {self.model_name}: {daily_token_count}/{self.limits['tpm']}"
            )
            raise ResourceExhausted(f"Token limit exceeded for {self.model_name}")

    def _enforce_rpm_limit(self):
        """Enforce the RPM limit by waiting if necessary."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rpm_interval:
            wait_time = self.rpm_interval - time_since_last_request 
            logger.info(
                f"Rate limit: Waiting {wait_time:.2f}s before next request for {self.model_name}"
            )
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _enforce_rpd_limit(self):
        """Enforce the RPD limit by checking the daily request count."""
        self.request_count_since_last_log += 1
        self.store.update_counts(requests=1)

        daily_request_count, _ = self.store.get_counts()

        if self.request_count_since_last_log >= self.LOG_FREQUENCY:
            logger.info(
                f"Request count update (last {self.LOG_FREQUENCY} requests): "
                f"Daily total: {daily_request_count}/{self.limits['rpd']}"
            )

        if daily_request_count > self.limits["rpd"]:
            logger.error(
                f"Daily request limit exceeded for {self.model_name}: {daily_request_count}/{self.limits['rpd']}"
            )
            raise ResourceExhausted(
                f"Daily request limit exceeded for {self.model_name}"
            )


class RateLimitedGemini(BaseGemini):
    """Rate-limited version of LlamaIndex's Gemini implementation."""

    rate_limiter: RateLimiter = Field(
        default_factory=lambda: RateLimiter("models/gemini-1.5-flash")
    )  # Default instance

    def __init__(self, *args, **kwargs):
        """
        Initialize the RateLimitedGemini with rate limiting.

        Args:
            *args: Positional arguments for the base Gemini class.
            **kwargs: Keyword arguments for the base Gemini class.
        """
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized RateLimitedGemini with model: {self.model}")

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ResourceExhausted),
    )
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Override complete with rate limiting.

        Args:
            prompt (str): The prompt to complete.
            formatted (bool): Whether to return a formatted response.
            **kwargs: Additional arguments for the completion request.

        Returns:
            CompletionResponse: The response from the completion request.
        """
        try:
            self.rate_limiter.wait()
            response = super().complete(prompt, formatted=formatted, **kwargs)
            # Estimate tokens from response length since CompletionResponse doesn't include token count
            estimated_tokens = len(str(response.text)) // 4
            self.rate_limiter.update_token_count(estimated_tokens)
            return response
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSONDecodeError or ValidationError encountered: {str(e)}")
            raise  # Allow retry to occur
        except Exception as e:
            logger.exception(f"Error in complete: {str(e)}")
            raise

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ResourceExhausted),
    )
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Override chat with rate limiting.

        Args:
            messages (Sequence[ChatMessage]): The messages for the chat.
            **kwargs: Additional arguments for the chat request.

        Returns:
            ChatResponse: The response from the chat request.
        """
        try:
            self.rate_limiter.wait()
            response = super().chat(messages, **kwargs)
            # Estimate tokens from response length
            estimated_tokens = len(str(response.message.content)) // 4
            self.rate_limiter.update_token_count(estimated_tokens)
            return response
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSONDecodeError or ValidationError encountered: {str(e)}")
            raise  # Allow retry to occur
        except Exception as e:
            logger.exception(f"Error in chat: {str(e)}")
            raise


class RateLimitedCerebras(LlamaIndexCerebras):
    """Rate-limited version of LlamaIndex's Cerebras implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized RateLimitedCerebras with model: {self.model}")

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Override complete with rate limiting."""
        try:
            time.sleep(2)  # Add 2 second delay
            return super().complete(prompt, formatted=formatted, **kwargs)
        except Exception as e:
            logger.exception(f"Error in complete: {str(e)}")
            raise

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Override chat with rate limiting."""
        try:
            time.sleep(2)  # Add 2 second delay
            return super().chat(messages, **kwargs)
        except Exception as e:
            logger.exception(f"Error in chat: {str(e)}")
            raise


class RateLimitedGroq(LlamaIndexGroq):
    """Rate-limited version of LlamaIndex's Groq implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized RateLimitedGroq with model: {self.model}")

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Override complete with rate limiting."""
        try:
            time.sleep(2)  # Add 2 second delay
            return super().complete(prompt, formatted=formatted, **kwargs)
        except Exception as e:
            logger.exception(f"Error in complete: {str(e)}")
            raise

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Override chat with rate limiting."""
        try:
            time.sleep(2)  # Add 2 second delay
            return super().chat(messages, **kwargs)
        except Exception as e:
            logger.exception(f"Error in chat: {str(e)}")
            raise
