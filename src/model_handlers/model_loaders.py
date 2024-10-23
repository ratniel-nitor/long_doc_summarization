import os
import time
import json
import instructor
from loguru import logger
from dotenv import load_dotenv
from typing import Any, Sequence
from pydantic import ValidationError, Field
from llama_index.llms.gemini import Gemini as BaseGemini
from google.api_core.exceptions import ResourceExhausted
from llama_index.core.base.llms.types import (
    CompletionResponse,
    ChatMessage,
    ChatResponse,
)
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

load_dotenv()

# Configure logging
logger.add("model_loaders.log", rotation="10 MB", level="DEBUG")


class RateLimiter:
    """Handle rate limits for Gemini models."""

    MODEL_LIMITS = {
        "models/gemini-1.5-flash": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},
        "models/gemini-1.5-flash-8b": {"rpm": 15, "tpm": 1_000_000, "rpd": 1_500},
        "models/gemini-1.5-pro": {"rpm": 2, "tpm": 32_000, "rpd": 50},
    }

    def __init__(self, model_name: str):
        """
        Initialize the RateLimiter with the specified model name.

        Args:
            model_name (str): The name of the model for which to enforce rate limits.
        """
        self.model_name = model_name
        self.limits = self.MODEL_LIMITS.get(
            model_name, self.MODEL_LIMITS["models/gemini-1.5-flash"]
        )
        self.rpm_interval = 60 / self.limits["rpm"]
        self.last_request_time = 0
        self.daily_request_count = 0
        self.daily_token_count = 0
        self.last_reset_day = time.localtime().tm_yday
        self.request_count_since_last_log = 0  # Add counter for logging
        self.token_count_since_last_log = 0    # Add counter for logging
        self.LOG_FREQUENCY = 10                # Log after every 10 documents
        logger.info(
            f"Initialized RateLimiter for {model_name} with limits: {self.limits}"
        )

    def wait(self):
        """Wait for the appropriate time to comply with RPM and RPD limits."""
        self._reset_daily_counts_if_needed()
        self._enforce_rpm_limit()
        self._enforce_rpd_limit()

    def update_token_count(self, token_count: int):
        """
        Update the daily token count and check against the TPM limit.

        Args:
            token_count (int): The number of tokens used in the request.

        Raises:
            ResourceExhausted: If the token limit is exceeded.
        """
        self.daily_token_count += token_count
        self.token_count_since_last_log += token_count
        
        # Log only after processing LOG_FREQUENCY documents
        if self.request_count_since_last_log >= self.LOG_FREQUENCY:
            logger.info(
                f"Token count update (last {self.LOG_FREQUENCY} requests): "
                f"+{self.token_count_since_last_log} tokens. "
                f"Total: {self.daily_token_count}/{self.limits['tpm']}"
            )
            self.token_count_since_last_log = 0
            self.request_count_since_last_log = 0

        if self.daily_token_count > self.limits["tpm"]:
            logger.error(
                f"Token limit exceeded for {self.model_name}: {self.daily_token_count}/{self.limits['tpm']}"
            )
            raise ResourceExhausted(f"Token limit exceeded for {self.model_name}")

    def _reset_daily_counts_if_needed(self):
        """Reset daily request and token counts if a new day has started."""
        current_day = time.localtime().tm_yday
        if current_day != self.last_reset_day:
            logger.info(f"Resetting daily counts for {self.model_name}")
            self.daily_request_count = 0
            self.daily_token_count = 0
            self.last_reset_day = current_day

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
        self.daily_request_count += 1
        self.request_count_since_last_log += 1
        
        # Log only after processing LOG_FREQUENCY documents
        if self.request_count_since_last_log >= self.LOG_FREQUENCY:
            logger.info(
                f"Request count update (last {self.LOG_FREQUENCY} requests): "
                f"Daily total: {self.daily_request_count}/{self.limits['rpd']}"
            )

        if self.daily_request_count > self.limits["rpd"]:
            logger.error(
                f"Daily request limit exceeded for {self.model_name}: {self.daily_request_count}/{self.limits['rpd']}"
            )
            raise ResourceExhausted(
                f"Daily request limit exceeded for {self.model_name}"
            )


class RateLimitedGemini(BaseGemini):
    """Rate-limited version of LlamaIndex's Gemini implementation."""

    rate_limiter: RateLimiter = Field(default_factory=lambda: RateLimiter("models/gemini-1.5-flash"))  # Default instance

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
        pass  # Implementation for initializing Groq model (not provided)

    @staticmethod
    def initialize_gemini(
        model_name: str = "models/gemini-1.5-flash",
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

        # Create rate-limited Gemini instance
        client = RateLimitedGemini(api_key=google_api_key, model=model_name)
        logger.info(f"Initialized rate-limited Gemini model: {model_name}")

        if use_instructor:
            logger.info("Using instructor with Gemini")
            return instructor.from_gemini(
                client=client._model,  # Access the underlying GenerativeModel
                mode=instructor.Mode.GEMINI_JSON,
            )

        if use_llamaindex:
            logger.info("Using LlamaIndex with Gemini")
            return client

        return client._model  # Return raw GenerativeModel for basic usage
