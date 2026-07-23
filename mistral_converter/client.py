"""Mistral SDK client singleton and retry configuration."""

import threading
from typing import Any, Dict, Optional, Tuple

import config
import utils

from .facade import attr
from .sdk_shims import httpx

logger = utils.logger


def _http_client_exceptions() -> Tuple[type, ...]:
    """Tuple of httpx errors to handle explicitly before a generic Exception."""
    if httpx is None:
        return ()
    return (
        httpx.HTTPError,
        httpx.TimeoutException,
    )


_client_lock = threading.Lock()
_client_instance: Optional[Any] = None


def get_mistral_client() -> Optional[Any]:
    """
    Create and configure a Mistral client instance.
    Uses a Lock-guarded singleton to prevent connection pool churn in batch
    operations while remaining safe under concurrent ``ThreadPoolExecutor``
    usage.

    Thread-safety note:
        The Mistral SDK client uses ``httpx`` under the hood, which is
        thread-safe for concurrent requests.  This singleton is therefore
        safe to share across threads.

    Returns:
        Configured Mistral client or None if unavailable
    """
    global _client_instance

    if _client_instance is not None:
        return _client_instance

    with _client_lock:
        if _client_instance is not None:
            return _client_instance

        Mistral = attr("Mistral")
        if Mistral is None:
            logger.error(
                "Mistral SDK not available. Run: pip install mistralai  "
                "(check logs/pip_install.log if you already installed it)"
            )
            return None

        if not config.MISTRAL_API_KEY:
            logger.error("MISTRAL_API_KEY not set. Add it to your .env file in the project root.")
            return None

        try:
            client_kwargs: Dict[str, Any] = {"api_key": config.MISTRAL_API_KEY}

            if config.MISTRAL_SERVER_URL:
                if config.MISTRAL_SERVER_URL.startswith("http://") and not config.ALLOW_INSECURE_MISTRAL_SERVER:
                    logger.error(
                        "MISTRAL_SERVER_URL uses insecure http://. "
                        "Set ALLOW_INSECURE_MISTRAL_SERVER=true to allow, or use https://."
                    )
                    return None
                client_kwargs["server_url"] = config.MISTRAL_SERVER_URL

            global_retry = attr("get_retry_config")()
            if global_retry:
                client_kwargs["retry_config"] = global_retry

            client_kwargs["timeout_ms"] = config.MISTRAL_CLIENT_TIMEOUT_MS

            client = Mistral(**client_kwargs)
            _client_instance = client
            return client

        except Exception as e:
            logger.exception("Error initializing Mistral client: %s", e)
            return None


def reset_mistral_client() -> None:
    """Clear the cached Mistral client so the next call creates a fresh one."""
    global _client_instance
    with _client_lock:
        _client_instance = None


def get_retry_config() -> Optional[Any]:
    """
    Create RetryConfig for Mistral API calls with exponential backoff.

    Returns:
        RetryConfig instance or None if retries module unavailable
    """
    retries = attr("retries")
    # MAX_RETRIES=0 disables (legacy). ENABLE_RETRIES=false also disables.
    # Either gate is sufficient so monkeypatched MAX_RETRIES=0 still works.
    if retries is None or config.MAX_RETRIES == 0:
        return None
    if not getattr(config, "ENABLE_RETRIES", True):
        return None

    try:
        backoff_strategy = retries.BackoffStrategy(
            initial_interval=config.RETRY_INITIAL_INTERVAL_MS,
            max_interval=config.RETRY_MAX_INTERVAL_MS,
            exponent=config.RETRY_EXPONENT,
            max_elapsed_time=config.RETRY_MAX_ELAPSED_TIME_MS,
        )

        retry_config = retries.RetryConfig(
            strategy="backoff",
            backoff=backoff_strategy,
            retry_connection_errors=config.RETRY_CONNECTION_ERRORS,
        )

        logger.debug(
            "Retry config: retries enabled, %dms initial interval (bounded by RETRY_MAX_ELAPSED_TIME_MS)",
            config.RETRY_INITIAL_INTERVAL_MS,
        )
        return retry_config

    except Exception as e:
        logger.warning("Error creating retry config: %s", e)
        return None
