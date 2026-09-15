import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Transient Gemini capacity / rate issues that are worth retrying.
_RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand")

# Daily (RPD) quota exhaustion — retrying will not help until reset.
_DAILY_QUOTA_MARKERS = (
    "GenerateRequestsPerDay",
    "PerDayPerProjectPerModel",
    "RequestsPerDay",
)


class GeminiDailyQuotaError(RuntimeError):
    """Raised when Gemini daily request quota (RPD) is exhausted."""


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_retryable(exc: BaseException) -> bool:
    message = _error_text(exc).upper()
    return any(marker.upper() in message for marker in _RETRYABLE_MARKERS)


def _is_daily_quota(exc: BaseException) -> bool:
    message = _error_text(exc)
    return any(marker in message for marker in _DAILY_QUOTA_MARKERS)


def _parse_retry_delay_seconds(exc: BaseException) -> float | None:
    """Extract retryDelay / 'retry in Xs' from the Gemini error payload."""
    message = _error_text(exc)

    match = re.search(
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?",
        message,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    match = re.search(
        r"retry in\s+(\d+(?:\.\d+)?)\s*s",
        message,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    return None


def invoke_with_retry(
    invoke_fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay_seconds: float = 2.0,
    label: str = "gemini",
) -> T:
    """Call a Gemini invoke with backoff on transient 503/RPM errors.

    Daily quota (RPD) exhaustion fails immediately with GeminiDailyQuotaError.
    Short-term rate limits (RPM) retry using the API's retryDelay when present.
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return invoke_fn()
        except Exception as exc:  # noqa: BLE001 — narrow via helpers below
            last_error = exc

            if _is_daily_quota(exc):
                raise GeminiDailyQuotaError(
                    f"[{label}] Gemini daily quota (RPD) exhausted for this model. "
                    "Retrying will not help until the quota resets (typically next day). "
                    "Wait for the reset, switch models, or enable billing. "
                    f"Original error: {exc}"
                ) from exc

            if not _is_retryable(exc) or attempt == max_attempts:
                raise

            api_delay = _parse_retry_delay_seconds(exc)
            delay = (
                api_delay
                if api_delay is not None
                else base_delay_seconds * (2 ** (attempt - 1))
            )
            print(
                f"[{label}] transient Gemini error (attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise RuntimeError(f"[{label}] exhausted retries") from last_error
