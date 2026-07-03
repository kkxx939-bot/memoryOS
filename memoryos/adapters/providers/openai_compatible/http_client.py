from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from memoryos.ports.providers.provider_errors import (
    ProviderAuthFailed,
    ProviderBadResponse,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
)

_CIRCUIT_STATE: dict[str, dict[str, float]] = {}


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
    retries: int = 2,
    backoff_seconds: float = 0.5,
    provider: str = "openai_compatible",
    circuit_breaker_failures: int = 5,
    circuit_breaker_cooldown_seconds: float = 30.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    circuit_key = f"{provider}:{base_url}:{endpoint}"
    circuit = _CIRCUIT_STATE.get(circuit_key, {"failures": 0.0, "opened_at": 0.0})
    opened_at = float(circuit.get("opened_at", 0.0))
    if opened_at and time.monotonic() - opened_at < circuit_breaker_cooldown_seconds:
        raise ProviderUnavailable(
            "Provider circuit breaker is open",
            provider=provider,
            metadata={
                "circuit_key": circuit_key,
                "failures": int(circuit.get("failures", 0)),
                "cooldown_seconds": circuit_breaker_cooldown_seconds,
            },
        )
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                _CIRCUIT_STATE[circuit_key] = {"failures": 0.0, "opened_at": 0.0}
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403}:
                raise ProviderAuthFailed(f"API auth failed with HTTP {exc.code}: {detail}", provider=provider, status_code=exc.code) from exc
            if exc.code == 429:
                last_error = ProviderRateLimited(f"API rate limited: {detail}", provider=provider, status_code=exc.code)
            elif exc.code in {500, 502, 503, 504}:
                last_error = ProviderUnavailable(
                    f"API unavailable with HTTP {exc.code}: {detail}",
                    provider=provider,
                    status_code=exc.code,
                )
            else:
                raise ProviderBadResponse(
                    f"API request failed with HTTP {exc.code}: {detail}",
                    provider=provider,
                    status_code=exc.code,
                ) from exc
            if attempt >= retries:
                _record_provider_failure(circuit_key, circuit_breaker_failures)
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = ProviderUnavailable(f"API request failed: {exc.reason}", provider=provider)
            if attempt >= retries:
                _record_provider_failure(circuit_key, circuit_breaker_failures)
                raise last_error from exc
        except json.JSONDecodeError as exc:
            raise ProviderBadResponse(f"API response is not valid JSON: {exc}", provider=provider) from exc
        time.sleep(backoff_seconds * (2**attempt))
    _record_provider_failure(circuit_key, circuit_breaker_failures)
    raise ProviderError(
        f"API request failed after retries: {last_error}",
        provider=provider,
        metadata={"retries": retries, "backoff_seconds": backoff_seconds, "circuit_key": circuit_key},
    )


def _record_provider_failure(circuit_key: str, threshold: int) -> None:
    state = _CIRCUIT_STATE.get(circuit_key, {"failures": 0.0, "opened_at": 0.0})
    failures = float(state.get("failures", 0.0)) + 1
    _CIRCUIT_STATE[circuit_key] = {
        "failures": failures,
        "opened_at": time.monotonic() if failures >= max(1, threshold) else 0.0,
    }
