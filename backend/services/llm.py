"""
llm.py

One function, generate_json(), that sends a prompt to an LLM and returns
JSON matching a schema. Two free providers are supported:

    gemini -- Google Gemini API (free tier), used when GEMINI_API_KEY
              (or GOOGLE_API_KEY) is set
    ollama -- a local model served by Ollama (offline, no key)

LLM_PROVIDER in config.py chooses: "auto" (default) uses Gemini when its
key is set and otherwise a local Ollama model, so no key is required.
"""

import os
import json
import sys
import urllib.error
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import (
    LLM_PROVIDER,
    GEMINI_MODEL,
    LLM_MAX_OUTPUT_TOKENS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_TIMEOUT_SECONDS,
)

_clients = {}  # provider -> lazily created SDK client


class LLMError(Exception):
    """The request reached the model but failed (bad output, refusal, API error)."""
    pass


class LLMUnavailableError(LLMError):
    """No provider is configured, or the provider rejected the key / is out of quota."""
    pass


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def active_provider():
    """Returns "gemini" or "ollama" (the keyless local fallback)."""
    if LLM_PROVIDER in ("gemini", "ollama"):
        return LLM_PROVIDER
    if _gemini_key():
        return "gemini"
    return "ollama"


def active_model():
    provider = active_provider()
    return {"gemini": GEMINI_MODEL, "ollama": OLLAMA_MODEL}[provider]


def generate_json(system: str, prompt: str, schema: dict) -> dict:
    provider = active_provider()
    if provider == "gemini":
        return _gemini_json(system, prompt, schema)
    return _ollama_json(system, prompt, schema)


def _parse_json(text):
    if not text:
        raise LLMError("The model returned no text.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LLMError("The model returned invalid JSON.")


# ---------------------------------------------------------------------------
# Gemini (Google) -- free tier available
# ---------------------------------------------------------------------------

def _gemini_schema(schema):
    """Drops "additionalProperties", which Gemini's schema support may reject.
    Callers only read the fields they expect, so extra fields are harmless."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


def _gemini_json(system: str, prompt: str, schema: dict) -> dict:
    try:
        import httpx
        from google import genai
        from google.genai import types, errors
    except ImportError:
        raise LLMUnavailableError("The google-genai package is not installed. Run: pip install google-genai")

    if not _gemini_key():
        raise LLMUnavailableError(
            "Gemini is not configured. Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)."
        )

    if "gemini" not in _clients:
        _clients["gemini"] = genai.Client(api_key=_gemini_key())
    client = _clients["gemini"]

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_json_schema=_gemini_schema(schema),
            ),
        )
    except errors.APIError as e:
        if e.code == 429:
            raise LLMUnavailableError(
                "Gemini quota/rate limit reached (free tier limits requests per minute and per day). "
                "Wait a minute and try again."
            )
        if e.code in (401, 403) or "API key" in str(e.message):
            raise LLMUnavailableError("Gemini rejected the API key. Check GEMINI_API_KEY.")
        if e.code == 404:
            raise LLMUnavailableError(
                f"Gemini model '{GEMINI_MODEL}' was not found. Change GEMINI_MODEL in backend/config.py."
            )
        raise LLMError(f"Gemini API error ({e.code}): {e.message}")
    except httpx.TransportError as e:  # network failures (google-genai uses httpx)
        raise LLMUnavailableError(f"Could not connect to the Gemini API: {e}")

    candidates = response.candidates or []
    finish = str(candidates[0].finish_reason) if candidates and candidates[0].finish_reason else ""
    if "MAX_TOKENS" in finish:
        raise LLMError("The model's response was cut off. Try a shorter section.")
    if not candidates or any(r in finish for r in ("SAFETY", "PROHIBITED", "BLOCKLIST", "RECITATION")):
        raise LLMError("The model declined to process this transcript.")

    return _parse_json(response.text)


# ---------------------------------------------------------------------------
# Ollama -- local model, free and offline
# ---------------------------------------------------------------------------

_NO_LLM_HINT = (
    "Either start Ollama and run `ollama pull {model}` (free, local), or set GEMINI_API_KEY "
    "(free key from https://aistudio.google.com/apikey)."
)


def _ollama_json(system: str, prompt: str, schema: dict) -> dict:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,  # Ollama constrains the output to this JSON schema
        "options": {"num_ctx": OLLAMA_CONTEXT_TOKENS, "temperature": 0.2},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    # Ollama runs on this machine: bypass any HTTP(S)_PROXY environment settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if e.code == 404 or "not found" in detail.lower():
            raise LLMUnavailableError(
                f"The Ollama model '{OLLAMA_MODEL}' is not installed. Run: ollama pull {OLLAMA_MODEL}"
            )
        raise LLMError(f"Ollama error ({e.code}): {detail[:300]}")
    except (urllib.error.URLError, ConnectionError) as e:
        raise LLMUnavailableError(
            f"No LLM is available: no API key is set and Ollama is not reachable at {OLLAMA_HOST} "
            f"({getattr(e, 'reason', e)}). " + _NO_LLM_HINT.format(model=OLLAMA_MODEL)
        )
    except TimeoutError:
        raise LLMError(f"Ollama did not answer within {OLLAMA_TIMEOUT_SECONDS}s. Try a smaller model.")
    except json.JSONDecodeError:
        raise LLMError("Ollama returned an unreadable response.")

    if data.get("done_reason") == "length":
        raise LLMError("The model's response was cut off. Try fewer questions or a shorter section.")
    return _parse_json((data.get("message") or {}).get("content"))
