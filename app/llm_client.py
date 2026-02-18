from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Generator, Optional

from dotenv import load_dotenv

# Suppress noisy deprecation warnings on older Python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Both GOOGLE_API_KEY and GEMINI_API_KEY.*")

# Load local env files for all entrypoints (UI + sanity + scripts).
load_dotenv(".env.local")
load_dotenv(".env")

# ── Resolve API key once at import time to prevent duplicate-key warnings ──
_RESOLVED_API_KEY: Optional[str] = None
_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _key:
    _RESOLVED_API_KEY = _key
    # SDK auto-reads GOOGLE_API_KEY and warns if both are set.
    # Remove BOTH env vars so the SDK doesn't find them; we pass the key explicitly.
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)


def _try_import_genai():
    try:
        from google import genai
        return genai
    except Exception:
        return None


@dataclass
class LLMResponse:
    text: str
    model_used: str
    used_fallback: bool


class GeminiClient:
    """Wrapper around the google-genai SDK (new API) with graceful fallback."""

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.model_name = model_name
        self._genai = _try_import_genai()
        self._client = None
        self._available = False

        if self._genai and _RESOLVED_API_KEY:
            try:
                self._client = self._genai.Client(api_key=_RESOLVED_API_KEY)
                self._available = True
            except Exception:
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, fallback: str = "") -> LLMResponse:
        if not self._available or not self._client:
            return LLMResponse(text=fallback, model_used="fallback", used_fallback=True)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if not text:
                return LLMResponse(text=fallback, model_used="fallback", used_fallback=True)
            return LLMResponse(text=text, model_used=self.model_name, used_fallback=False)
        except Exception:
            return LLMResponse(text=fallback, model_used="fallback", used_fallback=True)

    def stream(self, prompt: str, fallback: str = "") -> Generator[str, None, None]:
        if not self._available or not self._client:
            for token in fallback.split():
                yield token + " "
            return

        try:
            stream_iter = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
            )
            has_content = False
            for chunk in stream_iter:
                text = (chunk.text or "")
                if text:
                    has_content = True
                    yield text
            if not has_content and fallback:
                for token in fallback.split():
                    yield token + " "
        except Exception:
            for token in fallback.split():
                yield token + " "

    def json_decision(self, prompt: str, default_obj: Optional[dict] = None) -> dict:
        default_obj = default_obj or {}
        response = self.generate(prompt, fallback=json.dumps(default_obj))
        raw = response.text.strip()

        # Strip markdown fences
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return default_obj
        except Exception:
            return default_obj
