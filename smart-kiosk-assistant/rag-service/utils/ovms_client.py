"""OVMS LLM client — drop-in replacement for ov_genai.LLMPipeline.

This module provides OvmsLlmClient which exposes the same two methods
used in pipeline.py:

  generate(prompt, **gen_kwargs) -> str
  generate(prompt, streamer=callback, **gen_kwargs) -> None   [streaming]

It calls the OVMS text-generation pipeline via POST /v3/chat/completions
using Server-Sent Events for streaming.

Configuration (via config.container.yaml under models.llm):
  ovms_url: "http://ovms-service:8028"   # if set, enables OVMS mode
  ovms_model_name: "qwen25_7b_int8"      # must match --model_name in OVMS
  ovms_request_timeout: 300              # per-request HTTP timeout (s)
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Generator

import requests

logger = logging.getLogger(__name__)

_SENTINEL = object()


class OvmsLlmClient:
    """HTTP client that calls an OVMS text-generation endpoint.

    Mimics the ov_genai.LLMPipeline.generate() signature used in pipeline.py
    so the caller needs minimal changes.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "qwen25_7b_int8",
        request_timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = request_timeout
        logger.info(
            "[OvmsLlmClient] Initialized → %s  model=%s  timeout=%.0fs",
            self._base_url, self._model_name, self._timeout,
        )

    # ------------------------------------------------------------------
    # Public API (mirrors ov_genai.LLMPipeline.generate)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        do_sample: bool = False,
        streamer: Callable[[str], bool] | None = None,
        **_ignored,
    ) -> str | None:
        """
        If streamer is None  → returns the full generated string.
        If streamer is set   → calls streamer(chunk) for each SSE delta,
                               then calls streamer.end() if it exists,
                               and returns None (same as ov_genai behaviour).
        """
        if streamer is not None:
            self._stream_to_callback(prompt, max_new_tokens, temperature, streamer)
            return None
        return self._generate_blocking(prompt, max_new_tokens, temperature)

    def health_check(self) -> bool:
        """Return True if OVMS /v3/models responds 200 (model available)."""
        try:
            resp = requests.get(f"{self._base_url}/v3/models", timeout=10)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict:
        return {
            "model": self._model_name,
            # Wrap the raw prompt as a single user message.  OVMS applies the
            # chat template internally so the model sees the proper format.
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            # ov_genai clamps temperature to >0 internally; we do the same.
            "temperature": max(temperature, 1e-7),
            "stream": stream,
        }

    def _generate_blocking(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = self._make_payload(prompt, max_tokens, temperature, stream=False)
        try:
            resp = requests.post(
                f"{self._base_url}/v3/chat/completions",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"[OvmsLlmClient] Request timed out after {self._timeout:.0f}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"[OvmsLlmClient] HTTP error: {exc}"
            ) from exc

        try:
            body = resp.json()
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"[OvmsLlmClient] Unexpected response format: {resp.text[:200]}"
            ) from exc

    def _stream_to_callback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        streamer: Callable[[str], bool],
    ) -> None:
        """Stream SSE chunks from OVMS → call streamer(chunk) for each delta.

        Ends by calling streamer.end() if the callable exposes that method
        (matching the _TokenStream interface in pipeline.py).
        """
        payload = self._make_payload(prompt, max_tokens, temperature, stream=True)
        try:
            with requests.post(
                f"{self._base_url}/v3/chat/completions",
                json=payload,
                stream=True,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = (
                        raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    )
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        keep_going = streamer(delta)
                        # ov_genai streamers return False to continue — None also
                        # means continue; only explicit True means stop early.
                        if keep_going is True:
                            break
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"[OvmsLlmClient] Stream timed out after {self._timeout:.0f}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"[OvmsLlmClient] Stream HTTP error: {exc}") from exc
        finally:
            # Signal end of stream — matches _TokenStream.end() interface.
            end = getattr(streamer, "end", None)
            if callable(end):
                end()
