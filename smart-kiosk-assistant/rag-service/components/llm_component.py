from __future__ import annotations

import gc
import logging
import pathlib
from typing import Generator

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer


logger = logging.getLogger(__name__)


class OVLLMComponent:
    """OpenVINO IR LLM component backed by optimum-intel OVModelForCausalLM.

    Handles tokenization and model calls. Thread safety and reload logic
    remain the responsibility of the caller (RagPipeline).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        ov_config: dict | None = None,
    ) -> None:
        self._model_path = str(pathlib.Path(model_path).resolve())
        self._device = device.upper()
        self._ov_config = ov_config or {}

        logger.info("[LLM] Loading tokenizer from %s", self._model_path)
        try:
            self._tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                self._model_path, fix_mistral_regex=True
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        logger.info(
            "[LLM] Loading OVModelForCausalLM from %s on %s (ov_config=%s)",
            self._model_path,
            self._device,
            self._ov_config or "{}",
        )
        self._model: OVModelForCausalLM = OVModelForCausalLM.from_pretrained(
            self._model_path,
            device=self._device,
            ov_config=self._ov_config if self._ov_config else None,
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Tokenize *prompt*, run greedy/sampling generation, return decoded text (new tokens only)."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        output_ids = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_with_streamer(
        self, prompt: str, streamer: TextIteratorStreamer, **gen_kwargs
    ) -> None:
        """Run generation synchronously, feeding tokens into *streamer*.

        Intended to be called from a background thread so the caller can
        iterate over the streamer in the main thread concurrently.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt")
        self._model.generate(**inputs, streamer=streamer, **gen_kwargs)

    def destroy(self) -> None:
        """Release the compiled model and reclaim memory."""
        try:
            del self._model
            gc.collect()
            logger.info("[LLM] OVLLMComponent model destroyed, memory reclaimed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM] Failed to fully destroy OVLLMComponent: %s", exc)
