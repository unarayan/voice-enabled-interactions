from __future__ import annotations

import gc
import logging
import pathlib
from typing import Generator

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer, pipeline


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
                self._model_path, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True
            )

        logger.info(
            "[LLM] Loading OVModelForCausalLM from %s on %s (ov_config=%s)",
            self._model_path,
            self._device,
            self._ov_config or "{}",
        )
        self._model: OVModelForCausalLM = OVModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            device=self._device,
            ov_config=self._ov_config if self._ov_config else None,
        )
        self._pipe = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer)

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Run generation via the text-generation pipeline, return only new tokens."""
        import time as _time
        t_tok = _time.monotonic()
        input_len = len(self._tokenizer.encode(prompt))
        t_gen = _time.monotonic()
        logger.info(
            "[LLM] generate | input_tokens=%d | prompt_chars=%d | tokenize=%.3fs | gen_kwargs=%s",
            input_len, len(prompt), t_gen - t_tok, gen_kwargs,
        )
        output = self._pipe(prompt, return_full_text=False, **gen_kwargs)
        t_done = _time.monotonic()
        result = output[0]["generated_text"]
        result_tokens = len(self._tokenizer.encode(result))
        logger.info(
            "[LLM] generate | new_tokens=%d | generation=%.3fs | total=%.3fs",
            result_tokens, t_done - t_gen, t_done - t_tok,
        )
        return result

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
            del self._pipe
            del self._model
            gc.collect()
            logger.info("[LLM] OVLLMComponent model destroyed, memory reclaimed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM] Failed to fully destroy OVLLMComponent: %s", exc)
