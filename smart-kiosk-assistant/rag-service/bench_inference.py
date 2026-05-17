#!/usr/bin/env python3
"""
Standalone OpenVINO GenAI inference benchmark
──────────────────────────────────────────────
Model  : Qwen/Qwen2.5-7B-Instruct int8 (pre-exported by rag-service)
Device : GPU  (change DEVICE below to "CPU" or "NPU" to compare)
Goal   : measure pipeline load time, TTFT, total generation time, tok/s
         for a configurable-size prompt (default ~5 000 tokens)

Usage:
    python bench_inference.py
    python bench_inference.py --device CPU
    python bench_inference.py --tokens 2000 --max-new 512
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import queue
import threading
import time

import openvino_genai as ov_genai
from transformers import AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = HERE / "models/llm/Qwen_Qwen2.5-7B-Instruct__int8"
CACHE_DIR  = HERE / "storage/ov_cache"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DEVICE         = "GPU"
DEFAULT_TARGET_TOKENS  = 5_000   # approximate prompt token budget
DEFAULT_MAX_NEW_TOKENS = 256


# ── prompt construction ───────────────────────────────────────────────────────
_SYSTEM = (
    "You are a knowledgeable Smart Kiosk Assistant for a large electronics retail store. "
    "Answer questions clearly, concisely, and in a way suitable for spoken output. "
    "Use the product catalogue context provided to give accurate, helpful recommendations."
)

_QUESTION = (
    "Based on the product catalogue provided, please give a detailed comparison of the "
    "available laptop tiers, recommend the best option for a university student on a tight "
    "budget, explain the warranty and price-match policies, and list three smart-home "
    "starter kits that work well together."
)

# One catalogue paragraph ≈ 220–250 tokens (Qwen BPE, English text)
_CATALOGUE_PARA = (
    "The store carries a comprehensive selection of consumer electronics. "
    "Television sets range from 32-inch HD models at entry-level price points to "
    "85-inch 8K OLED panels from premium brands. "
    "Laptop computers are available in three tiers: budget Chromebooks starting at $299, "
    "mid-range Windows laptops from $599 to $999, and professional workstations above $1 200. "
    "Smartphones include the latest flagship models from Apple, Samsung, and Google, "
    "alongside affordable options from OnePlus, Motorola, and Nokia. "
    "Audio products span true-wireless earbuds, over-ear noise-cancelling headphones, "
    "portable Bluetooth speakers, and full home audio systems. "
    "Smart home devices include Wi-Fi-enabled thermostats, video doorbells, security cameras, "
    "smart lighting systems, and hub controllers compatible with Alexa, Google Home, and "
    "Apple HomeKit. "
    "Wearable technology covers fitness trackers, smartwatches, and GPS running watches. "
    "Gaming hardware includes the latest consoles, gaming laptops, mechanical keyboards, "
    "precision mice, high-refresh-rate monitors, and gaming chairs. "
    "Camera and photography equipment covers mirrorless bodies, DSLR cameras, action cameras, "
    "drones, tripods, and studio lighting kits. "
    "All products carry a minimum one-year manufacturer warranty; extended warranties and "
    "accidental-damage protection plans are available at the point of sale. "
    "The store price-match guarantee covers identical products sold by major national retailers. "
)


def build_prompt(hf_tokenizer, target_tokens: int) -> str:
    """Repeat the catalogue paragraph until the formatted prompt reaches target_tokens."""
    # Estimate tokens per paragraph (rough) to set initial repeat count
    para_tokens = len(hf_tokenizer.encode(_CATALOGUE_PARA))
    overhead = 120  # system + question + chat template overhead
    repeats = max(1, (target_tokens - overhead) // para_tokens)

    context = _CATALOGUE_PARA * repeats
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"Product Catalogue:\n{context}\n\nQuestion: {_QUESTION}"},
    ]
    return hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── streaming helper ──────────────────────────────────────────────────────────
class BenchStreamer(ov_genai.StreamerBase):
    """Streams output to stdout and captures timing/token metrics."""

    def __init__(self, hf_tokenizer) -> None:
        super().__init__()
        self._tok        = hf_tokenizer
        self._ids: list[int] = []
        self._print_len  = 0
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._ended      = False
        # metrics
        self.first_token_time: float | None = None
        self.token_count = 0

    def put(self, token_id: int) -> bool:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        self._ids.append(token_id)
        text = self._tok.decode(self._ids, skip_special_tokens=True)
        new  = text[self._print_len:]
        # emit only when we reach a safe boundary (space or punctuation)
        if new and (new[-1].isspace() or new[-1] in ".,:;!?\n"):
            self._queue.put(new)
            self._print_len = len(text)
        return False  # False = keep generating

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._ids:
            text = self._tok.decode(self._ids, skip_special_tokens=True)
            remaining = text[self._print_len:]
            if remaining:
                self._queue.put(remaining)
        self._queue.put(None)

    def __iter__(self):
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            yield chunk


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO GenAI inference benchmark")
    parser.add_argument("--device",   default=DEFAULT_DEVICE,         help="Inference device (GPU/CPU/NPU)")
    parser.add_argument("--tokens",   type=int, default=DEFAULT_TARGET_TOKENS,  help="Target prompt token count")
    parser.add_argument("--max-new",  type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens to generate")
    args = parser.parse_args()

    device         = args.device.upper()
    target_tokens  = args.tokens
    max_new_tokens = args.max_new

    print("=" * 62)
    print("  OpenVINO GenAI — Inference Benchmark")
    print("=" * 62)
    print(f"  Model  : Qwen2.5-7B-Instruct int8")
    print(f"  Path   : {MODEL_PATH}")
    print(f"  Device : {device}")
    print(f"  Target prompt tokens : ~{target_tokens:,}")
    print(f"  Max new tokens       : {max_new_tokens}")
    print("=" * 62)
    print()

    # ── 1. tokenizer ──────────────────────────────────────────────────────────
    print("[1/3] Loading HF tokenizer...", flush=True)
    try:
        hf_tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    except TypeError:
        hf_tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    prompt = build_prompt(hf_tok, target_tokens)
    prompt_token_count = len(hf_tok.encode(prompt))
    print(f"      Prompt built — actual token count: {prompt_token_count:,}\n")

    # ── 2. pipeline ───────────────────────────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    plugin_config: dict[str, str] = {"CACHE_DIR": str(CACHE_DIR)}
    if device == "GPU":
        plugin_config["INFERENCE_PRECISION_HINT"] = "f16"
        plugin_config["KV_CACHE_PRECISION"]       = "f16"
        plugin_config["NUM_STREAMS"]              = "1"

    print(f"[2/3] Loading LLMPipeline on {device}...", flush=True)
    t_load = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(MODEL_PATH), device, **plugin_config)
    load_time = time.perf_counter() - t_load
    print(f"      Loaded in {load_time:.1f}s\n")

    # ── 3. generate ───────────────────────────────────────────────────────────
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature":    1e-7,   # near-zero → greedy, avoids do_sample overhead
        "do_sample":      False,
    }

    streamer = BenchStreamer(hf_tok)

    def _run_generate() -> None:
        pipe.generate(prompt, streamer=streamer, **gen_kwargs)

    print("[3/3] Running inference...\n")
    print("─" * 62)

    thread = threading.Thread(target=_run_generate, daemon=True)
    t_gen_start = time.perf_counter()
    thread.start()

    for chunk in streamer:
        print(chunk, end="", flush=True)

    thread.join()
    t_gen_end = time.perf_counter()

    print("\n" + "─" * 62)

    # ── results ───────────────────────────────────────────────────────────────
    ttft      = (streamer.first_token_time - t_gen_start) if streamer.first_token_time else None
    total_gen = t_gen_end - t_gen_start
    gen_toks  = streamer.token_count
    tps       = gen_toks / total_gen if total_gen > 0 else 0.0

    print()
    print("═" * 62)
    print("  Results")
    print("═" * 62)
    print(f"  Pipeline load time   : {load_time:.1f} s")
    print(f"  Prompt tokens        : {prompt_token_count:,}")
    print(f"  Generated tokens     : {gen_toks}")
    if ttft is not None:
        print(f"  Time to first token  : {ttft:.2f} s")
    print(f"  Total generation time: {total_gen:.2f} s")
    print(f"  Throughput           : {tps:.1f} tok/s")
    print("═" * 62)

    del pipe
    gc.collect()


if __name__ == "__main__":
    main()
