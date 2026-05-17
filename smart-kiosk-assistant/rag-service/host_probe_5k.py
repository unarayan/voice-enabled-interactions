"""Standalone host probe — no rag-service imports needed.

Directly uses the exported model at:
  ~/.cache/ov_models/OpenVINO_Qwen2.5-7B-Instruct-int8-ov
  (or the rag-service models dir if that's preferred)

Usage:
  sg render -c "MALLOC_MMAP_THRESHOLD_=13107200 python3 host_probe_5k.py"
  or:
  MALLOC_MMAP_THRESHOLD_=13107200 .venv_probe/bin/python3 host_probe_5k.py
"""

from __future__ import annotations

import gc
import os
import pathlib
import sys
import threading
import time

import openvino_genai as ov_genai
from transformers import AutoTokenizer

# ── model path ────────────────────────────────────────────────────────────────
# Override via env var: PROBE_MODEL_PATH=/absolute/path
_env_path = os.environ.get("PROBE_MODEL_PATH", "").strip()
if _env_path:
    MODEL_PATH = pathlib.Path(_env_path)
    if not MODEL_PATH.exists():
        print(f"ERROR: PROBE_MODEL_PATH does not exist: {MODEL_PATH}")
        sys.exit(1)
else:
    CANDIDATES = [
        pathlib.Path.home() / ".cache/ov_models/OpenVINO_Qwen2.5-7B-Instruct-int8-ov",
        pathlib.Path(__file__).parent / "models/llm/Qwen_Qwen2.5-7B-Instruct__int8",
    ]
    MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)
    if MODEL_PATH is None:
        print("ERROR: No model found. Check CANDIDATES list or set PROBE_MODEL_PATH.")
        sys.exit(1)

DEVICE          = os.environ.get("PROBE_DEVICE", "GPU")
TIMEOUT_S       = float(os.environ.get("PROBE_TIMEOUT_S", "300"))
MAX_NEW_TOKENS  = int(os.environ.get("PROBE_MAX_NEW_TOKENS", "32"))
RAW_STEPS       = os.environ.get("PROBE_TOKEN_STEPS", "3000,5000,8000,12000")
STEPS           = [int(x) for x in RAW_STEPS.split(",") if x.strip()]


def build_prompt(tokenizer, target_tokens: int) -> str:
    seed = (
        "MegaRetail Hypermart stocks fresh produce, bakery items, dairy, "
        "frozen goods, household essentials, electronics, and seasonal "
        "promotions across fourteen ground floor aisles. "
    )
    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += seed
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True) + (
        "\n\nQuestion: In one short sentence, what does this store sell?\nAnswer:"
    )


def run_with_timeout(fn, timeout_s):
    res: dict = {}
    def _t():
        try: res["v"] = fn()
        except Exception as e: res["e"] = e
    th = threading.Thread(target=_t, daemon=True)
    th.start(); th.join(timeout=timeout_s)
    if th.is_alive():
        return None, TimeoutError(f"generate() exceeded {timeout_s:.0f}s")
    return res.get("v"), res.get("e")


def main():
    print(f"Model  : {MODEL_PATH}")
    print(f"Device : {DEVICE}")
    print(f"Timeout: {TIMEOUT_S}s per step")
    print(f"Steps  : {STEPS}")

    print("\nLoading tokenizer ...")
    try:
        tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    print(f"\n{'input_tok':<12} {'load_s':<10} {'gen_s':<10} {'tok/s':<10} {'status':<8} preview")
    print("-" * 85)

    last_ok = 0
    for target in STEPS:
        prompt = build_prompt(tok, target)
        n_in = len(tok.encode(prompt, add_special_tokens=False))

        t_load = time.perf_counter()
        try:
            # Minimal: no plugin config (mirrors Windows code)
            pipe = ov_genai.LLMPipeline(str(MODEL_PATH), device=DEVICE)
        except Exception as e:
            print(f"{n_in:<12} {'LOAD_FAIL':<10} {'':<10} {'':<10} {'FAIL':<8} {str(e)[:50]}")
            break
        load_s = time.perf_counter() - t_load

        def _gen():
            return pipe.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, do_sample=False)

        t_gen = time.perf_counter()
        result, err = run_with_timeout(_gen, TIMEOUT_S)
        gen_s = time.perf_counter() - t_gen

        if err:
            print(f"{n_in:<12} {load_s:<10.1f} {gen_s:<10.1f} {'—':<10} {'FAIL':<8} {repr(str(err))[:50]}")
            del pipe; gc.collect()
            break

        out = result if isinstance(result, str) else str(result)
        n_out = len(tok.encode(out, add_special_tokens=False))
        tps = (n_in + n_out) / gen_s if gen_s > 0 else 0
        print(f"{n_in:<12} {load_s:<10.1f} {gen_s:<10.1f} {tps:<10.1f} {'ok':<8} {out.strip()[:50]}")
        last_ok = n_in

        del pipe; gc.collect()

    print(f"\nMax input tokens within {TIMEOUT_S:.0f}s: {last_ok}")


if __name__ == "__main__":
    main()
