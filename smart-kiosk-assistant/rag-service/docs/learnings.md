# RAG Service — Key Learnings

## 1. Intel iGPU + OpenVINO GenAI on Linux

| Finding | Detail |
|---|---|
| **Driver version is the hard ceiling** | compute-runtime ≤ 24.39 causes `generate()` to hang indefinitely on prompts > ~3 000 tokens on Linux iGPU. No error — just hangs. Driver 26.18+ resolves this; 5 000-token prompts complete in ~24 s on Arrow Lake. |
| **WDDM paging is a Linux-only problem** | On Windows the driver pages out KV cache to system RAM. On Linux there is no equivalent, so every byte of the prompt must fit in the GPU's local memory. The driver version determines how much of that memory is addressable. |
| **Cold-start shader stall** | The first `generate()` call after model load triggers GPU JIT shader compilation. This can add 30–60 s to the first request. Fix: run a short warmup generation (~8 new tokens) on startup to prime the kernel cache before any real request arrives. |
| **Plugin config for memory** | Setting `INFERENCE_PRECISION_HINT=f16`, `KV_CACHE_PRECISION=f16`, and `NUM_STREAMS=1` roughly halves GPU memory usage vs f32 defaults and prevents multiple parallel execution buffers from exhausting VRAM. |
| **CACHE_DIR is essential for fast restarts** | Persisting compiled OpenVINO kernels with `CACHE_DIR` means subsequent `LLMPipeline` loads (e.g., after a CL_OUT_OF_RESOURCES reload) skip JIT compilation entirely — seconds instead of minutes. |
| **CL_OUT_OF_RESOURCES recovery** | On long-running workloads the iGPU can return CL_OUT_OF_RESOURCES. Catching this, deleting the pipeline, calling `gc.collect()`, and recreating it recovers without a full restart. With CACHE_DIR the reload takes a few seconds. |

## 2. OpenVINO GenAI Streaming API Changes

| Version | Streamer contract |
|---|---|
| ≤ 2025.x | Subclass `StreamerBase`, implement `put(token_id: int) → bool` and `end()`. |
| 2026.1+ | `put()` is gone. `write(token) → StreamingStatus` replaces it. The token argument can be a single int or a sequence. |
| **Best practice** | Pass a plain Python `Callable[[str], bool]` as the `streamer` argument instead of subclassing `StreamerBase`. `LLMPipeline.generate()` accepts both. The callable receives already-decoded text chunks, which is simpler and API-version-agnostic. |

## 3. Chunking Strategy

| Finding | Detail |
|---|---|
| **LLM-assisted chunking on iGPU is dangerous** | Sending 1 200–5 000-token chunking prompts to the same GPU that handles inference freezes the service during ingest, especially with older drivers. Gate it or remove it. |
| **Semantic (embedding) chunking is GPU-free** | Using the embedding model for similarity-based splitting runs entirely on CPU and is fast. It is the correct default for an iGPU deployment. |
| **Markdown headers are reliable hard boundaries** | Splitting on `# Heading` lines before embedding-based splitting keeps sections together, making it work well on structured documents without any tuning. |
| **Fixed-size chunking is underrated** | Recursive char split (paragraph → line → sentence → word) is deterministic, reproducible, and content-agnostic. It should always be an available fallback, especially for tabular or code-like content. |
| **Overlap should snap to line boundaries** | Prepending the last N chars of the previous chunk creates context continuity but can start mid-word. Snapping the overlap prefix to the next newline avoids garbled sentence starts. |

## 4. Container / Dockerfile

| Finding | Detail |
|---|---|
| **Driver install inside the image is fragile** | Intel's apt key URL and package names have changed across releases (`intel-igc-core` → `intel-igc-core-2`, key URL changes). Installing inside Docker ties the image to a specific driver version that may not match the host kernel. |
| **Correct approach: host driver + /dev/dri passthrough** | Install the driver once on the host. Mount `/dev/dri` into the container and add the `video` + `render` groups in `docker-compose.yml`. The container uses the host driver stack transparently. |
| **group_add matters** | Without `group_add: [video, render]` in docker-compose, OpenVINO cannot open the GPU device node from inside the container even when `/dev/dri` is mounted. |

## 5. Architecture / Code Quality

| Finding | Detail |
|---|---|
| **God-class pipeline is a smell** | Putting LLM lifecycle, retrieval, prompt building, chunker wiring, and generation all in one class makes each concern hard to test and change. A thin orchestrator that delegates to focused components is easier to maintain. |
| **Config keys should match current code** | Stale config keys (removed features, renamed fields) silently pass through and confuse operators. Keep config.yaml and the code in sync; remove any key that has no reader. |
| **Container config should only override deltas** | `config.container.yaml` should contain only the keys that differ from the base `config.yaml`. Repeating the entire config in both files doubles the maintenance surface. |
| **Smoke tests should cover both strategies** | Unit tests for the chunker should exercise fixed-size and semantic strategies independently, plus structural markers (headers), to catch regressions without a GPU. |
