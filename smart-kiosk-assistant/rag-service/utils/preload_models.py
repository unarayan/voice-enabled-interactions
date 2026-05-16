import logging
import time

from pipeline import get_shared_pipeline


logger = logging.getLogger(__name__)


def preload_models() -> None:
    """Trigger model load and a tiny warmup generation on the LLM device.

    The first inference after model load triggers GPU shader JIT compilation.
    Running a small warmup here means the first real user request — which
    may have a large RAG prompt — won't hit the cold-compile stall.
    """
    pl = get_shared_pipeline()
    if "GPU" not in pl.device.upper():
        return
    try:
        logger.info("[warmup] running GPU warmup inference")
        t0 = time.perf_counter()
        pl.generate_from_prompt("Warmup: list three common grocery items.", max_tokens=8)
        logger.info("[warmup] done in %.1fs", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[warmup] inference failed (non-fatal): %s", exc)
