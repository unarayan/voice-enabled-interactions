"""Standalone export script — calls ensure_llm_model(force=True) to re-export
the Qwen2.5-7B-Instruct model via optimum-intel main_export.

Usage:
  .venv_export/bin/python3 export_model.py [--force]

  --force  : wipe and re-export even if the model already exists
  (default): re-export only if missing or incomplete
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ── make sure rag-service root is on sys.path ─────────────────────────────────
_SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_model")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LLM to OpenVINO IR via ensure_model")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Wipe existing export and re-export from scratch",
    )
    ap.add_argument(
        "--cli",
        action="store_true",
        help="Use optimum-cli subprocess instead of main_export Python API",
    )
    args = ap.parse_args()

    if args.cli:
        _export_via_cli()
        return

    from utils.ensure_model import ensure_llm_model, get_llm_model_path
    from utils.config_loader import config

    out_path = get_llm_model_path()
    logger.info("Target model path : %s", out_path)
    logger.info("HF model id       : %s", config.models.llm.hf_id)
    logger.info("Weight format     : %s", getattr(config.models.llm, "weight_format", "int8"))
    logger.info("Force re-export   : %s", args.force)

    try:
        result = ensure_llm_model(force=args.force)
        logger.info("Export complete → %s", result)
    except Exception as exc:
        logger.error("Export failed: %s", exc, exc_info=True)
        sys.exit(1)


def _export_via_cli() -> None:
    """Re-export using the optimum-cli subprocess (alternative path)."""
    import subprocess

    from utils.ensure_model import get_llm_model_path
    from utils.config_loader import config

    hf_id = config.models.llm.hf_id
    weight_fmt = getattr(config.models.llm, "weight_format", "int8")
    out_path = get_llm_model_path()

    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", hf_id,
        "--weight-format", weight_fmt,
        "--trust-remote-code",
        "--task", "text-generation-with-past",
        out_path,
    ]
    logger.info("Running: %s", " ".join(cmd))

    # Also try the optimum-cli entry-point (preferred)
    optimum_cli = os.path.join(os.path.dirname(sys.executable), "optimum-cli")
    if os.path.isfile(optimum_cli):
        cmd = [
            optimum_cli, "export", "openvino",
            "--model", hf_id,
            "--weight-format", weight_fmt,
            "--trust-remote-code",
            "--task", "text-generation-with-past",
            out_path,
        ]
        logger.info("Found optimum-cli, using: %s", " ".join(cmd))

    import shutil
    if os.path.isdir(out_path):
        logger.info("Removing existing export at %s", out_path)
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("optimum-cli export exited with code %d", result.returncode)
        sys.exit(result.returncode)
    logger.info("optimum-cli export complete → %s", out_path)


if __name__ == "__main__":
    main()
