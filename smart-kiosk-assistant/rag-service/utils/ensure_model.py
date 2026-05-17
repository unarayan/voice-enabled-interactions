from __future__ import annotations

import logging
import os
import shutil

from huggingface_hub import snapshot_download
from optimum.exporters.openvino import main_export

from utils.config_loader import config


logger = logging.getLogger(__name__)
_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slugify(model_name: str, suffix: str | None = None) -> str:
    slug = model_name.replace("/", "_")
    return f"{slug}__{suffix}" if suffix else slug


def _resolve_service_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_SERVICE_ROOT, path))


def _llm_model_exists(output_dir: str) -> bool:
    xml_path = os.path.join(output_dir, "openvino_model.xml")
    bin_path = os.path.join(output_dir, "openvino_model.bin")
    return (
        os.path.isfile(xml_path)
        and os.path.getsize(xml_path) > 0
        and os.path.isfile(bin_path)
        and os.path.getsize(bin_path) > 0
    )


def _llm_hf_tokenizer_exists(output_dir: str) -> bool:
    """Check that HuggingFace tokenizer files are present (needed by OVModelForCausalLM)."""
    return os.path.isfile(os.path.join(output_dir, "tokenizer_config.json"))


def _llm_export_ready(output_dir: str) -> bool:
    return _llm_model_exists(output_dir) and _llm_hf_tokenizer_exists(output_dir)


def _download_repo(repo_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    return output_dir


def _reset_output_dir(output_dir: str) -> None:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def _export_openvino_model(
    model_name: str,
    output_dir: str,
    precision: str = "int8",
) -> str:
    _reset_output_dir(output_dir)
    logger.info(
        "Exporting %s → %s via main_export (precision=%s)",
        model_name,
        output_dir,
        precision,
    )
    main_export(
        model_name_or_path=model_name,
        output=output_dir,
        precision=precision,
    )
    if not _llm_model_exists(output_dir):
        raise RuntimeError(
            f"OpenVINO export failed for {model_name}: main_export did not produce a valid IR in {output_dir}"
        )
    return output_dir


def get_llm_model_path() -> str:
    llm_cfg = config.models.llm
    precision = getattr(llm_cfg, "weight_format", "int8")
    return os.path.join(
        _resolve_service_path(llm_cfg.models_base_path),
        _slugify(llm_cfg.hf_id, precision),
    )


def get_embedding_model_path() -> str:
    emb_cfg = config.models.embedding
    return os.path.join(
        _resolve_service_path(emb_cfg.models_base_path),
        "sentence_transformers",
        _slugify(emb_cfg.hf_id),
    )


def ensure_llm_model(force: bool = False) -> str:
    llm_cfg = config.models.llm
    output_dir = get_llm_model_path()
    hf_id = llm_cfg.hf_id

    if not force and _llm_export_ready(output_dir):
        logger.info("Using cached OpenVINO LLM export at %s", output_dir)
        return output_dir

    precision = getattr(llm_cfg, "weight_format", "int8")
    logger.info(
        "Exporting %s → OpenVINO IR at %s (precision=%s). "
        "This takes a few minutes on first run.",
        hf_id, output_dir, precision,
    )
    _export_openvino_model(hf_id, output_dir, precision=precision)

    if not _llm_export_ready(output_dir):
        raise RuntimeError(
            f"LLM model export completed but required OpenVINO artifacts are missing from {output_dir}"
        )
    logger.info("LLM ready at %s", output_dir)
    return output_dir


def ensure_embedding_model(force: bool = False) -> str:
    emb_cfg = config.models.embedding
    output_dir = get_embedding_model_path()
    if not force and os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        logger.info("Using cached embedding model at %s", output_dir)
        return output_dir

    logger.info("Downloading sentence-transformers embedding model %s to %s", emb_cfg.hf_id, output_dir)
    _download_repo(emb_cfg.hf_id, output_dir)
    return output_dir


def ensure_model(force: bool = False) -> None:
    ensure_llm_model(force=force)
    ensure_embedding_model(force=force)


def resolve_embedding_model_source() -> str:
    emb_cfg = config.models.embedding
    output_dir = get_embedding_model_path()
    if os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        return output_dir
    return emb_cfg.hf_id
