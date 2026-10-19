"""Shared Gemini client/model selection helpers."""
import logging

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


def _model_id(name: str) -> str:
    return name.removeprefix("models/")


def configured_gemini_models() -> list[str]:
    """Return the configured model candidates in priority order."""
    configured = _model_id(settings.GEMINI_MODEL)
    return list(dict.fromkeys([
        configured,
        *(_model_id(model) for model in settings.GEMINI_MODEL_FALLBACKS),
    ]))


def available_gemini_models() -> set[str] | None:
    """List models available to this key, or None when discovery is unavailable."""
    genai.configure(api_key=settings.GEMINI_API_KEY)
    try:
        return {
            _model_id(model.name)
            for model in genai.list_models()
            if "generateContent" in (model.supported_generation_methods or [])
        }
    except Exception:
        logger.warning(
            "Could not list Gemini models; generation will try configured fallbacks",
            exc_info=True,
        )
        return None


def gemini_model_candidates() -> list[str]:
    """Return usable candidates, preserving configured priority.

    Model discovery is best-effort because some API keys/projects can generate
    content while rejecting the model-list endpoint.
    """
    candidates = configured_gemini_models()
    available = available_gemini_models()
    if available is None:
        return candidates
    preferred = [candidate for candidate in candidates if candidate in available]
    discovered = sorted(
        model
        for model in available
        if model.startswith("gemini-") and model not in preferred
    )
    if discovered:
        logger.info(
            "Gemini model candidates for this key: preferred=%s discovered_fallbacks=%s",
            preferred,
            discovered,
        )
    return preferred + discovered


def resolve_gemini_model() -> str:
    """Return the first model enabled for this API key."""
    candidates = gemini_model_candidates()
    configured = _model_id(settings.GEMINI_MODEL)

    if candidates:
        if candidates[0] != configured:
            logger.warning(
                "Configured Gemini model %s is unavailable; using %s instead",
                configured,
                candidates[0],
            )
        return candidates[0]

    raise RuntimeError(
        f"No configured Gemini model is available for generateContent. "
        f"Configured={configured}; candidates={configured_gemini_models()}"
    )


def get_gemini_model(model_name: str | None = None):
    """Create a GenerativeModel for the selected compatible model."""
    return genai.GenerativeModel(model_name or resolve_gemini_model())
