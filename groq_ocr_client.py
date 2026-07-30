"""
Groq-backed drop-in replacement for the google.generativeai (Gemini) surface
this codebase used to call directly.

Mirrors just enough of the `genai` API — `configure()`, `GenerativeModel`,
`types.GenerationConfig`, `model.generate_content(parts).text` — that the
existing call sites in gemini_ocr_extract.py, sota_extraction_engine.py, and
confidence_aware_llm.py work unchanged after switching GEMINI_API_KEY over
to GROQ_API_KEY. Vision calls (image + prompt) route to a Groq vision model;
text-only calls route to a Groq text model.
"""

import io
import base64
from typing import Optional

from PIL import Image as _PILImage
from groq import Groq

# Groq vision-capable model (accepts image + text prompts)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Groq text-only model (plain prompts, no image)
TEXT_MODEL = "llama-3.3-70b-versatile"

_client: Optional[Groq] = None


def configure(api_key: str):
    """Mirrors genai.configure(api_key=...)."""
    global _client
    # max_retries defaults to 2 in the Groq SDK — on a transient 429/5xx that's up to
    # 3 total attempts with backoff, silently tripling latency on every call site's
    # timeout budget. One retry is enough to ride out a blip; capping it here keeps
    # a real failure (e.g. rate limit) fast so it surfaces as a job error quickly
    # instead of stalling the request for tens of seconds first.
    _client = Groq(api_key=api_key, max_retries=1)


class types:
    class GenerationConfig:
        """Mirrors genai.types.GenerationConfig(...)."""
        def __init__(self, temperature: float = 0.0, max_output_tokens: int = 2048):
            self.temperature = temperature
            self.max_output_tokens = max_output_tokens


class _Response:
    def __init__(self, text: str):
        self.text = text


# Longest side to send to the vision model. Phone photos routinely come in at
# 3000-4000px+; that's far more resolution than printed document text needs to
# stay legible, and every extra pixel adds upload time plus image-token
# processing time on Groq's side. Raised from an initial 1600 to 2048 after a
# user-reported accuracy regression — 1600 risked shrinking small print below
# legible size on uncropped phone photos where the document doesn't fill the
# frame. 2048 is a more conservative trade: still a real payload/latency win
# over full-resolution, with more margin for small text to stay readable.
_MAX_VISION_DIMENSION = 2048


def _to_data_url(image: "_PILImage.Image") -> str:
    image = image.convert("RGB")
    if max(image.size) > _MAX_VISION_DIMENSION:
        scale = _MAX_VISION_DIMENSION / max(image.size)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, _PILImage.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GenerativeModel:
    """Mirrors genai.GenerativeModel(model_name, generation_config=...)."""

    def __init__(self, model_name: str = None, generation_config: "types.GenerationConfig" = None):
        # model_name kept for interface compatibility but ignored — the
        # actual Groq model is picked per-call based on whether an image
        # is present (vision vs text-only).
        cfg = generation_config or types.GenerationConfig()
        self.temperature = cfg.temperature
        self.max_output_tokens = cfg.max_output_tokens

    def generate_content(self, parts, request_options: dict = None) -> _Response:
        """Mirrors genai's model.generate_content(parts, request_options={'timeout': N}).text"""
        if _client is None:
            raise RuntimeError("groq_ocr_client.configure(api_key) was not called — GROQ_API_KEY missing")

        timeout = (request_options or {}).get("timeout", 60)

        if isinstance(parts, str):
            parts = [parts]

        image = None
        prompt_parts = []
        for p in parts:
            if isinstance(p, _PILImage.Image):
                image = p
            else:
                prompt_parts.append(str(p))
        prompt = "\n".join(prompt_parts)

        if image is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
            ]
            model = VISION_MODEL
        else:
            content = prompt
            model = TEXT_MODEL

        completion = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            timeout=timeout,
        )
        return _Response(completion.choices[0].message.content.strip())
