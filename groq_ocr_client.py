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

# Groq vision-capable model (accepts image + text prompts).
# Llama 4 Maverick is Meta's larger, more capable multimodal model (128 experts
# vs Scout's 16) — used here instead of Scout for the strongest document/text
# reading accuracy Groq currently offers, per user request for the best
# available quality rather than the fastest/cheapest option.
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
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


# Sending images to Groq Vision at full resolution, unmodified. An earlier
# session tried downscaling to speed up requests (1600px, then 2048px) but it
# measurably hurt extraction accuracy on real phone photos — small print was
# landing below legible size — so per user direction this was reverted back
# to full-resolution, matching the pipeline's original (better) behavior.
def _to_data_url(image: "_PILImage.Image") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
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
        """Mirrors genai's model.generate_content(parts, request_options={'timeout': N}).text

        request_options also accepts 'json_mode': True (not part of the genai
        surface being mirrored, but call sites here opt into it) — this asks
        Groq to guarantee syntactically valid JSON output instead of relying on
        regex/markdown-fence cleanup of free-form text, which is a real source
        of parse failures on otherwise-correct extractions. Only set this for
        prompts that actually ask for JSON — forcing it on a plain-text prompt
        (e.g. document-type detection) would break that call.
        """
        if _client is None:
            raise RuntimeError("groq_ocr_client.configure(api_key) was not called — GROQ_API_KEY missing")

        opts = request_options or {}
        timeout = opts.get("timeout", 60)
        json_mode = opts.get("json_mode", False)

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

        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            timeout=timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = _client.chat.completions.create(**kwargs)
        return _Response(completion.choices[0].message.content.strip())
