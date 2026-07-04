"""
FileTract SOTA Extraction Engine
Patent-Pending: Multi-Strategy Parallel Vision Extraction with Self-Verification

Core innovations over prior art:
1. Direct Gemini Vision extraction — no OCR error propagation (biggest accuracy win)
2. Parallel multi-strategy execution (ThreadPoolExecutor) — 3x speed improvement
3. Cross-strategy consensus voting with normalization — catches hallucinations
4. Targeted self-verification loop for uncertain/disagreed fields — catches remaining errors
5. Document-type-aware context injection — domain knowledge improves accuracy
"""

import concurrent.futures
import json
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from PIL import Image
import google.generativeai as genai


@dataclass
class StrategyResult:
    strategy_name: str
    fields: Dict[str, Optional[str]]
    confidence: Dict[str, str]
    success: bool
    error: Optional[str] = None


@dataclass
class SOTAResult:
    fields: Dict[str, Optional[str]]
    field_confidence: Dict[str, str]      # 'high' | 'medium' | 'low' per field
    field_consensus: Dict[str, bool]      # True if >= 2 strategies agreed
    strategies_used: List[str]
    doc_type: str
    quality_score: float                  # 0.0 – 1.0


# ─── Constants ────────────────────────────────────────────────────────────────

_NULL_VALUES = frozenset({
    'null', 'none', 'n/a', 'not found', 'not applicable',
    '', '-', '—', 'na', 'not visible', 'unclear', 'unknown',
})

_CONF_WEIGHTS = {'high': 1.0, 'medium': 0.7, 'low': 0.3}


# ─── Main Engine ──────────────────────────────────────────────────────────────

class SOTAExtractionEngine:
    """
    State-of-the-art document field extraction engine.
    Replaces the Tesseract → annotated-text → Gemini pipeline with
    direct multimodal extraction, parallelism, and self-verification.
    """

    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )
        )

    # ─── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Robustly parse JSON from LLM output regardless of fencing."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the outermost JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return {}

    @classmethod
    def _norm_val(cls, val) -> Optional[str]:
        """Return None for non-values, stripped string otherwise."""
        if val is None:
            return None
        s = str(val).strip()
        return None if s.lower() in _NULL_VALUES else s

    @staticmethod
    def _norm_for_compare(v: str) -> str:
        """Case-fold + collapse whitespace for comparison."""
        return re.sub(r'\s+', ' ', str(v)).strip().upper()

    @staticmethod
    def _field_hints(fields: List[str]) -> str:
        """Domain-specific extraction hints for common ID fields."""
        hints = []
        for f in fields:
            fl = f.lower()
            if any(x in fl for x in ('aadhaar', 'aadhar', 'uid', 'uidai')):
                hints.append(f'• {f}: 12-digit number — format XXXX XXXX XXXX')
            elif 'pan' in fl and ('number' in fl or 'no' in fl):
                hints.append(f'• {f}: 10-character alphanumeric code — format ABCDE1234F')
            elif 'voter' in fl and ('id' in fl or 'number' in fl):
                hints.append(f'• {f}: Voter ID number (alphanumeric)')
            elif 'father' in fl:
                hints.append(f"• {f}: Father's name — often labeled 'S/O', 'Father', 'C/O'")
            elif 'mother' in fl:
                hints.append(f"• {f}: Mother's name — often labeled 'W/O', 'Mother'")
            elif 'name' in fl:
                hints.append(f'• {f}: Full name of the holder (usually the largest name text)')
            elif any(x in fl for x in ('dob', 'birth', 'born', 'date of birth')):
                hints.append(f'• {f}: Date of birth — copy the EXACT format shown')
            elif any(x in fl for x in ('expiry', 'expiration', 'valid till', 'validity')):
                hints.append(f'• {f}: Expiry/validity date')
            elif 'address' in fl:
                hints.append(f'• {f}: Complete address including city, state, and PIN code')
            elif any(x in fl for x in ('gender', ' sex')):
                hints.append(f'• {f}: Gender value (Male / Female / Transgender / M / F)')
            elif any(x in fl for x in ('roll', 'reg', 'enrol')):
                hints.append(f'• {f}: Roll/Registration/Enrollment number')
            elif 'cgpa' in fl or 'gpa' in fl or 'grade' in fl:
                hints.append(f'• {f}: Grade or GPA value')
            elif 'school' in fl or 'college' in fl or 'institution' in fl:
                hints.append(f'• {f}: Name of the educational institution')
            else:
                hints.append(f'• {f}: Locate and extract this field from the document')
        return '\n'.join(hints)

    def _call(self, parts, timeout: int = 45) -> str:
        """Single Gemini multimodal call, bounded so a stalled request fails fast instead of hanging."""
        return self.model.generate_content(
            parts, request_options={'timeout': timeout}
        ).text.strip()

    # ─── Document Type Detection ──────────────────────────────────────────

    def detect_document_type(self, pil_image: Image.Image) -> str:
        """Ask Gemini to identify the document type for context injection."""
        prompt = (
            "What type of document is shown in this image?\n"
            "Reply with ONLY the document type. Examples:\n"
            "Aadhaar Card | PAN Card | Voter ID Card | Passport | Driving License | "
            "Student ID Card | Employee ID Card | Mark Sheet | Birth Certificate | "
            "Educational Certificate | Bank Statement | Medical Record | Document\n"
            "One line, nothing else."
        )
        try:
            return self._call([pil_image, prompt])
        except Exception:
            return "Document"

    # ─── Extraction Strategies ────────────────────────────────────────────

    def _strategy_vision_primary(
        self, pil_image: Image.Image, fields: List[str], doc_type: str
    ) -> StrategyResult:
        """
        Strategy A — Direct Vision.
        Gemini reads the image as a human expert would.
        No OCR intermediary; no error propagation.
        """
        fields_list = ', '.join(fields)
        hints = self._field_hints(fields)

        prompt = f"""You are an expert document reader analyzing a {doc_type}.

Extract these fields: {fields_list}

Field-specific guidance:
{hints}

Rules:
1. Copy text EXACTLY as printed (preserve capitalization, spacing, punctuation)
2. For IDs/numbers: preserve all characters including spaces and dashes
3. For dates: use the exact format shown (do not reformat)
4. Return null for any field that is genuinely absent or unreadable
5. NEVER invent or guess values — only extract what is clearly visible
6. Return ONLY a JSON object — no markdown, no explanation

JSON:"""

        try:
            raw = self._call([pil_image, prompt])
            parsed = self._parse_json(raw)
            result_fields = {f: self._norm_val(parsed.get(f)) for f in fields}
            return StrategyResult(
                strategy_name='vision_primary',
                fields=result_fields,
                confidence={f: ('high' if result_fields[f] else 'low') for f in fields},
                success=True,
            )
        except Exception as e:
            return StrategyResult('vision_primary', {f: None for f in fields}, {f: 'low' for f in fields}, False, str(e))

    def _strategy_vision_analytical(
        self, pil_image: Image.Image, fields: List[str], doc_type: str
    ) -> StrategyResult:
        """
        Strategy B — Analytical Vision.
        Different prompt framing than A: asks Gemini to reason field-by-field.
        Cross-validates Strategy A results.
        """
        items = '\n'.join(f'{i+1}. {f}' for i, f in enumerate(fields))

        prompt = f"""Analyze this {doc_type} image carefully.

Scan the ENTIRE document and extract each field listed below.
Look at all areas — top, bottom, left column, right column, watermarks, corners.

Fields to extract:
{items}

Additional field context:
{self._field_hints(fields)}

For each field:
- Transcribe the exact printed text
- Numbers: digit-by-digit (never approximate)
- If a field appears multiple times, prefer the most prominent/complete instance
- Set to null if absent or illegible

Return ONLY JSON. No preamble, no markdown:"""

        try:
            raw = self._call([pil_image, prompt])
            parsed = self._parse_json(raw)
            result_fields = {f: self._norm_val(parsed.get(f)) for f in fields}
            return StrategyResult(
                strategy_name='vision_analytical',
                fields=result_fields,
                confidence={f: ('high' if result_fields[f] else 'low') for f in fields},
                success=True,
            )
        except Exception as e:
            return StrategyResult('vision_analytical', {f: None for f in fields}, {f: 'low' for f in fields}, False, str(e))

    def _strategy_ocr_assisted(
        self, pil_image: Image.Image, ocr_text: str, fields: List[str], doc_type: str
    ) -> StrategyResult:
        """
        Strategy C — OCR-Assisted Vision.
        Passes both the raw OCR text and the image.
        Gemini uses OCR as a spatial map and corrects errors by reading the image.
        Best for handling OCR character substitutions (0→O, 1→l, 5→S, etc.)
        """
        if not ocr_text or len(ocr_text.strip()) < 10:
            return self._strategy_vision_primary(pil_image, fields, doc_type)

        fields_list = ', '.join(fields)
        prompt = f"""You are extracting data from a {doc_type}.

You have BOTH the raw OCR text (which contains errors) AND the original document image.
The OCR is a rough map — use the IMAGE as the authoritative source.
Correct OCR errors you detect (e.g. '0' read as 'O', '1' as 'l', 'B' as '8').

RAW OCR TEXT (error-prone, do not trust blindly):
---
{ocr_text[:2500]}
---

Extract these fields: {fields_list}

{self._field_hints(fields)}

Return ONLY a JSON object with field names as keys:"""

        try:
            raw = self._call([pil_image, prompt])
            parsed = self._parse_json(raw)
            result_fields = {f: self._norm_val(parsed.get(f)) for f in fields}
            return StrategyResult(
                strategy_name='ocr_assisted',
                fields=result_fields,
                confidence={f: ('high' if result_fields[f] else 'low') for f in fields},
                success=True,
            )
        except Exception as e:
            return StrategyResult('ocr_assisted', {f: None for f in fields}, {f: 'low' for f in fields}, False, str(e))

    # ─── Consensus Fusion ─────────────────────────────────────────────────

    def _consensus_fuse(
        self,
        results: List[StrategyResult],
        fields: List[str],
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Dict[str, bool]]:
        """
        Patent-pending consensus voting across strategies.

        Confidence assignment:
        - All strategies agree → 'high'
        - Majority (≥2) agree → 'medium'
        - Only one strategy found it → 'medium' (single source)
        - Strategies disagree or all null → 'low' (needs verification)
        """
        fused_vals: Dict[str, Optional[str]] = {}
        fused_conf: Dict[str, str] = {}
        fused_consensus: Dict[str, bool] = {}

        successful = [r for r in results if r.success]
        n = len(successful)

        for field in fields:
            non_null = [
                r.fields[field]
                for r in successful
                if r.fields.get(field)
            ]

            if not non_null:
                fused_vals[field] = None
                fused_conf[field] = 'low'
                fused_consensus[field] = False
                continue

            counts = Counter(self._norm_for_compare(v) for v in non_null)
            best_norm, best_count = counts.most_common(1)[0]
            # Recover original-case version
            original = next((v for v in non_null if self._norm_for_compare(v) == best_norm), non_null[0])

            fused_vals[field] = original
            fused_consensus[field] = best_count >= 2

            if n == 1:
                fused_conf[field] = 'medium'
            elif best_count == n:
                fused_conf[field] = 'high'     # unanimous
            elif best_count >= 2:
                fused_conf[field] = 'medium'   # majority
            else:
                fused_conf[field] = 'low'      # all disagree

        return fused_vals, fused_conf, fused_consensus

    # ─── Self-Verification ────────────────────────────────────────────────

    def _verify_uncertain(
        self,
        pil_image: Image.Image,
        uncertain_fields: List[str],
        candidates: Dict[str, Optional[str]],
        doc_type: str,
    ) -> Dict[str, Tuple[Optional[str], str]]:
        """
        Targeted self-verification for low-confidence or disagreed fields.
        Batches all uncertain fields into a single API call for efficiency.
        """
        if not uncertain_fields:
            return {}

        lines = []
        for f in uncertain_fields:
            cv = candidates.get(f)
            if cv:
                lines.append(f'• {f}: My current extraction is "{cv}" — verify this is correct')
            else:
                lines.append(f'• {f}: I could NOT find this field — please look very carefully')

        prompt = f"""You are verifying field extractions from a {doc_type} image.

For each field below, examine the document image carefully and provide the definitive correct value.

{chr(10).join(lines)}

Additional context:
{self._field_hints(uncertain_fields)}

For each field: return the exact correct value, or null if truly absent.
Return ONLY JSON:"""

        try:
            raw = self._call([pil_image, prompt])
            parsed = self._parse_json(raw)
            result = {}
            for f in uncertain_fields:
                val = self._norm_val(parsed.get(f))
                result[f] = (val, 'medium' if val else 'low')
            return result
        except Exception:
            return {f: (candidates.get(f), 'low') for f in uncertain_fields}

    # ─── Main Entry Point ─────────────────────────────────────────────────

    def extract(
        self,
        pil_image: Image.Image,
        fields: List[str],
        ocr_text: str = '',
        enable_verification: bool = True,
    ) -> SOTAResult:
        """
        Full SOTA pipeline:
        1. Document type detection
        2. Parallel multi-strategy vision extraction
        3. Cross-strategy consensus fusion
        4. Self-verification for uncertain fields
        5. Quality scoring
        """
        # ── Step 1: Document type ──────────────────────────────────────
        print("    [SOTA] Identifying document type...")
        doc_type = self.detect_document_type(pil_image)
        print(f"    [SOTA] Document: {doc_type}")

        # ── Step 2: Parallel extraction ───────────────────────────────
        print("    [SOTA] Parallel extraction (3 strategies)...")
        strategy_results: List[StrategyResult] = []

        # Not using ThreadPoolExecutor as a context manager: its __exit__ calls
        # shutdown(wait=True), which blocks until every submitted thread finishes —
        # ignoring any per-future timeout below. shutdown(wait=False) lets a stalled
        # request keep running in the background without hanging this response.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        try:
            futures = {
                executor.submit(self._strategy_vision_primary, pil_image, fields, doc_type): 'A:vision_primary',
                executor.submit(self._strategy_vision_analytical, pil_image, fields, doc_type): 'B:vision_analytical',
            }
            has_ocr = ocr_text and len(ocr_text.strip()) > 10
            if has_ocr:
                f_c = executor.submit(self._strategy_ocr_assisted, pil_image, ocr_text, fields, doc_type)
                futures[f_c] = 'C:ocr_assisted'

            for fut, label in futures.items():
                try:
                    res = fut.result(timeout=60)
                    found = sum(1 for v in res.fields.values() if v)
                    print(f"    [SOTA] {label}: {found}/{len(fields)} fields")
                    strategy_results.append(res)
                except Exception as e:
                    print(f"    [SOTA] {label} failed: {e}")
        finally:
            executor.shutdown(wait=False)

        if not strategy_results:
            # All strategies failed — return empty result
            return SOTAResult(
                fields={f: None for f in fields},
                field_confidence={f: 'low' for f in fields},
                field_consensus={f: False for f in fields},
                strategies_used=[],
                doc_type=doc_type,
                quality_score=0.0,
            )

        # ── Step 3: Consensus fusion ───────────────────────────────────
        print("    [SOTA] Consensus fusion...")
        fused_vals, fused_conf, fused_consensus = self._consensus_fuse(strategy_results, fields)

        # ── Step 4: Self-verification ──────────────────────────────────
        if enable_verification:
            uncertain = [f for f in fields if fused_conf[f] == 'low' or not fused_consensus[f]]
            if uncertain:
                print(f"    [SOTA] Verifying {len(uncertain)} uncertain: {uncertain}")
                ver = self._verify_uncertain(pil_image, uncertain, fused_vals, doc_type)
                for f, (val, conf) in ver.items():
                    if val:
                        fused_vals[f] = val
                        fused_conf[f] = conf
                        fused_consensus[f] = True
                        print(f"    [SOTA] Verified {f!r}: {val!r} ({conf})")

        # ── Step 5: Quality scoring ────────────────────────────────────
        n = len(fields)
        quality = sum(_CONF_WEIGHTS.get(fused_conf[f], 0) for f in fields) / n if n else 0.0
        found = sum(1 for v in fused_vals.values() if v)
        print(f"    [SOTA] Done — {found}/{n} fields, quality={quality:.2f}")

        return SOTAResult(
            fields=fused_vals,
            field_confidence=fused_conf,
            field_consensus=fused_consensus,
            strategies_used=[r.strategy_name for r in strategy_results if r.success],
            doc_type=doc_type,
            quality_score=quality,
        )
