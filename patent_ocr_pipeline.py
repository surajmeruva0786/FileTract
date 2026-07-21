"""
FileTract - SOTA Patent-Eligible OCR Pipeline v3
Parallel Vision + Self-Verification + Smart Preprocessing

Patent-Pending Innovations:
1. Parallel multi-strategy Groq Vision extraction (ThreadPoolExecutor)
2. Cross-strategy consensus voting with normalization
3. Targeted self-verification loop for uncertain fields
4. Document-type-aware context injection
5. Full-image deskew + illumination normalization before extraction
6. Perspective correction for ID cards captured by phone cameras

Architecture (replaces sequential per-region Tesseract re-OCR):
  Input → Preprocess → [Vision_A ∥ Vision_B ∥ OCR_Assist] → Fuse → Verify → Score
  Expected time: 3–10s (vs 15–45s previously)
"""

import os
import sys
import json
import io
import time
import concurrent.futures
from typing import List, Dict, Any

import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import groq_ocr_client as genai
from dotenv import load_dotenv

from sota_extraction_engine import SOTAExtractionEngine, SOTAResult
from image_quality_analyzer import ImageQualityAnalyzer
from confidence_aware_llm import FieldWithQuality

load_dotenv()

# ─── Tesseract auto-detection ─────────────────────────────────────────────────

def _is_docker() -> bool:
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            return 'docker' in content or 'containerd' in content
    except Exception:
        return False

if _is_docker() or os.path.exists('/.dockerenv'):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
elif os.environ.get('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']
elif os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# ─── Groq configuration ────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env")
    sys.exit(1)

genai.configure(api_key=GROQ_API_KEY)

# ─── Image loading ────────────────────────────────────────────────────────────

def _load_pil_from_pdf(pdf_path: str) -> List[Image.Image]:
    """Render all PDF pages to PIL images at 300 DPI."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def _load_pil_from_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert('RGB')


# ─── Fast OCR helper (run in parallel with vision strategies) ─────────────────

def _run_tesseract_fast(pil_image: Image.Image) -> str:
    """
    Run Tesseract OCR on the full image as a supplemental text source.
    This runs in parallel with vision strategies — it is no longer the bottleneck.
    """
    try:
        text = pytesseract.image_to_string(pil_image, lang='eng',
                                            config='--psm 6 --oem 3')
        return text
    except Exception:
        return ''


# ─── Core pipeline ────────────────────────────────────────────────────────────

def extract_image_with_sota_pipeline(image_path: str) -> Dict:
    """
    Process a single image through the SOTA pipeline.

    Stages:
      1. Load and preprocess (deskew, illumination, card crop, upscale)
      2. Run Tesseract OCR in parallel (supplemental — not primary)
      3. Return preprocessed image + OCR text for the SOTA engine
    """
    print(f"  📸 Loading and preprocessing image...")
    quality_analyzer = ImageQualityAnalyzer()

    raw_img = _load_pil_from_image(image_path)

    # Preprocessing in parallel with Tesseract (saves ~1s)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_enhance = ex.submit(quality_analyzer.enhance_document_image, raw_img)
        f_ocr = ex.submit(_run_tesseract_fast, raw_img)

        enhanced_img = f_enhance.result()
        ocr_text = f_ocr.result()

    # Attempt card/document crop on the enhanced image
    try:
        cropped = quality_analyzer.detect_and_crop_card(enhanced_img)
        if cropped.size != enhanced_img.size:
            print("  ✓ Card boundary detected — perspective-corrected")
            final_img = cropped
        else:
            final_img = enhanced_img
    except Exception:
        final_img = enhanced_img

    return {
        'pil_image': final_img,
        'ocr_text': ocr_text,
        'raw_image': raw_img,
    }


def extract_pdf_with_sota_pipeline(pdf_path: str) -> Dict:
    """Load PDF (all pages), preprocess, and return image + OCR text."""
    print(f"  📄 Loading PDF...")
    quality_analyzer = ImageQualityAnalyzer()
    pages = _load_pil_from_pdf(pdf_path)
    print(f"    Pages: {len(pages)}")

    # Process all pages; for multi-page PDFs concatenate OCR text
    # and use page 1 image for vision (most ID documents are single-page)
    enhanced_pages = []
    ocr_parts = []

    for i, page in enumerate(pages):
        enhanced = quality_analyzer.enhance_document_image(page)
        enhanced_pages.append(enhanced)
        try:
            ocr_parts.append(f"--- PAGE {i+1} ---\n{_run_tesseract_fast(enhanced)}")
        except Exception:
            pass

    # For vision extraction, use page 1 (primary document face)
    primary_image = enhanced_pages[0]
    all_ocr_text = '\n'.join(ocr_parts)

    return {
        'pil_image': primary_image,
        'all_pages': enhanced_pages,
        'ocr_text': all_ocr_text,
    }


# ─── SOTA result → FieldWithQuality conversion ────────────────────────────────

def _sota_to_field_quality(sota: SOTAResult) -> Dict[str, FieldWithQuality]:
    """Convert SOTAResult to the FieldWithQuality format expected by app.py."""
    result = {}
    for field, value in sota.fields.items():
        conf = sota.field_confidence.get(field, 'low')
        consensus = sota.field_consensus.get(field, False)

        # Map confidence tiers
        ocr_conf_map = {'high': 0.95, 'medium': 0.78, 'low': 0.45}
        ocr_confidence = ocr_conf_map.get(conf, 0.5)

        llm_confidence = conf  # Already 'high'/'medium'/'low'

        # Quality flag
        if conf == 'high' and consensus:
            quality_flag = 'reliable'
        elif conf == 'medium' or (conf == 'high' and not consensus):
            quality_flag = 'good'
        elif value:
            quality_flag = 'uncertain'
        else:
            quality_flag = 'not-found'

        result[field] = FieldWithQuality(
            value=value,
            ocr_confidence=ocr_confidence,
            llm_confidence=llm_confidence,
            quality_flag=quality_flag,
        )
    return result


# ─── Public API (called by app.py) ───────────────────────────────────────────

def process_document_with_patent_pipeline(file_path: str, fields: List[str]) -> Dict:
    """
    SOTA patent-eligible document extraction pipeline.

    This is the main entry point called by app.py. Signature unchanged.

    Returns:
        {
            'fields': Dict[str, FieldWithQuality],
            'quality_report': Dict,
            'confidence_stats': Dict,
            'fusion_metadata': Dict,
        }
    """
    t0 = time.time()
    print(f"\n{'='*72}")
    print(f"  FILETRACT SOTA PIPELINE — {os.path.basename(file_path)}")
    print(f"{'='*72}")

    ext = os.path.splitext(file_path)[1].lower()

    # Stage 1: Preprocessing
    print("  Stage 1 — Smart preprocessing...")
    if ext == '.pdf':
        prep = extract_pdf_with_sota_pipeline(file_path)
    elif ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp'):
        prep = extract_image_with_sota_pipeline(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    pil_image = prep['pil_image']
    ocr_text = prep.get('ocr_text', '')

    # Stage 2-4: SOTA multi-strategy extraction
    print("  Stages 2-4 — Parallel extraction + consensus + verification...")
    engine = SOTAExtractionEngine()
    sota_result: SOTAResult = engine.extract(
        pil_image=pil_image,
        fields=fields,
        ocr_text=ocr_text,
        enable_verification=True,
    )

    # Stage 5: Build output
    print("  Stage 5 — Building quality report...")
    field_quality = _sota_to_field_quality(sota_result)
    quality_report = _build_quality_report(field_quality, sota_result)
    confidence_stats = _build_confidence_stats(field_quality, sota_result)
    fusion_metadata = _build_fusion_metadata(sota_result, time.time() - t0)

    # Console summary
    print(f"\n{'='*72}")
    print(f"  RESULTS — {os.path.basename(file_path)}")
    print(f"{'='*72}")
    for field_name, fq in field_quality.items():
        icon = {'reliable': '✅', 'good': '☑️', 'uncertain': '⚠️'}.get(fq.quality_flag, '❌')
        val = fq.value if fq.value else 'NOT FOUND'
        print(f"  {icon} {field_name}: {val}  [{fq.quality_flag}, llm={fq.llm_confidence}]")
    print(f"\n  Overall quality: {quality_report['overall_quality']} | "
          f"Reliable: {quality_report['reliable_fields']}/{quality_report['total_fields']} | "
          f"Score: {sota_result.quality_score:.2f}")
    print(f"  Strategies: {', '.join(sota_result.strategies_used)}")
    print(f"  Document type: {sota_result.doc_type}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"{'='*72}")

    return {
        'fields': field_quality,
        'quality_report': quality_report,
        'confidence_stats': confidence_stats,
        'fusion_metadata': fusion_metadata,
    }


def extract_text_with_confidence_pipeline(file_path: str) -> Dict:
    """
    Backward-compatible shim for code that calls this function directly.
    Returns text extracted by Tesseract (used as OCR supplement by SOTA engine).
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            pages = _load_pil_from_pdf(file_path)
            text = '\n'.join(_run_tesseract_fast(p) for p in pages)
        else:
            img = _load_pil_from_image(file_path)
            text = _run_tesseract_fast(img)
    except Exception:
        text = ''

    return {'text': text, 'annotated_text': text, 'fused_regions': [],
            'confidence_stats': {}, 'fusion_metadata': {}}


# ─── Report builders ──────────────────────────────────────────────────────────

def _build_quality_report(fields: Dict[str, FieldWithQuality], sota: SOTAResult) -> Dict:
    total = len(fields)
    reliable = sum(1 for f in fields.values() if f.quality_flag == 'reliable')
    good = sum(1 for f in fields.values() if f.quality_flag == 'good')
    uncertain = sum(1 for f in fields.values() if f.quality_flag == 'uncertain')
    low = sum(1 for f in fields.values() if f.quality_flag in ('low-quality', 'not-found'))

    score = sota.quality_score
    overall = 'High' if score >= 0.75 else 'Medium' if score >= 0.50 else 'Low'

    return {
        'overall_quality': overall,
        'quality_score': round(score, 3),
        'total_fields': total,
        'reliable_fields': reliable,
        'good_fields': good,
        'uncertain_fields': uncertain,
        'low_quality_fields': low,
        'document_type': sota.doc_type,
        'strategies_used': sota.strategies_used,
    }


def _build_confidence_stats(fields: Dict[str, FieldWithQuality], sota: SOTAResult) -> Dict:
    confs = [f.ocr_confidence for f in fields.values()]
    consensus_count = sum(1 for v in sota.field_consensus.values() if v)
    return {
        'mean_confidence': round(float(np.mean(confs)), 3) if confs else 0.0,
        'min_confidence': round(float(np.min(confs)), 3) if confs else 0.0,
        'max_confidence': round(float(np.max(confs)), 3) if confs else 0.0,
        'std_confidence': round(float(np.std(confs)), 3) if confs else 0.0,
        'consensus_fields': consensus_count,
        'total_fields': len(fields),
        'consensus_rate': round(consensus_count / len(fields) * 100, 1) if fields else 0.0,
        'pipeline_version': '3.0-SOTA',
    }


def _build_fusion_metadata(sota: SOTAResult, elapsed: float) -> Dict:
    total = len(sota.fields)
    high = sum(1 for c in sota.field_confidence.values() if c == 'high')
    medium = sum(1 for c in sota.field_confidence.values() if c == 'medium')
    low = sum(1 for c in sota.field_confidence.values() if c == 'low')
    consensus = sum(1 for v in sota.field_consensus.values() if v)

    return {
        'total_fields': total,
        'high_confidence_fields': high,
        'medium_confidence_fields': medium,
        'low_confidence_fields': low,
        'consensus_fields': consensus,
        'consensus_rate': round(consensus / total * 100, 1) if total else 0.0,
        'strategies_used': sota.strategies_used,
        'document_type': sota.doc_type,
        'quality_score': round(sota.quality_score, 3),
        'processing_time_seconds': round(elapsed, 2),
        'pipeline': 'SOTA v3.0 (parallel-vision + self-verification)',
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _get_file_paths() -> List[str]:
    print("\n" + "=" * 72)
    print("FILE SELECTION")
    defaults = ["10th_long_memo.pdf"]
    existing = [f for f in defaults if os.path.exists(f)]
    if existing:
        print(f"Default: {existing}")
        print("Press Enter to use defaults or type paths (comma-separated):")
    else:
        print("Enter file paths (comma-separated):")
    user = input("> ").strip()
    if not user and existing:
        return existing
    return [p.strip().strip('"\'') for p in user.split(',') if p.strip() and os.path.exists(p.strip().strip('"\''))]


def _get_fields() -> List[str]:
    print("\nFields to extract (comma-separated):")
    print("  e.g. Name, Father Name, Date of Birth, CGPA")
    user = input("> ").strip()
    if not user:
        return ["Name", "Father Name", "School", "Date of Birth"]
    return [f.strip() for f in user.split(',') if f.strip()]


def main():
    print("\n" + "=" * 72)
    print("  FILETRACT — SOTA PATENT-ELIGIBLE PIPELINE v3.0")
    print("=" * 72)

    file_paths = _get_file_paths()
    if not file_paths:
        print("❌ No valid files.")
        return

    fields = _get_fields()
    if not fields:
        print("❌ No fields specified.")
        return

    for fp in file_paths:
        try:
            results = process_document_with_patent_pipeline(fp, fields)
            # Save JSON
            base = os.path.splitext(fp)[0]
            out = base + "_patent_extracted.json"
            output_data = {
                'extracted_fields': {
                    k: {
                        'value': v.value,
                        'ocr_confidence': float(v.ocr_confidence),
                        'llm_confidence': v.llm_confidence,
                        'quality_flag': v.quality_flag,
                    }
                    for k, v in results['fields'].items()
                },
                'quality_report': results['quality_report'],
                'confidence_statistics': results['confidence_stats'],
                'fusion_metadata': results['fusion_metadata'],
            }
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved → {out}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted.")
        sys.exit(0)
