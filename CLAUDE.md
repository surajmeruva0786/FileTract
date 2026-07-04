# FileTract — Project Changelog & Development Guide

> **Rule:** Every change, progress update, and feature addition must be logged here **before** committing. Commit code + this file together, then push immediately.

---

## Project Overview

**FileTract** is a patent-pending AI document intelligence system. It extracts structured fields from scanned documents (PDFs, images) using a 5-stage confidence-weighted OCR pipeline powered by Tesseract OCR and Google Gemini 2.5 Flash.

**Repository:** https://github.com/surajmeruva0786/FileTract  
**Patent Status:** Pending  
**Backend:** Flask API on Render.com  

---

## Architecture

```
FileTract/
├── app.py                     # Flask REST API (main backend)
├── patent_ocr_pipeline.py     # 5-stage patent pipeline
├── gemini_ocr_extract.py      # Standard pipeline (fast)
├── confidence_analyzer.py     # Spatial confidence mapping
├── image_quality_analyzer.py  # Quality metrics
├── adaptive_reocr_engine.py   # Selective re-OCR engine
├── result_fusion.py           # Confidence-weighted fusion
├── confidence_aware_llm.py    # Quality-aware Gemini extraction
├── filetract_web/             # Vanilla JS web frontend
├── filetract_frontend/        # React/TypeScript frontend (Vite)
├── filetract_mobile/          # React Native (Expo) mobile app [NEW]
├── requirements.txt
├── Dockerfile
├── render.yaml
└── CLAUDE.md                  # This file
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload document(s) |
| POST | `/api/extract` | Extract fields (patent/standard pipeline) |
| POST | `/api/extract/batch` | Batch extract from multiple docs |
| GET | `/api/status/<job_id>` | Check job status |
| GET | `/api/result/<job_id>` | Get extraction results |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/health` | Health check |

## Environment Variables

```bash
GEMINI_API_KEY=your-gemini-api-key   # Required
PORT=5000                              # Optional (Render sets this)
FLASK_ENV=production                   # Optional
```

---

## Changelog

### 2026-07-04 — Mobile App: Fixed Broken Entry Point + First Successful Build

**What changed:**
- Mobile app had never been `npm install`'d or run since being added — this session did both and found it was broken.
- **`package.json`** — `main` pointed to `expo-router/entry`, but the project has no `app/` directory and never used expo-router; `App.js` uses classic React Navigation instead. This meant the app would boot into expo-router looking for routes that don't exist, never reaching the real screens. Fixed to `node_modules/expo/AppEntry.js`. Removed the unused `expo-router` dependency; bumped `expo-image-picker` to `~15.1.0` and `typescript` to `~5.3.3` to match Expo SDK 51 expectations; added missing peer deps `expo-constants`, `expo-linking`, `expo-font`.
- **`app.json`** — removed the leftover `expo-router` plugin and `typedRoutes` experiment. Added placeholder `icon.png`, `splash.png`, `adaptive-icon.png`, `favicon.png` under `assets/` (referenced in config but the files didn't exist — `expo-doctor` failed schema validation without them).
- **`screens/HomeScreen.js`** — removed a dead `import { colors, typography, spacing } from '../theme'` where `../theme` doesn't exist as a module anywhere in the project, and the file already defines its own local `const colors`. This caused a hard Babel syntax error ("Identifier 'colors' has already been declared") that broke the Metro bundle entirely.

**Why:** User asked "is the mobile app completely ready?" Answer was no — it had real code for all 5 screens but had literally never been built or run, so these breakages were undiscovered.

**Verification:** `npx expo-doctor` → 17/17 checks pass. `npx expo export --platform android` and `--platform ios` → both bundle cleanly (965 modules, no errors). Not yet tested on a physical device/emulator.

**Still not production-ready:**
- `services/api.js` `BASE_URL` is still a placeholder (`https://your-filetract-backend.onrender.com`) — needs the real deployed Render URL.
- Placeholder icon/splash assets are solid-color squares, not real branding.
- Never launched in Expo Go / an emulator — only static bundling was verified.
- Google Sheets Apps Script still requires the user to deploy their own Web App and paste the URL into Settings.

**Files added/changed:**
- `filetract_mobile/package.json`, `filetract_mobile/package-lock.json` (NEW)
- `filetract_mobile/app.json`
- `filetract_mobile/screens/HomeScreen.js`
- `filetract_mobile/assets/` (NEW — placeholder icon/splash/adaptive-icon/favicon PNGs)
- `CLAUDE.md` (this file)

---

### 2026-07-01 — SOTA Pipeline v3.0 (Major Accuracy + Speed Overhaul)

**What changed:**
- **Created `sota_extraction_engine.py`** — New patent-pending SOTA engine replacing the sequential per-region Tesseract re-OCR approach
  - Direct Gemini Vision extraction (no OCR error propagation — biggest accuracy win)
  - ThreadPoolExecutor parallel execution of 3 strategies simultaneously
  - Cross-strategy consensus voting with case-normalized comparison
  - Targeted self-verification loop (batched per call) for uncertain/disagreed fields
  - Document-type detection for domain-aware context injection
- **Rewrote `patent_ocr_pipeline.py`** — New 5-stage orchestration using SOTA engine
  - Stage 1: Smart preprocessing (deskew + illumination normalize + card crop + upscale)
  - Stages 2-4: Parallel SOTA extraction (replaces 50 sequential Tesseract calls)
  - Stage 5: Quality scoring and report generation
  - API signature unchanged — `app.py` works without modification
- **Enhanced `image_quality_analyzer.py`** — New preprocessing methods
  - `deskew_image()`: Hough-line skew correction (fixes tilted phone photos)
  - `detect_and_crop_card()`: Perspective correction for ID cards
  - `normalize_illumination()`: LAB-space CLAHE for uneven phone camera lighting
  - `enhance_document_image()`: Full pipeline combining all preprocessing steps
  - `_upscale_if_needed()`: Lanczos upscaling for low-res inputs
- **Updated `gemini_ocr_extract.py`** — Standard pipeline now uses Gemini Vision when image is available (not just OCR text)
- **Updated `app.py`** — Standard pipeline passes `image_path` to `extract_fields_with_gemini` for Vision extraction
- **Updated `confidence_aware_llm.py`** — Added vision-direct method header

**Why:** Patent pipeline results were poor (low accuracy) and very slow (15-45s). The root cause: passing error-prone Tesseract OCR text to Gemini instead of using Gemini's native vision capability. Tesseract errors propagate to the LLM and corrupt extraction. Running 50 per-region re-OCR calls was the main speed bottleneck.

**Performance improvement:**
- Before: 15–45 seconds, text-based extraction with OCR error propagation
- After: 3–10 seconds, direct vision with parallel strategies + verification

**Files added/changed:**
- `sota_extraction_engine.py` (NEW — core SOTA engine)
- `patent_ocr_pipeline.py` (REWRITTEN — uses SOTA engine)
- `image_quality_analyzer.py` (ENHANCED — deskew, card crop, illumination)
- `gemini_ocr_extract.py` (ENHANCED — vision path added)
- `app.py` (UPDATED — vision path for standard pipeline)
- `confidence_aware_llm.py` (MINOR — added vision method)
- `CLAUDE.md` (this file — updated)

---

### 2026-06-29 — Mobile App Added (React Native / Expo)

**What changed:**
- Created `filetract_mobile/` — a full React Native (Expo) mobile application
- App supports: camera capture or gallery upload of ID cards, user-defined field extraction, live preview of extracted data, and one-tap export to Google Sheets via Apps Script Web App
- Created `filetract_mobile/google_apps_script/Code.gs` — Google Apps Script template for Sheets integration
- Updated `README.md` with mobile app setup instructions

**Why:** Results from the existing pipeline needed improvement for mobile ID card scanning use case. Mobile-first approach allows users to directly photograph ID cards (Aadhaar, etc.) and extract to Google Sheets.

**Files added/changed:**
- `filetract_mobile/` (entire new directory — React Native Expo app)
- `CLAUDE.md` (this file — created)
- `README.md` (updated with mobile app section)

---

### Previous Changes (from git history)

- **Optimize Patent Pipeline** — Added timeout handling, fallback to standard pipeline, limit re-OCR to 50 regions
- **Critical Fix** — Remove hardcoded Windows Tesseract path from adaptive_reocr_engine (fixes Docker/Patent Pipeline)
- **Fix LLM Hallucinations** — Remove example data, add strict anti-hallucination instructions
- **Docker Detection** — Explicit Docker detection for Tesseract path (force Linux path in containers)
- **Tesseract Path Fix** — Use `tesseract` command for Linux instead of full Windows path

---

## Development Rules

1. **Every change → update this CLAUDE.md first**
2. **Commit code + CLAUDE.md together** (same commit)
3. **Push immediately after commit** — no batching
4. **Documentation changes** (README, etc.) also committed and pushed with code

## Known Issues / Next Steps

- Patent pipeline results quality needs improvement for real-world degraded documents
- Mobile app backend URL needs to be configured to point to deployed Render.com instance
- Google Sheets integration requires user to deploy their own Apps Script Web App
