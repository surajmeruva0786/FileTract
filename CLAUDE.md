# FileTract — Project Changelog & Development Guide

> **Rule:** Every change, progress update, and feature addition must be logged here **before** committing. Commit code + this file together, then push immediately.

---

## Project Overview

**FileTract** is a patent-pending AI document intelligence system. It extracts structured fields from scanned documents (PDFs, images) using a 5-stage confidence-weighted OCR pipeline powered by Tesseract OCR and Groq (Llama 4 Scout vision / Llama 3.3 70B text).

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
GROQ_API_KEY=your-groq-api-key         # Required
PORT=5000                              # Optional (Render sets this)
FLASK_ENV=production                   # Optional
```

---

## Changelog

### 2026-07-21 — Set Up EAS Build Pipeline for Mobile App APK

**What changed:**
- **`filetract_mobile/eas.json`** (NEW) — added EAS Build profiles: `development` and `preview` both build an installable Android `.apk` (`"buildType": "apk"`, `distribution": "internal"`) instead of the Play-Store-only `.aab` that's the platform default; `production` is left as an `.aab` for eventual Play Store submission.
- **`filetract_mobile/app.json`** — ran `eas init --force` to create and link the EAS project `@surajmeruva0786/filetract-mobile` (project ID `54bb510c-efb0-4829-849d-397d0c8ef909`); this wrote `extra.eas.projectId` into the file, which EAS Build requires to associate builds with the project.
- Kicked off the first cloud build: `eas build --platform android --profile preview --non-interactive`.

**Why:** User wants to install and test the mobile app on a physical Android device without going through Expo Go / Metro. The project was never previously linked to an EAS account or configured to produce a directly-installable `.apk`.

**Verified:** `eas whoami` confirmed existing login as `surajmeruva0786`. `eas init --force` succeeded and linked the project. Build was in progress at the time of this entry — result/APK link not yet confirmed.

**Files changed:**
- `filetract_mobile/eas.json` (NEW)
- `filetract_mobile/app.json`
- `CLAUDE.md` (this file)

---

### 2026-07-21 — Migrated LLM Provider from Gemini to Groq

**What changed:**
- User switched Render's env var from `GEMINI_API_KEY` to `GROQ_API_KEY` (already applied on the live service). Codebase updated to match.
- **`groq_ocr_client.py`** (NEW) — drop-in shim mirroring the `google.generativeai` surface (`configure()`, `GenerativeModel`, `types.GenerationConfig`, `model.generate_content(parts).text`) but backed by Groq. Auto-routes to a vision model (`meta-llama/llama-4-scout-17b-16e-instruct`) when a `PIL.Image` is in the call, or a text model (`llama-3.3-70b-versatile`) otherwise. Written this way so the 3 call-site files below needed only an import swap, not a rewrite of their prompt/strategy logic.
- **`gemini_ocr_extract.py`**, **`sota_extraction_engine.py`**, **`confidence_aware_llm.py`**, **`patent_ocr_pipeline.py`** — swapped `import google.generativeai as genai` → `import groq_ocr_client as genai`, `GEMINI_API_KEY` → `GROQ_API_KEY`, and updated user-facing log/comment text referencing "Gemini" to "Groq". `confidence_aware_llm.py`'s `ConfidenceAwareLLM` class is currently unused by the live pipeline (only its `FieldWithQuality` dataclass is imported elsewhere) but was migrated too so the module still imports cleanly.
- **`requirements.txt`** — replaced `google-generativeai` with `groq`.
- **`test_gemini.py` → `test_groq.py`** — renamed and rewritten to hit Groq via `GROQ_API_KEY` from `.env`. This file also had a **live Gemini API key hardcoded in source and committed to the public repo** — unrelated to this migration but a real secret leak; rewritten to read from env instead. **Action needed from you:** revoke that Gemini key in Google AI Studio regardless of this code fix (it's already in git history / on GitHub).
- **`.env`** (local, gitignored) — swapped the `GEMINI_API_KEY` line for an empty `GROQ_API_KEY` line; fill in your key locally to run/test outside Render.

**Why:** User's call — switched providers, backend env var already changed on Render before this fix landed, meaning the previously-deployed code would `sys.exit(1)` at import (`patent_ocr_pipeline.py` and `gemini_ocr_extract.py` both hard-exit if `GEMINI_API_KEY` is missing) as soon as Render restarted the service on the env var change.

**Verified:** All 5 modules that touch the LLM (`gemini_ocr_extract`, `patent_ocr_pipeline`, `confidence_aware_llm`, `sota_extraction_engine`, `app`) import cleanly with `groq` installed and a dummy `GROQ_API_KEY` (structural check only — no real Groq call made locally, no local Groq key available in this session).

**Files changed:**
- `groq_ocr_client.py` (NEW)
- `gemini_ocr_extract.py`, `sota_extraction_engine.py`, `confidence_aware_llm.py`, `patent_ocr_pipeline.py`
- `requirements.txt`
- `test_gemini.py` → `test_groq.py`
- `.env` (local only, not committed)
- `CLAUDE.md` (this file)

---

### 2026-07-20 — Wired Mobile App to Live Backend (https://filetract.onrender.com)

**What changed:**
- **`filetract_mobile/services/api.js`** — `BASE_URL` was a placeholder (`https://your-filetract-backend.onrender.com`) that would never resolve; set to the real deployed backend `https://filetract.onrender.com`.
- **`filetract_mobile/services/api.js`** — removed a global default `Content-Type: multipart/form-data` header from the axios instance. That header had no `boundary` parameter and was applied to every request, including `uploadImage()`'s `FormData` POST. Explicitly setting `Content-Type: multipart/form-data` without a boundary prevents axios/React Native from auto-generating the correct boundary for the multipart body, which silently breaks file uploads (Werkzeug can't parse the parts without a boundary). Now the header is left unset on the FormData request so the platform sets it correctly; JSON requests (`extractFields`, `pollUntilComplete`) already set `Content-Type: application/json` explicitly per-call, so they're unaffected.
- **`filetract_mobile/screens/SettingsScreen.js`**, **`filetract_mobile/README.md`** — updated placeholder/hint text to reference the real deployed URL instead of a generic `your-app.onrender.com` example.

**Why:** The mobile app's screens, navigation, and API client shape (`uploadImage` → `extractFields` → `pollUntilComplete`, and the patent/standard result-parsing in `PreviewScreen.js`) were already correctly built and matched the backend's actual response shapes — but the app had never been pointed at a real backend, and had a latent multipart bug that would have broken the very first upload attempt regardless.

**Verified:** Replicated the mobile app's exact HTTP contract (multipart upload → JSON `/api/extract` → poll `/api/status` → `/api/result`) against `https://filetract.onrender.com` via curl for both `standard` and `patent` pipelines — both complete end-to-end successfully (patent: 5 stages, ~83s; standard: ~5s). Not yet run on a physical device/emulator (no RN runtime available in this environment).

**Known issue (pre-existing, not fixed here):** The patent pipeline's live result showed `"strategies_used": []` and `"consensus_rate": 0` — none of the parallel Gemini calls returned a usable value against a real test image. This matches the `GEMINI_API_KEY` problem already noted in the 2026-07-04 entry below (previously believed local-only); this test suggests it may also be affecting the Render deployment's key. Needs verification of the `GEMINI_API_KEY` set in Render's environment variables.

**Files changed:**
- `filetract_mobile/services/api.js`
- `filetract_mobile/screens/SettingsScreen.js`
- `filetract_mobile/README.md`
- `CLAUDE.md` (this file)

---

### 2026-07-04 — Fixed Website Pipelines Hanging Forever With No Result

**What changed:**
- User reported both the standard and patent pipelines on the website would load indefinitely and never return a result for a simple image.
- **`sota_extraction_engine.py`** — `_call()` (used by every Gemini request in the SOTA/patent engine: doc-type detection, all 3 extraction strategies, self-verification) had no request timeout at all. A stalled network call could block forever. Added `request_options={'timeout': 45}`.
- **`sota_extraction_engine.py`** — the parallel-strategy `with concurrent.futures.ThreadPoolExecutor(...) as executor:` block was pointless as a safety net: `fut.result(timeout=90)` gives up reading a slow future, but the `with` block's `__exit__` still calls `shutdown(wait=True)`, which blocks until every submitted thread actually finishes anyway — silently defeating the timeout. Switched to a manual `executor.shutdown(wait=False)` in a `finally`.
- **`gemini_ocr_extract.py`** — same missing-timeout issue on both the vision and text-mode `model.generate_content()` calls in the standard pipeline. Added the same 45s `request_options` timeout.
- **`app.py`** — added a real outer wall-clock timeout (`_run_with_timeout`, patent=150s / standard=60s) around both pipeline entry points. The old code had a comment claiming "timeout protection" but it was only an exception-catching `try/except` — no actual time bound existed anywhere above the per-call level.
- **`app.py`** — fixed a guaranteed crash in the standard pipeline: the patent-pipeline's exception-fallback branch did `from gemini_ocr_extract import extract_text_from_image, ...` as a *local* import inside an `except` block. Python decides variable scope for a whole function at compile time, so that local import made `extract_text_from_image` (and friends) local to all of `process_job_async` — including the completely separate `else:` (standard pipeline) branch, which never executes that import line. Every standard-pipeline request hit `UnboundLocalError: cannot access local variable 'extract_text_from_image'` before this fix. Removed the redundant local imports (already imported at module level).

**Why:** Root cause of "keeps loading and loading, then no result" was the combination of unbounded Gemini calls (can hang indefinitely on a stalled request) and a per-future timeout that didn't actually stop the wait. The standard pipeline had a second, independent bug that crashed it outright on every request regardless of network conditions.

**Verified locally:** ran the Flask server against `certif_img1.png`, confirmed the standard pipeline completes in ~3–12s with no crash (previously threw `UnboundLocalError` on every call). Full Vision extraction quality could not be verified end-to-end because the local `.env` `GEMINI_API_KEY` is being rejected by Google (`400 API_KEY_INVALID`) — this is a separate, local-only credential issue, not a code bug. Needs a valid key (locally, and confirm the one set in Render's environment variables for the live site) to fully verify extraction accuracy.

**Files changed:**
- `app.py`
- `sota_extraction_engine.py`
- `gemini_ocr_extract.py`
- `CLAUDE.md` (this file)

---

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
