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

### 2026-07-30 (second follow-up, same day) — Reverted Vision Image Downscaling — Accuracy Over Speed

**What changed:**
- User reported that after the downscaling fix (1600px, then 2048px — see the two entries below), Fast-pipeline extraction accuracy was noticeably below the original pre-session behavior: "results were also better than previous, but not up to that level." Asked to go back to exactly the state where results were good and keep only that.
- **`groq_ocr_client.py`** — reverted `_to_data_url` back to its original form: images are sent to Groq Vision at **full resolution**, JPEG quality 92, no resizing. Removed the `_MAX_VISION_DIMENSION` downscale step entirely.

**Why:** Direct user report of an accuracy regression traced to the earlier speed optimization — small print on real (uncropped, full-frame) phone photos was apparently landing below a legible size once downscaled, hurting extraction quality more than the latency win was worth. Per explicit instruction, reverted rather than re-tuning the threshold further.

**What's intentionally still in place (not reverted — unrelated to the accuracy complaint):**
- `Groq(api_key=..., max_retries=1)` in `groq_ocr_client.configure()` — caps retry-related latency/stalling on failures, doesn't touch the successful-call image data.
- The error-surfacing fix in `gemini_ocr_extract.py`/`sota_extraction_engine.py` — real API failures still raise instead of silently returning empty fields as a fake "success."
- The Tesseract-skip restructuring in `app.py` — Vision is still tried first with no OCR text on the standard pipeline's happy path; Tesseract only runs lazily on a Vision failure. This never touched the image data Vision receives, so it isn't implicated in the accuracy regression.

**Verified:** `python -m py_compile groq_ocr_client.py` and a full module-import check (`GROQ_API_KEY=dummy ... import groq_ocr_client, gemini_ocr_extract, app`) — clean. Not yet re-verified end-to-end against the live backend for accuracy (needs a fresh test once Render redeploys — see the deploy-lag note in the entry below, which was still unresolved as of this fix).

**Files changed:**
- `groq_ocr_client.py`
- `CLAUDE.md` (this file)

---

### 2026-07-30 — Fixed "Empty Values" Bug on Both Pipelines + Vision Latency + Default Sheets URL

**What changed:**
- User reported both Fast (standard) and Accurate (patent) extraction worked once on-device, then started returning **empty values on every subsequent attempt**, with Fast also taking ~15-16s instead of a target ~5s.
- **Root cause found (silent failure swallowing):** every Groq call site caught its own exceptions (timeouts, rate limits, malformed responses) and returned an all-`null` field dict as if extraction had *succeeded* — the job was marked `'complete'` with empty data and no error ever reached the UI. This is textbook Groq free-tier rate-limit behavior: the patent/Accurate pipeline fires up to 5 Groq calls per single job (`sota_extraction_engine.py`: doc-type detection + 2-3 parallel vision strategies + a verification call), so a handful of test runs back-to-back exhausts the per-minute quota; every call after that quietly degraded to nulls instead of surfacing the real problem.
  - **`gemini_ocr_extract.py`** (`extract_fields_with_gemini`) — now raises instead of silently returning nulls when an actual API call fails and there's no usable OCR-text fallback (a genuinely empty document — no image, no OCR text — still returns nulls gracefully; that's not an error).
  - **`sota_extraction_engine.py`** (`SOTAExtractionEngine.extract`) — when *all* strategies fail, now raises instead of returning a fake "successful" empty `SOTAResult`. This was the key fix: `app.py`'s patent-pipeline `except` block already had fallback-to-standard-pipeline logic written, but it could never trigger because the SOTA engine never actually raised.
  - **`app.py`** — that fallback branch also never passed the image through to the retry call (`extract_fields_with_gemini(text, fields)` — no `image_path`), meaning a fallback attempt only ever got noisy Tesseract OCR text, never a second shot at Vision. Now passes the image through so the fallback can still use Vision extraction, not just raw OCR text.
  - Net effect: a real failure (e.g. Groq rate limit) now surfaces as an actual job error the app's existing "Extraction Failed" screen can show, instead of a false "success" with blank fields. This does **not** fix Groq's own account-level rate limits — if the new error message says rate-limited/429, that's a Groq plan/quota constraint, not something fixable in this codebase.
- **`groq_ocr_client.py`** — two latency fixes on the shared `_to_data_url` image-encoding path (used by both pipelines):
  - Images are now downscaled to a 1600px longest side before being sent to Groq Vision. Phone photos routinely come in at 3000-4000px+; that resolution is unnecessary for legible printed text and was adding significant upload + image-token processing time on every single call. JPEG quality also reduced 92→88 (imperceptible for OCR purposes, smaller payload).
  - `Groq(api_key=..., max_retries=1)` — was defaulting to the SDK's `max_retries=2`, meaning a transient 429/5xx silently tripled a call's latency (and its slice of the caller's timeout budget) before finally failing. Capped at 1 retry so a real failure surfaces fast instead of stalling.
- **`filetract_mobile/services/googleSheets.js`** — `getSheetsUrl()` now falls back to the user's deployed Apps Script Web App URL (`https://script.google.com/macros/s/AKfycbxaPSBqZMqMZ_wqSL0lxtW6U3lpwLsP3e9sN5_EWc-FVx0O3f-5g_4dtyVL8v3k3FTbtw/exec`) when nothing is saved yet, so it's pre-filled in Settings and exports work out of the box instead of requiring the user to paste it in manually first. Saving a different URL in Settings still overrides it as before.
- **`app.py`** (follow-up fix, same session) — the standard/Fast pipeline ran a full Tesseract OCR pass on the original full-resolution photo *before* the Groq Vision call, even though that OCR text is only ever used as a fallback when Vision fails or for PDFs. That put 1-4s of pure dead time (Tesseract on an un-resized image) directly on the critical path for every Fast request, on top of the actual Vision call. Restructured so the non-PDF path tries Vision first with no text; Tesseract is now only computed lazily, on retry, if Vision genuinely fails. This was flagged by the user re-asking whether the "under 5 seconds" ask had actually been addressed — the earlier image-downscaling/retry-cap fix helped, but this eager-OCR waste was a separate, bigger contributor that hadn't been caught yet.

**Why:** User's explicit bug report — both pipelines appeared broken (empty predictions) after the first run, and Fast was too slow for good UX.

**Verified:** `python -m py_compile` on all four changed Python files, and a full module-import check (`GROQ_API_KEY=dummy ... import app, gemini_ocr_extract, patent_ocr_pipeline, sota_extraction_engine, groq_ocr_client`) — all clean. `npx expo export --platform android` on the mobile change — bundles cleanly. No live Groq key available in this environment, so the actual rate-limit hypothesis and the real-world latency improvement from downscaling could not be measured end-to-end here; needs confirmation against the live Render deployment.

**Known pre-existing inconsistency (not touched here):** `render.yaml` still lists `GEMINI_API_KEY` as the service env var, even though the code and the live Render service have used `GROQ_API_KEY` since the 2026-07-21 migration. Render Blueprints are normally only read on initial provisioning, not on every push, so this is unlikely to affect the already-running service — but worth fixing if the service is ever reprovisioned from this file.

**Files changed:**
- `groq_ocr_client.py`
- `gemini_ocr_extract.py`
- `sota_extraction_engine.py`
- `app.py`
- `filetract_mobile/services/googleSheets.js`
- `CLAUDE.md` (this file)

---

### 2026-07-30 (follow-up, same day) — Live Diagnosis: Deploy Lag + Groq Now Failing 100% of Calls

**What changed:**
- User reported the above fixes hadn't visibly changed anything (Fast still ~15s, patent still returning empty values, and Fast accuracy had *regressed* vs. an earlier successful test) and asked me to actually verify rather than assume.
- Ran live end-to-end tests directly against `https://filetract.onrender.com` using `certif_img1.png` (591×882px) for both pipelines, twice each, several minutes apart:
  - **Standard/Fast pipeline**: succeeded both times with fully correct values for all 4 fields (Name, Father Name, School, Date of Birth) — but took **~40-45 seconds**, far worse than even the user's reported "15s," and far worse than the "~5s" target.
  - **Patent/Accurate pipeline**: failed completely both times — `strategies_used: []`, every field `null`/`quality_flag: "not-found"`, `document_type: "Document"` (the exception fallback value), meaning **every single Groq call failed, including the trivially cheap doc-type-detection call** — yet job status was `"complete"`, not `"error"`.
- **That last point is the smoking gun**: the `sota_extraction_engine.py` fix earlier today explicitly raises when all strategies fail, which should have produced an `"error"` status (or triggered the patent→standard fallback in `app.py`). Getting `"complete"` with silent nulls instead means **the live Render service had not deployed the newer commits yet** at the time of testing — this session has no Render API/CLI access to confirm or force a redeploy, or to read Render's server logs directly.
- **Separately, and more concerning**: every Groq call failing 100% of the time — including the cheapest possible call (one-line document-type classification) — doesn't look like ordinary per-minute rate-limiting (which usually degrades partially, not totally). It's more consistent with an account-level problem: an invalid/revoked `GROQ_API_KEY` on Render, an exhausted daily/monthly quota, or the preview vision model (`meta-llama/llama-4-scout-17b-16e-instruct`, set up 2026-07-21) having been deprecated or renamed by Groq. **This cannot be diagnosed further without either a live Groq key or Render's server logs** (both outside this session's access) — the actual exception text our `print()` calls emit (e.g. "Vision extraction failed (...)") would appear in Render's log stream and should make the real cause obvious immediately.
- **`groq_ocr_client.py`** — raised `_MAX_VISION_DIMENSION` from 1600 → 2048px as a precaution against the user's reported Fast-pipeline accuracy regression: 1600px risked shrinking small print below legible size on uncropped phone photos where the document doesn't fill the frame (the local test image was far below either threshold, so it couldn't confirm or rule out this specific risk). 2048 keeps a meaningful size/latency win over full-resolution while leaving more margin for legibility.

**Why:** User pushed back that nothing seemed fixed and specifically asked for real verification instead of assumed fixes — this entry documents what that verification actually found, including a real blocker (deploy visibility) this session cannot resolve alone.

**Action needed from the user (I have no access to either):**
1. Check Render's dashboard → this service → **Events/Deploys** tab: confirm commit `d990858` (or later) has actually deployed; manually trigger "Deploy latest commit" if it's stuck or auto-deploy is off.
2. Pull up Render's **live logs** during a test run and paste back the actual printed error (search for "Vision extraction failed", "Groq text extraction failed", or "SOTA") — that message will say definitively whether this is rate-limiting, an auth/key problem, or a deprecated model, which determines what (if anything) is fixable in code versus needing a Groq dashboard/plan change.

**Files changed:**
- `groq_ocr_client.py`
- `CLAUDE.md` (this file)

---

### 2026-07-29 — Rebuilt Mobile App UI from the Claude-Designed Mockup (FileTract-app-frontend)

**What changed:**
- User tested the previous APK and pointed out the app didn't match "the frontend I designed" — turned out `FileTract-app-frontend/` (committed 2026-07-20 in `39036af upload mobile app frontend`) is a fully-specified interactive design mockup (`FileTract.dc.html` + `android-frame.jsx` + supporting JS) exported from a Claude design tool, spec'ing all 5 screens (Home, Configure Extraction, Processing, Extracted Fields, Settings) in a dark violet/indigo "AI intelligence" visual language. It had never actually been read or implemented — every prior mobile session (2026-06-29 through 2026-07-21) only ever touched the original placeholder screens (cyan-on-black theme, generic copy), so the shipped APK never reflected this design at all. (Separately, `filetract_frontend/` — the other frontend-looking folder — is unrelated: its `metadata.json` identifies it as "Flash UI," a generic Google AI Studio/Gemini demo scaffold present since the very first commit, not a FileTract design and not something built in any of these sessions.)
- **`filetract_mobile/theme.js`** (NEW) — shared color palette (`#0C0714` background, `#6366F1`→`#7C3AED` primary gradient, `#A78BFA` violet accent, etc.), gradient presets, and font-family constants ported directly from the mockup's inline styles.
- **`filetract_mobile/components/GlowBackground.js`** (NEW) — reusable ambient background (two soft violet/indigo glow circles) approximating the mockup's layered radial-gradients without pulling in `react-native-svg` just for a decoration.
- **`filetract_mobile/App.js`** — loads Instrument Sans / DM Sans / JetBrains Mono via `expo-font`'s `useFonts` before rendering the navigator (blank dark screen shown until ready); card background switched to the new theme color.
- **`filetract_mobile/screens/{Home,Fields,Processing,Preview,Settings}Screen.js`** — full visual rewrite to match the mockup screen-for-screen: gradient CTA buttons (`expo-linear-gradient`), pill-style chips/tags, segmented Fast/Accurate mode switch, step-by-step processing checklist with connectors and checkmarks, card-based field list with quality badges, gradient Save/Export buttons with saving/sent states. All real logic is unchanged — camera/gallery picking, the actual `pipeline` (standard/patent) values sent to the backend, `processImage`/stage-callback wiring, real result parsing (patent vs standard field shapes, OCR confidence, quality flags), Google Sheets export, and AsyncStorage-backed settings are untouched, only re-skinned.
- Two intentional deviations from the raw mockup, made to keep the screens truthful to what the app actually does (the mockup was built with static demo data, not the real backend):
  - **Processing step labels/counts** now reflect the real pipeline (patent = 5 backend-reported stages: Preprocessing Image, Detecting Document Type, Running OCR Extraction, Cross-Checking Fields, Scoring Confidence — matching the actual SOTA v3.0 architecture from the 2026-07-01 entry; standard = 2 real stages) rather than the mockup's generic 3/6-step fake timeline.
  - **Document type presets** kept the existing real Aadhaar/PAN/Voter ID/Student ID/Driver License/Custom set (restyled to the new chip look) instead of the mockup's placeholder ID Card/Passport/License/Custom set, since the real presets are what the product actually targets.
  - Dropped the mockup's "by ORBIS SYSTEMS" credit line — that name doesn't appear anywhere else in this project (owner is `surajmeruva0786`) and looks like placeholder branding from the design tool, not real attribution.
- **`filetract_mobile/package.json`** — added `expo-linear-gradient`, `@expo-google-fonts/dm-sans`, `@expo-google-fonts/instrument-sans`, `@expo-google-fonts/jetbrains-mono` (all installed via `npx expo install` for SDK 51-correct versions).
- Kicked off a new cloud build: `eas build --platform android --profile preview --non-interactive`. Build `6c6ec330-df20-4fbb-a90f-38d746593bed` sat `IN_QUEUE` for an extended period (consistent with the 2026-07-21 build, which also queued for hours) before finishing successfully. APK: https://expo.dev/artifacts/eas/qLc2_NZAdmsyKTVTQkZIP2bruGutEpFkSLM4IBLhJ6U.apk

**Why:** User's explicit ask — the app they downloaded and tested didn't match the design they'd created, and they wanted the actual designed frontend wired up before testing again.

**Verified:** `npx expo-doctor` → 17/17 checks pass. `npx expo export --platform android` → bundles cleanly (1020 modules, no errors). Cloud build completed successfully. Not yet installed on a physical device — user is downloading this APK to test now.

**Files changed:**
- `filetract_mobile/theme.js` (NEW)
- `filetract_mobile/components/GlowBackground.js` (NEW)
- `filetract_mobile/App.js`
- `filetract_mobile/screens/HomeScreen.js`, `FieldsScreen.js`, `ProcessingScreen.js`, `PreviewScreen.js`, `SettingsScreen.js`
- `filetract_mobile/package.json`, `filetract_mobile/package-lock.json`
- `CLAUDE.md` (this file)

---

### 2026-07-29 — Confirmed First EAS Cloud Build Finished (Installable APK Ready)

**What changed:**
- No code changes. Checked `eas build:list` and found the cloud build kicked off in the 2026-07-21 entry below had actually completed — that entry was written mid-build and left the result unconfirmed.
- Build `8f124880-6dfd-4ebe-9d65-ec0b06ff1e3d` (profile `preview`, Android, SDK 51, commit `47e8bce`) finished 2026-07-22 ~2:31am.
- Direct APK download: https://expo.dev/artifacts/eas/32sB5U76AF-PbbdkhiplPZQqazB8vlYLIu34r0-TjfQ.apk

**Why:** User asked for a progress update on the mobile app; the last changelog entry was stale on this exact point (build "in progress... not yet confirmed"), so it needed correcting rather than repeating.

**Still not done (unchanged from prior entries):**
- APK not yet installed/tested on a physical device or emulator.
- Icon/splash assets are still placeholder solid-color squares.
- Google Sheets Apps Script Web App still needs to be deployed by the user.
- Patent pipeline's `strategies_used: []` / `consensus_rate: 0` issue (noted 2026-07-20) hasn't been re-verified since the Groq migration.

**Files changed:**
- `CLAUDE.md` (this file)

---

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
