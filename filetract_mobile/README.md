# FileTract Mobile App

A React Native (Expo) mobile application for extracting structured fields from ID cards and documents using the FileTract patent-pending AI pipeline, with direct export to Google Sheets.

## Features

- **Camera Capture**: Photograph ID cards (Aadhaar, PAN, Voter ID, Student ID, Driver's License, etc.)
- **Gallery Upload**: Select existing photos from your phone
- **Flexible Field Extraction**: Choose preset field templates or define custom fields
- **Patent Pipeline**: Option to use FileTract's 5-stage confidence-weighted pipeline for maximum accuracy
- **Live Preview**: View extracted fields with OCR confidence scores
- **Inline Editing**: Edit extracted values before export
- **Google Sheets Export**: One-tap export to any Google Sheet via Apps Script Web App

## Screen Flow

```
Home → Fields → Processing → Preview → (Export to Sheets)
                                  ↓
                              Settings
```

## Setup

### 1. Install Dependencies

```bash
cd filetract_mobile
npm install
```

### 2. Configure Backend URL

In Settings (accessible from Home screen), enter your FileTract backend URL:
- Local dev: `http://192.168.x.x:5000` (your machine's LAN IP)
- Deployed: `https://your-app.onrender.com`

### 3. Set Up Google Sheets Integration

1. Open any Google Sheet where you want data to land
2. Go to **Extensions → Apps Script**
3. Paste the contents of `google_apps_script/Code.gs`
4. Click **Deploy → New Deployment**
5. Type: **Web App**, Execute as: **Me**, Who has access: **Anyone**
6. Click Deploy, copy the Web App URL
7. Paste the URL in the FileTract mobile app Settings screen

### 4. Run the App

```bash
npx expo start
```

Scan the QR code with Expo Go (iOS/Android) or press `a` for Android emulator / `i` for iOS simulator.

## Architecture

```
filetract_mobile/
├── App.js                          # Navigation root
├── screens/
│   ├── HomeScreen.js               # Camera / gallery pick
│   ├── FieldsScreen.js             # Field configuration + pipeline choice
│   ├── ProcessingScreen.js         # Upload + extraction progress
│   ├── PreviewScreen.js            # Results preview + Google Sheets export
│   └── SettingsScreen.js           # Backend URL, Sheets URL, defaults
├── services/
│   ├── api.js                      # FileTract backend API client
│   ├── googleSheets.js             # Google Apps Script integration
│   └── storage.js                  # Persistent settings (AsyncStorage)
├── google_apps_script/
│   └── Code.gs                     # Deploy this in Google Apps Script
├── package.json
└── app.json
```

## API Integration

The app connects to the existing FileTract Flask backend:

| Action | Endpoint |
|--------|----------|
| Upload image | `POST /api/upload` |
| Start extraction | `POST /api/extract` |
| Poll status | `GET /api/status/<job_id>` |
| Fetch results | `GET /api/result/<job_id>` |

## Building for Production

```bash
# Install EAS CLI
npm install -g eas-cli

# Build Android APK
eas build --platform android --profile preview

# Build iOS IPA
eas build --platform ios --profile preview
```

## Requirements

- Node.js 18+
- Expo CLI (`npm install -g expo-cli`)
- Expo Go app on your phone (for testing)
- FileTract backend running (local or deployed)
- Google account (for Sheets integration)
