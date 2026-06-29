import AsyncStorage from '@react-native-async-storage/async-storage';

const SHEETS_URL_KEY = '@filetract_sheets_url';

/**
 * Save the Google Apps Script Web App URL to persistent storage.
 */
export async function saveSheetsUrl(url) {
  await AsyncStorage.setItem(SHEETS_URL_KEY, url);
}

/**
 * Retrieve the stored Google Apps Script Web App URL.
 */
export async function getSheetsUrl() {
  return await AsyncStorage.getItem(SHEETS_URL_KEY);
}

/**
 * Flatten extraction results into a simple key→value object.
 * Handles both standard and patent pipeline result formats.
 */
function flattenResults(results) {
  const flat = {};

  // Patent pipeline: results.extracted_fields[field].value
  if (results.extracted_fields) {
    for (const [field, data] of Object.entries(results.extracted_fields)) {
      flat[field] = typeof data === 'object' ? (data.value ?? '') : data;
    }
    return flat;
  }

  // Standard pipeline: results.results[field] = value directly
  if (results.results) {
    return typeof results.results === 'object' ? results.results : {};
  }

  // Already flat
  return results;
}

/**
 * Send extracted data to Google Sheets via Apps Script Web App.
 *
 * @param {object} extractionResults - Raw API result object
 * @param {string} imageFilename - Original filename for reference
 * @param {string} [customSheetsUrl] - Override stored URL
 * @returns {Promise<{success: boolean, message: string}>}
 */
export async function sendToGoogleSheets(extractionResults, imageFilename, customSheetsUrl) {
  const url = customSheetsUrl || (await getSheetsUrl());

  if (!url || !url.startsWith('https://')) {
    throw new Error(
      'Google Sheets Web App URL not configured.\nGo to Settings and enter your Apps Script deployment URL.'
    );
  }

  const flat = flattenResults(extractionResults);

  const payload = {
    timestamp: new Date().toISOString(),
    source_file: imageFilename || 'unknown',
    fields: flat,
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    redirect: 'follow',
  });

  // Apps Script redirects POST to GET; follow the redirect
  if (!response.ok && response.status !== 302) {
    const text = await response.text();
    throw new Error(`Sheets API error ${response.status}: ${text}`);
  }

  let responseText = '';
  try {
    responseText = await response.text();
  } catch (_) {}

  return { success: true, message: responseText || 'Data sent to Google Sheets' };
}
