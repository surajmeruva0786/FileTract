/**
 * FileTract — Google Apps Script Web App
 *
 * PURPOSE:
 *   Receives POST requests from the FileTract mobile app and appends
 *   extracted field data as a new row in this Google Spreadsheet.
 *
 * SETUP:
 *   1. Open your Google Sheet.
 *   2. Go to Extensions → Apps Script.
 *   3. Paste this entire file, replacing any existing code.
 *   4. Click Deploy → New Deployment.
 *   5. Type: Web App.
 *   6. Execute as: Me.
 *   7. Who has access: Anyone.
 *   8. Click Deploy → copy the Web App URL.
 *   9. Paste that URL in the FileTract mobile app Settings screen.
 *
 * PAYLOAD FORMAT (sent by the mobile app):
 *   {
 *     "timestamp": "2026-06-29T12:00:00.000Z",
 *     "source_file": "aadhaar_card.jpg",
 *     "fields": {
 *       "Name": "Suraj Meruva",
 *       "Aadhaar Number": "XXXX XXXX XXXX",
 *       "Date of Birth": "01/01/2000"
 *     }
 *   }
 *
 * SHEET STRUCTURE:
 *   Row 1 (auto-created): Headers — Timestamp | Source File | Field1 | Field2 | ...
 *   Row 2+: Data rows appended by each extraction.
 */

// ─── CONFIG ────────────────────────────────────────────────────────────────
// Name of the sheet tab to write data to. Change if needed.
var SHEET_NAME = 'FileTract Extractions';
// ───────────────────────────────────────────────────────────────────────────

/**
 * Handle GET requests (required by Apps Script for web app health checks).
 */
function doGet(e) {
  return ContentService
    .createTextOutput(JSON.stringify({ status: 'ok', service: 'FileTract Sheets Integration' }))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Handle POST requests from the FileTract mobile app.
 */
function doPost(e) {
  try {
    var payload = JSON.parse(e.postData.contents);
    appendToSheet(payload);
    return ContentService
      .createTextOutput(JSON.stringify({ success: true, message: 'Row appended successfully' }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ success: false, error: err.toString() }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * Append a new row of extracted fields to the spreadsheet.
 */
function appendToSheet(payload) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName(SHEET_NAME);

  // Create the sheet if it doesn't exist
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
  }

  var fields = payload.fields || {};
  var fieldKeys = Object.keys(fields);

  // If the sheet is empty, write headers in row 1
  if (sheet.getLastRow() === 0) {
    var headers = ['Timestamp', 'Source File'].concat(fieldKeys);
    sheet.appendRow(headers);

    // Style header row
    var headerRange = sheet.getRange(1, 1, 1, headers.length);
    headerRange.setBackground('#0a0a12');
    headerRange.setFontColor('#00e5ff');
    headerRange.setFontWeight('bold');
    headerRange.setFontSize(11);
    sheet.setFrozenRows(1);
  } else {
    // Check if new field columns need to be added
    var existingHeaders = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    fieldKeys.forEach(function(key) {
      if (existingHeaders.indexOf(key) === -1) {
        // Add new column header
        var newCol = sheet.getLastColumn() + 1;
        sheet.getRange(1, newCol).setValue(key);
        sheet.getRange(1, newCol).setBackground('#0a0a12');
        sheet.getRange(1, newCol).setFontColor('#00e5ff');
        sheet.getRange(1, newCol).setFontWeight('bold');
        existingHeaders.push(key);
      }
    });
  }

  // Build the data row in the same order as headers
  var currentHeaders = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  var row = currentHeaders.map(function(header) {
    if (header === 'Timestamp') return payload.timestamp || new Date().toISOString();
    if (header === 'Source File') return payload.source_file || '';
    return fields[header] !== undefined ? fields[header] : '';
  });

  sheet.appendRow(row);

  // Auto-resize columns for readability
  sheet.autoResizeColumns(1, sheet.getLastColumn());
}
