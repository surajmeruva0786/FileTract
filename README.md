# FileTract: AI-Powered OCR Document Data Extraction System

FileTract is an end-to-end automated solution for extracting structured data from large sets of scanned documents, certificates, mark sheets, or ID cards. Leveraging AI-powered OCR and intelligent text processing, FileTract streamlines data collection and digitization, making it easy to convert stacks of documents into organized digital datasets.

## Features
- **Bulk Upload**: Upload multiple image or PDF documents via a clean, web-based interface.
- **Custom Field Extraction**: Specify key fields to extract (e.g., Name, Date of Birth, Marks, Certificate Number, etc.) for flexible, document-agnostic processing.
- **AI-Powered OCR**: Uses Tesseract OCR to convert images and PDFs into machine-readable text.
- **Intelligent Parsing**: Contextual keyword detection and pattern matching for accurate extraction, even from unstructured layouts.
- **Editable Results Table**: Review and edit extracted data in a responsive HTML/JS table before export.
- **Easy Export**: Download results as CSV or Excel files for further analysis or integration.
- **Simple Deployment**: All files reside in a single root directory—no subfolders—ensuring portability and ease of setup.

## Workflow
1. **Upload Documents**: Drag and drop or select multiple files (images or PDFs) in the web interface.
2. **Define Fields**: Enter the key fields you want to extract from the documents.
3. **Automated Extraction**: The Python backend processes each file, performs OCR, and searches for specified key-value pairs.
4. **Review & Edit**: View results in an editable table; make corrections or adjustments as needed.
5. **Export Data**: Download the final dataset in CSV or Excel format.

## Technology Stack
- **Backend**: Python, Flask, Tesseract OCR
- **Frontend**: HTML, CSS, JavaScript
- **File Export**: CSV, Excel

## Getting Started
1. **Install Requirements**:
   - Python 3.x
   - Flask
   - pytesseract
   - pdf2image
   - pandas
   - (Optional) openpyxl for Excel export
2. **Run the Server**:
   ```powershell
   python ocr_extract.py
   ```
3. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:5000`
4. **Upload, Extract, Review, and Export!**

## Project Structure
All files are located in the root directory for simplicity:
- `ocr_extract.py` — Main backend script (Flask server, OCR logic)
- `index.html`, `style.css`, `script.js` — Frontend files
- `README.md` — Project documentation

## Why FileTract?
- **Flexible**: Works with any document type and user-defined fields.
- **Accurate**: Combines OCR with smart text parsing for reliable results.
- **User-Friendly**: Simple interface, editable results, and easy export.
- **Portable**: No subfolders; deploy anywhere with minimal setup.

## License
This project is open-source and available under the MIT License.

---

**FileTract** helps you turn stacks of certificates and scanned documents into structured, actionable data—quickly and accurately.