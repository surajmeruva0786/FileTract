# FileTract: AI-Powered Multi-File OCR with Gemini Integration

FileTract is an intelligent document processing system that extracts text from multiple documents (PDFs and images) using OCR, then leverages Google's Gemini AI to automatically extract structured field data. Perfect for processing certificates, forms, ID cards, and any scanned documents at scale.

## 🌟 Features

### Core Capabilities
- **Multi-File Processing**: Process multiple PDFs and images in a single run
- **High-Quality OCR**: Tesseract OCR at 300 DPI for accurate text extraction
- **AI-Powered Field Extraction**: Gemini 2.5 Flash intelligently identifies and extracts field values
- **Interactive CLI**: User-friendly command-line interface with clear prompts
- **Flexible Field Selection**: Define custom fields for any document type
- **Persistent Storage**: Saves extracted text and structured data to files

### Supported Formats
- **PDF Documents**: Multi-page PDF support with page-by-page processing
- **Images**: PNG, JPG, JPEG, TIFF, BMP

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/surajmeruva0786/FileTract.git
cd FileTract
```

2. **Create and activate virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Configure API Key**
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Then edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your-actual-api-key-here
```

### Usage

Run the main script:
```powershell
python gemini_ocr_extract.py
```

**Step 1: Select Files**
- Press Enter to use default files in the directory, OR
- Type file paths separated by commas

**Step 2: Define Fields**
Enter the fields you want to extract (comma-separated):
```
Name, Father Name, Mother Name, School, Date of Birth, CGPA
```

**Step 3: View Results**
- Extracted text displayed in terminal
- Field values extracted by Gemini AI
- Results automatically saved to files

## 📊 Example Output

### Input
- Document: SSC Certificate (Screenshot or PDF)

### Fields Requested
```
Name, Father Name, Mother Name, School, Date of Birth, CGPA
```

### Extracted Results
```json
{
    "Name": "RAPOLU SHIVA TEJA",
    "Father Name": "RAPOLU MARUTHE RAO",
    "Mother Name": "RAPOLU MALATHE",
    "School": "NEW VISION CONCEPT SCHOOL. KHAMMAM",
    "Date of Birth": "08/08/2002",
    "CGPA": "9.8"
}
```

## 🛠️ Technology Stack

- **OCR Engine**: Tesseract OCR with PyMuPDF for PDF processing
- **AI Model**: Google Gemini 2.5 Flash
- **Image Processing**: Pillow (PIL)
- **Data Validation**: Pydantic
- **Language**: Python 3.11+

## 📁 Project Structure

```
FileTract/
├── gemini_ocr_extract.py      # Main script with Gemini AI integration
├── ocr_extract.py              # Legacy OCR script (basic extraction)
├── test_gemini.py              # API testing utility
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── venv/                       # Virtual environment (not in repo)
```

## 📝 Output Files

For each processed document, FileTract creates:

1. **`filename_extracted_text.txt`**
   - Complete OCR text output
   - Includes all pages for PDFs
   - Raw text for debugging

2. **`filename_extracted_fields.json`**
   - Structured field values
   - JSON format for easy integration
   - AI-extracted data

## 🔧 Configuration

### API Key
The Gemini API key is stored securely in a `.env` file (not committed to Git):

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your-actual-api-key-here
   ```

> **Note**: Never commit your `.env` file to version control. It's already excluded in `.gitignore`.

### Tesseract Path
Update the Tesseract path if installed elsewhere:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### OCR Settings
- **DPI**: 300 (configurable in `get_pixmap(dpi=300)`)
- **Language**: English (`lang="eng"`)

## 🎯 Use Cases

- **Educational Institutions**: Extract student data from certificates and mark sheets
- **HR Departments**: Process employee documents and ID cards
- **Government Offices**: Digitize citizen records and applications
- **Healthcare**: Extract patient information from medical forms
- **Legal**: Process contracts and legal documents

## 🔍 How It Works

1. **File Selection**: User provides file paths or uses defaults
2. **OCR Processing**: Tesseract extracts text from each document
3. **Text Display**: Extracted text shown in terminal with separators
4. **Field Input**: User specifies which fields to extract
5. **AI Extraction**: Gemini AI analyzes text and identifies field values
6. **Results Output**: Structured data displayed and saved to JSON

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Google Gemini AI for intelligent field extraction
- Tesseract OCR for text recognition
- PyMuPDF for PDF processing

---

**FileTract** - Transform stacks of documents into structured, actionable data with the power of AI.