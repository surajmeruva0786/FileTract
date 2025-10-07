import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Path to Tesseract executable (change if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------------------
# 🔹 1. File path setup
# ------------------------------------------
FILE_PATH = r"10th_long_memo.pdf"  # Change this to your file

# ------------------------------------------
# 🔹 2. OCR extraction function
# ------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using OCR (via PyMuPDF + Tesseract)"""
    text_pages = []

    pdf_document = fitz.open(pdf_path)
    print(f"📄 Total pages found: {pdf_document.page_count}")

    for page_num in range(pdf_document.page_count):
        print(f"\n🌀 Processing page {page_num + 1}...")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        text = pytesseract.image_to_string(img, lang="eng")
        text_pages.append(text)

    pdf_document.close()
    print("\n✅ Text extraction complete.\n")
    return "\n--- PAGE BREAK ---\n".join(text_pages)

# ------------------------------------------
# 🔹 3. Main execution
# ------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
    else:
        extracted_text = extract_text_from_pdf(FILE_PATH)
        print("\n================= 🧾 EXTRACTED TEXT =================\n")
        print(extracted_text)
        print("\n=====================================================\n")
