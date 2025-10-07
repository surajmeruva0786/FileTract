import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------------------------
# 🔹 1. File paths setup
# ------------------------------------------
# Add all PDF/image file paths here
FILES = [
    r"10th_long_memo.pdf",
    r"Screenshot 2025-10-07 211503.png"
]

# ------------------------------------------
# 🔹 2. OCR extraction functions
# ------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using OCR (via PyMuPDF + Tesseract)"""
    text_pages = []
    pdf_document = fitz.open(pdf_path)
    print(f"\n📄 Total pages found in {os.path.basename(pdf_path)}: {pdf_document.page_count}")

    for page_num in range(pdf_document.page_count):
        print(f"🌀 Processing page {page_num + 1}...")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
        text_pages.append(text)

    pdf_document.close()
    return "\n--- PAGE BREAK ---\n".join(text_pages)

def extract_text_from_image(image_path):
    """Extract text from a single image using OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng")
    return text

# ------------------------------------------
# 🔹 3. Main execution for multiple files
# ------------------------------------------
if __name__ == "__main__":
    for file_path in FILES:
        if not os.path.exists(file_path):
            print(f"\n❌ File not found: {file_path}")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        print(f"\n================= 📝 PROCESSING {os.path.basename(file_path)} =================\n")

        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            extracted_text = extract_text_from_image(file_path)
        else:
            print(f"⚠️ Unsupported file type: {ext}")
            continue

        print(extracted_text)
        print("\n============================================================\n")
