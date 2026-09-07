import re
import pdfplumber

def clean_extracted_text(text: str) -> str:
    """Cleans raw text extracted from PDF pages to remove noise before LLM extraction."""
    if not text:
        return ""
    
    # Remove common header/footer artifacts like "Page 1 of 50"
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page \d+", "", text, flags=re.IGNORECASE)
    
    # Fix hyphenated words split across line breaks (e.g., "pa-\ntient" -> "patient")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    
    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r"\n\s*\n", "\n", text)
    
    # Collapse consecutive spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip()

def load_pdf_text(pdf_path: str) -> str:
    """Extracts and cleans all text content from a given PDF file as a single string."""
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                raw_text += extracted + "\n"
                
    cleaned_text = clean_extracted_text(raw_text)
    return cleaned_text

if __name__ == "__main__":
    sample_pdf_path = r"C:\Users\Lenovo\Desktop\pader-extractor\data\PADER-FDA-Y0AHP_PADER_Full_sample_data_B-1_CLIENT_DEV_01_FDA_v1_20260810.pdf"
    
    try:
        full_text = load_pdf_text(sample_pdf_path)
        print("✅ Step 1 complete: PDF text successfully extracted and cleaned in a single string!")
        print(f"Total character length: {len(full_text)}")
        if full_text:
            print("\n--- Document Preview (First 400 characters) ---")
            print(full_text[:400])
    except FileNotFoundError:
        print(f"File not found at {sample_pdf_path}.")
    except Exception as e:
        print(f"Error during PDF extraction: {e}")