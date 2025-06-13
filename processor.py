from typing import List, Dict, Any
from pypdf import PdfReader

class PDFProcessor:
    """Handles PDF content extraction and chunking."""

    def extract_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts content from PDF without OCR, preserving sections, headings, and tables."""
        reader = PdfReader(pdf_path)
        content = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            # This is a simplified extraction. Real-world scenario needs advanced parsing
            # to identify headings, paragraphs, and tables accurately.
            # For now, we'll treat each page as a section.
            content.append({
                "page_number": page_num + 1,
                "text": text,
                "sections": [
                    {
                        "heading": f"Page {page_num + 1}",
                        "paragraphs": [p.strip() for p in text.split('\n\n') if p.strip()],
                        "tables": [] # pypdf doesn't directly extract structured tables easily
                    }
                ]
            })
        return content

    def semantic_chunking(self, extracted_content: List[Dict[str, Any]]) -> List[str]:
        """Performs semantic chunking of the extracted data suitable for embedding."""
        chunks = []
        for page_content in extracted_content:
            for section in page_content["sections"]:
                # Combine heading and paragraphs for chunking
                chunk_text = f"{section['heading']}\n" + "\n".join(section["paragraphs"])
                chunks.append(chunk_text)
                # In a more advanced scenario, tables would also be processed and chunked
        return chunks


