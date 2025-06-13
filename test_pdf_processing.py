import unittest
import os
from unittest.mock import MagicMock, patch
from src.pdf_processing.processor import PDFProcessor
from pypdf import PdfReader, PdfWriter

class TestPDFProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = PDFProcessor()
        self.mock_pdf_path = "dummy.pdf"
        
        # Create a valid dummy PDF file for testing
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(self.mock_pdf_path, "wb") as f:
            writer.write(f)

    def tearDown(self):
        # Clean up the dummy PDF file
        if os.path.exists(self.mock_pdf_path):
            os.remove(self.mock_pdf_path)

    @patch("pypdf.PdfReader")
    def test_extract_content(self, MockPdfReader):
        mock_reader_instance = MagicMock()
        MockPdfReader.return_value = mock_reader_instance
        
        # Mocking pages and extract_text for PdfReader
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = """Page 1 content.\n\nSection 1 heading.\nSection 1 paragraph 1.\nSection 1 paragraph 2.\n"""
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = """Page 2 content.\n\nSection 2 heading.\nSection 2 paragraph 1.\n"""
        mock_reader_instance.pages = [mock_page1, mock_page2]

        extracted_data = self.processor.extract_content(self.mock_pdf_path)

        self.assertEqual(len(extracted_data), 2)
        self.assertEqual(extracted_data[0]["page_number"], 1)
        self.assertIn("Page 1 content", extracted_data[0]["text"])
        self.assertIn("Section 1 heading", extracted_data[0]["sections"][0]["heading"])

    def test_semantic_chunking(self):
        mock_extracted_content = [
            {"page_number": 1, "text": "This is a sentence. This is another sentence.", 
             "sections": [{"heading": "Section 1", "paragraphs": ["Paragraph 1", "Paragraph 2"]}]}
        ]
        chunks = self.processor.semantic_chunking(mock_extracted_content)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()


