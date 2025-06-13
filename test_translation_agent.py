import unittest
from unittest.mock import MagicMock, patch
from src.agents.translation_agent import TranslationAgent
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor

class TestTranslationAgent(unittest.TestCase):

    def setUp(self):
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        self.mock_pdf_processor = MagicMock(spec=PDFProcessor)
        self.agent = TranslationAgent(llm_provider=self.mock_llm_provider, pdf_processor=self.mock_pdf_processor)

    def test_run(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {
                "page_number": 1,
                "text": "",
                "sections": [
                    {"heading": "Title", "paragraphs": ["Hello world."]}
                ]
            }
        ]
        self.mock_llm_provider.generate_text.return_value = "Hola mundo."

        result = self.agent.run("dummy.pdf", "Spanish")

        self.assertIn("translated_content", result)
        self.assertEqual(result["translated_content"][0]["translated_sections"][0]["translated_text"], "Hola mundo.")
        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_llm_provider.generate_text.assert_called_once()

if __name__ == "__main__":
    unittest.main()


