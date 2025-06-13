import unittest
from unittest.mock import MagicMock, patch
from src.agents.summarization_agents import DocumentSummaryAgent, SectionSummaryAgent, PageBasedSummaryAgent
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor

class TestSummarizationAgents(unittest.TestCase):

    def setUp(self):
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        self.mock_pdf_processor = MagicMock(spec=PDFProcessor)

    def test_document_summary_agent(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {"page_number": 1, "text": "This is page 1 content.", "sections": []},
            {"page_number": 2, "text": "This is page 2 content.", "sections": []}
        ]
        self.mock_llm_provider.generate_text.return_value = "Overall summary."

        agent = DocumentSummaryAgent(llm_provider=self.mock_llm_provider, pdf_processor=self.mock_pdf_processor)
        summary = agent.run("dummy.pdf")

        self.assertEqual(summary, "Overall summary.")
        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_llm_provider.generate_text.assert_called_once()

    def test_section_summary_agent(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {
                "page_number": 1,
                "text": "",
                "sections": [
                    {"heading": "Introduction", "paragraphs": ["Intro paragraph 1.", "Intro paragraph 2."]},
                    {"heading": "Conclusion", "paragraphs": ["Conclusion paragraph 1."]}
                ]
            }
        ]
        self.mock_llm_provider.generate_text.return_value = "Section summary."

        agent = SectionSummaryAgent(llm_provider=self.mock_llm_provider, pdf_processor=self.mock_pdf_processor)
        summary = agent.run("dummy.pdf", "Introduction")

        self.assertEqual(summary, "Section summary.")
        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_llm_provider.generate_text.assert_called_once()

    def test_page_based_summary_agent(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {"page_number": 1, "text": "Page 1 text.", "sections": []},
            {"page_number": 2, "text": "Page 2 text.", "sections": []}
        ]
        self.mock_llm_provider.generate_text.return_value = "Page summary."

        agent = PageBasedSummaryAgent(llm_provider=self.mock_llm_provider, pdf_processor=self.mock_pdf_processor)
        summary = agent.run("dummy.pdf", [1])

        self.assertEqual(summary, "Page summary.")
        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_llm_provider.generate_text.assert_called_once()

if __name__ == "__main__":
    unittest.main()


