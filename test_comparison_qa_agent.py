import unittest
from unittest.mock import MagicMock, patch
from src.agents.comparison_qa_agent import ComparisonQAAgent
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor
from src.vector_store.qdrant_store import VectorStore

class TestComparisonQAAgent(unittest.TestCase):

    def setUp(self):
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        self.mock_pdf_processor = MagicMock(spec=PDFProcessor)
        self.mock_vector_store = MagicMock(spec=VectorStore)
        self.agent = ComparisonQAAgent(
            llm_provider=self.mock_llm_provider,
            pdf_processor=self.mock_pdf_processor,
            vector_store=self.mock_vector_store
        )

    def test_embed_and_store_pdf(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {"page_number": 1, "text": "Page 1 content.", "sections": [{"heading": "Intro", "paragraphs": ["p1"]}]}
        ]
        self.mock_pdf_processor.semantic_chunking.return_value = ["chunk1"]
        self.mock_llm_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]

        self.agent._embed_and_store_pdf("dummy.pdf", "doc_test")

        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_pdf_processor.semantic_chunking.assert_called_once()
        self.mock_llm_provider.generate_embedding.assert_called_once_with("chunk1")
        self.mock_vector_store.store_embeddings.assert_called_once()

    def test_run(self):
        self.mock_llm_provider.generate_embedding.side_effect = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] # For query and chunk
        self.mock_vector_store.retrieve_similar.return_value = [
            {"text": "Relevant text from doc1", "score": 0.9, "metadata": {"doc_id": "doc1", "page_number": 1, "heading": "Section A"}}
        ]
        self.mock_llm_provider.generate_text.return_value = "Answer from LLM."

        # Mock _embed_and_store_pdf to prevent actual embedding during test
        with patch.object(self.agent, "_embed_and_store_pdf") as mock_embed:
            response = self.agent.run("pdf1.pdf", "pdf2.pdf", "What is the question?")

            mock_embed.assert_any_call("pdf1.pdf", "doc1")
            mock_embed.assert_any_call("pdf2.pdf", "doc2")
            self.mock_llm_provider.generate_embedding.assert_called_with("What is the question?")
            self.mock_vector_store.retrieve_similar.assert_called_once()
            self.mock_llm_provider.generate_text.assert_called_once()
            self.assertIn("Answer from LLM.", response["answer"])
            self.assertIn({"document": "doc1", "page_number": 1, "section": "Section A", "text_snippet": "Relevant text from doc1"}, response["references"])

if __name__ == "__main__":
    unittest.main()


