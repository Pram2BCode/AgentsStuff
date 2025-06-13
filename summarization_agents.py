from typing import List, Dict, Any
from pydantic import BaseModel

from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor
from src.vector_store.qdrant_store import VectorStore
from .base import Agent

class SummarizationAgent(Agent):
    """Base class for summarization agents."""
    llm_provider: LLMProvider

    def _summarize(self, text: str, summary_type: str = "document") -> str:
        prompt = f"Please provide a concise {summary_type} summary of the following text:\n\n{text}"
        return self.llm_provider.generate_text(prompt)

class DocumentSummaryAgent(SummarizationAgent):
    """Summarizes the entire PDF document."""
    name: str = "Document Summary Agent"
    description: str = "Summarizes the entire PDF."
    pdf_processor: PDFProcessor

    def run(self, pdf_path: str) -> str:
        extracted_content = self.pdf_processor.extract_content(pdf_path)
        full_text = "\n".join([page["text"] for page in extracted_content])
        return self._summarize(full_text, summary_type="document")

class SectionSummaryAgent(SummarizationAgent):
    """Summarizes a specific section of the PDF."""
    name: str = "Section Summary Agent"
    description: str = "Accepts a section or heading and summarizes its content."
    pdf_processor: PDFProcessor

    def run(self, pdf_path: str, section_heading: str) -> str:
        extracted_content = self.pdf_processor.extract_content(pdf_path)
        section_text = ""
        for page_content in extracted_content:
            for section in page_content["sections"]:
                if section["heading"] == section_heading:
                    section_text = section["heading"] + "\n" + "\n".join(section["paragraphs"])
                    break
            if section_text:
                break
        if not section_text:
            return f"Section with heading \'{section_heading}\' not found."
        return self._summarize(section_text, summary_type="section")

class PageBasedSummaryAgent(SummarizationAgent):
    """Summarizes content based on a page number or range."""
    name: str = "Page-based Summary Agent"
    description: str = "Summarizes content based on a page number or range."
    pdf_processor: PDFProcessor

    def run(self, pdf_path: str, page_numbers: List[int]) -> str:
        extracted_content = self.pdf_processor.extract_content(pdf_path)
        pages_text = []
        for page_num in page_numbers:
            if 0 < page_num <= len(extracted_content):
                pages_text.append(extracted_content[page_num - 1]["text"])
        if not pages_text:
            return f"No content found for page numbers: {page_numbers}"
        full_pages_text = "\n".join(pages_text)
        return self._summarize(full_pages_text, summary_type="page")


