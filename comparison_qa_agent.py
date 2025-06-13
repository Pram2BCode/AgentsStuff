from typing import List, Dict, Any
from pydantic import BaseModel
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor
from src.vector_store.qdrant_store import VectorStore
from .base import Agent

class ComparisonQAAgent(Agent):
    """Compares two uploaded PDFs and supports QA-style interaction."""
    name: str = "Comparison + QA Chatbot Agent"
    description: str = "Compares two uploaded PDFs and supports QA-style interaction, providing answers with references."
    llm_provider: LLMProvider
    pdf_processor: PDFProcessor
    vector_store: VectorStore

    def _embed_and_store_pdf(self, pdf_path: str, doc_id: str):
        extracted_content = self.pdf_processor.extract_content(pdf_path)
        chunks = self.pdf_processor.semantic_chunking(extracted_content)
        embeddings = [self.llm_provider.generate_embedding(chunk) for chunk in chunks]
        metadata = []
        for i, page_content in enumerate(extracted_content):
            for section in page_content["sections"]:
                metadata.append({"doc_id": doc_id, "page_number": page_content["page_number"], "heading": section["heading"]})
        self.vector_store.store_embeddings(chunks, embeddings, metadata)

    def run(self, pdf1_path: str, pdf2_path: str, question: str) -> Dict[str, Any]:
        # For simplicity, re-embedding each time. In a real app, manage stored PDFs.
        self._embed_and_store_pdf(pdf1_path, "doc1")
        self._embed_and_store_pdf(pdf2_path, "doc2")

        query_embedding = self.llm_provider.generate_embedding(question)
        similar_chunks = self.vector_store.retrieve_similar(query_embedding, top_k=10)

        context = ""
        references = []
        for chunk in similar_chunks:
            context += "Document {doc_id}, Page {page_number}: {text}\n".format(
                doc_id=chunk["metadata"]["doc_id"],
                page_number=chunk["metadata"]["page_number"],
                text=chunk["text"]
            )
            references.append({
                "document": chunk["metadata"]["doc_id"],
                "page_number": chunk["metadata"]["page_number"],
                "section": chunk["metadata"]["heading"],
                "text_snippet": chunk["text"]
            })
        
        prompt = f"Given the following context, answer the question. State which document, section/page the answer is from, and include references.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = self.llm_provider.generate_text(prompt)

        return {"answer": answer, "references": references}




