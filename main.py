import os
from src.llm_providers.provider_factory import get_llm_provider
from src.pdf_processing.processor import PDFProcessor
from src.vector_store.qdrant_store import VectorStore
from src.agents.summarization_agents import DocumentSummaryAgent, SectionSummaryAgent, PageBasedSummaryAgent
from src.agents.translation_agent import TranslationAgent
from src.agents.comparison_qa_agent import ComparisonQA Agent
from src.agents.evaluation_agent import EvaluationAgent

# Configuration (replace with your actual Azure OpenAI details)
# For a real application, use environment variables or a config file
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "YOUR_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "YOUR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Initialize LLM Provider
llm_provider = get_llm_provider(
    "azure_openai",
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME # For chat completions
)

# Initialize PDF Processor and Vector Store
pdf_processor = PDFProcessor()
vector_store = VectorStore()

# Initialize Agents
document_summary_agent = DocumentSummaryAgent(llm_provider=llm_provider, pdf_processor=pdf_processor)
section_summary_agent = SectionSummaryAgent(llm_provider=llm_provider, pdf_processor=pdf_processor)
page_based_summary_agent = PageBasedSummaryAgent(llm_provider=llm_provider, pdf_processor=pdf_processor)
translation_agent = TranslationAgent(llm_provider=llm_provider, pdf_processor=pdf_processor)
comparison_qa_agent = ComparisonQAAgent(llm_provider=llm_provider, pdf_processor=pdf_processor, vector_store=vector_store)
evaluation_agent = EvaluationAgent(llm_provider=llm_provider, pdf_processor=pdf_processor)

def main():
    print("PDF Intelligence System - Agent Demonstration")
    print("Please ensure you have a PDF file named 'sample.pdf' in the root directory for testing.")
    print("Also, ensure your Azure OpenAI environment variables are set or replace placeholders in main.py.")

    sample_pdf_path = "sample.pdf"

    if not os.path.exists(sample_pdf_path):
        print(f"Error: {sample_pdf_path} not found. Please create one for testing.")
        return

    # --- Demonstrate Document Summary Agent ---
    print("\n--- Document Summary ---")
    doc_summary = document_summary_agent.run(sample_pdf_path)
    print(f"Document Summary: {doc_summary}")

    # --- Demonstrate Section Summary Agent ---
    print("\n--- Section Summary (assuming a section heading 'Introduction') ---")
    # This requires a known section heading from your PDF. Adjust as needed.
    section_summary = section_summary_agent.run(sample_pdf_path, "Introduction")
    print(f"Section Summary: {section_summary}")

    # --- Demonstrate Page-based Summary Agent ---
    print("\n--- Page-based Summary (Page 1) ---")
    page_summary = page_based_summary_agent.run(sample_pdf_path, [1])
    print(f"Page Summary: {page_summary}")

    # --- Demonstrate Translation Agent ---
    print("\n--- Translation (to Spanish) ---")
    translated_content = translation_agent.run(sample_pdf_path, "Spanish")
    print(f"Translated Content (first page): {translated_content['translated_content'][0]['translated_sections'][0]['translated_text'][:200]}...")

    # --- Demonstrate Comparison + QA Chatbot Agent ---
    print("\n--- Comparison + QA Chatbot ---")
    print("Please ensure you have 'sample2.pdf' for comparison testing.")
    sample_pdf2_path = "sample2.pdf"
    if os.path.exists(sample_pdf2_path):
        qa_response = comparison_qa_agent.run(sample_pdf_path, sample_pdf2_path, "What is the main topic of the documents?")
        print(f"QA Answer: {qa_response['answer']}")
        print(f"References: {qa_response['references']}")
    else:
        print(f"Skipping Comparison + QA: {sample_pdf2_path} not found.")

    # --- Demonstrate Evaluation Agent ---
    print("\n--- Evaluation Agent ---")
    evaluation_report = evaluation_agent.run(sample_pdf_path, num_questions=2)
    print(f"Generated Questions: {evaluation_report['questions_generated']}")
    print(f"Overall Assessment: {evaluation_report['overall_assessment']}")

if __name__ == "__main__":
    main()


