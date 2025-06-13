from typing import List, Dict, Any
from pydantic import BaseModel
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor
from .base import Agent

class EvaluationAgent(Agent):
    """Generates N questions from a PDF, runs an interactive Q&A, and evaluates answers."""
    name: str = "Evaluation Agent"
    description: str = "Generates N questions from a PDF, runs an interactive Q&A session, and evaluates answers."
    llm_provider: LLMProvider
    pdf_processor: PDFProcessor

    def generate_questions(self, pdf_path: str, num_questions: int = 5) -> List[str]:
        extracted_content = self.pdf_processor.extract_content(pdf_path)
        full_text = "\n".join([page["text"] for page in extracted_content])
        
        prompt = f"Generate {num_questions} insightful questions based on the following document content. Provide only the questions, one per line.\n\nDocument:\n{full_text}"
        questions_str = self.llm_provider.generate_text(prompt)
        return [q.strip() for q in questions_str.split('\n') if q.strip()]

    def evaluate_answer(self, question: str, user_answer: str, correct_answer: str) -> Dict[str, Any]:
        prompt = f"Evaluate the user's answer to the question. Provide feedback on accuracy, understanding, and areas for improvement. \n\nQuestion: {question}\nCorrect Answer: {correct_answer}\nUser's Answer: {user_answer}"
        evaluation_report = self.llm_provider.generate_text(prompt)
        return {"question": question, "user_answer": user_answer, "evaluation": evaluation_report}

    def run(self, pdf_path: str, num_questions: int = 5) -> Dict[str, Any]:
        questions = self.generate_questions(pdf_path, num_questions)
        assessment_results = []

        # In a real interactive scenario, you would prompt the user for answers here.
        # For this implementation, we'll simulate correct answers for demonstration.
        # A more robust solution would involve a separate mechanism for user interaction.

        for i, question in enumerate(questions):
            # Simulate getting the correct answer (e.g., from an internal knowledge base or by asking the LLM)
            correct_answer_prompt = f"Provide a concise answer to the following question based on the document. Question: {question}"
            correct_answer = self.llm_provider.generate_text(correct_answer_prompt)
            
            # For demonstration, let's assume the user provides a placeholder answer
            user_answer = f"User's simulated answer to: {question}"

            evaluation = self.evaluate_answer(question, user_answer, correct_answer)
            assessment_results.append(evaluation)
        
        overall_report_prompt = f"Based on the following individual question evaluations, generate an overall assessment report for the user, highlighting accuracy, understanding, and areas to improve.\n\nEvaluations:\n{assessment_results}"
        overall_assessment = self.llm_provider.generate_text(overall_report_prompt)

        return {"questions_generated": questions, "assessment_results": assessment_results, "overall_assessment": overall_assessment}


