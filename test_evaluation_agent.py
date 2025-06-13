import unittest
from unittest.mock import MagicMock, patch
from src.agents.evaluation_agent import EvaluationAgent
from src.llm_providers.base import LLMProvider
from src.pdf_processing.processor import PDFProcessor

class TestEvaluationAgent(unittest.TestCase):

    def setUp(self):
        self.mock_llm_provider = MagicMock(spec=LLMProvider)
        self.mock_pdf_processor = MagicMock(spec=PDFProcessor)
        self.agent = EvaluationAgent(
            llm_provider=self.mock_llm_provider,
            pdf_processor=self.mock_pdf_processor
        )

    def test_generate_questions(self):
        self.mock_pdf_processor.extract_content.return_value = [
            {"page_number": 1, "text": "This is a document about AI. It discusses machine learning.", "sections": []}
        ]
        self.mock_llm_provider.generate_text.return_value = "Q1: What is AI?\nQ2: What is machine learning?"

        questions = self.agent.generate_questions("dummy.pdf", 2)

        self.assertEqual(questions, ["Q1: What is AI?", "Q2: What is machine learning?"])
        self.mock_pdf_processor.extract_content.assert_called_once_with("dummy.pdf")
        self.mock_llm_provider.generate_text.assert_called_once()

    def test_evaluate_answer(self):
        self.mock_llm_provider.generate_text.return_value = "Evaluation report: Good answer."

        evaluation = self.agent.evaluate_answer("What is AI?", "AI is artificial intelligence.", "AI is artificial intelligence.")

        self.assertEqual(evaluation["evaluation"], "Evaluation report: Good answer.")
        self.mock_llm_provider.generate_text.assert_called_once()

    def test_run(self):
        # Patch the generate_questions method on the class
        with patch.object(EvaluationAgent, 'generate_questions', return_value=["Q1", "Q2"]) as mock_generate_questions:
            # Re-instantiate the agent after patching the class method
            self.agent = EvaluationAgent(
                llm_provider=self.mock_llm_provider,
                pdf_processor=self.mock_pdf_processor
            )
            # Mock generate_text for correct answer and overall assessment
            self.mock_llm_provider.generate_text.side_effect = [
                "Correct answer to Q1.", # For correct_answer_prompt
                "Evaluation for Q1.",    # For evaluate_answer
                "Correct answer to Q2.", # For correct_answer_prompt
                "Evaluation for Q2.",    # For evaluate_answer
                "Overall assessment report."
            ]

            report = self.agent.run("dummy.pdf", 2)

            self.assertEqual(report["questions_generated"], ["Q1", "Q2"])
            self.assertEqual(len(report["assessment_results"]), 2)
            self.assertEqual(report["overall_assessment"], "Overall assessment report.")
            self.assertEqual(self.mock_llm_provider.generate_text.call_count, 5) # 2 correct answers + 2 evaluations + 1 overall

if __name__ == "__main__":
    unittest.main()


