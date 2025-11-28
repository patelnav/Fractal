import unittest
from unittest.mock import MagicMock, patch
from fractal_repair_loop import solve_with_repair

class MockQwenInterface:
    def generate_sketch(self, sig, doc):
        return "Plan v0", {"prompt_tokens": 1, "completion_tokens": 1}
        
    def _generate(self, prompt, max_tokens, temperature):
        # This is called by generate_guided_code
        return "def foo(): return 0", {"prompt_tokens": 1, "completion_tokens": 1}
        
    def repair_sketch(self, sig, doc, old_sketch, failing_code, error):
        return "Plan v1 (Fixed)", {"prompt_tokens": 1, "completion_tokens": 1}

class TestRepairPipeline(unittest.TestCase):
    
    @patch('fractal_repair_loop.check_correctness')
    def test_repair_success(self, mock_check):
        # Mock execution results
        # Call 1 (Attempt 0): Fail
        # Call 2 (Attempt 1): Pass
        mock_check.side_effect = [
            {"passed": False, "error": "AssertionError"},
            {"passed": True, "error": None}
        ]
        
        model = MockQwenInterface()
        problem = {"task_id": "T1", "prompt": "def foo():\n    pass"}
        
        result = solve_with_repair(model, problem)
        
        self.assertEqual(result['final_status'], "passed_at_1")
        self.assertEqual(len(result['trajectory']), 2)
        self.assertEqual(result['trajectory'][0]['step'], 0)
        self.assertEqual(result['trajectory'][0]['passed'], False)
        self.assertEqual(result['trajectory'][1]['step'], 1)
        self.assertEqual(result['trajectory'][1]['passed'], True)
        self.assertEqual(result['trajectory'][1]['sketch'], "Plan v1 (Fixed)")
        
        print("Repair Pipeline Test: Successfully simulated Fail -> Repair -> Pass loop.")

    @patch('fractal_repair_loop.check_correctness')
    def test_max_retries_exceeded(self, mock_check):
        # Always fail
        mock_check.return_value = {"passed": False, "error": "Still failing"}
        
        model = MockQwenInterface()
        problem = {"task_id": "T1", "prompt": "def foo():\n    pass"}
        
        result = solve_with_repair(model, problem)
        
        self.assertEqual(result['final_status'], "failed")
        # Step 0 + 5 Retries = 6 total entries
        self.assertEqual(len(result['trajectory']), 6)
        print("Repair Pipeline Test: Successfully hit max retries.")

if __name__ == "__main__":
    unittest.main()
