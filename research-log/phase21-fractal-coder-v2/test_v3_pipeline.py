import unittest
from unittest.mock import MagicMock
from fractal_guided_v3 import solve_problem_guided

class MockQwenInterface:
    def generate_sketch(self, signature, docstring):
        # Mock a sketch response
        return "# 1. Do this\n# 2. Do that", {"prompt_tokens": 10, "completion_tokens": 10}

    def _generate(self, prompt, max_tokens, temperature):
        # Mock a code response
        code = "    return 42"
        return code, {"prompt_tokens": 20, "completion_tokens": 5}

class TestV3Pipeline(unittest.TestCase):
    def test_solve_problem_guided(self):
        model = MockQwenInterface()
        problem = {
            "prompt": "def foo():\n    \"\"\"Doc\"\"\"\n",
            "task_id": "Test/0",
            "entry_point": "foo"
        }
        
        result = solve_problem_guided(model, problem)
        
        self.assertEqual(result['method'], "fractal_v3_guided")
        self.assertIn("sketch", result)
        self.assertIn("generated_code", result)
        self.assertIn("usage", result)
        self.assertEqual(result['generated_code'], "    return 42")
        print("Pipeline Test Passed: V3 Guided Logic works (with mock model).")

if __name__ == "__main__":
    unittest.main()

