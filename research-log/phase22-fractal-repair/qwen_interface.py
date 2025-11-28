import os
from typing import List, Optional, Dict

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class QwenInterface:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct", tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.llm = None
        if VLLM_AVAILABLE:
            print(f"Initializing vLLM with {model_name}...")
            self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        else:
            print("Warning: vLLM not available. Interface will fail if used for generation.")

    def generate_hierarchical_sketch(self, signature: str, docstring: str) -> tuple[str, dict]:
        """
        Generates a hierarchical step-by-step sketch using nested numbering.
        """
        prompt = f"""<|im_start|>system
You are an expert Python architect. Your goal is to plan the implementation of a function given its signature and docstring.
Create a Hierarchical Execution Plan using nested numbering (e.g., 1., 1.1., 1.1.1.) to explicitly show control flow and nesting.

Rules:
- Use '1. Step' for top-level actions.
- Use '1.1. Sub-step' for actions inside a block (loop/if).
- Use '1.1.1. Leaf-step' for deeper nesting.
- The indentation of your plan should reflect the python code indentation.
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Please provide the hierarchical plan.
<|im_end|>
<|im_start|>assistant
"""
        return self._generate(prompt, max_tokens=512, temperature=0.2)

    def repair_sketch(self, signature: str, docstring: str, old_sketch: str, failing_code: str, error_trace: str) -> tuple[str, dict]:
        """
        Analyzes a failure and generates a revised sketch.
        """
        prompt = f"""<|im_start|>system
You are an expert Python developer debugging a function.
The previous implementation failed tests. Your goal is to Analyze the error and Rewrite the Implementation Plan (Sketch) to fix the logic.
Do NOT write code. Just write the new Plan (numbered steps).
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Previous Plan:
{old_sketch}

Failing Code:
{failing_code}

Error Trace:
{error_trace}

Please provide a REVISED Implementation Plan that addresses the error.
<|im_end|>
<|im_start|>assistant
"""
        return self._generate(prompt, max_tokens=512, temperature=0.2)

    def generate_sketch(self, signature: str, docstring: str) -> tuple[str, dict]:
        """
        Generates a high-level step-by-step sketch (pseudocode/comments) for the function.
        """
        prompt = f"""<|im_start|>system
You are an expert Python architect. Your goal is to plan the implementation of a function given its signature and docstring.
Break down the logic into sequential steps using Python comments (#). Do not write the actual code, just the high-level plan.
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Please provide a step-by-step implementation plan as comments.
<|im_end|>
<|im_start|>assistant
"""
        return self._generate(prompt, max_tokens=512, temperature=0.2)

    def render_step(self, signature: str, docstring: str, sketch: str, step_to_render: str, context_so_far: str = "") -> tuple[str, dict]:
        """
        Generates the code for a specific step in the sketch, given the context.
        """
        prompt = f"""<|im_start|>system
You are an expert Python developer. Implement the code for the specified comment step.
Ensure the code is correct, efficient, and fits within the provided context.
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Full Plan:
{sketch}

Current Context (Code so far):
{context_so_far}

Target Step to Implement:
{step_to_render}

Write only the Python code for this step.
<|im_end|>
<|im_start|>assistant
"""
        return self._generate(prompt, max_tokens=256, temperature=0.2)
    
    def render_full_body(self, signature: str, docstring: str, sketch: str) -> tuple[str, dict]:
        """
        Alternative: Render the full body based on the sketch in one go (Baseline).
        """
        prompt = f"""<|im_start|>system
You are an expert Python developer. Implement the full function body based on the provided plan.
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Plan:
{sketch}

Write the complete Python code for the function body.
<|im_end|>
<|im_start|>assistant
"""
        return self._generate(prompt, max_tokens=1024, temperature=0.2)

    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
        if not self.llm:
            raise RuntimeError("vLLM not initialized.")
        
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["<|im_end|>"])
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        text = output.outputs[0].text.strip()
        usage = {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids)
        }
        return text, usage

if __name__ == "__main__":
    # Mock test if running locally without vLLM just to check syntax
    if not VLLM_AVAILABLE:
        print("Mocking QwenInterface for syntax check...")
        iface = QwenInterface()
        try:
            iface.generate_sketch("def add(a, b):", "Adds two numbers.")
        except RuntimeError as e:
            print(f"Caught expected error: {e}")
