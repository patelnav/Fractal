
import sys
import torch
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))
sys.path.insert(0, str(Path(__file__).parent)) # For synthetic_data

from run_fractal_engine import FractalDiffusionModel, FractalModelConfig
from synthetic_data import CodeConfig, OPS, decode_text, CHAR_TO_ID

class FractalCoder:
    """
    The Self-Healing Program Synthesizer (Vector 6).
    Combines Flash Flood (Generation) + Vector 7 (Repair).
    """
    def __init__(self, model_path, device='cpu'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        self.model = FractalDiffusionModel(self.config).to(device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # "Manager" is simulated: Randomly proposes roots
        
    def generate_flash_flood(self, roots: List[int], k=16) -> List[str]:
        """
        Flash Flood Decoder: Render roots to text in parallel (simulated Best-of-K).
        Since we trained a deterministic mapping (Root->Text), K=1 is fine, but we support K.
        """
        B = len(roots)
        # To simulate Flash Flood properly, we'd batch them.
        # Here we just loop for simplicity of the demo harness, 
        # but the model is the parallel component.
        
        cond = torch.tensor(roots, device=self.device)
        
        # Input: [Cond, Noise]
        # cond_offset = chunk_offset (from training)
        cond_input = (cond + self.config.chunk_offset).unsqueeze(1)
        
        # Denoise (One-step generation for demo)
        # We pass pure noise at T-1
        noise = torch.randint(0, self.config.num_chars, (B, 4), device=self.device)
        x = torch.cat([cond_input, noise], dim=1)
        t = torch.tensor([self.config.num_timesteps - 1], device=self.device).repeat(B)
        
        with torch.no_grad():
            logits, _ = self.model(x, t, level=1)
            
        # Argmax decoding
        preds = torch.argmax(logits, dim=-1)
        
        texts = []
        for i in range(B):
            char_ids = preds[i].tolist()
            # Filter pad? No padding in this synthetic set really
            texts.append(decode_text(char_ids))
            
        return texts

    def execute_program(self, text_parts: List[str]) -> Tuple[bool, int, int]:
        """
        Execute the synthetic program.
        Returns (Success, Result, FailingIndex).
        """
        current_val = 0
        full_program = "".join(text_parts)
        
        # Parse and execute step by step to find error
        # Format: "+ 5", "* 2" (each 4 chars)
        
        for i, part in enumerate(text_parts):
            part = part.strip()
            if not part: continue
            
            op = part[0]
            try:
                val = int(part[1:])
            except ValueError:
                return False, current_val, i # Syntax error
            
            if op == '+':
                current_val += val
            elif op == '-':
                current_val -= val
            elif op == '*':
                current_val *= val
            else:
                return False, current_val, i # Invalid op
                
        return True, current_val, -1

    def repair_program(self, roots: List[int], target_val: int, max_steps=10) -> Tuple[bool, str, int]:
        """
        The Fractal Repair Loop.
        1. Render
        2. Verify
        3. Patch (if fail)
        """
        current_roots = list(roots)
        
        print(f"Target: {target_val}")
        
        for step in range(max_steps):
            # 1. Flash Flood Render
            # In a real system, we only re-render changed parts.
            text_parts = self.generate_flash_flood(current_roots)
            full_text = "".join(text_parts)
            
            # 2. Verify
            success, result, fail_idx = self.execute_program(text_parts)
            
            error = abs(target_val - result)
            print(f"Step {step}: {full_text} = {result} (Err: {error})")
            
            if success and result == target_val:
                return True, full_text, step
                
            # 3. Patch (Heuristic for Synthetic Task)
            # Identify which root to change.
            # Strategy: Randomly pick one root and change it?
            # Or Gradient Descent?
            # Let's use "Random Mutation of One Root" (Evolutionary Strategy)
            
            idx_to_patch = random.randint(0, len(current_roots)-1)
            
            # Mutation: Pick a neighbor root ID (likely similar op or value)
            # Or just random
            old_root = current_roots[idx_to_patch]
            new_root = random.randint(0, 29) # 30 roots
            
            current_roots[idx_to_patch] = new_root
            # print(f"  Patching idx {idx_to_patch}: {old_root} -> {new_root}")
            
        return False, full_text, max_steps

def test_fractal_coder():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    coder = FractalCoder("research-log/phase18-fractal-coder/fractal_coder_model.pt", device)
    
    print("\n[1] Testing Generation & Execution")
    # Test: 5 + 3 * 2 = 16
    # Roots: ADD 5 (4), ADD 3 (2), MUL 2 (20)
    roots = [4, 2, 20] 
    text = coder.generate_flash_flood(roots)
    print(f"Roots: {roots} -> Text: {text}")
    ok, res, _ = coder.execute_program(text)
    print(f"Execution: {res} (Expected 16)")
    
    print("\n[2] Testing Self-Healing (Repair Loop)")
    # Problem: Reach 20
    # Start: +5, +5 = 10 (Wrong)
    roots = [4, 4] # +5, +5
    target = 20
    
    success, final_prog, steps = coder.repair_program(roots, target)
    
    if success:
        print(f"\nSUCCESS! Repaired in {steps} steps.")
        print(f"Final Program: {final_prog}")
    else:
        print("\nFailed to repair.")

if __name__ == "__main__":
    test_fractal_coder()
