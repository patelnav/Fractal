
import sys
import torch
import random
from pathlib import Path
import torch.nn.functional as F
from typing import List, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent)) # For critic
sys.path.insert(0, str(Path(__file__).parent.parent / "phase18-fractal-coder")) # For coder

from fractal_critic import FractalCritic
from fractal_coder import FractalCoder
from synthetic_data import OPS

class GuidedFractalCoder(FractalCoder):
    def __init__(self, coder_model_path, critic_model_path, device='cpu'):
        super().__init__(coder_model_path, device)
        
        self.critic = FractalCritic(num_roots=31, embed_dim=64, max_len=8).to(device)
        self.critic.load_state_dict(torch.load(critic_model_path, map_location=device, weights_only=True))
        self.critic.eval()
        
    def repair_program_guided(self, roots: List[int], target_val: int, max_steps=10) -> Tuple[bool, str, int]:
        """
        Critic-Guided Repair Loop (Improved).
        Uses top-k sampling for location and heuristic for mutation.
        """
        current_roots = list(roots)
        
        # Heuristic mappings
        # 0-8: ADD 1..9
        # 10-18: SUB 1..9
        # 20-22: MUL 2..4
        
        for step in range(max_steps):
            # 1. Render & Execute
            text_parts = self.generate_flash_flood(current_roots)
            full_text = "".join(text_parts)
            success, result, fail_idx = self.execute_program(text_parts)
            
            error = target_val - result
            
            if success and result == target_val:
                return True, full_text, step
                
            # 2. Consult Critic
            padded_roots = current_roots + [30] * (8 - len(current_roots))
            padded_roots = padded_roots[:8]
            
            roots_tensor = torch.tensor([padded_roots], device=self.device)
            error_tensor = torch.tensor([error], device=self.device)
            
            with torch.no_grad():
                logits = self.critic(roots_tensor, error_tensor)
                probs = F.softmax(logits, dim=1)
            
            # Mask out padding/invalid
            valid_len = len(current_roots)
            probs[0, valid_len:] = 0
            
            # 3. Select Faulty Root (Top-3 Sampling)
            # Get top 3 indices
            topk = torch.topk(probs[0], min(3, valid_len))
            indices = topk.indices.tolist()
            weights = topk.values.tolist()
            
            # Sample from top 3
            faulty_idx = random.choices(indices, weights=weights, k=1)[0]
            
            # 4. Patch with Heuristic
            # If Error > 0 (Need more): Try changing SUB->ADD or increasing ADD
            # If Error < 0 (Need less): Try changing ADD->SUB or decreasing ADD
            
            old_root = current_roots[faulty_idx]
            
            if error > 0:
                # We need to increase value.
                # Prefer ADD (0-8) or MUL (20-22)
                # Bias towards ADD
                if random.random() < 0.7:
                    new_root = random.randint(0, 8) # ADD 1..9
                else:
                    new_root = random.randint(20, 22) # MUL
            else:
                # We need to decrease value.
                # Prefer SUB (10-18)
                if random.random() < 0.8:
                    new_root = random.randint(10, 18) # SUB 1..9
                else:
                    new_root = random.randint(0, 4) # Small ADD?
            
            current_roots[faulty_idx] = new_root
            
        return False, full_text, max_steps

def compare_search_strategies(num_trials=50):
    device = torch.device("cpu") # MPS missing op for nested tensor mask
    
    coder_path = "research-log/phase18-fractal-coder/fractal_coder_model.pt"
    critic_path = "research-log/phase19-fractal-critic/critic_model.pt"
    
    # We need to initialize GuidedFractalCoder which inherits from FractalCoder
    # But FractalCoder's init loads the generation model.
    
    system = GuidedFractalCoder(coder_path, critic_path, device)
    
    random_success = 0
    guided_success = 0
    random_steps_total = 0
    guided_steps_total = 0
    
    print(f"\nRunning {num_trials} trials...")
    
    for i in range(num_trials):
        # Generate a problem
        # Target: 20. Length 4.
        target = 20
        length = 4
        
        # Random start
        start_roots = [random.randint(0, 8) for _ in range(length)] # Just ADDs to start?
        
        # 1. Random Search (Baseline)
        # We utilize the base class method repair_program (which uses random idx)
        ok, _, steps = system.repair_program(list(start_roots), target, max_steps=15)
        if ok:
            random_success += 1
            random_steps_total += steps
        else:
            random_steps_total += 15
            
        # 2. Guided Search (Critic)
        ok, _, steps = system.repair_program_guided(list(start_roots), target, max_steps=15)
        if ok:
            guided_success += 1
            guided_steps_total += steps
        else:
            guided_steps_total += 15
            
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Random Search Success: {random_success}/{num_trials} ({random_success/num_trials*100:.1f}%)")
    print(f"Avg Steps (Random):    {random_steps_total/num_trials:.1f}")
    print("-" * 40)
    print(f"Guided Search Success: {guided_success}/{num_trials} ({guided_success/num_trials*100:.1f}%)")
    print(f"Avg Steps (Guided):    {guided_steps_total/num_trials:.1f}")
    print("="*40)

from typing import List, Tuple

if __name__ == "__main__":
    compare_search_strategies()
