
import sys
import torch
import random
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent)) # For critic
sys.path.insert(0, str(Path(__file__).parent.parent / "phase18-fractal-coder")) # For coder

from fractal_critic_full import FractalCriticFull
from fractal_coder import FractalCoder

class FullGuidedCoder(FractalCoder):
    def __init__(self, coder_model_path, critic_model_path, device='cpu'):
        super().__init__(coder_model_path, device)
        
        self.critic = FractalCriticFull(num_roots=31, embed_dim=64, max_len=8).to(device)
        self.critic.load_state_dict(torch.load(critic_model_path, map_location=device, weights_only=True))
        self.critic.eval()
        
    def repair_random(self, roots, target, max_steps=10):
        """Pure Random Search Baseline"""
        return self.repair_program(roots, target, max_steps) # From Phase 18 base class
        
    def repair_heuristic(self, roots, target, max_steps=10):
        """
        Vector 3 Baseline: Critic for Loc + Hard-Coded Heuristic for Mut.
        (Simulated here without loading the old model, using the new model's loc head)
        """
        current_roots = list(roots)
        for step in range(max_steps):
            text_parts = self.generate_flash_flood(current_roots)
            success, result, _ = self.execute_program(text_parts)
            error = target - result
            
            if success and result == target:
                return True, step
                
            # Loc: Use Critic Head 1
            padded_roots = current_roots + [30] * (8 - len(current_roots))
            padded_roots = padded_roots[:8]
            roots_t = torch.tensor([padded_roots], device=self.device)
            error_t = torch.tensor([error], device=self.device)
            
            with torch.no_grad():
                loc_logits, _ = self.critic(roots_t, error_t)
                
            # Top-3 Loc
            valid_len = len(current_roots)
            probs = F.softmax(loc_logits, dim=1)
            probs[0, valid_len:] = 0
            topk = torch.topk(probs[0], min(3, valid_len))
            faulty_idx = random.choices(topk.indices.tolist(), weights=topk.values.tolist(), k=1)[0]
            
            # Mut: Heuristic
            if error > 0:
                if random.random() < 0.7: new_root = random.randint(0, 8)
                else: new_root = random.randint(20, 22)
            else:
                if random.random() < 0.8: new_root = random.randint(10, 18)
                else: new_root = random.randint(0, 4)
                
            current_roots[faulty_idx] = new_root
            
        return False, max_steps

    def repair_full_learned(self, roots, target, max_steps=10):
        """
        Vector 3.5: Full Learned Critic (Loc + Mut).
        No heuristics.
        """
        current_roots = list(roots)
        for step in range(max_steps):
            text_parts = self.generate_flash_flood(current_roots)
            success, result, _ = self.execute_program(text_parts)
            error = target - result
            
            if success and result == target:
                return True, step
                
            # Loc & Mut: Use Critic Head 1 & 2
            padded_roots = current_roots + [30] * (8 - len(current_roots))
            padded_roots = padded_roots[:8]
            roots_t = torch.tensor([padded_roots], device=self.device)
            error_t = torch.tensor([error], device=self.device)
            
            with torch.no_grad():
                loc_logits, mut_logits = self.critic(roots_t, error_t)
                
            # 1. Sample Location (Top-3)
            valid_len = len(current_roots)
            loc_probs = F.softmax(loc_logits, dim=1)
            loc_probs[0, valid_len:] = 0
            topk_loc = torch.topk(loc_probs[0], min(3, valid_len))
            faulty_idx = random.choices(topk_loc.indices.tolist(), weights=topk_loc.values.tolist(), k=1)[0]
            
            # 2. Sample Mutation (Top-3)
            mut_probs = F.softmax(mut_logits, dim=1)
            # Mask out PAD token (30)
            mut_probs[0, 30] = 0
            topk_mut = torch.topk(mut_probs[0], 3)
            new_root = random.choices(topk_mut.indices.tolist(), weights=topk_mut.values.tolist(), k=1)[0]
            
            current_roots[faulty_idx] = new_root
            
        return False, max_steps

def test_full_repair(num_trials=100):
    device = torch.device("cpu")
    coder_path = "research-log/phase18-fractal-coder/fractal_coder_model.pt"
    critic_path = "research-log/phase19.5-full-critic/critic_full.pt"
    
    system = FullGuidedCoder(coder_path, critic_path, device)
    
    results = {
        "Random": 0,
        "Heuristic": 0,
        "FullLearned": 0
    }
    
    print(f"\nRunning {num_trials} trials comparing 3 strategies...")
    
    for i in range(num_trials):
        # Problem
        target = 20
        start_roots = [random.randint(0, 8) for _ in range(4)]
        
        # 1. Random
        ok, _, _ = system.repair_random(list(start_roots), target, max_steps=10)
        if ok: results["Random"] += 1
        
        # 2. Heuristic
        ok, _ = system.repair_heuristic(list(start_roots), target, max_steps=10)
        if ok: results["Heuristic"] += 1
        
        # 3. Full Learned
        ok, _ = system.repair_full_learned(list(start_roots), target, max_steps=10)
        if ok: results["FullLearned"] += 1
        
    print("\n" + "="*40)
    print(f"RESULTS (N={num_trials})")
    print("="*40)
    print(f"Random Search:       {results['Random']}/{num_trials} ({results['Random']/num_trials*100:.1f}%)")
    print(f"Heuristic Critic:    {results['Heuristic']}/{num_trials} ({results['Heuristic']/num_trials*100:.1f}%)")
    print(f"Full Learned Critic: {results['FullLearned']}/{num_trials} ({results['FullLearned']/num_trials*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    test_full_repair()
