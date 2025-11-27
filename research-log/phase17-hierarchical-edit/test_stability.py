
import sys
from pathlib import Path
import difflib

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from fractal_editor import FractalEditor
# Import Configs to satisfy pickle
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

def test_stability(num_trials=10):
    editor = FractalEditor()
    
    print(f"\nRunning {num_trials} random edit trials...")
    print("-" * 40)
    
    success_count = 0
    
    for i in range(num_trials):
        seed = 42 + i
        
        # 1. Generate Base Story
        trace_original = editor.generate_fresh(num_roots=10, seed=seed)
        
        # 2. Random Edit
        # Avoid first and last to ensure we have prefix/suffix to check
        import random
        idx = random.randint(1, 8)
        original_rid = trace_original.roots[idx]
        new_rid = (original_rid + random.randint(1, 100)) % 2000
        
        trace_patched = editor.patch_root(trace_original, idx, new_rid)
        
        # 3. Verify Stability
        prefix_match = (
            trace_original.text_segments[:idx] == trace_patched.text_segments[:idx]
        )
        
        suffix_match = (
            trace_original.text_segments[idx+1:] == trace_patched.text_segments[idx+1:]
        )
        
        # Check that target actually changed (unless we randomly picked same ID or model mapped both to same text)
        # But strict byte equality of the entire segment list except index
        
        if prefix_match and suffix_match:
            success_count += 1
            # print(f"Trial {i+1}: Index {idx} Edit -> STABLE")
        else:
            print(f"Trial {i+1}: Index {idx} Edit -> FAILURE")
            print(f"  Prefix: {prefix_match}, Suffix: {suffix_match}")

    print("-" * 40)
    print(f"RESULTS (N={num_trials})")
    print(f"Stability Rate: {success_count}/{num_trials} ({success_count/num_trials*100:.1f}%)")
    
    if success_count == num_trials:
        print("CONCLUSION: Fractal Editing is 100% Stable.")
    else:
        print("CONCLUSION: Instability detected.")

if __name__ == "__main__":
    test_stability(num_trials=10)
