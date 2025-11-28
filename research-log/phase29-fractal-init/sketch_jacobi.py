import torch
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def run_jacobi(model, tokenizer, prompt, ground_truth_ids, init_guess_ids, name="Method"):
    device = ground_truth_ids.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_len = input_ids.shape[1]
    target_len = ground_truth_ids.shape[0]
    
    print(f"\n--- Running {name} ---")
    
    # Setup Sequence: [Prompt, Guess]
    curr_seq = torch.cat([input_ids, init_guess_ids], dim=1)
    
    steps = 0
    for step in range(30): # Max steps
        steps = step + 1
        
        with torch.no_grad():
            outputs = model(curr_seq)
            logits = outputs.logits
            
        # Argmax prediction
        preds = torch.argmax(logits, dim=-1)
        
        # Extract new guesses for the target region
        # Logits at [input_len-1 ... input_len+target_len-2] predict target[0...target_len-1]
        start_idx = input_len - 1
        end_idx = input_len + target_len - 1
        new_guess = preds[:, start_idx:end_idx]
        
        # Calculate Accuracy vs Ground Truth
        matches = (new_guess == ground_truth_ids).sum().item()
        acc = matches / target_len
        
        print(f"Step {step+1}: Acc={acc:.0%}")
        
        # Update Sequence
        prev_seq = curr_seq.clone()
        curr_seq = torch.cat([input_ids, new_guess], dim=1)
        
        # Check Convergence (Stability)
        # We check if the output *stabilized*, or if it matches Ground Truth.
        # Ideally we want to match Ground Truth for this benchmark.
        if matches == target_len:
            print(f"Converged to Truth in {step+1} steps.")
            return step + 1
            
        if torch.equal(curr_seq, prev_seq):
             print(f"Converged (Stable but not 100% Truth) in {step+1} steps.")
             return step + 1
             
    print("Did not converge.")
    return 30

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    
    prompt = "def fibonacci(n):"
    # Generate Ground Truth (50 tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gt_len = 50
    
    print("Generating Ground Truth...")
    ar_out = model.generate(inputs.input_ids, max_new_tokens=gt_len, do_sample=False)
    ground_truth = ar_out[0, inputs.input_ids.shape[1]:]
    
    print(f"Truth: {tokenizer.decode(ground_truth)}")
    
    # 1. Naive Init (PAD/EOS)
    naive_guess = torch.full((1, gt_len), tokenizer.eos_token_id, dtype=torch.long).to(device)
    steps_naive = run_jacobi(model, tokenizer, prompt, ground_truth, naive_guess, "Naive Init (EOS)")
    
    # 2. Sketch Init (50% Masked)
    # "Islands of Correctness"
    # We keep every even token, mask every odd token with EOS
    sketch_guess = ground_truth.clone().unsqueeze(0)
    # Mask odd indices: 1, 3, 5...
    sketch_guess[:, 1::2] = tokenizer.eos_token_id
    
    steps_sketch = run_jacobi(model, tokenizer, prompt, ground_truth, sketch_guess, "Fractal Init (50% Masked)")
    
    # 3. Poor Sketch (10% Known)
    # Keep every 10th token
    poor_guess = torch.full((1, gt_len), tokenizer.eos_token_id, dtype=torch.long).to(device)
    poor_guess[:, ::10] = ground_truth[::10]
    
    steps_poor = run_jacobi(model, tokenizer, prompt, ground_truth, poor_guess, "Sparse Init (10% Known)")

    print("\n--- Final Results ---")
    print(f"Naive Steps: {steps_naive}")
    print(f"Sparse Steps: {steps_poor}")
    print(f"Fractal Steps: {steps_sketch}")
    print(f"Speedup (Fractal vs Naive): {steps_naive / steps_sketch:.2f}x")

if __name__ == "__main__":
    main()
