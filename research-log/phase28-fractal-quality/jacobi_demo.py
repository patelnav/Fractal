import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def demo_jacobi(model, tokenizer, prompt, max_new_tokens=20, device="cuda"):
    print(f"\n--- Jacobi Decoding Demo ({max_new_tokens} tokens) ---")
    
    # 1. Setup Inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    input_len = input_ids.shape[1]
    
    # 2. Ground Truth (AR)
    print("Generating Ground Truth (AR)...")
    ar_out = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    truth_text = tokenizer.decode(ar_out[0], skip_special_tokens=True)
    print(f"Truth: {truth_text}\n")
    truth_ids = ar_out[0, input_len:]
    
    # 3. Jacobi Initialization
    # We guess the future tokens. 
    # Naive Guess: Repeat the last token of prompt? 
    # Or Pad tokens?
    # Let's use PAD (or EOS) as the initial guess.
    guess_ids = torch.full((1, max_new_tokens), tokenizer.eos_token_id, dtype=torch.long).to(device)
    
    # Full sequence: [Prompt, Guess]
    curr_seq = torch.cat([input_ids, guess_ids], dim=1)
    
    # 4. The Loop
    print("Starting Parallel Loop...")
    for step in range(15): # Max 15 steps
        # Forward pass on WHOLE sequence
        # Note: Causal Mask is strictly applied.
        # position i sees 0..i-1.
        # So prediction at `input_len` sees `prompt`.
        # Prediction at `input_len+1` sees `prompt + guess[0]`.
        
        with torch.no_grad():
            outputs = model(curr_seq)
            logits = outputs.logits # [1, SeqLen, Vocab]
            
        # Greedy decode
        preds = torch.argmax(logits, dim=-1) # [1, SeqLen]
        
        # The prediction for position `i` comes from logits at `i-1`.
        # We want predictions for the *new* tokens.
        # The logits at `input_len-1` predict the first new token.
        # The logits at `input_len+max-2` predict the last new token.
        
        # Extract new guesses
        # The slice of logits corresponding to the *inputs* of the new tokens
        # We want to predict tokens at indices: input_len ... input_len + max - 1
        # These are predicted by logits at: input_len - 1 ... input_len + max - 2
        
        start_idx = input_len - 1
        end_idx = input_len + max_new_tokens - 1
        new_guess_ids = preds[:, start_idx:end_idx]
        
        # Update sequence
        prev_seq = curr_seq.clone()
        curr_seq = torch.cat([input_ids, new_guess_ids], dim=1)
        
        # Visualization
        decoded = tokenizer.decode(new_guess_ids[0], skip_special_tokens=False)
        # Replace newlines for clean print
        vis = decoded.replace("\n", "\\n")
        
        # Check matching
        matches = (new_guess_ids == truth_ids).sum().item()
        acc = matches / max_new_tokens
        
        print(f"Step {step+1}: Acc={acc:.0%} | Text: {vis[:50]}...")
        
        # Convergence Check
        if torch.equal(curr_seq, prev_seq):
            print("Converged!")
            break
            
    return truth_text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    
    prompt = "def fibonacci(n):"
    demo_jacobi(model, tokenizer, prompt, max_new_tokens=20, device=device)

if __name__ == "__main__":
    main()
