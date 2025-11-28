import os
import torch
import random
from tqdm import tqdm
from fractal_data import generate_dyck
from fractal_models import GPT, ModelConfig
import matplotlib.pyplot as plt

# Config
VOCAB = sorted(list(set("()[]{}<>"))) + ['->', '<pad>'] # No spaces, Arrow separator
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
PAD_TOKEN = stoi['<pad>']
SEP_TOKEN = stoi['->']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def encode(s):
    """Encode string, handling '->' as a special token."""
    tokens = []
    i = 0
    while i < len(s):
        # Check for '->' as a 2-char token first
        if i + 1 < len(s) and s[i:i+2] == '->':
            tokens.append(stoi['->'])
            i += 2
        elif s[i] in stoi:
            tokens.append(stoi[s[i]])
            i += 1
        else:
            i += 1  # Skip unknown chars
    return tokens

def decode(l):
    return ''.join([itos[i] for i in l if i in itos])

def generate_dataset(num_samples=10000, max_depth=6):
    data = []
    print(f"Generating {num_samples} samples (Depth 1-{max_depth})...")
    for _ in range(num_samples):
        d = random.randint(1, max_depth)
        inp, out = generate_dyck(d, d)
        # Format: "([{->}])" (No spaces in fractal_data now)
        text = f"{inp}->{out}"
        data.append(text)
    return data

def train_model(model, train_data, steps=5000, batch_size=64, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    pbar = tqdm(range(steps))

    for i in pbar:
        batch_idx = torch.randint(0, len(train_data), (batch_size,))
        batch_texts = [train_data[i] for i in batch_idx]

        encoded_batch = [encode(t) for t in batch_texts]
        max_len = max(len(e) for e in encoded_batch)

        x = torch.full((batch_size, max_len), PAD_TOKEN, dtype=torch.long)
        y = torch.full((batch_size, max_len), -1, dtype=torch.long)  # -1 = ignore in loss

        for j, enc in enumerate(encoded_batch):
            l = len(enc)
            x[j, :l-1] = torch.tensor(enc[:-1])

            # Find where '->' token is and only compute loss on OUTPUT tokens
            try:
                sep_idx = enc.index(SEP_TOKEN)  # Find '->' in token sequence
                # Output tokens start after the separator token
                output_start = sep_idx + 1
                # Only set targets for output tokens (after ->)
                for k in range(output_start - 1, l - 1):  # -1 for x/y offset
                    if k >= 0:
                        y[j, k] = enc[k + 1]
            except ValueError:
                # No separator found, supervise all tokens
                y[j, :l-1] = torch.tensor(enc[1:])

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")

    return losses

def evaluate_depths(model, depths=range(1, 13), samples_per_depth=50, dynamic_fractal=False):
    model.eval()
    model = model.to(DEVICE)
    accuracies = []
    
    print(f"Evaluating Generalization (Dynamic={dynamic_fractal})...")
    for d in depths:
        correct = 0
        for _ in range(samples_per_depth):
            inp, out_truth = generate_dyck(d, d)
            prompt = f"{inp}->"
            x = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)
            
            loop_cnt = None
            if dynamic_fractal:
                loop_cnt = max(d, 6) 
                
            # Generate
            max_new = len(out_truth) + 2
            
            output_ids = model.generate(x, max_new_tokens=max_new, loop_count=loop_cnt)
            
            gen_text = decode(output_ids[0].tolist())
            
            gen_res = ""
            if '->' in gen_text:
                gen_res = gen_text.split('->')[1].strip()
                # Truncate to length of truth
                gen_res = gen_res[:len(out_truth)]
                
                if gen_res == out_truth:
                    correct += 1
            
            if correct == 0 and _ < 1: # Print first failure/sample per depth
                 print(f"DEBUG D{d}: '{prompt}' -> EXP: '{out_truth}' | GOT: '{gen_res}' (Raw: '{gen_text}')")
            
        acc = correct / samples_per_depth
        accuracies.append(acc)
        print(f"Depth {d}: {acc*100:.1f}%")
        
    return accuracies

def main():
    # 1. Data (Depths 1-6)
    train_data = generate_dataset(num_samples=10000, max_depth=6)
    
    config = ModelConfig(
        vocab_size=len(VOCAB),
        n_layer=6,
        n_head=4,
        n_embd=256, # Increased embedding size for capacity
        block_size=512
    )
    
    # 2. Train Baseline
    print("\n--- Training Baseline (Standard GPT) ---")
    baseline = GPT(config)
    train_model(baseline, train_data, steps=5000, batch_size=64)
    
    # 3. Train Fractal
    print("\n--- Training Fractal (Shared Weights) ---")
    config.is_fractal = True
    fractal = GPT(config)
    train_model(fractal, train_data, steps=5000, batch_size=64)
    
    # 4. Evaluate
    depths = range(1, 15)
    
    print("\n--- Evaluating Baseline ---")
    acc_b = evaluate_depths(baseline, depths)
    
    print("\n--- Evaluating Fractal (Fixed Loops=6) ---")
    acc_f_fixed = evaluate_depths(fractal, depths, dynamic_fractal=False)
    
    print("\n--- Evaluating Fractal (Dynamic Loops=Depth) ---")
    acc_f_dynamic = evaluate_depths(fractal, depths, dynamic_fractal=True)
    
    # Save results (use absolute path or script dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "results_dyck.txt"), "w") as f:
        f.write(f"Depths: {list(depths)}\n")
        f.write(f"Baseline: {acc_b}\n")
        f.write(f"Fractal_Fixed: {acc_f_fixed}\n")
        f.write(f"Fractal_Dynamic: {acc_f_dynamic}\n")

if __name__ == "__main__":
    main()
