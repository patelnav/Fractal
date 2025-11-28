import torch
import torch.nn.functional as F
from fractal_data import FractalMathDataset
from model import FractalTransformer, Config

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ds = FractalMathDataset(1, 4) 
stoi = ds.stoi
itos = ds.itos

def load_model(path):
    config = Config()
    config.vocab_size = len(ds.vocab)
    config.block_size = 128
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.causal = False
    model = FractalTransformer(config)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(DEVICE)
    model.eval()
    return model

def decode(ids):
    return "".join([itos[i.item()] for i in ids if i.item() not in [stoi["<PAD>"], stoi["<BOS>"], stoi["<EOS>"]]])

def probe(model, text, mask_char='?'):
    # Replace ? with MASK
    tokens = [stoi["<BOS>"]]
    for c in text:
        if c == ' ': continue
        if c == mask_char:
            tokens.append(stoi["<MASK>"])
        else:
            tokens.append(stoi[c])
    tokens.append(stoi["<EOS>"])
    
    # Pad
    tokens += [stoi["<PAD>"]] * (128 - len(tokens))
    x = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    
    pad_id = stoi["<PAD>"]
    
    print(f"Input: {text}")
    
    for step in range(3):
        with torch.no_grad():
            logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        pred[x == pad_id] = pad_id # Lock pad
        
        print(f"Step {step+1}: {decode(pred[0])}")
        x = pred

model = load_model("checkpoints/fractal_bidirectional.pt")

print("--- Probing ---")
# Simple Addition
probe(model, "(+ 3 4) = ??")
# Recursive
probe(model, "(+ 2 (* 3 2)) = ??")
# Filling operands
probe(model, "(+ ? 5) = 8")
