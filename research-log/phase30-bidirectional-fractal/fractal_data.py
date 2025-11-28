import random
import torch
from torch.utils.data import Dataset

OPS = ['+', '*']
# Tokens: (, ), +, *, =, 0-9, space (maybe ignore space?)
# Let's include space for readability but maybe tokenization handles it.
# Vocab: 0,1,2,3,4,5,6,7,8,9, (, ), +, *, =
# Plus a PAD, MASK, BOS, EOS token.

class FractalMathDataset(Dataset):
    def __init__(self, num_samples, max_depth, min_depth=1, max_int=10, max_len=128):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_int = max_int
        self.max_len = max_len
        
        self.vocab = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", 
                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                      "(", ")", "+", "*", "="]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        
        self.data = []
        for _ in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            expr, val = self.generate_expression(depth)
            # Format: <BOS> expr = val <EOS>
            text = f"{expr}={val}"
            self.data.append(text)

    def generate_expression(self, depth):
        if depth == 0 or (depth > 0 and random.random() < 0.2):
            # Base case: simple integer
            val = random.randint(0, self.max_int)
            return str(val), val
        
        # Recursive step
        op = random.choice(OPS)
        
        left_depth = depth - 1
        right_depth = random.randint(0, depth - 1)
        
        # Randomly swap
        if random.random() < 0.5:
            left_depth, right_depth = right_depth, left_depth
            
        left_str, left_val = self.generate_expression(left_depth)
        right_str, right_val = self.generate_expression(right_depth)
        
        # Lisp-style Prefix: ( op left right )
        expr = f"({op} {left_str} {right_str})"
        
        if op == '+':
            val = left_val + right_val
        elif op == '*':
            val = left_val * right_val
        else:
            val = 0
            
        return expr, val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = []
        tokens.append(self.stoi["<BOS>"])
        for char in text:
            if char == ' ':
                continue
            if char in self.stoi:
                tokens.append(self.stoi[char])
        tokens.append(self.stoi["<EOS>"])
        
        # Pad
        if len(tokens) < self.max_len:
            tokens += [self.stoi["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
            
        return torch.tensor(tokens, dtype=torch.long)

if __name__ == "__main__":
    ds = FractalMathDataset(10, 4)
    for i in range(5):
        print(ds.data[i])
        print(ds[i])
