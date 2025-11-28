import random

BRACKETS = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
OPENERS = [b[0] for b in BRACKETS]
CLOSERS = [b[1] for b in BRACKETS]
PAIRS = dict(BRACKETS)

def generate_dyck(depth, max_depth, current_depth=0):
    # Generate a sequence of OPEN brackets
    # We want a pure stack task.
    # Input: " ( [ { "
    # Output: " } ] ) "
    
    seq = []
    for _ in range(depth):
        seq.append(random.choice(OPENERS))
        
    # The target is the reverse sequence mapped to closers
    target = [PAIRS[c] for c in reversed(seq)]
    
    input_str = "".join(seq)
    target_str = "".join(target)
    
    return input_str, target_str

if __name__ == "__main__":
    for d in range(1, 5):
        i, t = generate_dyck(d, d)
        print(f"Depth {d}: {i} -> {t}")