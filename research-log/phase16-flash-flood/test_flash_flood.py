
import sys
import time
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))

from generate_hybrid import load_hybrid_system, DEVICE
# Import Configs to satisfy pickle
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

from flash_flood import FlashFloodDecoder

def test_flash_flood(num_roots=50, k=16):
    """
    Benchmark the Flash Flood parallel decoder.
    """
    # Load system
    manager, fractal_model, tokenizer, config = load_hybrid_system() 
    
    print(f"\nStarting Flash Flood Benchmark...")
    print(f"Target: Generate {num_roots} roots and render them (Parallel).")
    print(f"Best-of-K: {k}")
    
    # 1. Generate Roots (Manager)
    # We generate a bit more to ensure we have valid ones, 
    # or we just take whatever comes out.
    start_time = time.time()
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated_roots = manager.generate(
            start_idx,
            max_new_tokens=num_roots,
            temperature=0.8,
            top_k=50
        )
    root_seq = generated_roots[0, 1:] # (NumRoots,)
    manager_time = time.time() - start_time
    print(f"Manager Generation Time: {manager_time:.4f}s")
    
    # Filter OOV roots for fair comparison (though the decoder handles them, 
    # the tokenizer won't like them).
    # Actually, let's just clamp them or mask them.
    # The model expects valid root IDs.
    valid_mask = root_seq < config.num_roots
    root_seq = root_seq[valid_mask]
    print(f"Valid roots: {len(root_seq)}")
    
    if len(root_seq) == 0:
        print("No valid roots generated.")
        return

    # 2. Render Roots (Flash Flood)
    decoder = FlashFloodDecoder(fractal_model, config, DEVICE)
    
    render_start_time = time.time()
    
    # Run Parallel Rendering
    # level0_out (chunks): (B, ExpansionSize)
    # level1_out (chars): (B, ExpansionSize, MaxCharLen)
    
    # We need the chunk IDs to check for padding
    chunks_out, _ = decoder.expand_level_parallel(root_seq, level=0, k=k)
    
    # Flatten chunks for Level 1
    chunk_ids_flat = chunks_out.reshape(-1)
    
    # Expand Level 1
    chars_flat, _ = decoder.expand_level_parallel(chunk_ids_flat, level=1, k=k)
    chars_out = chars_flat.view(len(root_seq), config.expansion_size, config.max_char_len)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    render_time = time.time() - render_start_time
    
    # 3. Decode and Count
    total_chars = 0
    decoded_texts = []
    
    # chunks_out: (B, Exp)
    # chars_out: (B, Exp, CharLen)
    B, E = chunks_out.shape
    
    for b in range(B):
        root_text = ""
        for e in range(E):
            chunk_id = chunks_out[b, e].item()
            
            # Skip pad chunks
            if chunk_id == config.pad_chunk_id or chunk_id >= config.num_chunks:
                continue
                
            chunk_chars = chars_out[b, e]
            # Filter pad chars
            valid_chars = chunk_chars[chunk_chars < config.num_chars]
            
            s = "".join([tokenizer.id_to_char.get(c.item(), '?') for c in valid_chars])
            root_text += s
        
        decoded_texts.append(root_text)
        total_chars += len(root_text)
        
    total_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Manager Time: {manager_time:.4f}s")
    print(f"  Render Time: {render_time:.4f}s")
    print(f"  Total Chars: {total_chars}")
    print(f"  Chars/Sec (End-to-End): {total_chars / total_time:.2f}")
    print(f"  Chars/Sec (Render Only): {total_chars / render_time:.2f}")
    
    tokens_approx = total_chars / 4
    print(f"  Tokens/Sec (Approx @ 4chars/tok): {tokens_approx / total_time:.2f}")
    print(f"  Render Tokens/Sec: {tokens_approx / render_time:.2f}")
    
    print(f"\nSample Output (First 3 roots):")
    for i in range(min(3, len(decoded_texts))):
        print(f"  Root {i}: {repr(decoded_texts[i])}")

if __name__ == "__main__":
    test_flash_flood(num_roots=200, k=16)
