
import sys
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))

from generate_hybrid import load_hybrid_system, render_root, DEVICE
# Import Configs for pickle
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

@dataclass
class FractalTrace:
    """
    Stores the hierarchical structure of a generated text.
    Allows for surgical editing.
    """
    roots: List[int]
    text_segments: List[str]
    stats: List[Dict[str, Any]]

    @property
    def full_text(self) -> str:
        return "".join(self.text_segments)

class FractalEditor:
    def __init__(self):
        print("Initializing Fractal Editor...")
        self.manager, self.fractal_model, self.tokenizer, self.config = load_hybrid_system()
        
    def generate_fresh(self, num_roots: int = 10, seed: int = None) -> FractalTrace:
        """
        Generate a completely new story structure.
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Plan (Manager)
        start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated = self.manager.generate(
                start_idx,
                max_new_tokens=num_roots,
                temperature=0.8
            )
        roots = generated[0].tolist()[1:] # Skip start token
        
        # 2. Render (Fractal Engine)
        segments = []
        stats_list = []
        
        print(f"Rendering {len(roots)} roots...")
        for i, rid in enumerate(roots):
            if rid >= len(self.tokenizer.root_vocab):
                text = ""
                stat = {'error': 'OOV'}
            else:
                text, stat = render_root(
                    self.fractal_model, 
                    self.tokenizer, 
                    self.config, 
                    rid, 
                    DEVICE
                )
            segments.append(text)
            stats_list.append(stat)
            
        return FractalTrace(roots, segments, stats_list)

    def patch_root(self, trace: FractalTrace, index: int, new_root_id: int) -> FractalTrace:
        """
        Surgically replace one root and re-render ONLY that segment.
        Returns a NEW trace (functional style), preserving the old one.
        """
        if index < 0 or index >= len(trace.roots):
            raise ValueError(f"Index {index} out of bounds for trace of length {len(trace.roots)}")
            
        print(f"Patching Root {index}: {trace.roots[index]} -> {new_root_id}")
        
        # Copy lists
        new_roots = list(trace.roots)
        new_segments = list(trace.text_segments)
        new_stats = list(trace.stats)
        
        # Update Root
        new_roots[index] = new_root_id
        
        # Re-render ONLY this segment
        if new_root_id >= len(self.tokenizer.root_vocab):
            new_text = ""
            new_stat = {'error': 'OOV'}
        else:
            new_text, new_stat = render_root(
                self.fractal_model,
                self.tokenizer,
                self.config,
                new_root_id,
                DEVICE
            )
            
        # Update Segment
        new_segments[index] = new_text
        new_stats[index] = new_stat
        
        return FractalTrace(new_roots, new_segments, new_stats)
