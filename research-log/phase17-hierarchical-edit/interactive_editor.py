
import sys
import torch
import time
import readline # For better input handling
from pathlib import Path
from typing import Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))

from fractal_editor import FractalEditor, FractalTrace
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

class InteractiveEditor:
    def __init__(self):
        self.editor = FractalEditor()
        self.trace: Optional[FractalTrace] = None
        self.last_trace: Optional[FractalTrace] = None
        self.last_edit_idx: int = -1
        
    def run(self):
        print("\n" + "=" * 60)
        print("FRACTAL STABLE EDITOR (Vector 7)")
        print("Surgical editing with structural guarantees.")
        print("=" * 60)
        
        self.do_new()
        
        while True:
            try:
                cmd_raw = input("\nCommand (show, edit <N>, diff, new, quit) > ").strip().lower()
            except EOFError:
                break
                
            parts = cmd_raw.split()
            if not parts:
                continue
                
            cmd = parts[0]
            
            if cmd == 'quit' or cmd == 'exit':
                break
            elif cmd == 'new':
                self.do_new()
            elif cmd == 'show':
                self.do_show()
            elif cmd == 'edit':
                if len(parts) < 2:
                    print("Usage: edit <index>")
                    continue
                try:
                    idx = int(parts[1])
                    self.do_edit(idx)
                except ValueError:
                    print("Invalid index.")
            elif cmd == 'diff':
                self.do_diff()
            else:
                print("Unknown command.")

    def do_new(self):
        print("\nGenerating new story...")
        self.trace = self.editor.generate_fresh(num_roots=8)
        self.last_trace = None
        self.do_show()

    def do_show(self):
        if not self.trace:
            print("No story generated.")
            return
            
        print("\n--- Current Story Structure ---")
        for i, (root, text) in enumerate(zip(self.trace.roots, self.trace.text_segments)):
            # Visualize text snippet
            snippet = text.strip()
            if not snippet:
                snippet = "[Empty/Padding]"
            # Truncate for display
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
                
            print(f"[{i}] Root {root:4d} | \"{snippet}\"\n")
        print("-------------------------------")

    def do_edit(self, idx: int):
        if not self.trace:
            return
        if idx < 0 or idx >= len(self.trace.roots):
            print(f"Index {idx} out of bounds.")
            return
            
        # Save history
        self.last_trace = self.trace
        self.last_edit_idx = idx
        
        # Pick a new root ID (simulating a semantic edit request)
        # Ensure it's different
        old_root = self.trace.roots[idx]
        new_root = (old_root + 123) % self.editor.config.num_roots
        
        print(f"\nPatching Segment {idx} (Root {old_root} -> {new_root})...")
        start_t = time.time()
        self.trace = self.editor.patch_root(self.trace, idx, new_root)
        dt = time.time() - start_t
        
        print(f"Done in {dt:.3f}s.")
        self.do_diff()

    def do_diff(self):
        if not self.last_trace or not self.trace:
            print("No edit history to diff.")
            return
            
        print("\n--- Stability Diff ---")
        
        unchanged_chars = 0
        total_chars = 0
        segments_stable = 0
        
        for i in range(len(self.trace.roots)):
            old_seg = self.last_trace.text_segments[i]
            new_seg = self.trace.text_segments[i]
            
            is_target = (i == self.last_edit_idx)
            is_match = (old_seg == new_seg)
            
            if is_match:
                status = "STABLE"
                unchanged_chars += len(new_seg)
                segments_stable += 1
            else:
                status = "CHANGED" if is_target else "DRIFT!!"
                
            total_chars += len(new_seg)
            
            # Only show detail if it changed or is drift
            if is_target:
                print(f"[{i}] {status} | Root {self.last_trace.roots[i]} -> {self.trace.roots[i]}")
                print(f"    Old: \"{old_seg[:50]}...\"")
                print(f"    New: \"{new_seg[:50]}...\"")
            elif not is_match:
                print(f"[{i}] {status} | (Unintended side effect!)")
            # else:
            #     print(f"[{i}] {status}")

        print("----------------------")
        print(f"Segments Stable: {segments_stable}/{len(self.trace.roots)}")
        print(f"Unintended Side Effects: {0 if segments_stable == len(self.trace.roots)-1 else 'DETECTED'}")
        
if __name__ == "__main__":
    app = InteractiveEditor()
    app.run()
