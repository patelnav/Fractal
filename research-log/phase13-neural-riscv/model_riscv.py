
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from model_logic import FractalRecurrentALU, BitConfig

class NeuralCPU(nn.Module):
    """
    A Differentiable CPU with a learned Controller.
    Components:
    - Registers: R0, R1, R2, R3 (Fixed size embeddings)
    - ALU: Frozen FractalRecurrentALU (ADD operation)
    - Shifter: Differentiable Left Shift
    - Controller: Transformer that outputs Instruction + Operands
    """
    def __init__(self, config, adder_checkpoint=None):
        super().__init__()
        self.config = config
        
        # 1. The ALU (Frozen)
        self.adder = FractalRecurrentALU(config)
        if adder_checkpoint:
            print(f"Loading Adder from {adder_checkpoint}")
            state_dict = torch.load(adder_checkpoint, map_location='cpu')
            self.adder.load_state_dict(state_dict, strict=False)
        for param in self.adder.parameters():
            param.requires_grad = False
            
        # 2. Registers
        # We don't store registers as parameters. They are state passed in forward().
        # But we need embeddings for them. The Adder handles embeddings.
        # We assume 4 Registers.
        self.num_registers = 4
        
        # 3. Controller
        # Input: Concatenation of all Registers? Or a sequence of [R0, R1, R2, R3]?
        # Let's do Sequence of Registers.
        # Input to Controller: [R0_emb, R1_emb, R2_emb, R3_emb] -> [Batch, 4*L, D]
        # Output: 
        #   Opcode (3): NO-OP, ADD, SHIFT
        #   Src1 (4): R0..R3
        #   Src2 (4): R0..R3
        #   Dest (4): R0..R3
        
        # Controller Config
        self.ctrl_dim = config.dim
        self.ctrl_core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.ctrl_dim, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        
        # Heads
        self.head_opcode = nn.Linear(self.ctrl_dim, 3) # NOOP(0), ADD(1), SHIFT(2)
        self.head_src1 = nn.Linear(self.ctrl_dim, self.num_registers)
        self.head_src2 = nn.Linear(self.ctrl_dim, self.num_registers)
        self.head_dest = nn.Linear(self.ctrl_dim, self.num_registers)
        
        # Learnable "Program Counter" / Step embedding?
        # The controller runs for T steps.
        self.step_emb = nn.Embedding(32, self.ctrl_dim) # Max 32 steps

    def shift_embedding(self, emb):
        # Left shift by 1
        batch_size, seq_len, dim = emb.size()
        zero_vec = self.adder.bit_emb(torch.tensor(0, device=emb.device)).view(1, 1, -1)
        zeros = zero_vec.expand(batch_size, 1, -1)
        shifted = torch.cat([zeros, emb], dim=1)
        return shifted[:, :seq_len, :]

    def forward(self, r0, r1, max_steps=16):
        """
        r0, r1: Initial register values [B, L] (Indices)
        """
        batch_size, seq_len = r0.size()
        
        # Initialize Registers (Embeddings)
        # R0=A, R1=B, R2=0, R3=0
        emb_r0 = self.adder.bit_emb(r0)
        emb_r1 = self.adder.bit_emb(r1)
        
        zeros_idx = torch.zeros_like(r0)
        emb_r2 = self.adder.bit_emb(zeros_idx)
        emb_r3 = self.adder.bit_emb(zeros_idx)
        
        registers = [emb_r0, emb_r1, emb_r2, emb_r3]
        
        # Execution Trace (for debugging/loss)
        trace = []
        
        for t in range(max_steps):
            # 1. Controller State Construction
            # Flatten registers into one sequence? [B, 4*L, D]
            # Or mean pool each register? [B, 4, D]
            # Pooling is safer for variable length.
            # Let's mean pool.
            reg_states = [r.mean(dim=1).unsqueeze(1) for r in registers] # [B, 1, D]
            state_seq = torch.cat(reg_states, dim=1) # [B, 4, D]
            
            # Add step embedding
            step_v = self.step_emb(torch.tensor(t, device=r0.device)).view(1, 1, -1)
            state_seq = state_seq + step_v
            
            # 2. Controller Decision
            # Process
            out = self.ctrl_core(state_seq) # [B, 4, D]
            # Pool for decision (Take first token? or Mean?)
            decision_vec = out.mean(dim=1) # [B, D]
            
            # Logits
            logits_op = self.head_opcode(decision_vec)   # [B, 3]
            logits_s1 = self.head_src1(decision_vec)     # [B, 4]
            logits_s2 = self.head_src2(decision_vec)     # [B, 4]
            logits_d  = self.head_dest(decision_vec)     # [B, 4]
            
            # 3. Action Selection (Gumbel Softmax for Differentiability)
            op_onehot = F.gumbel_softmax(logits_op, tau=1.0, hard=True) # [B, 3]
            s1_onehot = F.gumbel_softmax(logits_s1, tau=1.0, hard=True) # [B, 4]
            s2_onehot = F.gumbel_softmax(logits_s2, tau=1.0, hard=True) # [B, 4]
            dest_onehot = F.gumbel_softmax(logits_d, tau=1.0, hard=True) # [B, 4]
            
            # 4. Execution (Differentiable Routing)
            
            # Fetch Operands
            # We compute weighted sum of all registers based on s1_onehot
            # registers is list of [B, L, D]. Stack -> [B, 4, L, D]
            reg_stack = torch.stack(registers, dim=1)
            
            # Operand 1: [B, 4, 1, 1] * [B, 4, L, D] -> Sum over dim 1 -> [B, L, D]
            val_s1 = (s1_onehot.view(batch_size, 4, 1, 1) * reg_stack).sum(dim=1)
            val_s2 = (s2_onehot.view(batch_size, 4, 1, 1) * reg_stack).sum(dim=1)
            
            # Compute Results for ALL possible ops
            # OP 0: NOOP -> Result is Old Dest (handled later)
            # OP 1: ADD  -> Adder(s1, s2)
            # OP 2: SHIFT -> Shift(s1) (Ignore s2)
            
            # ADD
            # We need clean embeddings for adder? Digital Restoration?
            # If we want gradients to flow to Controller, we use the Soft Embeddings.
            # But Adder might drift.
            # Phase 12 showed we NEED Digital Restoration (Argmax) for stability.
            # But Argmax kills gradients to the Controller.
            # Straight-Through Estimator (Gumbel) on the bits?
            
            # For now, let's assume Soft Embeddings work for short programs (16 steps).
            # If not, we need Gumbel-Softmax on the Bits too.
            res_add_logits, _ = self.adder.forward_embeddings(val_s1, val_s2)
            # Convert logits to embedding (Soft)
            # Softmax -> Matmul(EmbWeight)
            probs_add = F.softmax(res_add_logits, dim=-1)
            res_add_emb = torch.matmul(probs_add, self.adder.bit_emb.weight[:2])
            
            # SHIFT
            res_shift_emb = self.shift_embedding(val_s1)
            
            # NOOP
            # Actually, we just don't update. Or we update with old value.
            # Let's say Result is val_s1 for NOOP (Identity).
            res_noop_emb = val_s1
            
            # Multiplex Result based on Opcode
            # Opcode OneHot: [NOOP, ADD, SHIFT]
            # Result = op[0]*noop + op[1]*add + op[2]*shift
            op_view = op_onehot.view(batch_size, 3, 1, 1)
            result_emb = (op_view[:, 0] * res_noop_emb +
                          op_view[:, 1] * res_add_emb +
                          op_view[:, 2] * res_shift_emb)
            
            # 5. Write Back
            # Update ONLY the destination register
            # New_Reg[i] = (i == dest) ? Result : Old_Reg[i]
            new_registers = []
            for i in range(4):
                is_dest = dest_onehot[:, i].view(batch_size, 1, 1)
                updated_reg = is_dest * result_emb + (1 - is_dest) * registers[i]
                new_registers.append(updated_reg)
                
            registers = new_registers
            
            # Store trace
            trace.append({
                "op": torch.argmax(logits_op, -1),
                "dest": torch.argmax(logits_d, -1)
            })
            
        # Return final value of R2 (Convention: Result in R2)
        # We return logits for R2 to compute loss against integer C
        # We need to project R2 embedding back to logits.
        # We can use the Adder's head? Or a new head?
        # Adder head maps Hidden -> Logits. R2 is Embedding.
        # We need Embedding -> Logits.
        # Cosine similarity with 0/1 embeddings?
        # Simple: Train a readout head.
        # But for Zero-shot composition, we assume the embeddings are valid.
        # Let's assume Adder.head works if we pass it through the core once? No.
        
        # Use Distance to 0/1 embeddings
        # dist0 = (R2 - Emb0)^2
        # dist1 = (R2 - Emb1)^2
        # logits = [ -dist0, -dist1 ]
        
        r2 = registers[2] # [B, L, D]
        w = self.adder.bit_emb.weight[:2] # [2, D]
        
        # r2: [B, L, D], w: [2, D]
        # logits: [B, L, 2]
        # (x-y)^2 = x^2 + y^2 - 2xy
        # We just want dot product (similarity)
        logits = torch.matmul(r2, w.t())
        
        return logits, trace
