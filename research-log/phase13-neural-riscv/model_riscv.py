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
    
    FIXED VERSION:
    - Removed Mean Pooling on registers. The Controller now sees the full bit sequence.
    - Input: [B, 4*L, D] (Concatenated Registers)
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
            
        self.num_registers = 4
        
        # 2. Controller
        # Input: [R0, R1, R2, R3] concatenated along sequence dim.
        # If L=16, Input is [B, 64, D].
        self.ctrl_dim = config.dim
        
        # We use a Transformer Encoder.
        # It needs Positional Embeddings to know which bit is which.
        # The Adder has implicit pos due to recurrence, but Controller is parallel?
        # Or is Controller also recurrent?
        # Standard Transformer Encoder is parallel. We need Pos Embeddings.
        self.pos_emb = nn.Parameter(torch.randn(1, 64, self.ctrl_dim)) # Max 4 regs * 16 bits
        
        self.ctrl_core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.ctrl_dim, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        
        # Output Heads
        # We need to pool the controller output to get a single decision vector.
        # Or use a CLS token.
        # Let's use a CLS token approach.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.ctrl_dim))
        
        self.head_opcode = nn.Linear(self.ctrl_dim, 3) # NOOP(0), ADD(1), SHIFT(2)
        self.head_src1 = nn.Linear(self.ctrl_dim, self.num_registers)
        self.head_src2 = nn.Linear(self.ctrl_dim, self.num_registers)
        self.head_dest = nn.Linear(self.ctrl_dim, self.num_registers)
        
        # Step Embedding (To know "Phase" of the algorithm)
        self.step_emb = nn.Embedding(64, self.ctrl_dim) 

    def shift_embedding(self, emb):
        batch_size, seq_len, dim = emb.size()
        zero_vec = self.adder.bit_emb(torch.tensor(0, device=emb.device)).view(1, 1, -1)
        zeros = zero_vec.expand(batch_size, 1, -1)
        shifted = torch.cat([zeros, emb], dim=1)
        return shifted[:, :seq_len, :]

    def forward(self, r0, r1, max_steps=32, supervision_trace=None):
        """
        Forward pass. 
        If supervision_trace is provided, we return the logits for the actions 
        at each step (Teacher Forcing / Imitation).
        """
        batch_size, seq_len = r0.size()
        
        # Init Registers
        emb_r0 = self.adder.bit_emb(r0)
        emb_r1 = self.adder.bit_emb(r1)
        
        zeros_idx = torch.zeros_like(r0)
        emb_r2 = self.adder.bit_emb(zeros_idx)
        emb_r3 = self.adder.bit_emb(zeros_idx)
        
        registers = [emb_r0, emb_r1, emb_r2, emb_r3]
        
        # Action Logits Storage
        all_logits = [] # List of dicts {op, s1, s2, dest}
        
        for t in range(max_steps):
            # 1. Construct State
            # Concat all registers [B, 4*L, D]
            state_seq = torch.cat(registers, dim=1)
            
            # Add CLS token [B, 1+4L, D]
            cls_t = self.cls_token.expand(batch_size, -1, -1)
            
            # Add Step Embedding to CLS
            step_v = self.step_emb(torch.tensor(t, device=r0.device)).view(1, 1, -1)
            cls_t = cls_t + step_v
            
            full_input = torch.cat([cls_t, state_seq], dim=1)
            
            # Add Positional Embeddings (to register part)
            # We reuse pos_emb. simple additive.
            # Note: L might vary. We slice pos_emb.
            curr_len = state_seq.size(1)
            if curr_len <= self.pos_emb.size(1):
                full_input[:, 1:, :] += self.pos_emb[:, :curr_len, :]
            
            # 2. Controller Forward
            out = self.ctrl_core(full_input)
            
            # 3. Extract Decision (CLS token output)
            decision_vec = out[:, 0, :] # [B, D]
            
            # 4. Heads
            l_op = self.head_opcode(decision_vec)
            l_s1 = self.head_src1(decision_vec)
            l_s2 = self.head_src2(decision_vec)
            l_d = self.head_dest(decision_vec)
            
            all_logits.append({
                "op": l_op, "s1": l_s1, "s2": l_s2, "dest": l_d
            })
            
            # 5. Execute (Differentiable or Hard?)
            # For Imitation Learning, we can use Teacher Forcing (Execute the TRUE action)
            # or Execute the Predicted action.
            # Since we want to verify "Learning", let's execute the Predicted (Gumbel).
            
            op_onehot = F.gumbel_softmax(l_op, tau=1.0, hard=True) 
            s1_onehot = F.gumbel_softmax(l_s1, tau=1.0, hard=True)
            s2_onehot = F.gumbel_softmax(l_s2, tau=1.0, hard=True) 
            dest_onehot = F.gumbel_softmax(l_d, tau=1.0, hard=True) 
            
            # Fetch Operands
            reg_stack = torch.stack(registers, dim=1) # [B, 4, L, D]
            val_s1 = (s1_onehot.view(batch_size, 4, 1, 1) * reg_stack).sum(dim=1)
            val_s2 = (s2_onehot.view(batch_size, 4, 1, 1) * reg_stack).sum(dim=1)
            
            # Execute Ops
            # ADD
            # We use Hard Snapping (Argmax) inside the execution to keep state clean
            # Just like Phase 12.
            res_add_logits, _ = self.adder.forward_embeddings(val_s1, val_s2)
            res_add_idx = torch.argmax(res_add_logits, dim=-1)
            res_add_emb = self.adder.bit_emb(res_add_idx)
            
            # SHIFT
            res_shift_emb = self.shift_embedding(val_s1)
            
            # NOOP (Identity)
            res_noop_emb = val_s1 # Or registers[dest]? 
            # Ideally NOOP means "Do not write".
            # But our Write-Back logic is "New = Mask*Result + (1-Mask)*Old".
            # So if Op is NOOP, we want Result to be Old?
            # Actually, the Write-Back logic below handles Dest selection.
            # If Op is NOOP, we should probably force "Write Mask" to 0?
            # Or just say Result = Old_Dest.
            # Let's say Result = val_s1. And we hope Dest is irrelevant.
            
            # Multiplex Result
            op_view = op_onehot.view(batch_size, 3, 1, 1)
            result_emb = (op_view[:, 0] * res_noop_emb +
                          op_view[:, 1] * res_add_emb +
                          op_view[:, 2] * res_shift_emb)
            
            # Write Back
            # If Op is NOOP (0), we should NOT write?
            # Let's imply that Op 0 means "Don't Change Anything".
            # The update equation:
            # New_Reg = Is_Dest * (Is_Not_NOOP * Result + Is_NOOP * Old) + Is_Not_Dest * Old
            
            is_noop = op_view[:, 0] # [B, 1, 1]
            
            new_registers = []
            for i in range(4):
                is_dest = dest_onehot[:, i].view(batch_size, 1, 1)
                
                # If this is dest:
                #   If NOOP: keep old
                #   Else: take result
                
                update_val = is_noop * registers[i] + (1 - is_noop) * result_emb
                
                updated_reg = is_dest * update_val + (1 - is_dest) * registers[i]
                new_registers.append(updated_reg)
                
            registers = new_registers
            
        return all_logits, registers