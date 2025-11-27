# Phase 15: Reinforcement Learning (Closing the Loop)

**Objective:** Train the Generator (Qwen-1.5B) to maximize correctness using the Reward Signal from Phase 14.
**Method:** Group Relative Policy Optimization (GRPO).

## The Insight
In Phase 14, we proved that we can distinguish correct code from incorrect code (Critic Acc ~89%).
Instead of just *filtering* (Rejection Sampling), we should *teach* the model to generate correct code in the first place.

## Strategy: GRPO
We will use **Group Relative Policy Optimization (GRPO)** (introduced by DeepSeek-Math).
1.  **Sampling:** For each prompt $q$, generate a group of outputs {o_1, o_2, ..., o_G} from the old policy $\pi_{\theta_{old}}$.
2.  **Reward:** Calculate reward $r_i$ for each output.
    *   $r_i = 1$ if `Execution(o_i) == Pass`
    *   $r_i = 0$ if `Execution(o_i) == Fail`
    *   (Optional) Add Critic Score as shaping reward? GRPO works best with sparse hard rewards if $G$ is large enough.
3.  **Advantage:** Compute advantage by normalizing within the group:
    $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\}) + \epsilon}$
4.  **Update:** Optimize $\pi_\theta$ to maximize $\sum A_i \log \pi_\theta(o_i | q)$ subject to KL constraint.

## Why GRPO?
*   **No Value Model:** PPO requires a separate Value Model (Critic) to estimate baseline. GRPO uses the group mean as the baseline. Saves 50% VRAM.
*   **Direct Optimization:** Optimizes for the metric we care about (Pass Rate).

## Implementation Plan
1.  **Environment:** We need a fast training loop that can:
    *   Generate batch (vLLM or pure PyTorch).
    *   Execute batch (Python Multiprocessing).
    *   Update weights (PyTorch).
2.  **Codebase:** We will write a custom `train_grpo.py` because standard libraries (`trl`) might be heavy/inflexible for our custom Execution Reward.

## Constraints (A100 40GB)
*   **Model:** Qwen-1.5B-Instruct (bf16).
*   **Batch Size:** Group Size $G=4$ or $8$, Micro-batch 1.
*   **Gradient Checkpointing:** Enabled.

## Success Criteria
*   **Pass@1 (Zero-Shot) > Baseline.**
    *   Baseline: 60.94%
    *   Target: > 65% (Matching the Rejection Sampling performance, but in a single sample).
