# Plan: Vector 3 "Fractal Critic"

## Goal
Train a **Critic** model to guide the **Fractal Repair Loop** (Vector 6). Instead of randomly mutating roots, the Critic predicts *which* root is likely responsible for the error based on the execution feedback.

## The Hypothesis
The error signal (`Target - Result`) combined with the program structure (`Roots`) contains enough information to localize the bug.
*   Example: `Target=20`, `Result=10`. Error=`+10`.
*   Program: `[ADD 5, ADD 5]`.
*   Critic intuition: "We need more. Change one of these to something bigger."

## Implementation

### 1. Data Generation (`synthetic_critic_data.py`)
*   Generate valid programs (Ground Truth).
*   Create **Buggy Examples** by perturbing one root at index $i$.
*   Execute the buggy program to get `CurrentResult`.
*   Dataset Item: `(BuggyRoots, TargetVal, CurrentResult) -> Label: i` (The index to fix).

### 2. Critic Model (`fractal_critic.py`)
*   Architecture: Small Transformer Encoder or MLP.
*   Input: `Roots` (Token Embeddings) + `Error` (Scalar/Embedding).
*   Output: Logits over sequence length (Pointer Network style). "Which position is bad?"

### 3. Training (`train_critic.py`)
*   Train on 10k synthetic examples.

### 4. Guided Repair (`test_guided_repair.py`)
*   Hook the trained Critic into `FractalCoder.repair_program`.
*   Compare `Random Search` vs `Critic Search`.
*   Metric: Mean steps to repair.

## Files
- `research-log/phase19-fractal-critic/synthetic_critic_data.py`
- `research-log/phase19-fractal-critic/fractal_critic.py`
- `research-log/phase19-fractal-critic/train_critic.py`
- `research-log/phase19-fractal-critic/test_guided_repair.py`
- `research-log/phase19-fractal-critic/RESULTS.md`
