# Phase 11: Fractal Logic Gates

**"The Gradient Problem is Solved."**

This phase explores using Chen's **Perturbed Reverse Heat Process** to build a differentiable Neural Arithmetic Logic Unit (NALU).

## The Problem
Neural networks are bad at math.
*   **Approximation:** They try to fit a curve to discrete logic.
*   **No Extrapolation:** A model trained on numbers 1-100 fails on 101.
*   **Existing Solutions:** NALUs (Neural Logic Units) exist but are notoriously hard to train because discrete logic (0/1) has zero gradient.

## The Solution
Our Discrete Diffusion Engine **is** the solution.
*   It operates on the Boolean Hypercube (perfect for binary logic).
*   It uses a Heat Semigroup to smooth the space, providing **valid gradients** for discrete transitions.
*   We can train it to *be* a logic gate.

## Experiments
1.  **Binary Addition:** Can we learn to add 8-bit numbers?
2.  **Extrapolation:** Can a model trained on 8-bit numbers add 12-bit numbers? (This proves it learned the *algorithm*, not the data).

## Run Instructions
*   `python generate_binary_data.py`
*   `python train_logic.py`
