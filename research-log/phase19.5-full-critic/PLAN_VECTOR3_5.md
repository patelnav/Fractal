# Plan: Vector 3.5 "Full Learned Critic"

## Goal
Eliminate the hard-coded `if error > 0` heuristic from the Fractal Repair Loop. Train a Critic that predicts both **Where** to fix (Localization) and **What** to write (Mutation).

## Methodology
1.  **Data Upgrade (`synthetic_critic_data_full.py`):**
    -   Generate triplets: `(BuggyProgram, Error) -> (FaultyIndex, CorrectRootID)`.
2.  **Model Upgrade (`fractal_critic_full.py`):**
    -   Add a second head: `MutationHead`.
    -   Input: Transformer features + (Optionally) Positional focus on the Faulty Index.
    -   Output: `Logits(VocabularySize)`.
3.  **Training (`train_critic_full.py`):**
    -   Loss = `CrossEntropy(Index) + CrossEntropy(RootID)`.
4.  **Evaluation (`test_full_repair.py`):**
    -   Compare:
        1.  **Random Search** (Baseline)
        2.  **Heuristic Search** (Previous Vector 3 result)
        3.  **Full Learned Search** (Vector 3.5)

## Files
-   `research-log/phase19.5-full-critic/synthetic_critic_data_full.py`
-   `research-log/phase19.5-full-critic/fractal_critic_full.py`
-   `research-log/phase19.5-full-critic/train_critic_full.py`
-   `research-log/phase19.5-full-critic/test_full_repair.py`
-   `research-log/phase19.5-full-critic/RESULTS.md`
