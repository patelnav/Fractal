# Fractal Research Arc Summary

This document sits between `research-log/LOG.md` (chronological lab notes) and `Vectors/README.md` (forward-looking directions).

Goals:
- Summarize what has been **demonstrated**, **falsified**, and **still open** across phases.
- Connect those results to the Vector roadmap.
- Identify a small set of **high‑leverage directions** for future work.

### What “Fractal” Means Now

Early work treated “fractal” mainly as **weight sharing across depth or positions** (looping the same layer/bit‑module). The later phases clarified a more useful view:

- **Recursion in time:** iterative refinement steps that move from fuzzy to sharp, noisy to clean.
- **Self‑similar updates:** the same denoiser / refinement rule applied repeatedly over a structure.
- **Multi‑scale refinement:** optionally applied across levels (roots → chunks → tokens), not just along a single sequence.

In this sense, Phase 30’s bidirectional diffusion decoder is “fractal” in the refinement process, even when the network depth is modest.

---

## 1. High‑Level Innovations So Far

These are the main technical ideas that have actually been instantiated and tested:

- **Fractal Diffusion Engine (Phases 1–6)**  
  Shared‑weight discrete diffusion across abstraction levels (roots → chunks → characters) with a contrastive energy head that can verify expansions.

- **Contrastive Energy Head (Phases 2.5–4)**  
  Amortized energy function trained on (correct, wrong) pairs that separates valid from invalid expansions and supports rejection‑based decoding.

- **Neural ALU / CPU (Phases 11–12)**  
  Recurrent / weight‑shared transformer that learns full‑adder logic and extrapolates addition to longer bit‑widths; composed into a multiplier with digital restoration.

- **Execution‑Based Code Critic (Phases 14–15)**  
  Large‑model critic trained on pass/fail outcomes from unit tests, improving MBPP Pass@1 by ≈+6–7% over baseline via Best‑of‑k ranking.

- **Flash Flood Decoder (Phase 16)**  
  Root–renderer separation plus batched expansion; demonstrated 10–60× speedups on synthetic and Shakespeare‑like tasks when rendering is parallelized.

- **Fractal Editor (Phase 17)**  
  Hierarchical editing where patching a “root” leaves surrounding text byte‑for‑byte identical, in contrast to standard AR drift.

- **Fractal Coder + Critic (Phases 18–19.5)**  
  Synthetic “math code” loop that generates, executes, localizes, and repairs code using a fully learned dual‑head critic (location + mutation), substantially beating random and heuristic search.

- **Causal Limits + Flash Flood (Phases 21–29)**  
  Systematic evidence that sketching, plan‑level repair, and causal “fractal sampling” do not beat strong baselines on HumanEval; causal wavefronts destroy “islands of correctness”.

- **Bidirectional Flash Flood Prototype (Phase 30)**  
  Small bidirectional diffusion‑style model that preserves anchors and fills gaps in a recursive arithmetic language in effectively one parallel refinement step.

---

## 1.1 Key Quantitative Highlights

These are approximate headline metrics; see phase‑specific `RESULTS.md` files for details.

| Phase | Result / Mechanism           | Metric / Observation                          | Status |
|-------|------------------------------|-----------------------------------------------|--------|
| 3–4   | Contrastive energy head      | ≈100% detection on mismatched pairs           | ✅     |
| 11.1  | Recurrent ALU (adder)       | 100% train (8‑bit), ≈99.5% test (12‑bit)      | ✅     |
| 12.3  | Neural CPU (multiplier)     | 100% accuracy on held‑out multiplication      | ✅     |
| 14–15 | Execution‑based code critic | +5.9–6.6% absolute Pass@1 on MBPP/HumanEval   | ✅     |
| 16,26 | Flash Flood speed           | 10–94× renderer throughput vs sequential      | ✅     |
| 17    | Fractal editor              | 100% prefix/suffix stability (synthetic stories) | ✅  |
| 19.5  | Full learned critic (synthetic) | ≈3× improvement over random repair       | ✅     |
| 21–22 | Sketch‑guided + repair      | No Pass@1 gain on “hard 5” HumanEval problems | ❌     |
| 28–29 | Causal refinement limit     | Fractal init 22% → 6% collapse at step 2      | ❌     |
| 30    | Bidirectional Flash Flood   | ≈80.6% structural accuracy; anchors preserved | ✅     |

Phase 30’s details currently live in `research-log/phase30-bidirectional-fractal/RESULTS.md` and should be mirrored into `research-log/LOG.md` in a future logging pass.

---

## 2. Findings by Theme

### 2.1 Fractal Engine & Energy (Phases 1–6)

- **What worked**
  - Discrete diffusion over hierarchical vocabularies can learn deterministic mappings perfectly on synthetic and BPE decompression tasks.
  - A contrastive energy head trained on (correct, wrong) pairs yields near‑perfect discrimination for decompression and hierarchical expansions.
  - A unified fractal engine can reuse weights across levels, with a single energy head that verifies expansions at multiple abstraction levels.
  - A tiny autoregressive “Manager” plus fractal renderer produces novel Shakespeare‑like generations with energy‑based rejection.

- **What did not pan out (so far)**
  - Energy‑based hallucination detection on open‑ended next‑char language modeling is fragile; performance collapsed when UNKs or noisy roots were present.

- **Implication**
  - The fractal diffusion + energy head is a solid **verification and rendering primitive** for structured tasks where the correct expansion is well‑defined.
  - To fully test the “scale‑invariant processing” hypothesis, a natural next experiment is to probe **compression as well as expansion** (chars → roots as well as roots → chars) and quantify how far a single shared‑weight engine can go in both directions.

### 2.2 Neural ALU & Algorithmic Generalization (Phases 11–13)

- **What worked**
  - A recurrent / fractal architecture (weight sharing across bit positions) learns a full adder that extrapolates to longer bit‑widths with near‑perfect accuracy.
  - Composing this adder into a hard‑coded shift‑and‑add multiplier, with digital restoration, yields perfect generalization on multiplication without retraining.

- **What did not work**
  - Training a controller to *discover* the shift‑and‑add algorithm end‑to‑end over the adder failed due to credit assignment and search‑space issues.

- **Implication**
  - Fractal / recurrent architectures are well‑suited to **algorithmic modules** (ALU‑like components). Making a full “Neural RISC‑V” requires better curriculum and RL, not just more capacity.
  - Conceptually, this is a “fractal in time / bits” result: the same computation is reused across positions. Bidirectional Flash Flood (Phase 30) can be seen as a complementary “fractal in time / tokens” mechanism at the sequence level, while target domains (code, text) provide the hierarchical structure.
  - Digital restoration (snapping intermediate activations back to discrete codes) was critical for robust composition in the Neural CPU; it is a plausible ingredient to explore more broadly (e.g., discrete codes or thresholds in diffusion and verification modules).

### 2.3 Execution‑Based Code Generation (Phases 14–19.5)

- **What worked**
  - On MBPP, an execution‑trained critic improved Pass@1 by ≈+6–7% over a strong base model by ranking samples using test outcomes.
  - In the synthetic “math code” domain, the Fractal Coder loop plus a fully learned dual‑head critic (faulty index + correct root) more than doubled repair success rate vs random search and beat hand‑coded heuristics.
  - The small‑scale prototype (Vector 1 + 3 + 6 + 7 on synthetic data) is **closed and honest**: generation, execution, localization, and mutation are all learned, no hard‑coded shortcuts.

- **What did not work**
  - Text‑only energy heads and explanation‑based critics for real‑world code (Phase 24) failed to beat simple baseline scoring; asking the model to “explain itself” did not improve its judgment.
  - Plan‑level repair loops on HumanEval (Phase 22) failed to rescue the hardest problems; updating the sketch without new knowledge just rephrased the same misconception.

- **Implication**
  - **Execution + critic** is a proven recipe for code selection and synthetic repair. Scaling it to real‑world code remains promising, but explanation‑based or sketch‑level critics alone are not enough.
  - So far, critics primarily act at inference time (selection and repair). A more aggressive but interesting direction is to **train generators directly against critics** (e.g., EBM/contrastive or RL‑style objectives) to improve single‑sample quality, not just Best‑of‑k.

### 2.4 Flash Flood & Speed (Phases 16, 21, 25–29)

- **What worked**
  - On synthetic and Shakespeare‑style tasks, the Flash Flood decoder achieved 10–60× speedups in rendering throughput vs sequential baselines.
  - With careful batching and sketching, end‑to‑end speedups of 5–8× were achievable while preserving quality.

- **What failed**
  - Sketch‑then‑fill v2/v2.5 pipelines for code (HumanEval) often underperformed single‑shot baselines due to assembly and nesting issues.
  - Length and depth generalization experiments (Phase 25) showed that shared weights alone did not guarantee out‑of‑distribution success; standard transformers did not magically become “fractal generalizers”.
  - Initialization experiments (Phase 29) revealed that causal models rapidly **destroy islands of correctness** during iterative refinement: a wrong prefix propagates errors and overwrites correct tokens downstream.

- **Implication**
  - Parallelism is genuinely fast and architecturally elegant. But naïve Flash Flood on top of a causal LLM bumps into fundamental limitations of the causal mask for O(1) decoding and iterative refinement.
  - Looping the same weights deeper in a causal stack (more “fractal depth”) increases **computation**, but does not automatically increase the **effective working memory or stack** needed for hard reasoning tasks. Scaling laws and architectural choices both matter.
  - The “islands of correctness” phenomenon has so far been diagnosed at decoding time; it is natural to consider **training objectives and curricula that explicitly reward island preservation**, e.g., by constructing denoising tasks with fixed anchors and weighting losses to keep them intact.

### 2.5 Hierarchical Editing (Phase 17)

- **What worked**
  - The Fractal Editor can surgically edit a single root in a generated story while provably leaving the prefix and suffix segments byte‑for‑byte identical.
  - Baseline autoregressive models showed “butterfly effect” behavior: even small local edits led to near‑total drift in the continuation.

- **Implication**
  - The fractal representation is a strong fit for **stable local editing** of tree‑structured content (stories, code, documents). This is a differentiated capability relative to standard LLM editors.

### 2.6 Causal Limits & Bidirectional Flash Flood (Phases 21–30)

- **What we learned about causal models**
  - v3 sketch‑guided generation can match strong baselines but does not significantly exceed them on hard logic/code tasks.
  - Plan‑level repair and reverse‑sketch critics do not reliably fix deep misunderstandings; the model’s own explanations typically mirror its misconceptions.
  - Causal wavefront experiments (Phases 28–29) showed that even favorable initializations collapse: wrong tokens in the prefix corrupt the context and overwrite correct anchors.

- **What changed with bidirectional diffusion**
  - A small bidirectional transformer trained in an MLM / diffusion style on a recursive arithmetic language can:
    - Preserve existing structural anchors (`+`, `*`, parentheses, some operands).
    - Fill in masked positions in effectively one global parallel step.
    - Maintain “islands of correctness” instead of destroying them, achieving ≈80% structural token accuracy after one refinement step on the benchmarked task (while still missing some numeric details on harder expressions).

- **Implication**
  - For **true Flash Flood decoding and refinement**, a **bidirectional diffusion‑style decoder** is the correct architectural target. Causal LMs can still act as planners or proposal generators, but they are structurally unsuited to the O(1) flood phase identified in the experiments.
  - This lines up with the broader diffusion / masked‑LM literature (e.g., BERT‑style bidirectional attention, UL2‑style denoising LMs, and discrete text diffusion approaches such as MaskGIT and D3PM), but here the focus is on structured, hierarchical sequences with explicit anchoring and energy‑based verification.
  - Taken together, the results suggest a division of labor: **fractal hierarchies and energy heads** are strongest for verification, editing, and structured rendering, while **bidirectional attention and diffusion‑style refinement** are required for robust parallel generation. These ingredients are complementary rather than identical. 

---

## 3. Vector Status Snapshot

This is a coarse mapping from the Vector roadmap to actual evidence.

> **Labels:**  
> **Validated** – demonstrated on at least one non‑toy benchmark with clear metrics.  
> **Prototype** – convincingly shown on synthetic / small‑scale tasks.  
> **Blocked / Negative** – multiple attempts with no clear path forward.  
> **Open** – conceptually promising but not yet seriously attempted.

- **Vector 1 – Flash Flood Decoder (Speed)**  
  - Status: **Prototype → Architecturally clarified**.  
  - Evidence: Phase 16 (10–60× renderer speedups), Phases 26–27 (5–8× end‑to‑end), Phase 29 (causal limit), Phase 30 (bidirectional prototype).  
  - Takeaway: Parallel rendering is real and valuable, but usable Flash Flood requires a bidirectional diffusion decoder.

- **Vector 2 – Holographic Learner (Data Efficiency)**  
  - Status: **Open / partially explored**.  
  - Evidence: Phase 4–6 (shared weights across levels), Phase 11–12 (recurrent ALU).  
  - Takeaway: “Holographic” here means **weight sharing across levels or positions** to reuse structure; this clearly helps for algorithmic modules and hierarchical decompression, but there is not yet a large‑scale “equal performance with fewer tokens/params” result on real data.

- **Vector 3 – Ouroboros / Critic**  
  - Status: **Mixed**.  
  - Evidence: Phase 7–10 (text reasoning critic failed to beat baseline), Phases 14–15 (execution‑trained code critic strong), Phases 18–19.5 (synthetic dual‑head critic very strong), Phase 24 (reverse‑sketch critic negative).  
  - Takeaway: Text‑only critics are fragile; **execution‑based critics** for code are promising. The “energy‑only text critic” variant is effectively falsified.

- **Vector 6 – Program Synthesis & Execution Loop**  
  - Status: **Validated (small‑scale) + promising (real‑world)**.  
  - Evidence: Phase 14–15 (MBPP/HumanEval critic results), Phases 18–19.5 (fully learned synthetic repair loop).  
  - Takeaway: The generate–execute–critic–repair loop works; the main open questions are scaling, cost, and integration with better decoders.

- **Vector 7 – Hierarchical Editing**  
  - Status: **Prototype**.  
  - Evidence: Phase 17 (fractal editor stability vs AR drift).  
  - Takeaway: There is a strong story and demo in synthetic/story domains; mapping this capability to real code/doc editing remains open.

- **Vector 1.x / 16 – Diversity & Flash Flood Sampling**  
  - Status: **Partially explored → de‑prioritized**.  
  - Evidence: Phases 21–24 (diversity/selection tests on HumanEval, mostly negative for “fractal sampling” over naive sampling).  
  - Takeaway: Parallel sampling is easy; selecting and repairing effectively is harder than it looks when using the same causal model, so this variant of the Flash Flood idea currently sits in the de‑prioritized bucket (see Section 6).

---

## 4. Where the Interesting Future Work Is

Given the above, here are the most promising directions that stay true to the “Fractal Computer” vision and avoid repeating dead ends.

### 4.1 Hybrid Planner–Renderer: Causal + Bidirectional

Objective: Combine the strengths of causal LMs (open‑ended planning) with bidirectional Flash Flood decoders (fast, anchor‑preserving rendering).

- Approach:
  - Use a small causal “Manager” or planner to emit a skeleton: sketches, roots, or high‑level structure (specs, AST shapes, section headings).
  - Feed that skeleton into a bidirectional diffusion decoder that performs parallel refinement and filling, preserving given anchors.
  - Optionally, run a short refinement loop (a few denoising steps) rather than a single pass, leveraging the “fractal in time” behavior.
- Why this is interesting:
  - Plays to each architecture’s strengths: causal for creativity and global intent, bidirectional for fast, consistent realization.
  - Connects the original Manager+Renderer story (Phase 6, Vector 1) with the Phase 30 insight that the renderer should be bidirectional, not causal.
  - Longer term, it remains an open question whether planning itself should also be reframed as a **bidirectional / diffusion‑style process** rather than purely causal; the hybrid design gives a concrete intermediate step.

### 4.2 Small but Real: Bidirectional Fractal Decoder on a Practical Domain

Objective: Take the Phase 30 bidirectional Flash Flood idea and move it one notch toward reality.

- Candidate domains:
  - Structured text: JSON/YAML configs with tens of fields, tables, templated docs.
  - Small‑scale code: function‑sized Python snippets (<100 lines) with simple tests.
- Concrete goals:
  - Demonstrate **>5–10× faster whole‑sequence generation** at comparable quality vs a causal baseline for that domain (e.g., within ≈2 percentage points of task accuracy), using O(1) refinement steps.
  - Show that initialized anchors (locked lines/fields) remain stable under refinement, enabling fast, locality‑preserving edits.
  - Report both **training cost** (compute used to fit the decoder) and **inference cost** (wall‑clock latency and FLOPs per sequence) so speedups are measured at similar or lower total cost.
- Why this is interesting:
  - It directly exploits the **bidirectional diffusion** insight and connects it to an application where latency, stability, and cost all matter.

### 4.3 Fractal Editor v2: Real‑World Hierarchical Editing

Objective: Turn the synthetic Fractal Editor into a concrete editing capability for code or documents.

- Approach:
  - Define a tree representation for a real artifact (AST for code, section/paragraph tree for docs).
  - Use a fractal or bidirectional decoder to regenerate only selected subtrees while freezing the rest.
  - Quantitatively compare drift vs a strong causal LLM editor on same tasks (“edit function X but leave Y untouched”).
- Why this is interesting:
  - Stable, local edits are something practitioners feel daily; a clear win here is both scientifically and productively compelling.

### 4.4 Execution‑Based Critic at Scale (Vector 6 on Stronger Models)

Objective: Consolidate the execution‑loop success into a more decisive real‑world result.

- Approach:
  - Re‑run the MBPP/HumanEval pipeline with a stronger base model and a more efficient generator (possibly aided by parallel decoding).
  - Tighten the evaluation: fixed compute/time budget, clear Pass@1 improvement, and ablations (no critic vs critic vs critic + repair).
- Why this is interesting:
  - It leverages a **validated mechanism** (execution + critic) and can be combined later with better decoders (causal + bidirectional hybrid).

### 4.5 Holographic Learning (Vector 2) on a Controlled Benchmark

Objective: Give the original “holographic” claim a fair but scoped test.

- Approach:
  - Design a synthetic or mid‑scale hierarchical dataset where weight sharing across levels is plausibly beneficial (e.g., multi‑scale documents or programs).
  - Compare a fractal shared‑weight model vs a depth‑matched non‑shared transformer on sample efficiency and depth generalization.
- Why this is interesting:
  - It clarifies whether the “fractal learner” is a real statistical advantage or mainly an architectural convenience for certain tasks.

### 4.6 Universal Denoising Engine (Recurrent Bidirectional Diffusion)

Objective: Explore a more ambitious unification where generation, repair, and editing are all treated as denoising at different noise levels.

- Approach:
  - Train a bidirectional diffusion‑style decoder on a **continuous noise curriculum**: from fully masked / noisy sequences (generation) through synthetically corrupted drafts (repair) to local masked subtrees (editing).
  - Implement the denoiser as a **recurrent bidirectional transformer** (Universal‑Transformer‑style), reusing the same block across refinement steps to trade compute for quality at inference time.
  - Use a synthetic mutation engine (inspired by Phase 18) to create (corrupted, clean) pairs for code and structured text.
- Why this is interesting:
  - Unifies Flash Flood, program repair, and hierarchical editing as points on a single denoising curve.
  - Revives the “shared‑weight depth” idea (Vector 2) in a bidirectional setting where recursive refinement is a fixed‑point iteration over a global state, rather than a fragile causal stack.

---

## 5. The “Fractal Computer” in One Sentence

The emerging picture is:

> **Fractal Computer = Causal Manager (planner) + Bidirectional Flash Flood Decoder (parallel renderer) + Energy/Execution Critics (verifiers) + Hierarchical Editor (stable local edits)**  
> with fractal behavior expressed as self‑similar refinement in time and across levels, not just as deep stacks of unique layers.

This architecture unifies the main validated ingredients: the Phase 6 Manager+Renderer split, the energy‑based fractal engine, the execution‑trained critics, the hierarchical editor, and the Phase 30 bidirectional Flash Flood decoder.

---

## 6. What Can Be Safely De‑Prioritized

Based on the accumulated negative results:

- **Pure prompt / sketch / explanation tricks on top of causal LLMs**  
  Multiple phases (21–24) suggest diminishing returns here; the model’s own misunderstandings propagate through plan, code, and explanation.

- **Text‑only energy heads for reasoning**  
  Ouroboros‑style text critics did not beat baselines; execution‑based critics are more promising.

- **Naïve causal Flash Flood for O(1) decoding**  
  Phases 28–29 make a strong case that causal masking fundamentally conflicts with island‑preserving refinement.

These ideas can still be used as **baselines or ablations**, but they no longer look like primary research directions.

---

This file is meant to be updated **sparingly**, when a phase produces a genuinely new thematic insight (not just a small metric bump). For detailed chronology, see `research-log/LOG.md`. For forward‑looking project ideas and scoring, see `Vectors/README.md`.
