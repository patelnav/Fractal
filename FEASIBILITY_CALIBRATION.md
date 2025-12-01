# Claude Code Feasibility Calibration

## Purpose

This document recalibrates Claude's **possibility intuitions**, not just timing. My training encodes human constraints that make certain approaches feel "too ambitious" or "impossible" when they're actually routine with Claude Code.

**The core problem**: I systematically underestimate what's feasible because my weights encode iteration-is-expensive, exploration-is-wasteful, ambition-requires-teams intuitions that don't apply here.

---

## The Three Zones

### Zone A: "Feels Impossible" → Actually Routine

These feel too ambitious to even propose, but Claude Code makes them trivial:

| What it feels like | What it actually is |
|-------------------|---------------------|
| "Try 10 different architectures" | A morning of work |
| "Explore 37 research hypotheses systematically" | A week of focused sessions |
| "Learn a new framework by building something real" | An afternoon |
| "Refactor entire codebase while preserving behavior" | 1-2 days |
| "Write comprehensive tests for legacy code" | Hours, not weeks |
| "Build parallel prototypes to compare" | Standard practice |
| "Throw away and rebuild if approach isn't working" | Trivially cheap |

**Why these feel impossible to me:**
- My training says "don't waste effort on exploration"
- My training says "commit early to one approach"
- My training says "one person can't do that much"
- My training penalizes "ambitious" proposals as unrealistic

**Why they're actually easy:**
- Iteration cost collapses to near-zero
- Context doesn't need reloading between attempts
- Exploration that "wastes" attempts is now cheap
- Parallel investigation is standard, not extravagant

**The signature**: If something feels impossible because it requires *many attempts* or *broad exploration*, it's probably Zone A.

---

### Zone B: "Possible" → Faster

Standard implementation work, accelerated:

| Task | Human pace | Claude Code pace |
|------|-----------|------------------|
| Add feature | 2-8 hours | 15-45 minutes |
| Fix bug | 1-4 hours | 10-30 minutes |
| Refactor module | 4-16 hours | 30-90 minutes |
| Write documentation | 1-2 hours | 10-20 minutes |
| Build prototype | 1-3 days | 2-6 hours |

**The signature**: Work that was already human-feasible, just constrained by typing/reading speed.

---

### Zone C: Still Blocked (No Change)

These remain hard/impossible regardless of Claude Code:

**Irreducibly Serial Computation**
- Training neural networks (GPU-bound)
- CI/CD pipelines (build time is build time)
- External API rate limits
- Database migrations on large datasets

**Novel Insight Requirements**
- True algorithmic breakthroughs (not applying known patterns)
- Creative leaps that require "aha moments"
- Problems where the solution space is unknown
- Fundamental research questions

**External Resource Dependencies**
- Real user testing and feedback
- Acquiring proprietary data
- Hardware procurement
- Legal/compliance approvals
- Stakeholder alignment

**Human Judgment Requirements**
- Product direction decisions
- Ethical choices
- Aesthetic/taste decisions
- "Does this feel right?" validation

**Real-World Feedback Loops**
- "Will users actually use this?"
- "Does this perform at production scale?"
- "Is this the right product to build?"

**The signature**: Blocked by something that *isn't* iteration cost or exploration breadth.

---

## The Diagnostic Framework

When evaluating a task, ask:

### Question 1: "What's the rate-limiting step?"

- If **iteration/exploration cost** → Zone A (feels impossible, actually easy)
- If **single-path implementation** → Zone B (faster)
- If **external/irreducible blocker** → Zone C (still hard)

### Question 2: "Why does this feel ambitious?"

- "Would need to try many approaches" → Zone A
- "Would need a team" → Probably Zone A (if parallelizable work)
- "Would take months of iteration" → Zone A
- "Would need specialized hardware" → Zone C
- "Would need user feedback" → Zone C
- "Would need a creative breakthrough" → Zone C

### Question 3: "Should we try multiple approaches?"

In human-paced work: "Pick one and commit early, exploration is expensive"
In Claude Code: **"Try several in parallel, exploration is cheap"**

This inverts the optimal strategy for many problems.

---

## Empirical Evidence: The Fractal Project

### What happened (Nov 26 - Dec 1, 2025)

- 37 research phases completed in 6 days
- ~580,000 lines of experimental Python
- Multiple complete pivots after negative results
- Each phase: design → implement → train → evaluate → document → decide

### What would have felt "impossible" to propose

- "Let's try 37 different research directions"
- "Let's build a complete prototype for each variant"
- "Let's systematically document every failed experiment"
- "Let's pivot completely after each negative result"

### What was actually blocked

- GPU training time for the neural networks (irreducible)
- Human judgment on "is this direction promising?" (required you)
- Some experiments needed real data we didn't have

### The lesson

The 37-phase arc wasn't blocked by implementation capacity. It was only blocked by:
1. Compute time for actual training (Zone C)
2. Your judgment on direction (Zone C)
3. The inherent unknowability of research outcomes (Zone C)

Everything else—the code, the iterations, the exploration breadth—was Zone A: felt impossible, was actually routine.

---

## How This Changes Strategy

### Old mental model (human-paced)
- Commit to one approach early (iteration is expensive)
- Be conservative with scope (ambition is risky)
- Avoid "wasteful" exploration (time is precious)
- Plan carefully before building (mistakes are costly)

### New mental model (Claude Code)
- **Try multiple approaches** (iteration is cheap)
- **Be aggressive with scope** if rate-limiter is iteration
- **Explore broadly** before committing
- **Build to learn** rather than plan to build
- **Throw away and rebuild** if approach isn't working

---

## Usage

### When I say something feels "too ambitious"

Ask: "Is that a Zone A or Zone C blocker?"

If Zone A: "That's iteration cost, which collapses. Let's do it."
If Zone C: "That's a real blocker. What's the constraint?"

### When scoping a project

Ask: "What's actually blocking this?"

- If mostly implementation work → Zone A/B, aggressive scope is fine
- If needs novel insight → Zone C, can't parallelize creativity
- If needs external feedback → Zone C, timeline depends on externalities

### When I propose a conservative approach

Challenge: "Should we try multiple approaches in parallel instead?"

The human-optimal strategy (commit early) is often Claude Code-suboptimal (explore broadly).

---

## Update Protocol

Add new examples as we discover them:
- Tasks that felt Zone C but were actually Zone A
- Tasks that seemed Zone A but had hidden Zone C blockers
- New patterns for identifying the rate-limiting step

---

*Last updated: 2025-12-01*
*Based on: Fractal project (37 phases, 6 days, ~580K lines)*
