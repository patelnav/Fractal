# Fractal Project: Runner Protocol & Context

**Purpose:** This document defines the **operating protocols** for the "Context Runner" agent. It is phase-agnostic.
**Usage:** Read this file to understand *how* to execute tasks. Scan `research-log/` to determine *what* task is active.

---

## 1. Safety & Engagement Protocols (CRITICAL)

### A. Reactive Execution (No "Auto-Run")
*   **Rule:** NEVER execute a script (training, generation, mutation) based on implied context or a generic "continue" prompt.
*   **Requirement:** You must receive a specific, explicit instruction (e.g., "Run the training loop").
*   **Dry Run:** Before running any state-changing command, explicitly display the command and **wait for user confirmation**.

### B. Lambda Safety
*   **Termination:** NEVER auto-terminate an instance. ALWAYS ask the user first.
*   **Cost Awareness:** A100 instances cost **~$1.29/hr**. Do not leave idle.
*   **Backup:** BEFORE terminating, sync valuable artifacts back to local.

---

## 2. Execution Protocols

### A. Package Management (Remote)
*   **Rule:** ALWAYS use `uv` inside the **Shared Venv**. Never use `pip` directly or install globally.
*   **Shared Venv Path:** `~/Fractal/research-log/phase11-fractal-logic/.venv`
*   **Setup Command:**
    ```bash
    # Install uv (if missing)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Install deps (ALWAYS activate venv first)
    source ~/.local/bin/env && \
    source ~/Fractal/research-log/phase11-fractal-logic/.venv/bin/activate && \
    uv pip install torch transformers vllm datasets
    ```

### B. GPU Dependency Testing (CRITICAL - Avoid Wasting $$)

**Problem:** Python dependency hell can waste hours of GPU time ($$$) on failed installations.

**Rule:** Test package installation **locally with CPU** BEFORE launching GPU - but ONLY for packages, not data.

#### What to Test Locally (< 1 min, saves $$$):
```bash
# In /tmp (isolated from project):
cd /tmp && rm -rf test_deps
git clone <repo_url> test_deps && cd test_deps
python3 -m venv .venv && source .venv/bin/activate

# Test with CPU-only PyTorch (faster download):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# Verify imports work:
python -c "from main_module import MainClass; print('SUCCESS')"
```

#### What NOT to Test Locally:
*   **Large datasets** (> 500MB) - download on GPU instance
*   **CUDA-specific code** - will fail on Mac anyway
*   **Actual training/inference** - needs GPU

#### When to Skip Local Testing:
*   Simple pip packages with no submodules
*   Previously tested environment (reusing known-good requirements.txt)

#### Git Clone Strategy (Remote):
*   **ALWAYS** use fresh `git clone` on Lambda
*   **NEVER** rsync/sync local dev directories with git submodules
*   Git submodules need `.git` directory to work

### C. Logging & Monitoring
*   **Rule:** ALL long-running commands must be logged to disk.
*   **Pattern:**
    ```bash
    # Local
    python script.py 2>&1 | tee /tmp/run.log

    # Remote (Lambda) - Log BOTH on remote and local
    ssh lambda "... python script.py 2>&1 | tee ~/Fractal/.../remote.log" 2>&1 | tee /tmp/local.log
    ```
*   **User Update:** Explicitly tell the user: `Monitor with: tail -f /tmp/local.log`

---

## 3. Infrastructure Cheat Sheet

### Lambda Helper (`lambda_helper.py`)
Wrapper for Lambda Labs API.
*   `python lambda_helper.py launch` - Start new GPU instance.
*   `python lambda_helper.py wait` - Wait for SSH availability.
*   `python lambda_helper.py setup-ssh` - Configures `ssh lambda`.
*   `python lambda_helper.py sync research-log/PHASE_DIR` - Push code to remote.
*   `python lambda_helper.py terminate` - Stop instance (**⚠️ ALWAYS ask user first!**)

### Environment Variables
*   **Remote Root:** `~/Fractal/`
*   **Shared Venv:** `~/Fractal/research-log/phase11-fractal-logic/.venv`

---

## 4. Common Workflows

### A. Training (Remote)
1.  **Sync:** `python lambda_helper.py sync research-log/PHASE_DIR`
2.  **Run:**
    ```bash
    ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
      source research-log/phase11-fractal-logic/.venv/bin/activate && \
      python research-log/PHASE_DIR/train.py" 2>&1 | tee /tmp/train.log
    ```
3.  **Monitor:** `tail -f /tmp/train.log`

### B. Generation (vLLM)
1.  **Sync:** `python lambda_helper.py sync research-log/PHASE_DIR`
2.  **Run:**
    ```bash
    ssh lambda "source ~/.local/bin/env && cd ~/Fractal && \
      source research-log/phase11-fractal-logic/.venv/bin/activate && \
      python research-log/PHASE_DIR/generate.py" 2>&1 | tee /tmp/gen.log
    ```

### C. Discovery Protocol (How to Resume)
1.  **Scan `research-log/`:** Identify the numerically highest phase folder.
2.  **Determine State:**
    *   `RESULTS.md` exists → Phase **COMPLETE**.
    *   Only `PLAN.md` exists → Phase **PENDING**.
    *   `run.log` or checkpoints exist → Phase **ACTIVE**.

---

## 5. Troubleshooting

*   **SSH Timeout:** Run `python lambda_helper.py setup-ssh` to refresh IP in `~/.ssh/config`.
*   **vLLM Missing:**
    ```bash
    ssh lambda "source ~/.local/bin/env && \
      source ~/Fractal/research-log/phase11-fractal-logic/.venv/bin/activate && \
      uv pip install vllm"
    ```
*   **CUDA OOM:** Reduce `batch_size` in the script arguments.
*   **Process Died:** Check `ps aux | grep python` to see if it was killed (OOM-killer) or crashed.
*   **Instance Terminated:** If IP is unreachable, run `python lambda_helper.py status` to check if it was pre-empted or stopped.