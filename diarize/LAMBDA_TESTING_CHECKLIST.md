# Lambda GPU Testing Checklist

**CRITICAL**: Complete EVERY item in this checklist before terminating the instance. Instances cost $1.29+/hr - terminating early wastes time and money.

---

## Pre-Launch Preparation

### 1. Define Complete Task List
**Write down EVERY task that must be completed before shutdown:**

- [ ] Task 1: ___________________________
- [ ] Task 2: ___________________________
- [ ] Task 3: ___________________________
- [ ] Task 4: ___________________________
- [ ] Task 5: ___________________________

**Example (DiariZen VoxConverse):**
- [ ] Run benchmark on all 216 files
- [ ] Compute DER scores
- [ ] Run boundary error analysis
- [ ] Create results archive
- [ ] Download archive to local machine
- [ ] Verify archive integrity

### 2. Identify Known Setup Issues
**Document all compatibility fixes needed:**

| Issue | Fix | Verification |
|-------|-----|--------------|
| Example: PyTorch CUDA version | Use CUDA 12 bundled | `python -c "import torch; print(torch.version.cuda)"` |
| Example: NumPy 2.0 incompatibility | `pip install 'numpy<2.0'` | `python -c "import numpy; print(numpy.__version__)"` |

### 3. Prepare Required Files
**List all files that need to be uploaded to Lambda:**

- [ ] Modified/patched scripts: ___________________________
- [ ] Configuration files: ___________________________
- [ ] Custom analysis scripts: ___________________________
- [ ] Requirements files: ___________________________

### 4. Define Expected Outputs
**Specify exactly what files/results should exist at the end:**

- [ ] Output 1: Path `___________`, Size `___________`, Contents: ___________
- [ ] Output 2: Path `___________`, Size `___________`, Contents: ___________
- [ ] Output 3: Path `___________`, Size `___________`, Contents: ___________

**Example:**
- [ ] RTTM files: `~/results/*.rttm`, 216 files, speaker diarization outputs
- [ ] DER log: `/tmp/der_scores.log`, ~50KB, per-file DER metrics
- [ ] Archive: `~/results.tar.gz`, ~2-10MB, compressed results bundle

---

## During Testing - Live Checklist

### Phase 1: Launch & Setup
- [ ] Launch instance: `python lambda_helper.py launch`
- [ ] Note instance ID: `___________________________`
- [ ] Note IP address: `___________________________`
- [ ] Setup SSH: `python lambda_helper.py setup-ssh`
- [ ] Verify GPU: `ssh lambda "nvidia-smi"`

### Phase 2: Environment Setup
- [ ] Clone/upload code
- [ ] Create venv: `python3 -m venv .venv`
- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Install dependencies (see Known Issues section)
- [ ] Verify imports: `python -c "import torch, numpy; print('OK')"`

### Phase 3: Run Tasks (MARK EACH AS DONE)
**From Pre-Launch section - DO NOT SKIP ANY:**

- [ ] Task 1: ___________________________
  - Command: ___________________________
  - Expected duration: ___________________________
  - Success indicator: ___________________________

- [ ] Task 2: ___________________________
  - Command: ___________________________
  - Expected duration: ___________________________
  - Success indicator: ___________________________

### Phase 4: Pre-Shutdown Verification
**BEFORE running terminate command:**

- [ ] All tasks from Phase 3 are checked off
- [ ] All expected output files exist (verify with `ls -lh`)
- [ ] All results downloaded to local machine
- [ ] Archive integrity verified: `tar -tzf archive.tar.gz | wc -l`
- [ ] No "missing task" concerns remain

### Phase 5: Shutdown
**Only proceed if Phase 4 is 100% complete:**

- [ ] Download final logs: `scp lambda:/tmp/*.log /tmp/`
- [ ] Terminate: `python lambda_helper.py terminate <instance_id>`
- [ ] Verify termination status
- [ ] Final cost estimate: `___________________________`

---

## Emergency Protocols

### If a Task Fails
1. **DO NOT terminate immediately**
2. Investigate the error: save logs to /tmp/error.log
3. Attempt fix or workaround
4. If unfixable: explicitly document what was skipped and WHY
5. Confirm with user before terminating

### If a Task Tool is Missing
1. **DO NOT assume it's optional**
2. Search for alternatives: `find ~ -name "*keyword*"`
3. Check documentation: `ls docs/`, `grep -r "keyword" .`
4. Ask user: "Tool X not found - skip, create custom, or investigate?"
5. Only terminate after user confirmation

### If Setup Takes Longer Than Expected
1. **DO NOT rush the remaining tasks**
2. Let long-running processes finish (use `nohup`, `screen`, or `tmux`)
3. Monitor via: `ssh lambda "tail -f /tmp/process.log"`
4. Cost is already incurred - completing correctly matters more than speed

---

## Post-Mortem Template

**After instance terminates, fill this out:**

### What Worked
- ___________________________
- ___________________________

### What Failed
- ___________________________
- ___________________________

### Surprises / Unexpected Issues
- ___________________________
- ___________________________

### Time Breakdown
- Setup: ___ minutes
- Task 1: ___ minutes
- Task 2: ___ minutes
- Task N: ___ minutes
- **Total: ___ minutes = $_____ cost**

### Would Do Differently Next Time
- ___________________________
- ___________________________

### Files to Keep for Next Time
- ___________________________
- ___________________________

---

## Example: DiariZen VoxConverse Testing

### Pre-Launch Task List
- [x] Run benchmark on 216 audio files
- [x] Compute DER scores with dscore
- [ ] ❌ **FAILED TO COMPLETE**: Run boundary error analysis (tool didn't exist, terminated anyway)
- [x] Create results archive
- [x] Download archive locally
- [x] Verify archive contents

### Known Setup Issues (DiariZen)
| Issue | Fix | Verification |
|-------|-----|--------------|
| PyTorch 2.8.0 + CUDA 11.8 | Use CUDA 12 bundled | `pip install torch==2.8.0` (no index-url) |
| pyannote.audio 4.0.2 incompatible | Use bundled 3.1.1 submodule | `cd pyannote-audio && pip install -e .` |
| NumPy 2.0 breaks pyannote | Downgrade to 1.26.4 | `pip install 'numpy<2.0'` |
| torch.load weights_only=True | Monkey patch before imports | Add patch to script top |
| dscore np.int deprecated | Edit metrics.py line 63 | `sed -i 's/np.int)/int)/g' dscore/scorelib/metrics.py` |

### Expected Outputs
- [x] 216 RTTM files at `~/DiariZen/results/voxconverse/*.rttm`
- [x] DER scores at `/tmp/voxconverse_der_final.log` (~15KB)
- [x] Benchmark log at `/tmp/voxconverse_full.log` (~2MB)
- [ ] ❌ **MISSING**: Boundary error analysis results
- [x] Archive at `~/DiariZen_results.tar.gz` (88KB)

### Mistake Made
**Terminated instance without completing boundary analysis.** The tool (`analyze_boundary_errors.py`) didn't exist in the repo, so I assumed it was optional. Should have asked user or created custom analysis script before terminating.

**Cost of mistake**: ~$1.29 for 60 minutes of work that needs to be partially redone.

---

## Checklist Usage Instructions

1. **Before launching Lambda**: Fill out "Pre-Launch Preparation" section completely
2. **During testing**: Mark off items in "During Testing" section as you complete them
3. **Before terminate**: Verify 100% completion of Phase 4
4. **After shutdown**: Complete "Post-Mortem Template"
5. **Keep this file updated**: Add new issues/fixes as you discover them

**Remember**: When in doubt about whether a task is complete, IT ISN'T. Ask the user.
