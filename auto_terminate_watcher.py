#!/usr/bin/env python3
"""
Background watcher that:
1. Polls Lambda every 5 minutes to check if training is running
2. When not running (success OR crash): syncs ALL artifacts back to local
3. Terminates the Lambda instance
4. Logs everything to a file

Simple logic: running → keep polling, not running → sync & terminate
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path

POLL_INTERVAL = 300  # 5 minutes
LOG_FILE = "/tmp/lambda_watcher.log"
REMOTE_DIR = "~/Fractal/research-log/phase32-json-planner-refiner"
LOCAL_DIR = Path(__file__).parent / "research-log/phase32-json-planner-refiner"


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_training_running():
    """Check if train_denoiser_a100.py is running on Lambda.

    Returns:
        True: training running
        False: training not running (success or crash)
        None: SSH error (will retry)
    """
    try:
        # Check SSH connectivity first
        ssh_check = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "lambda", "echo ok"],
            capture_output=True, text=True, timeout=30
        )
        if ssh_check.returncode != 0:
            log(f"SSH connection failed: {ssh_check.stderr.strip()}")
            return None

        # Check if process is running
        result = subprocess.run(
            ["ssh", "lambda", "pgrep -f train_denoiser_a100.py"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        log("SSH command timed out")
        return None
    except Exception as e:
        log(f"Error checking status: {e}")
        return None


def sync_all_artifacts():
    """Rsync ALL artifacts back to local - checkpoints, logs, everything."""
    log("Syncing all artifacts from Lambda...")

    # Sync checkpoints directories (both possible names)
    log("  - checkpoints_a100/")
    subprocess.run([
        "rsync", "-avz", "--progress",
        f"lambda:{REMOTE_DIR}/checkpoints_a100/",
        f"{LOCAL_DIR}/checkpoints_a100/"
    ])

    log("  - checkpoints_v2/ (if exists)")
    subprocess.run([
        "rsync", "-avz", "--progress",
        f"lambda:{REMOTE_DIR}/checkpoints_v2/",
        f"{LOCAL_DIR}/checkpoints_v2/"
    ], stderr=subprocess.DEVNULL)

    # Sync training log
    log("  - train_v2.log")
    subprocess.run([
        "rsync", "-avz",
        f"lambda:{REMOTE_DIR}/train_v2.log",
        f"{LOCAL_DIR}/train_v2.log"
    ])

    # Also sync any other potentially useful files
    log("  - Any .pt files in root")
    subprocess.run([
        "rsync", "-avz",
        f"lambda:{REMOTE_DIR}/*.pt",
        f"{LOCAL_DIR}/"
    ], stderr=subprocess.DEVNULL)  # Ignore "no files" error

    log("Sync complete.")


def terminate_lambda():
    """Terminate the Lambda instance."""
    log("Terminating Lambda instance...")
    result = subprocess.run(
        ["python", "lambda_helper.py", "terminate"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent
    )
    log(f"Terminate result: {result.stdout.strip()}")


def main():
    log("=" * 60)
    log("Lambda watcher started")
    log(f"Poll interval: {POLL_INTERVAL}s (5 min)")
    log(f"Remote: {REMOTE_DIR}")
    log(f"Local: {LOCAL_DIR}")
    log("=" * 60)

    consecutive_errors = 0
    max_errors = 10  # After 10 SSH failures (~10 min), give up

    while True:
        status = is_training_running()

        if status is True:
            # Training running - keep waiting
            log("Training running. Sleeping 5 min...")
            consecutive_errors = 0
            time.sleep(POLL_INTERVAL)

        elif status is False:
            # Training not running - sync and terminate
            log("Training not running. Syncing and terminating...")
            consecutive_errors = 0
            sync_all_artifacts()
            terminate_lambda()
            log("=" * 60)
            log("DONE. Lambda terminated. Check results at:")
            log(f"  {LOCAL_DIR}/checkpoints_v2/")
            log(f"  {LOCAL_DIR}/train_v2.log")
            log("=" * 60)
            break

        else:
            # SSH error - retry with shorter interval
            consecutive_errors += 1
            log(f"SSH error ({consecutive_errors}/{max_errors}). Retry in 1 min...")

            if consecutive_errors >= max_errors:
                log("=" * 60)
                log("ERROR: Too many SSH failures. Instance may be down.")
                log("Check manually: python lambda_helper.py status")
                log("=" * 60)
                break

            time.sleep(60)


if __name__ == "__main__":
    main()
