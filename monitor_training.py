#!/usr/bin/env python3
"""Live training monitor - watches log file and plots metrics in terminal."""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import plotext as plt


def parse_metrics(line: str) -> dict:
    """Extract metrics from a log line."""
    metrics = {}
    # loss patterns (ce=, loss=, val_loss:)
    for match in re.finditer(r'(?:loss|ce|val_loss)[=:\s]+([0-9.]+)', line, re.I):
        metrics['loss'] = float(match.group(1))
    # accuracy patterns
    for match in re.finditer(r'(?:acc|accuracy|token_acc)[=:\s]+([0-9.]+)', line, re.I):
        metrics['acc'] = float(match.group(1))
    # parse success (parse@10=0.61)
    for match in re.finditer(r'parse@?(\d*)[=:\s]+([0-9.]+)', line, re.I):
        key = f"parse@{match.group(1)}" if match.group(1) else "parse"
        metrics[key] = float(match.group(2))
    # rl reward
    for match in re.finditer(r'rl[=:\s]+(-?[0-9.]+)', line, re.I):
        val = float(match.group(1))
        if val != 0:  # skip "rl=off"
            metrics['rl'] = val
    return metrics


def monitor(logfile: str, refresh: float = 1.0):
    """Watch log file and plot live metrics."""
    history = defaultdict(list)
    last_size = 0

    print(f"Monitoring {logfile}... (Ctrl+C to exit)")

    while True:
        try:
            # Check if file changed
            current_size = Path(logfile).stat().st_size
            if current_size < last_size:
                # File was truncated, reset
                history.clear()
                last_size = 0

            # Read entire file and use strings to handle tqdm
            import subprocess
            result = subprocess.run(
                ['strings', logfile],
                capture_output=True, text=True
            )

            # Parse all lines
            history.clear()
            for line in result.stdout.split('\n'):
                metrics = parse_metrics(line)
                for k, v in metrics.items():
                    history[k].append(v)

            last_size = current_size

            # Status line
            now = datetime.now().strftime("%H:%M:%S")
            mtime = Path(logfile).stat().st_mtime
            age = int(time.time() - mtime)
            if age < 60:
                age_str = f"{age}s ago"
            else:
                age_str = f"{age // 60}m {age % 60}s ago"
            stale_warn = " ⚠️  STALE - training may have died!" if age > 30 else ""
            metrics_summary = ", ".join(f"{k}={len(v)}" for k, v in sorted(history.items()))
            status = f"[{now}] log updated {age_str}{stale_warn} | {metrics_summary or 'no metrics yet'}"

            # Plot
            if any(len(v) > 1 for v in history.values()):
                plt.clear_terminal()
                plt.clear_figure()
                plt.subplots(1, 2)

                # Left plot: loss
                plt.subplot(1, 1)
                if 'loss' in history and len(history['loss']) > 1:
                    plt.plot(history['loss'], label='loss')
                plt.title("Loss")

                # Right plot: parse success
                plt.subplot(1, 2)
                for key in sorted(history.keys()):
                    if key.startswith('parse') and len(history[key]) > 1:
                        plt.plot(history[key], label=key)
                plt.title("Parse Success")

                plt.show()
                print(status)
            else:
                print(f"\r{status}", end="", flush=True)

            time.sleep(refresh)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(refresh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live training monitor')
    parser.add_argument('logfile', help='Log file to monitor')
    args = parser.parse_args()
    monitor(args.logfile)
