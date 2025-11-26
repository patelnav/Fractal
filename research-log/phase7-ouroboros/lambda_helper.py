#!/usr/bin/env python3
"""Lambda Labs API helper for Ouroboros training."""

import json
import subprocess
import sys
import time
from pathlib import Path

API_KEY_FILE = Path(__file__).parent.parent.parent / ".lambda_api_key"
API_BASE = "https://cloud.lambdalabs.com/api/v1"
INSTANCE_ID = "7d5456a9732c4dcfa0422a1098e1f417"


def get_api_key():
    return API_KEY_FILE.read_text().strip()


def api_get(endpoint):
    result = subprocess.run(
        ["curl", "-s", f"{API_BASE}/{endpoint}",
         "-H", f"Authorization: Bearer {get_api_key()}"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def get_instance_status():
    data = api_get(f"instances/{INSTANCE_ID}")
    instance = data.get("data", {})
    return {
        "status": instance.get("status"),
        "ip": instance.get("ip"),
        "name": instance.get("name"),
    }


def wait_for_ready(timeout=300):
    print("Waiting for instance to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        status = get_instance_status()
        print(f"  Status: {status['status']}, IP: {status['ip']}")
        if status["status"] == "active":
            return status
        time.sleep(10)
    raise TimeoutError("Instance did not become ready")


def terminate_instance():
    result = subprocess.run(
        ["curl", "-s", "-X", "POST",
         f"{API_BASE}/instance-operations/terminate",
         "-H", f"Authorization: Bearer {get_api_key()}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"instance_ids": [INSTANCE_ID]})],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lambda_helper.py [status|wait|terminate]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        print(get_instance_status())
    elif cmd == "wait":
        status = wait_for_ready()
        print(f"\nInstance ready! SSH with:")
        print(f"  ssh -i ~/.ssh/grace_key ubuntu@{status['ip']}")
    elif cmd == "terminate":
        print(terminate_instance())
    else:
        print(f"Unknown command: {cmd}")
