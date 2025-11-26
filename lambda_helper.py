#!/usr/bin/env python3
"""Lambda Labs API helper for managing GPU instances."""

import json
import subprocess
import sys
import time
from pathlib import Path

API_KEY_FILE = Path(__file__).parent / ".lambda_api_key"
API_BASE = "https://cloud.lambdalabs.com/api/v1"
STATE_FILE = Path(__file__).parent / ".lambda_instance_id"
SSH_CONFIG = Path.home() / ".ssh" / "config"
PROJECT_ROOT = Path(__file__).parent

# SSH host alias
SSH_ALIAS = "lambda"

# Default instance preferences
DEFAULT_INSTANCE_TYPES = [
    "gpu_1x_a100_sxm4",  # $1.29/hr
    "gpu_1x_a100",       # $1.29/hr
    "gpu_1x_h100_pcie",  # $2.49/hr
    "gpu_1x_h100_sxm5",  # $3.29/hr
]


def get_api_key():
    return API_KEY_FILE.read_text().strip()


def api_get(endpoint):
    result = subprocess.run(
        ["curl", "-s", f"{API_BASE}/{endpoint}",
         "-H", f"Authorization: Bearer {get_api_key()}"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def api_post(endpoint, data):
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{API_BASE}/{endpoint}",
         "-H", f"Authorization: Bearer {get_api_key()}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps(data)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def get_ssh_keys():
    """Get available SSH key names."""
    data = api_get("ssh-keys")
    return [key["name"] for key in data.get("data", [])]


def get_available_instances():
    """Get instance types with availability."""
    data = api_get("instance-types")
    available = []
    for name, info in data.get("data", {}).items():
        regions = info.get("regions_with_capacity_available", [])
        if regions:
            available.append({
                "name": name,
                "description": info["instance_type"]["description"],
                "price": info["instance_type"]["price_cents_per_hour"] / 100,
                "regions": [r["name"] for r in regions]
            })
    return sorted(available, key=lambda x: x["price"])


def get_instance_id():
    """Get saved instance ID."""
    if STATE_FILE.exists():
        return STATE_FILE.read_text().strip()
    return None


def save_instance_id(instance_id):
    """Save instance ID to state file."""
    STATE_FILE.write_text(instance_id)


def clear_instance_id():
    """Clear saved instance ID."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def get_instance_status(instance_id=None):
    """Get status of an instance."""
    instance_id = instance_id or get_instance_id()
    if not instance_id:
        return {"error": "No instance ID. Run 'launch' first."}

    data = api_get(f"instances/{instance_id}")
    if "error" in data:
        return {"status": "terminated", "ip": None, "name": None}

    instance = data.get("data", {})
    return {
        "id": instance_id,
        "status": instance.get("status"),
        "ip": instance.get("ip"),
        "name": instance.get("name"),
        "instance_type": instance.get("instance_type", {}).get("name"),
        "region": instance.get("region", {}).get("name"),
    }


def launch_instance(instance_type=None, region=None, name="fractal-gpu"):
    """Launch a new GPU instance."""
    # Get SSH keys
    ssh_keys = get_ssh_keys()
    if not ssh_keys:
        return {"error": "No SSH keys found. Add one at cloud.lambdalabs.com"}

    # Find available instance
    available = get_available_instances()
    if not available:
        return {"error": "No instances available"}

    # Select instance type
    selected = None
    if instance_type:
        for inst in available:
            if inst["name"] == instance_type:
                selected = inst
                break
        if not selected:
            return {"error": f"Instance type {instance_type} not available"}
    else:
        # Try preferred types in order
        for pref in DEFAULT_INSTANCE_TYPES:
            for inst in available:
                if inst["name"] == pref:
                    selected = inst
                    break
            if selected:
                break
        if not selected:
            selected = available[0]  # Cheapest available

    # Select region
    if region and region not in selected["regions"]:
        return {"error": f"Region {region} not available for {selected['name']}"}
    target_region = region or selected["regions"][0]

    print(f"Launching {selected['name']} in {target_region} @ ${selected['price']:.2f}/hr...")

    # Launch
    result = api_post("instance-operations/launch", {
        "region_name": target_region,
        "instance_type_name": selected["name"],
        "ssh_key_names": ssh_keys[:1],  # Use first SSH key
        "name": name,
    })

    if "error" in result:
        return result

    instance_ids = result.get("data", {}).get("instance_ids", [])
    if not instance_ids:
        return {"error": "No instance ID returned", "response": result}

    instance_id = instance_ids[0]
    save_instance_id(instance_id)

    return {
        "id": instance_id,
        "instance_type": selected["name"],
        "region": target_region,
        "price": selected["price"],
        "message": "Instance launching. Run 'wait' to wait for SSH."
    }


def wait_for_ready(timeout=300):
    """Wait for instance to be ready for SSH."""
    instance_id = get_instance_id()
    if not instance_id:
        return {"error": "No instance ID. Run 'launch' first."}

    print("Waiting for instance to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        status = get_instance_status(instance_id)
        print(f"  Status: {status.get('status')}, IP: {status.get('ip')}")

        if status.get("status") == "active" and status.get("ip"):
            # Test SSH connection
            print("  Testing SSH connection...")
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                     f"ubuntu@{status['ip']}", "echo ready"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    print(f"\nInstance ready!")
                    print(f"  SSH: ssh ubuntu@{status['ip']}")
                    return status
            except subprocess.TimeoutExpired:
                pass

        time.sleep(10)

    return {"error": "Timeout waiting for instance"}


def terminate_instance(instance_id=None):
    """Terminate an instance."""
    instance_id = instance_id or get_instance_id()
    if not instance_id:
        return {"error": "No instance ID. Run 'launch' first."}

    print(f"Terminating instance {instance_id}...")
    result = api_post("instance-operations/terminate", {
        "instance_ids": [instance_id]
    })

    if "error" not in result:
        clear_instance_id()

    return result


def list_instances():
    """List all running instances."""
    data = api_get("instances")
    instances = data.get("data", [])

    if not instances:
        return {"message": "No running instances"}

    result = []
    for inst in instances:
        result.append({
            "id": inst["id"],
            "name": inst.get("name"),
            "status": inst.get("status"),
            "ip": inst.get("ip"),
            "type": inst.get("instance_type", {}).get("name"),
            "region": inst.get("region", {}).get("name"),
        })
    return result


def ssh_command():
    """Print SSH command for current instance."""
    status = get_instance_status()
    if status.get("ip"):
        return f"ssh ubuntu@{status['ip']}"
    return {"error": "No active instance with IP"}


def setup_ssh_config():
    """Setup SSH config for easy 'ssh lambda' access."""
    status = get_instance_status()
    if not status.get("ip"):
        return {"error": "No active instance with IP"}

    ip = status["ip"]

    # SSH config block with optimizations from LAMBDA_SESSION.md
    ssh_block = f"""
# Lambda Labs GPU Instance (auto-managed by lambda_helper.py)
Host {SSH_ALIAS}
    HostName {ip}
    User ubuntu
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    # Connection multiplexing (first: ~600ms, subsequent: ~80ms)
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    # Keepalive for auto-reconnect
    ServerAliveInterval 60
    ServerAliveCountMax 3
"""

    # Ensure sockets directory exists
    sockets_dir = Path.home() / ".ssh" / "sockets"
    sockets_dir.mkdir(parents=True, exist_ok=True)

    # Read existing config
    if SSH_CONFIG.exists():
        config_content = SSH_CONFIG.read_text()
    else:
        config_content = ""

    # Remove existing lambda block if present
    import re
    pattern = r'\n# Lambda Labs GPU Instance \(auto-managed.*?(?=\n# |\nHost |\Z)'
    config_content = re.sub(pattern, '', config_content, flags=re.DOTALL)

    # Also remove standalone Host lambda block
    pattern2 = r'\nHost lambda\n.*?(?=\nHost |\n# |\Z)'
    config_content = re.sub(pattern2, '', config_content, flags=re.DOTALL)

    # Append new config
    config_content = config_content.rstrip() + ssh_block

    SSH_CONFIG.write_text(config_content)

    print(f"SSH config updated. Connect with: ssh {SSH_ALIAS}")
    return {"ssh_alias": SSH_ALIAS, "ip": ip}


def run_remote(command):
    """Run a command on the remote instance."""
    status = get_instance_status()
    if not status.get("ip"):
        return {"error": "No active instance with IP"}

    result = subprocess.run(
        ["ssh", SSH_ALIAS, command],
        capture_output=True, text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def sync_code(subdir=None):
    """Sync project code to the remote instance."""
    status = get_instance_status()
    if not status.get("ip"):
        return {"error": "No active instance with IP"}

    # Create remote directory
    remote_dir = "~/Fractal"
    subprocess.run(["ssh", SSH_ALIAS, f"mkdir -p {remote_dir}"], check=True)

    # Determine what to sync
    if subdir:
        local_path = PROJECT_ROOT / subdir
        remote_path = f"{remote_dir}/{subdir}"
        subprocess.run(["ssh", SSH_ALIAS, f"mkdir -p {remote_path}"], check=True)
    else:
        local_path = PROJECT_ROOT
        remote_path = remote_dir

    # Rsync with exclusions
    excludes = [
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.venv",
        "--exclude=venv",
        "--exclude=checkpoints",
        "--exclude=*.pt",
        "--exclude=.DS_Store",
        "--exclude=nanoGPT",
        "--exclude=data",
    ]

    cmd = [
        "rsync", "-avz", "--progress",
        *excludes,
        f"{local_path}/",
        f"{SSH_ALIAS}:{remote_path}/"
    ]

    print(f"Syncing {local_path} -> {SSH_ALIAS}:{remote_path}")
    result = subprocess.run(cmd)

    return {"synced": str(local_path), "to": remote_path, "returncode": result.returncode}


def print_help():
    print("""Lambda Labs GPU Instance Manager

Usage: python lambda_helper.py <command>

Instance Lifecycle:
  list        - List all running instances
  available   - Show available instance types
  launch      - Launch a new GPU instance
  wait        - Wait for instance to be SSH-ready
  status      - Get current instance status
  terminate   - Terminate current instance

SSH & Remote:
  setup-ssh   - Configure ~/.ssh/config for 'ssh lambda' access
  ssh         - Print SSH command
  run <cmd>   - Run command on remote instance
  sync [dir]  - Sync code to remote (optionally just a subdir)

Workflow Example:
  python lambda_helper.py launch           # Launch A100 ($1.29/hr)
  python lambda_helper.py wait             # Wait for SSH
  python lambda_helper.py setup-ssh        # Configure 'ssh lambda'
  python lambda_helper.py sync             # Sync all code
  ssh lambda                               # Connect!
  python lambda_helper.py terminate        # Tear down when done

Quick Run:
  python lambda_helper.py sync research-log/phase7-ouroboros
  python lambda_helper.py run "cd ~/Fractal/research-log/phase7-ouroboros && python solve_math.py"
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        print(json.dumps(list_instances(), indent=2))

    elif cmd == "available":
        available = get_available_instances()
        for inst in available:
            print(f"{inst['name']}: {inst['description']} @ ${inst['price']:.2f}/hr")
            print(f"  Regions: {', '.join(inst['regions'])}")

    elif cmd == "launch":
        instance_type = sys.argv[2] if len(sys.argv) > 2 else None
        result = launch_instance(instance_type=instance_type)
        print(json.dumps(result, indent=2))

    elif cmd == "wait":
        result = wait_for_ready()
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "status":
        print(json.dumps(get_instance_status(), indent=2))

    elif cmd == "ssh":
        result = ssh_command()
        if isinstance(result, str):
            print(result)
        else:
            print(json.dumps(result, indent=2))

    elif cmd == "setup-ssh":
        result = setup_ssh_config()
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "run":
        if len(sys.argv) < 3:
            print("Usage: python lambda_helper.py run <command>")
            sys.exit(1)
        remote_cmd = " ".join(sys.argv[2:])
        result = run_remote(remote_cmd)
        if result.get("stdout"):
            print(result["stdout"])
        if result.get("stderr"):
            print(result["stderr"], file=sys.stderr)
        sys.exit(result.get("returncode", 0))

    elif cmd == "sync":
        subdir = sys.argv[2] if len(sys.argv) > 2 else None
        result = sync_code(subdir)
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "terminate":
        print(json.dumps(terminate_instance(), indent=2))

    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)
