#!/usr/bin/env python3
"""Lambda Labs API helper for managing multiple GPU instances in parallel."""

import json
import subprocess
import sys
import time
from pathlib import Path

API_KEY_FILE = Path(__file__).parent / ".lambda_api_key"
API_BASE = "https://cloud.lambdalabs.com/api/v1"
STATE_FILE = Path(__file__).parent / ".lambda_instances.json"
OLD_STATE_FILE = Path(__file__).parent / ".lambda_instance_id"  # Migration
SSH_CONFIG = Path.home() / ".ssh" / "config"
PROJECT_ROOT = Path(__file__).parent

# Default instance name when none specified
DEFAULT_INSTANCE = "default"

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


# ============================================================================
# Multi-instance state management
# ============================================================================

def load_instances():
    """Load all instance states from JSON file."""
    # Migrate from old single-instance file if exists
    if OLD_STATE_FILE.exists() and not STATE_FILE.exists():
        old_id = OLD_STATE_FILE.read_text().strip()
        if old_id:
            # Direct write to avoid recursion
            STATE_FILE.write_text(json.dumps({DEFAULT_INSTANCE: {"id": old_id}}, indent=2))
        OLD_STATE_FILE.unlink()

    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_instances(instances):
    """Save all instance states to JSON file."""
    STATE_FILE.write_text(json.dumps(instances, indent=2))


def get_instance(name=None):
    """Get a specific instance by name."""
    name = name or DEFAULT_INSTANCE
    instances = load_instances()
    return instances.get(name, {})


def save_instance(name, data):
    """Save/update a specific instance."""
    name = name or DEFAULT_INSTANCE
    instances = load_instances()
    instances[name] = data
    save_instances(instances)


def delete_instance(name):
    """Remove an instance from state."""
    name = name or DEFAULT_INSTANCE
    instances = load_instances()
    if name in instances:
        del instances[name]
        save_instances(instances)
        return True
    return False


def get_ssh_alias(name=None):
    """Get SSH alias for an instance."""
    name = name or DEFAULT_INSTANCE
    if name == DEFAULT_INSTANCE:
        return "lambda"
    return f"lambda-{name}"


# ============================================================================
# Instance operations
# ============================================================================

def get_instance_status(name=None):
    """Get status of a named instance."""
    name = name or DEFAULT_INSTANCE
    inst = get_instance(name)
    instance_id = inst.get("id")

    if not instance_id:
        return {"error": f"No instance '{name}'. Run 'launch {name}' first."}

    data = api_get(f"instances/{instance_id}")
    if "error" in data:
        return {"name": name, "status": "terminated", "ip": None}

    instance = data.get("data", {})
    return {
        "name": name,
        "id": instance_id,
        "status": instance.get("status"),
        "ip": instance.get("ip"),
        "instance_type": instance.get("instance_type", {}).get("name"),
        "region": instance.get("region", {}).get("name"),
        "ssh_alias": get_ssh_alias(name),
    }


def launch_instance(name=None, instance_type=None, region=None):
    """Launch a new GPU instance with a name."""
    name = name or DEFAULT_INSTANCE

    # Check if already exists
    existing = get_instance(name)
    if existing.get("id"):
        status = get_instance_status(name)
        if status.get("status") not in ["terminated", None]:
            return {"error": f"Instance '{name}' already exists (status: {status.get('status')}). Terminate it first or use a different name."}

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

    lambda_name = f"fractal-{name}" if name != DEFAULT_INSTANCE else "fractal-gpu"
    print(f"Launching '{name}' ({selected['name']}) in {target_region} @ ${selected['price']:.2f}/hr...")

    # Launch
    result = api_post("instance-operations/launch", {
        "region_name": target_region,
        "instance_type_name": selected["name"],
        "ssh_key_names": ssh_keys[:1],
        "name": lambda_name,
    })

    if "error" in result:
        return result

    instance_ids = result.get("data", {}).get("instance_ids", [])
    if not instance_ids:
        return {"error": "No instance ID returned", "response": result}

    instance_id = instance_ids[0]
    save_instance(name, {
        "id": instance_id,
        "instance_type": selected["name"],
        "region": target_region,
        "price": selected["price"],
    })

    return {
        "name": name,
        "id": instance_id,
        "instance_type": selected["name"],
        "region": target_region,
        "price": selected["price"],
        "ssh_alias": get_ssh_alias(name),
        "message": f"Instance '{name}' launching. Run 'wait {name}' to wait for SSH."
    }


def wait_for_ready(name=None, timeout=300):
    """Wait for a named instance to be ready for SSH."""
    name = name or DEFAULT_INSTANCE
    inst = get_instance(name)
    instance_id = inst.get("id")

    if not instance_id:
        return {"error": f"No instance '{name}'. Run 'launch {name}' first."}

    print(f"Waiting for instance '{name}' to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        status = get_instance_status(name)
        print(f"  [{name}] Status: {status.get('status')}, IP: {status.get('ip')}")

        if status.get("status") == "active" and status.get("ip"):
            # Update stored IP
            inst["ip"] = status["ip"]
            save_instance(name, inst)

            # Test SSH connection
            print(f"  [{name}] Testing SSH connection...")
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                     f"ubuntu@{status['ip']}", "echo ready"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    print(f"\nInstance '{name}' ready!")
                    print(f"  SSH: ssh ubuntu@{status['ip']}")
                    print(f"  Alias: ssh {get_ssh_alias(name)}")
                    return status
            except subprocess.TimeoutExpired:
                pass

        time.sleep(10)

    return {"error": f"Timeout waiting for instance '{name}'"}


def terminate_instance(name=None):
    """Terminate a named instance."""
    name = name or DEFAULT_INSTANCE
    inst = get_instance(name)
    instance_id = inst.get("id")

    if not instance_id:
        return {"error": f"No instance '{name}'. Nothing to terminate."}

    print(f"Terminating instance '{name}' ({instance_id})...")
    result = api_post("instance-operations/terminate", {
        "instance_ids": [instance_id]
    })

    if "error" not in result:
        delete_instance(name)
        # Clean up SSH config
        remove_ssh_config(name)

    return result


def list_instances():
    """List all tracked instances with their status."""
    instances = load_instances()

    if not instances:
        # Also check API for any running instances
        data = api_get("instances")
        api_instances = data.get("data", [])
        if api_instances:
            result = []
            for inst in api_instances:
                result.append({
                    "name": "(untracked)",
                    "id": inst["id"],
                    "lambda_name": inst.get("name"),
                    "status": inst.get("status"),
                    "ip": inst.get("ip"),
                    "type": inst.get("instance_type", {}).get("name"),
                    "region": inst.get("region", {}).get("name"),
                })
            return result
        return {"message": "No instances tracked or running"}

    result = []
    for name, inst in instances.items():
        status = get_instance_status(name)
        result.append({
            "name": name,
            "id": inst.get("id"),
            "status": status.get("status"),
            "ip": status.get("ip"),
            "type": inst.get("instance_type"),
            "region": inst.get("region"),
            "price": inst.get("price"),
            "ssh_alias": get_ssh_alias(name),
        })

    return result


# ============================================================================
# SSH config management
# ============================================================================

def setup_ssh_config(name=None):
    """Setup SSH config for a named instance."""
    name = name or DEFAULT_INSTANCE
    status = get_instance_status(name)
    if not status.get("ip"):
        return {"error": f"No active instance '{name}' with IP"}

    ip = status["ip"]
    ssh_alias = get_ssh_alias(name)

    # SSH config block
    ssh_block = f"""
# Lambda Labs GPU Instance: {name} (auto-managed by lambda_helper.py)
Host {ssh_alias}
    HostName {ip}
    User ubuntu
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
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

    # Remove existing block for this instance
    import re
    pattern = rf'\n# Lambda Labs GPU Instance: {re.escape(name)} \(auto-managed.*?(?=\n# Lambda Labs GPU Instance:|\nHost (?!{re.escape(ssh_alias)})|\Z)'
    config_content = re.sub(pattern, '', config_content, flags=re.DOTALL)

    # Also remove standalone Host block for this alias
    pattern2 = rf'\nHost {re.escape(ssh_alias)}\n.*?(?=\nHost |\n# |\Z)'
    config_content = re.sub(pattern2, '', config_content, flags=re.DOTALL)

    # Append new config
    config_content = config_content.rstrip() + ssh_block

    SSH_CONFIG.write_text(config_content)

    print(f"SSH config updated for '{name}'. Connect with: ssh {ssh_alias}")
    return {"name": name, "ssh_alias": ssh_alias, "ip": ip}


def remove_ssh_config(name=None):
    """Remove SSH config for a named instance."""
    name = name or DEFAULT_INSTANCE
    ssh_alias = get_ssh_alias(name)

    if not SSH_CONFIG.exists():
        return

    config_content = SSH_CONFIG.read_text()

    import re
    # Remove the block for this instance
    pattern = rf'\n# Lambda Labs GPU Instance: {re.escape(name)} \(auto-managed.*?(?=\n# Lambda Labs GPU Instance:|\nHost (?!{re.escape(ssh_alias)})|\Z)'
    config_content = re.sub(pattern, '', config_content, flags=re.DOTALL)

    SSH_CONFIG.write_text(config_content)


def setup_all_ssh_configs():
    """Setup SSH configs for all active instances."""
    instances = load_instances()
    results = []
    for name in instances:
        result = setup_ssh_config(name)
        results.append(result)
    return results


# ============================================================================
# Remote operations
# ============================================================================

def run_remote(name, command):
    """Run a command on a named remote instance."""
    name = name or DEFAULT_INSTANCE
    status = get_instance_status(name)
    if not status.get("ip"):
        return {"error": f"No active instance '{name}' with IP"}

    ssh_alias = get_ssh_alias(name)
    result = subprocess.run(
        ["ssh", ssh_alias, command],
        capture_output=True, text=True
    )

    return {
        "name": name,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def sync_code(name=None, subdir=None):
    """Sync project code to a named remote instance."""
    name = name or DEFAULT_INSTANCE
    status = get_instance_status(name)
    if not status.get("ip"):
        return {"error": f"No active instance '{name}' with IP"}

    ssh_alias = get_ssh_alias(name)
    remote_dir = "~/Fractal"

    subprocess.run(["ssh", ssh_alias, f"mkdir -p {remote_dir}"], check=True)

    if subdir:
        local_path = PROJECT_ROOT / subdir
        remote_path = f"{remote_dir}/{subdir}"
        subprocess.run(["ssh", ssh_alias, f"mkdir -p {remote_path}"], check=True)
    else:
        local_path = PROJECT_ROOT
        remote_path = remote_dir

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
        f"{ssh_alias}:{remote_path}/"
    ]

    print(f"Syncing {local_path} -> {ssh_alias}:{remote_path}")
    result = subprocess.run(cmd)

    return {"name": name, "synced": str(local_path), "to": remote_path, "returncode": result.returncode}


# ============================================================================
# Compound operations
# ============================================================================

def launch_and_wait(name=None, instance_type=None):
    """Launch instance, wait for SSH, and setup SSH config. All-in-one blocking call."""
    name = name or DEFAULT_INSTANCE

    # Launch
    result = launch_instance(name=name, instance_type=instance_type)
    if "error" in result:
        return result
    print(json.dumps(result, indent=2))

    # Wait for SSH
    status = wait_for_ready(name)
    if "error" in status:
        return status

    # Setup SSH config
    ssh_result = setup_ssh_config(name)
    if "error" in ssh_result:
        return ssh_result

    ssh_alias = get_ssh_alias(name)
    print(f"\n✓ Instance '{name}' ready! Run: ssh {ssh_alias}")
    return status


def terminate_all():
    """Terminate all tracked instances."""
    instances = load_instances()
    results = []
    for name in list(instances.keys()):
        print(f"Terminating '{name}'...")
        result = terminate_instance(name)
        results.append({"name": name, "result": result})
    return results


# ============================================================================
# CLI
# ============================================================================

def print_help():
    print("""Lambda Labs Multi-Instance GPU Manager

Usage: python lambda_helper.py <command> [instance_name] [options]

Instance names are optional - defaults to 'default' (alias: 'lambda')
Named instances get aliases like 'lambda-mlp', 'lambda-transformer', etc.

Instance Lifecycle:
  list              - List all tracked instances with status
  available         - Show available instance types
  launch [name]     - Launch a new GPU instance
  wait [name]       - Wait for instance to be SSH-ready
  start [name]      - Launch + wait + setup-ssh (ALL-IN-ONE)
  status [name]     - Get instance status
  terminate [name]  - Terminate instance
  terminate-all     - Terminate ALL tracked instances

SSH & Remote:
  setup-ssh [name]  - Configure SSH alias for instance
  setup-ssh-all     - Configure SSH aliases for all instances
  run <name> <cmd>  - Run command on named instance
  sync [name] [dir] - Sync code to instance

Parallel Training Example:
  # Launch 3 instances for parallel training
  python lambda_helper.py start mlp
  python lambda_helper.py start transformer
  python lambda_helper.py start contrastive

  # Each gets its own SSH alias
  ssh lambda-mlp
  ssh lambda-transformer
  ssh lambda-contrastive

  # Sync code to all (run in parallel)
  python lambda_helper.py sync mlp diarize &
  python lambda_helper.py sync transformer diarize &
  python lambda_helper.py sync contrastive diarize &
  wait

  # Run training on each
  ssh lambda-mlp "cd ~/Fractal/diarize && ./train_mlp.sh"
  ssh lambda-transformer "cd ~/Fractal/diarize && ./train_transformer.sh"

  # Terminate all when done
  python lambda_helper.py terminate-all

Single Instance (backward compatible):
  python lambda_helper.py start      # Uses 'default', alias 'lambda'
  ssh lambda
  python lambda_helper.py terminate
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        result = list_instances()
        if isinstance(result, list):
            for inst in result:
                status_str = inst.get('status', 'unknown')
                price_str = f"${inst.get('price', 0):.2f}/hr" if inst.get('price') else ""
                print(f"  {inst.get('name', '?'):15} {status_str:10} {inst.get('ip', 'no-ip'):15} {inst.get('ssh_alias', ''):20} {price_str}")
        else:
            print(json.dumps(result, indent=2))

    elif cmd == "available":
        available = get_available_instances()
        for inst in available:
            print(f"{inst['name']}: {inst['description']} @ ${inst['price']:.2f}/hr")
            print(f"  Regions: {', '.join(inst['regions'])}")

    elif cmd == "launch":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        instance_type = sys.argv[3] if len(sys.argv) > 3 else None
        result = launch_instance(name=name, instance_type=instance_type)
        print(json.dumps(result, indent=2))

    elif cmd == "wait":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        result = wait_for_ready(name)
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "start":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        instance_type = sys.argv[3] if len(sys.argv) > 3 else None
        result = launch_and_wait(name=name, instance_type=instance_type)
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "status":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(get_instance_status(name), indent=2))

    elif cmd == "setup-ssh":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        result = setup_ssh_config(name)
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "setup-ssh-all":
        results = setup_all_ssh_configs()
        print(json.dumps(results, indent=2))

    elif cmd == "run":
        if len(sys.argv) < 4:
            print("Usage: python lambda_helper.py run <instance_name> <command>")
            print("Example: python lambda_helper.py run mlp 'nvidia-smi'")
            sys.exit(1)
        name = sys.argv[2]
        remote_cmd = " ".join(sys.argv[3:])
        result = run_remote(name, remote_cmd)
        if result.get("stdout"):
            print(result["stdout"])
        if result.get("stderr"):
            print(result["stderr"], file=sys.stderr)
        sys.exit(result.get("returncode", 0))

    elif cmd == "sync":
        # Parse: sync [name] [subdir] or sync [subdir] (assume default)
        if len(sys.argv) == 2:
            name, subdir = None, None
        elif len(sys.argv) == 3:
            # Could be name or subdir - check if it looks like a path
            arg = sys.argv[2]
            if "/" in arg or arg in ["diarize", "research-log"]:
                name, subdir = None, arg
            else:
                name, subdir = arg, None
        else:
            name, subdir = sys.argv[2], sys.argv[3]

        result = sync_code(name, subdir)
        if "error" in result:
            print(json.dumps(result, indent=2))
            sys.exit(1)

    elif cmd == "terminate":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(terminate_instance(name), indent=2))

    elif cmd == "terminate-all":
        results = terminate_all()
        print(json.dumps(results, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)
