#!/usr/bin/env python3
"""
JSON Repair CLI - Command-line interface for neural JSON repair.

Usage:
    python json_repair_cli.py broken.json              # Print repaired to stdout
    python json_repair_cli.py broken.json -o fixed.json # Write to file
    python json_repair_cli.py -i broken.json           # Repair in-place
    python json_repair_cli.py -                        # Read from stdin
    python json_repair_cli.py broken.json --diff       # Show diff

Examples:
    echo '{"key" "value"}' | python json_repair_cli.py -
    python json_repair_cli.py config.json -o config_fixed.json
"""

import sys
import json
import argparse
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Repair broken JSON using neural denoiser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'input',
        help='Input JSON file (use - for stdin)',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)',
    )
    parser.add_argument(
        '-i', '--inline',
        action='store_true',
        help='Repair file in-place',
    )
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Show unified diff instead of output',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if JSON is valid, exit 0 if valid, 1 if repaired',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run model on (default: cpu)',
    )
    parser.add_argument(
        '--model',
        help='Path to model checkpoint (default: auto-detect)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print repair statistics',
    )

    args = parser.parse_args()

    # Import here to avoid slow startup if just checking args
    from json_repair_api import load_model, repair

    # Read input
    if args.input == '-':
        broken_json = sys.stdin.read()
        input_path = '<stdin>'
    else:
        input_path = args.input
        with open(input_path, 'r') as f:
            broken_json = f.read()

    # Check if already valid
    try:
        json.loads(broken_json)
        is_valid = True
    except json.JSONDecodeError:
        is_valid = False

    if args.check and is_valid:
        if args.verbose:
            print(f'{input_path}: valid JSON', file=sys.stderr)
        sys.exit(0)

    # Load model
    if args.verbose:
        print(f'Loading model...', file=sys.stderr)

    model, tokenizer = load_model(
        checkpoint_path=args.model,
        device=args.device,
    )

    # Repair
    if args.verbose:
        print(f'Repairing...', file=sys.stderr)

    from inference_repair import repair_json_full_denoise

    result = repair_json_full_denoise(
        model=model,
        tokenizer=tokenizer,
        broken_json=broken_json,
        sigma=0.2,
        device=args.device,
        max_len=model.config.block_size,
    )

    repaired_json = result.repaired

    # Check result
    if args.check:
        if result.success:
            if args.verbose:
                print(f'{input_path}: repaired ({result.tokens_changed} tokens changed)', file=sys.stderr)
            sys.exit(1)  # Exit 1 = needed repair
        else:
            print(f'{input_path}: repair failed', file=sys.stderr)
            sys.exit(2)

    # Show diff
    if args.diff:
        diff = difflib.unified_diff(
            broken_json.splitlines(keepends=True),
            repaired_json.splitlines(keepends=True),
            fromfile=f'{input_path} (broken)',
            tofile=f'{input_path} (repaired)',
        )
        sys.stdout.writelines(diff)
        if args.verbose:
            print(f'\nSuccess: {result.success}', file=sys.stderr)
            print(f'Tokens changed: {result.tokens_changed}', file=sys.stderr)
            print(f'Confidence: {result.confidence:.2f}', file=sys.stderr)
        return

    # Output
    if args.inline:
        if args.input == '-':
            print('Error: cannot use --inline with stdin', file=sys.stderr)
            sys.exit(1)
        with open(input_path, 'w') as f:
            f.write(repaired_json)
        if args.verbose:
            print(f'Repaired {input_path} in-place', file=sys.stderr)
    elif args.output:
        with open(args.output, 'w') as f:
            f.write(repaired_json)
        if args.verbose:
            print(f'Wrote repaired JSON to {args.output}', file=sys.stderr)
    else:
        print(repaired_json)

    # Print stats
    if args.verbose and not args.diff:
        print(f'Success: {result.success}', file=sys.stderr)
        print(f'Tokens changed: {result.tokens_changed}', file=sys.stderr)
        print(f'Confidence: {result.confidence:.2f}', file=sys.stderr)


if __name__ == '__main__':
    main()
