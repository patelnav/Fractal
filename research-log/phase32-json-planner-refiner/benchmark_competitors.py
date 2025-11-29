#!/usr/bin/env python3
"""
Benchmark Neural JSON Repair vs Popular Heuristic Tools

Compares:
- Neural (ours): REINFORCE-trained denoiser (98.5% on training distribution)
- json-repair (Python): Most popular, 9.2M downloads/month
- fast-json-repair (Python/Rust): Performance-optimized
- jsonrepair (JavaScript): Feature-rich, handles many edge cases

Metrics:
- Parse Success: Does json.loads() succeed on output?
- Semantic Match: Does output match expected JSON (when known)?
- Speed: Milliseconds per repair
- Edit Distance: How many characters changed?
"""

import json
import time
import subprocess
import random
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from collections import defaultdict

# Competitors
import json_repair as json_repair_lib
try:
    import fast_json_repair
    HAS_FAST_JSON_REPAIR = True
except ImportError:
    HAS_FAST_JSON_REPAIR = False

# Our neural repair
from json_repair_api import load_model, repair as neural_repair


@dataclass
class TestCase:
    """A single test case for benchmarking."""
    broken: str                    # Broken JSON input
    expected: Optional[str]        # Expected correct JSON (if known)
    category: str                  # Error category
    description: str               # Human description


@dataclass
class RepairResult:
    """Result from a repair attempt."""
    output: str
    success: bool                  # json.loads() succeeded
    semantic_match: Optional[bool] # Matches expected (if known)
    time_ms: float
    edit_distance: int


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for one tool."""
    name: str
    total: int
    parse_success: int
    semantic_matches: int
    semantic_total: int           # Cases where expected was known
    total_time_ms: float
    total_edits: int
    by_category: Dict[str, Dict] = field(default_factory=dict)


def edit_distance(s1: str, s2: str) -> int:
    """Simple character-level edit distance."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# =============================================================================
# TEST CORPUS GENERATION
# =============================================================================

def generate_test_corpus() -> List[TestCase]:
    """Generate comprehensive test corpus across all error categories."""
    cases = []

    # Category 1: Structural Errors
    cases.extend([
        TestCase(
            broken='{"name": "Alice", "age": 30',
            expected='{"name": "Alice", "age": 30}',
            category='structural',
            description='Missing closing brace'
        ),
        TestCase(
            broken='{"items": [1, 2, 3}',
            expected='{"items": [1, 2, 3]}',
            category='structural',
            description='Missing closing bracket'
        ),
        TestCase(
            broken='"name": "Alice", "age": 30}',
            expected='{"name": "Alice", "age": 30}',
            category='structural',
            description='Missing opening brace'
        ),
        TestCase(
            broken='{"items": 1, 2, 3]}',
            expected='{"items": [1, 2, 3]}',
            category='structural',
            description='Missing opening bracket'
        ),
        TestCase(
            broken='{"a": {"b": {"c": 1}}',
            expected='{"a": {"b": {"c": 1}}}',
            category='structural',
            description='Missing nested closing brace'
        ),
        TestCase(
            broken='{"list": [[1, 2], [3, 4]}',
            expected='{"list": [[1, 2], [3, 4]]}',
            category='structural',
            description='Missing nested closing bracket'
        ),
    ])

    # Category 2: Punctuation Errors
    cases.extend([
        TestCase(
            broken='{"name": "Alice" "age": 30}',
            expected='{"name": "Alice", "age": 30}',
            category='punctuation',
            description='Missing comma between key-value pairs'
        ),
        TestCase(
            broken='{"items": [1 2 3]}',
            expected='{"items": [1, 2, 3]}',
            category='punctuation',
            description='Missing commas in array'
        ),
        TestCase(
            broken='{"name": "Alice",, "age": 30}',
            expected='{"name": "Alice", "age": 30}',
            category='punctuation',
            description='Extra comma'
        ),
        TestCase(
            broken='{"name": "Alice", "age": 30,}',
            expected='{"name": "Alice", "age": 30}',
            category='punctuation',
            description='Trailing comma in object'
        ),
        TestCase(
            broken='{"items": [1, 2, 3,]}',
            expected='{"items": [1, 2, 3]}',
            category='punctuation',
            description='Trailing comma in array'
        ),
        TestCase(
            broken='{"name" "Alice"}',
            expected='{"name": "Alice"}',
            category='punctuation',
            description='Missing colon'
        ),
        TestCase(
            broken='{"name":: "Alice"}',
            expected='{"name": "Alice"}',
            category='punctuation',
            description='Extra colon'
        ),
    ])

    # Category 3: Quote Errors
    cases.extend([
        TestCase(
            broken='{name: "Alice"}',
            expected='{"name": "Alice"}',
            category='quotes',
            description='Unquoted key'
        ),
        TestCase(
            broken='{"name": Alice}',
            expected='{"name": "Alice"}',
            category='quotes',
            description='Unquoted string value'
        ),
        TestCase(
            broken="{'name': 'Alice'}",
            expected='{"name": "Alice"}',
            category='quotes',
            description='Single quotes instead of double'
        ),
        TestCase(
            broken='{"name": "Alice"s house"}',
            expected='{"name": "Alice\'s house"}',
            category='quotes',
            description='Unescaped quote in value'
        ),
        TestCase(
            broken='{"name": "Alice}',
            expected='{"name": "Alice"}',
            category='quotes',
            description='Missing closing quote'
        ),
        TestCase(
            broken='{"name: "Alice"}',
            expected='{"name": "Alice"}',
            category='quotes',
            description='Missing quote after key'
        ),
    ])

    # Category 4: Value Errors
    cases.extend([
        TestCase(
            broken='{"active": True}',
            expected='{"active": true}',
            category='values',
            description='Python True instead of true'
        ),
        TestCase(
            broken='{"active": False}',
            expected='{"active": false}',
            category='values',
            description='Python False instead of false'
        ),
        TestCase(
            broken='{"value": None}',
            expected='{"value": null}',
            category='values',
            description='Python None instead of null'
        ),
        TestCase(
            broken='{"name": }',
            expected='{"name": ""}',
            category='values',
            description='Missing value'
        ),
        TestCase(
            broken='{"name": "Alice", "age":}',
            expected='{"name": "Alice", "age": null}',
            category='values',
            description='Missing value at end'
        ),
        TestCase(
            broken='{"text": "Hello worl',
            expected='{"text": "Hello world"}',
            category='values',
            description='Truncated string'
        ),
    ])

    # Category 5: LLM-Specific Errors
    cases.extend([
        TestCase(
            broken='```json\n{"name": "Alice"}\n```',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON in markdown code fence'
        ),
        TestCase(
            broken='Here is the JSON:\n{"name": "Alice"}\nThat is the data.',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON with surrounding prose'
        ),
        TestCase(
            broken='{"a": 1}{"b": 2}',
            expected='[{"a": 1}, {"b": 2}]',
            category='llm_specific',
            description='Concatenated JSON objects'
        ),
        TestCase(
            broken='{"name": "Alice"} // user data',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON with line comment'
        ),
        TestCase(
            broken='{"name": "Alice"} /* user data */',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON with block comment'
        ),
        TestCase(
            broken='{"name": "Alice"}\n\nHope this helps!',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON with trailing text'
        ),
        TestCase(
            broken='Sure! Here you go:\n```\n{"name": "Alice"}\n```',
            expected='{"name": "Alice"}',
            category='llm_specific',
            description='JSON in plain code fence with intro'
        ),
    ])

    # Category 6: Complex/Multi-Error
    cases.extend([
        TestCase(
            broken='{"name": "Alice" "age": 30',
            expected='{"name": "Alice", "age": 30}',
            category='multi_error',
            description='Missing comma AND missing brace'
        ),
        TestCase(
            broken='{name: Alice, age: 30}',
            expected='{"name": "Alice", "age": 30}',
            category='multi_error',
            description='Multiple unquoted keys and values'
        ),
        TestCase(
            broken="{'items': [1 2 3}",
            expected='{"items": [1, 2, 3]}',
            category='multi_error',
            description='Single quotes + missing commas + wrong bracket'
        ),
        TestCase(
            broken='{"a": 1, "b": {"c": 2 "d": 3}}',
            expected='{"a": 1, "b": {"c": 2, "d": 3}}',
            category='multi_error',
            description='Missing comma in nested object'
        ),
        TestCase(
            broken='```json\n{"active": True, "count": None}\n```',
            expected='{"active": true, "count": null}',
            category='multi_error',
            description='Code fence + Python constants'
        ),
    ])

    # Add more random variations using synthetic generation
    cases.extend(generate_synthetic_cases(50))

    return cases


def generate_synthetic_cases(n: int) -> List[TestCase]:
    """Generate random synthetic test cases."""
    cases = []

    templates = [
        ('{"name": "Alice", "age": 30}', 'Simple object'),
        ('{"items": [1, 2, 3], "total": 6}', 'Object with array'),
        ('{"user": {"id": 1, "name": "Bob"}}', 'Nested object'),
        ('[{"a": 1}, {"b": 2}]', 'Array of objects'),
        ('{"x": true, "y": false, "z": null}', 'Boolean and null values'),
    ]

    corruptions = [
        (lambda s: s[:-1], 'remove_last_char'),
        (lambda s: s.replace(',', '', 1), 'remove_first_comma'),
        (lambda s: s.replace('"', "'"), 'single_quotes'),
        (lambda s: s.replace('true', 'True').replace('false', 'False'), 'python_bools'),
        (lambda s: s.replace('null', 'None'), 'python_none'),
        (lambda s: '```json\n' + s + '\n```', 'markdown_fence'),
    ]

    random.seed(42)
    for _ in range(n):
        template, desc = random.choice(templates)
        corrupt_fn, corrupt_name = random.choice(corruptions)

        broken = corrupt_fn(template)
        cases.append(TestCase(
            broken=broken,
            expected=template,
            category='synthetic',
            description=f'{desc} + {corrupt_name}'
        ))

    return cases


# =============================================================================
# REPAIR FUNCTIONS (wrappers for each tool)
# =============================================================================

def repair_with_json_repair(broken: str) -> Tuple[str, float]:
    """Repair using json-repair library."""
    start = time.perf_counter()
    try:
        result = json_repair_lib.repair_json(broken)
    except Exception:
        result = broken
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def repair_with_fast_json_repair(broken: str) -> Tuple[str, float]:
    """Repair using fast-json-repair library."""
    if not HAS_FAST_JSON_REPAIR:
        return broken, 0.0

    start = time.perf_counter()
    try:
        result = fast_json_repair.repair_json(broken)
    except Exception:
        result = broken
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def repair_with_jsonrepair_js(broken: str) -> Tuple[str, float]:
    """Repair using jsonrepair (Node.js) via subprocess."""
    # Create temp file with broken JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(broken)
        temp_path = f.name

    try:
        start = time.perf_counter()
        result = subprocess.run(
            ['node', '-e', f'''
const {{ jsonrepair }} = require('jsonrepair');
const fs = require('fs');
const input = fs.readFileSync("{temp_path}", "utf8");
try {{
    console.log(jsonrepair(input));
}} catch (e) {{
    console.log(input);
}}
'''],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(__file__),
        )
        elapsed = (time.perf_counter() - start) * 1000
        output = result.stdout.strip() if result.returncode == 0 else broken
    except Exception:
        output = broken
        elapsed = 0.0
    finally:
        os.unlink(temp_path)

    return output, elapsed


# Neural repair wrapper
_neural_model = None
_neural_tokenizer = None

def repair_with_neural(broken: str) -> Tuple[str, float]:
    """Repair using our neural denoiser."""
    global _neural_model, _neural_tokenizer

    if _neural_model is None:
        _neural_model, _neural_tokenizer = load_model()

    start = time.perf_counter()
    try:
        result = neural_repair(broken, _neural_model, _neural_tokenizer)
    except Exception:
        result = broken
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_single_test(
    test: TestCase,
    repair_fn: Callable[[str], Tuple[str, float]],
) -> RepairResult:
    """Run a single repair test."""
    output, time_ms = repair_fn(test.broken)

    # Check parse success
    try:
        json.loads(output)
        success = True
    except json.JSONDecodeError:
        success = False

    # Check semantic match (if expected is known)
    semantic_match = None
    if test.expected is not None:
        try:
            expected_obj = json.loads(test.expected)
            output_obj = json.loads(output) if success else None
            semantic_match = (expected_obj == output_obj)
        except:
            semantic_match = False

    # Edit distance
    edits = edit_distance(test.broken, output)

    return RepairResult(
        output=output,
        success=success,
        semantic_match=semantic_match,
        time_ms=time_ms,
        edit_distance=edits,
    )


def run_benchmark(
    corpus: List[TestCase],
    repair_fn: Callable[[str], Tuple[str, float]],
    name: str,
) -> BenchmarkResult:
    """Run benchmark on entire corpus."""
    result = BenchmarkResult(
        name=name,
        total=len(corpus),
        parse_success=0,
        semantic_matches=0,
        semantic_total=0,
        total_time_ms=0.0,
        total_edits=0,
    )

    for test in corpus:
        r = run_single_test(test, repair_fn)

        if r.success:
            result.parse_success += 1

        if r.semantic_match is not None:
            result.semantic_total += 1
            if r.semantic_match:
                result.semantic_matches += 1

        result.total_time_ms += r.time_ms
        result.total_edits += r.edit_distance

        # Track by category
        cat = test.category
        if cat not in result.by_category:
            result.by_category[cat] = {
                'total': 0, 'parse_success': 0, 'semantic_matches': 0,
                'semantic_total': 0, 'time_ms': 0.0, 'edits': 0
            }

        result.by_category[cat]['total'] += 1
        if r.success:
            result.by_category[cat]['parse_success'] += 1
        if r.semantic_match is not None:
            result.by_category[cat]['semantic_total'] += 1
            if r.semantic_match:
                result.by_category[cat]['semantic_matches'] += 1
        result.by_category[cat]['time_ms'] += r.time_ms
        result.by_category[cat]['edits'] += r.edit_distance

    return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice table."""
    print()
    print('=' * 80)
    print('JSON REPAIR BENCHMARK - Neural vs Heuristic')
    print('=' * 80)
    print()

    # Header
    print(f"{'Tool':<20} {'Parse%':>10} {'Semantic%':>10} {'Speed(ms)':>10} {'Edits':>10}")
    print('-' * 60)

    for r in results:
        parse_pct = 100 * r.parse_success / r.total if r.total > 0 else 0
        sem_pct = 100 * r.semantic_matches / r.semantic_total if r.semantic_total > 0 else 0
        avg_time = r.total_time_ms / r.total if r.total > 0 else 0
        avg_edits = r.total_edits / r.total if r.total > 0 else 0

        print(f"{r.name:<20} {parse_pct:>9.1f}% {sem_pct:>9.1f}% {avg_time:>10.2f} {avg_edits:>10.1f}")

    print()
    print('-' * 80)
    print('BY CATEGORY (Parse Success %)')
    print('-' * 80)

    # Collect all categories
    categories = set()
    for r in results:
        categories.update(r.by_category.keys())
    categories = sorted(categories)

    # Header
    header = f"{'Category':<15}"
    for r in results:
        header += f" {r.name[:12]:>12}"
    print(header)
    print('-' * (15 + 13 * len(results)))

    for cat in categories:
        row = f"{cat:<15}"
        for r in results:
            if cat in r.by_category:
                c = r.by_category[cat]
                pct = 100 * c['parse_success'] / c['total'] if c['total'] > 0 else 0
                row += f" {pct:>11.1f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print()


def main():
    print("Generating test corpus...")
    corpus = generate_test_corpus()
    print(f"Generated {len(corpus)} test cases")
    print()

    # Count by category
    by_cat = defaultdict(int)
    for t in corpus:
        by_cat[t.category] += 1
    print("Test cases by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    print()

    results = []

    # Benchmark json-repair
    print("Benchmarking: json-repair (Python)...")
    results.append(run_benchmark(corpus, repair_with_json_repair, 'json-repair'))

    # Benchmark fast-json-repair
    if HAS_FAST_JSON_REPAIR:
        print("Benchmarking: fast-json-repair (Rust)...")
        results.append(run_benchmark(corpus, repair_with_fast_json_repair, 'fast-json-repair'))

    # Benchmark jsonrepair (JS)
    print("Benchmarking: jsonrepair (JS)...")
    results.append(run_benchmark(corpus, repair_with_jsonrepair_js, 'jsonrepair-js'))

    # Benchmark neural
    print("Benchmarking: Neural (ours)...")
    results.append(run_benchmark(corpus, repair_with_neural, 'Neural'))

    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
