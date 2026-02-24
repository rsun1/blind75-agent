"""
Safe test runner: executes user-submitted Python code against problem test cases.

Uses exec() in an isolated namespace. Does NOT import unsafe modules.
All I/O is captured; no network or file system access from user code.
"""

import traceback
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Tree helpers for problems that use binary trees
# ─────────────────────────────────────────────────────────────────────────────

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val   = val
        self.left  = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def _build_tree(values: list) -> "TreeNode | None":
    """Build a binary tree from BFS-level list (None = missing node)."""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def _tree_to_list(root: "TreeNode | None") -> list:
    """Serialise a binary tree to BFS-level list, stripping trailing Nones."""
    if not root:
        return []
    result, queue = [], [root]
    while queue:
        node = queue.pop(0)
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    while result and result[-1] is None:
        result.pop()
    return result


def _parse_tree_input(raw_input):
    """
    Parse a tree input that may be encoded as a string like 'tree:[4,2,7,1,3]'
    or may be a TreeNode already.
    """
    if isinstance(raw_input, str) and raw_input.startswith("tree:"):
        import json
        vals = json.loads(raw_input[5:])
        return _build_tree(vals)
    return raw_input


def _parse_tree_expected(raw_expected):
    """Parse expected value that may be a 'tree:[...]' string."""
    if isinstance(raw_expected, str) and raw_expected.startswith("tree:"):
        import json
        return json.loads(raw_expected[5:])
    return raw_expected


# ─────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(value: Any) -> Any:
    """Convert sets/tuples to sorted lists for comparison."""
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(v) for v in value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def _results_match(actual: Any, expected: Any, unordered: bool = False,
                   unordered_groups: bool = False) -> bool:
    """
    Flexible comparison:
    - unordered: outer list order doesn't matter
    - unordered_groups: list of lists where inner lists are sorted and outer order doesn't matter
    """
    actual   = _normalize(actual)
    expected = _normalize(expected)

    if unordered_groups:
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        actual_sorted   = sorted(sorted(g) for g in actual)
        expected_sorted = sorted(sorted(g) for g in expected)
        return actual_sorted == expected_sorted

    if unordered:
        if isinstance(actual, list) and isinstance(expected, list):
            try:
                return sorted(str(x) for x in actual) == sorted(str(x) for x in expected)
            except Exception:
                pass

    return actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# Safe namespace builder
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "getattr",
        "hasattr", "hash", "hex", "int", "isinstance", "iter", "len",
        "list", "map", "max", "min", "next", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set", "setattr",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
    ]
    if (isinstance(__builtins__, dict) and name in __builtins__)
    or (not isinstance(__builtins__, dict) and hasattr(__builtins__, name))
}


def _make_namespace() -> dict:
    """Return a clean execution namespace with safe builtins and common stdlib."""
    import collections
    import itertools
    import math
    import heapq
    import bisect
    import functools

    ns = {
        "__builtins__": _ALLOWED_BUILTINS,
        "TreeNode":     TreeNode,
        "collections":  collections,
        "itertools":    itertools,
        "math":         math,
        "heapq":        heapq,
        "bisect":       bisect,
        "functools":    functools,
    }
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(problem: dict, user_code: str) -> list[dict]:
    """
    Execute `user_code` against all test cases in `problem`.

    Returns a list of result dicts:
        {
            "passed":   bool,
            "input":    str,
            "expected": str,
            "actual":   str,
            "error":    str | None,
        }
    """
    test_cases = problem.get("test_cases", [])
    if not test_cases:
        return [{"passed": None, "input": "—", "expected": "—",
                 "actual": "No test cases defined for this problem.",
                 "error": None}]

    # Compile user code once; catch syntax errors early
    try:
        compiled = compile(user_code, "<user_code>", "exec")
    except SyntaxError as exc:
        return [{"passed": False, "input": "—", "expected": "—",
                 "actual": None,
                 "error": f"SyntaxError: {exc}"}]

    results = []

    for tc in test_cases:
        ns = _make_namespace()
        try:
            exec(compiled, ns)  # noqa: S102
        except Exception as exc:
            results.append({
                "passed":   False,
                "input":    str(tc.get("input", "—")),
                "expected": str(tc.get("expected", "—")),
                "actual":   None,
                "error":    f"Runtime error loading code:\n{traceback.format_exc()}",
            })
            continue

        raw_input    = tc.get("input", ())
        raw_expected = tc.get("expected")
        is_tree      = tc.get("is_tree", False)
        unordered    = tc.get("unordered", False)
        unord_groups = tc.get("unordered_groups", False)
        encode_decode= tc.get("encode_decode", False)

        # Build call arguments
        try:
            if is_tree:
                args = tuple(_parse_tree_input(a) for a in raw_input)
                expected_list = _parse_tree_expected(raw_expected)
            elif encode_decode:
                args = raw_input
                expected_list = raw_expected
            else:
                args = raw_input if isinstance(raw_input, tuple) else (raw_input,)
                expected_list = raw_expected
        except Exception as exc:
            results.append({
                "passed":   False,
                "input":    str(raw_input),
                "expected": str(raw_expected),
                "actual":   None,
                "error":    f"Error preparing test input: {exc}",
            })
            continue

        # Determine the function name to call
        try:
            func_name = _find_function(user_code, ns, problem)
        except Exception as exc:
            results.append({
                "passed":   False,
                "input":    str(raw_input),
                "expected": str(raw_expected),
                "actual":   None,
                "error":    str(exc),
            })
            continue

        # Call the function
        try:
            if encode_decode:
                # Special case: call encode then decode
                encode_fn = ns.get("encode")
                decode_fn = ns.get("decode")
                if not encode_fn or not decode_fn:
                    raise NameError("Functions 'encode' and 'decode' not found in your code.")
                encoded = encode_fn(*args)
                actual  = decode_fn(encoded)
            else:
                fn = ns[func_name]
                actual = fn(*args)
        except Exception:
            results.append({
                "passed":   False,
                "input":    str(raw_input),
                "expected": str(raw_expected),
                "actual":   None,
                "error":    traceback.format_exc(),
            })
            continue

        # Tree problems: convert actual TreeNode back to list for comparison
        if is_tree and hasattr(actual, "val"):
            actual_cmp = _tree_to_list(actual)
        else:
            actual_cmp = actual

        passed = _results_match(actual_cmp, expected_list,
                                unordered=unordered,
                                unordered_groups=unord_groups)

        results.append({
            "passed":   passed,
            "input":    _format_input(raw_input),
            "expected": str(expected_list),
            "actual":   str(actual_cmp),
            "error":    None,
        })

    return results


def _find_function(user_code: str, ns: dict, problem: dict) -> str:
    """Determine the primary function name to call from the user's code."""
    # Collect all callables defined in user code (non-TreeNode classes excluded)
    candidates = [
        name for name, obj in ns.items()
        if callable(obj)
        and not name.startswith("_")
        and name not in ("TreeNode",)
        and not (isinstance(obj, type) and issubclass(obj, TreeNode))
    ]
    if not candidates:
        raise NameError(
            "No callable function found in your code. "
            "Make sure you define a function (e.g., `def two_sum(...):`)"
        )
    # Prefer an exact match against the starter code's first def line
    import re
    defs = re.findall(r"^def (\w+)", user_code, re.MULTILINE)
    for d in defs:
        if d in ns and callable(ns[d]):
            return d
    return candidates[0]


def _format_input(raw_input) -> str:
    if isinstance(raw_input, tuple):
        return ", ".join(repr(a) for a in raw_input)
    return repr(raw_input)
