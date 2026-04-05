"""
Safe test runner: executes user-submitted Python code against problem test cases.

Uses exec() in an isolated namespace. Does NOT import unsafe modules.
Runs each test in a separate process with a timeout; if the process does not finish
in time it is terminated so infinite loops cannot hang or consume the app.
"""

import builtins as _builtins_mod
import traceback
import multiprocessing
from typing import Any

# Max seconds per test case; if exceeded, the worker process is killed.
TEST_TIMEOUT_SECONDS = 5


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
# Linked-list helpers for problems that use singly linked lists
# ─────────────────────────────────────────────────────────────────────────────

class ListNode:
    def __init__(self, val=0, next=None):
        self.val  = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


def _build_linked_list(values: list) -> "ListNode | None":
    """Build a singly linked list from a list of values."""
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def _build_linked_list_with_cycle(values: list, cycle_pos: int) -> "ListNode | None":
    """Build a linked list where the tail connects to the node at cycle_pos (-1 = no cycle)."""
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if 0 <= cycle_pos < len(nodes):
        nodes[-1].next = nodes[cycle_pos]
    return nodes[0]


def _linked_list_to_list(head: "ListNode | None") -> list:
    """Serialise a linked list to a plain list, with cycle detection."""
    result = []
    seen = set()
    while head and id(head) not in seen:
        seen.add(id(head))
        result.append(head.val)
        head = head.next
    return result


def _parse_list_input(raw_input):
    """Parse a linked-list input: 'list:[1,2,3]' → ListNode chain, or recurse for lists of lists."""
    if isinstance(raw_input, str) and raw_input.startswith("list:"):
        import json
        vals = json.loads(raw_input[5:])
        return _build_linked_list(vals)
    if isinstance(raw_input, list):
        return [_parse_list_input(item) for item in raw_input]
    return raw_input


def _parse_list_expected(raw_expected):
    """Parse expected value: 'list:[1,2,3]' → [1,2,3]."""
    if isinstance(raw_expected, str) and raw_expected.startswith("list:"):
        import json
        return json.loads(raw_expected[5:])
    return raw_expected


def _find_node(root: "TreeNode | None", val: int) -> "TreeNode | None":
    """Find a TreeNode by value (BFS) — used for LCA-style problems."""
    if not root:
        return None
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.val == val:
            return node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return None


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
    name: getattr(_builtins_mod, name)
    for name in [
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "getattr",
        "hasattr", "hash", "hex", "int", "isinstance", "iter", "len",
        "list", "map", "max", "min", "next", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set", "setattr",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
        # Required for `class` definitions in user code
        "__build_class__",
    ]
    if hasattr(_builtins_mod, name)
}

# Modules that user code is allowed to import (safe for DSA practice)
_IMPORT_WHITELIST = frozenset({
    "collections", "itertools", "math", "heapq", "bisect", "functools",
    "re", "typing", "dataclasses", "copy", "string", "random", "json",
    "decimal", "fractions", "operator", "array",
})

_builtin_import = getattr(_builtins_mod, "__import__")

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow only whitelisted modules so user code can e.g. 'from collections import Counter'."""
    if level != 0:
        raise ImportError("Relative imports are not allowed in practice code.")
    # Top-level module only (e.g. "collections" from "collections.abc")
    top = name.split(".")[0]
    if top not in _IMPORT_WHITELIST:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed: {sorted(_IMPORT_WHITELIST)}")
    return _builtin_import(name, globals, locals, fromlist, level)

_ALLOWED_BUILTINS["__import__"] = _safe_import


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
        "__name__":     "__user_code__",   # required for `class` definitions
        "TreeNode":     TreeNode,
        "ListNode":     ListNode,
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

def _run_class_test(user_code: str, tc: dict, ns: dict, result_queue: multiprocessing.Queue) -> None:
    """Run a class-based test (e.g. MinStack, LRUCache, Trie)."""
    class_name = tc.get("class_name", "")
    operations = tc.get("operations", [])
    arguments  = tc.get("arguments", [])
    expected   = tc.get("expected", [])

    cls = ns.get(class_name)
    if cls is None:
        result_queue.put({
            "passed": False,
            "input":  f"operations: {operations}",
            "expected": str(expected),
            "actual": None,
            "error":  f"Class '{class_name}' not found in your code.",
        })
        return

    try:
        obj = cls(*arguments[0]) if arguments and arguments[0] else cls()
    except Exception as e:
        result_queue.put({
            "passed": False,
            "input":  f"operations: {operations}",
            "expected": str(expected),
            "actual": None,
            "error":  f"Error creating {class_name}: {e}",
        })
        return

    results = [None]  # constructor always returns None
    try:
        for op, args in zip(operations[1:], arguments[1:]):
            method = getattr(obj, op, None)
            if method is None:
                raise AttributeError(f"Method '{op}' not found on {class_name}")
            results.append(method(*args))
    except Exception as e:
        err_tb = traceback.format_exception(type(e), e, e.__traceback__) if hasattr(e, "__traceback__") else [str(e)]
        result_queue.put({
            "passed": False,
            "input":  f"operations: {operations}",
            "expected": str(expected),
            "actual": str(results),
            "error":  "".join(err_tb),
        })
        return

    # Compare results — only check positions where expected is not None
    first_mismatch = None
    for i, (act, exp) in enumerate(zip(results, expected)):
        if exp is not None and act != exp:
            first_mismatch = (i, operations[i] if i < len(operations) else "?", act, exp)
            break

    if first_mismatch is None:
        result_queue.put({
            "passed": True,
            "input":  f"operations: {operations[:8]}{'…' if len(operations)>8 else ''}",
            "expected": str(expected),
            "actual": str(results),
            "error":  None,
        })
    else:
        idx, op, act, exp = first_mismatch
        result_queue.put({
            "passed": False,
            "input":  f"operations: {operations[:8]}{'…' if len(operations)>8 else ''}",
            "expected": str(expected),
            "actual": str(results),
            "error":  f"After {op}({arguments[idx] if idx < len(arguments) else ''}): expected {exp}, got {act}",
        })


def _run_one_test_in_process(user_code: str, tc: dict, problem: dict, result_queue: multiprocessing.Queue) -> None:
    """
    Run a single test case in the current (child) process.
    Puts one result dict on result_queue, or does not put if the process is killed.
    """
    raw_input    = tc.get("input", ())
    raw_expected = tc.get("expected")
    is_tree      = tc.get("is_tree", False)
    is_list      = tc.get("is_list", False)
    unordered    = tc.get("unordered", False)
    unord_groups = tc.get("unordered_groups", False)
    encode_decode= tc.get("encode_decode", False)
    class_test   = tc.get("class_test", False)
    check_head   = tc.get("check_head", False)
    cycle_pos    = tc.get("cycle_pos", None)

    try:
        compiled = compile(user_code, "<user_code>", "exec")
    except SyntaxError as exc:
        result_queue.put({
            "passed":   False,
            "input":    str(raw_input),
            "expected": str(raw_expected),
            "actual":   None,
            "error":    f"SyntaxError: {exc}",
        })
        return

    ns = _make_namespace()
    try:
        exec(compiled, ns)  # noqa: S102
    except Exception:
        result_queue.put({
            "passed":   False,
            "input":    str(raw_input),
            "expected": str(raw_expected),
            "actual":   None,
            "error":    f"Runtime error loading code:\n{traceback.format_exc()}",
        })
        return

    # ── Class-based tests (Trie, LRU Cache, etc.) ──
    if class_test:
        _run_class_test(user_code, tc, ns, result_queue)
        return

    try:
        if is_tree:
            args = tuple(_parse_tree_input(a) for a in raw_input)
            # Resolve "treenode:N" references (for LCA-style problems)
            first_tree = args[0] if args and hasattr(args[0], "val") else None
            args_list = list(args)
            for i, raw in enumerate(raw_input):
                if isinstance(raw, str) and raw.startswith("treenode:"):
                    target_val = int(raw[9:])
                    args_list[i] = _find_node(first_tree, target_val)
            args = tuple(args_list)
            expected_list = _parse_tree_expected(raw_expected)
        elif is_list:
            if cycle_pos is not None and raw_input:
                # Build linked list with a cycle for cycle-detection problems
                import json
                raw0 = raw_input[0]
                vals = json.loads(raw0[5:]) if isinstance(raw0, str) and raw0.startswith("list:") else raw0
                head = _build_linked_list_with_cycle(vals, cycle_pos)
                args = (head,) + tuple(_parse_list_input(a) for a in raw_input[1:])
            else:
                args = tuple(_parse_list_input(a) for a in raw_input)
            expected_list = _parse_list_expected(raw_expected)
        elif encode_decode:
            args = raw_input
            expected_list = raw_expected
        else:
            args = raw_input if isinstance(raw_input, tuple) else (raw_input,)
            expected_list = raw_expected
    except Exception as exc:
        result_queue.put({
            "passed":   False,
            "input":    str(raw_input),
            "expected": str(raw_expected),
            "actual":   None,
            "error":    f"Error preparing test input: {exc}",
        })
        return

    try:
        func_name = _find_function(user_code, ns, problem)
    except Exception as exc:
        result_queue.put({
            "passed":   False,
            "input":    str(raw_input),
            "expected": str(raw_expected),
            "actual":   None,
            "error":    str(exc),
        })
        return

    try:
        if encode_decode:
            # Support both serialize/deserialize and encode/decode naming
            encode_fn = ns.get("serialize") or ns.get("encode")
            decode_fn = ns.get("deserialize") or ns.get("decode")
            if not encode_fn or not decode_fn:
                raise NameError("Functions 'serialize'/'deserialize' (or 'encode'/'decode') not found in your code.")
            if is_tree:
                encoded = encode_fn(args[0])
                actual  = decode_fn(encoded)
            else:
                encoded = encode_fn(*args)
                actual  = decode_fn(encoded)
        else:
            fn = ns[func_name]
            actual = fn(*args)
    except Exception as e:
        err_tb = traceback.format_exception(type(e), e, e.__traceback__) if hasattr(e, "__traceback__") else [str(e)]
        result_queue.put({
            "passed":   False,
            "input":    _format_input(raw_input),
            "expected": str(expected_list),
            "actual":   None,
            "error":    "".join(err_tb),
        })
        return

    # ── Convert result for comparison ──
    if is_tree and hasattr(actual, "val"):
        # If expected is a simple value (int/bool), compare node.val; else compare full tree
        if not isinstance(expected_list, list):
            actual_cmp = actual.val
        else:
            actual_cmp = _tree_to_list(actual)
    elif is_tree and actual is None:
        actual_cmp = []   # None return = empty tree, compare as []
    elif is_list and check_head:
        # In-place modification: check the head node after the call
        actual_cmp = _linked_list_to_list(args[0])
    elif is_list and hasattr(actual, "val"):
        actual_cmp = _linked_list_to_list(actual)
    elif is_list and actual is None and not check_head:
        actual_cmp = []
    else:
        actual_cmp = actual

    passed = _results_match(actual_cmp, expected_list, unordered=unordered, unordered_groups=unord_groups)
    result_queue.put({
        "passed":   passed,
        "input":    _format_input(raw_input),
        "expected": str(expected_list),
        "actual":   str(actual_cmp),
        "error":    None,
    })


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
        try:
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_run_one_test_in_process,
                args=(user_code, tc, problem, result_queue),
                daemon=True,
            )
            p.start()
            p.join(timeout=TEST_TIMEOUT_SECONDS)

            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1.0)
                results.append({
                    "passed":   False,
                    "input":    str(tc.get("input", "—")),
                    "expected": str(tc.get("expected", "—")),
                    "actual":   None,
                    "error":    f"Timeout ({TEST_TIMEOUT_SECONDS}s). Your code was stopped — possible infinite loop.",
                })
                continue

            try:
                result = result_queue.get_nowait()
            except Exception:
                result = {
                    "passed":   False,
                    "input":    str(tc.get("input", "—")),
                    "expected": str(tc.get("expected", "—")),
                    "actual":   None,
                    "error":    "Timeout or process error.",
                }
            results.append(result)
        except Exception as e:
            results.append({
                "passed":   False,
                "input":    str(tc.get("input", "—")),
                "expected": str(tc.get("expected", "—")),
                "actual":   None,
                "error":    f"Test runner error: {e}. If this persists, try restarting the app.",
            })

    return results


def _find_function(user_code: str, ns: dict, problem: dict) -> str:
    """Determine the primary function name to call from the user's code."""
    # Collect all callables defined in user code (non-TreeNode classes excluded)
    candidates = [
        name for name, obj in ns.items()
        if callable(obj)
        and not name.startswith("_")
        and name not in ("TreeNode", "ListNode")
        and not (isinstance(obj, type) and issubclass(obj, TreeNode))
        and not (isinstance(obj, type) and issubclass(obj, ListNode))
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
