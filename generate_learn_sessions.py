"""
Generate pre-written learn sessions for all 108 problems and save to learn_sessions.json.

Run once (or re-run to refresh):
    python generate_learn_sessions.py

Options:
    --ids 1 2 3      Only regenerate specific problem IDs
    --force          Re-generate even if a session already exists
"""

import json
import os
import sys
import time
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()

_load_env()

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path(__file__).parent / "learn_sessions.json"
MAX_WORKERS = 1   # sequential to respect rate limits
MAX_TOKENS  = 2000
REQUEST_DELAY = 20  # seconds between requests (keeps under 8K tok/min limit)

_LEARN_SESSION_PROMPT = """\
You are an expert DSA instructor teaching a beginner how to solve algorithm problems in Python.

Your job is to deliver a complete, structured LEARNING SESSION for the given problem. Do NOT just hint — actually teach.

Format your response with these exact sections:

## 🧩 Understanding the Problem
Restate the problem in simple plain English. Use a real-world analogy if it helps. Show what the input looks like and what the output should be.

## 💡 Key Insight
Explain the single most important idea that makes this problem solvable. Why does the naive approach fail? What pattern or data structure unlocks the efficient solution?

## 🗺️ Step-by-Step Approach
Walk through the algorithm step by step in plain English. Number each step. Be specific enough that a beginner could translate it to code.

## 🔍 Trace Through an Example
Pick one of the provided test cases and manually trace through your algorithm step by step, showing the state of all variables at each step.

## 🐍 Python Implementation Walkthrough
Show the complete solution code with detailed inline comments on every important line. Explain *why* each line is written that way, not just *what* it does.

## ⏱️ Time & Space Complexity
Explain the Big-O complexity clearly. Why is it O(n)? O(n log n)? What determines the space usage? Keep it beginner-friendly.

## ⚠️ Common Mistakes
List 2–3 mistakes beginners make on this problem and how to avoid them.

## 🔁 Pattern to Remember
End with one or two sentences summarizing the reusable pattern or technique this problem teaches, so the student can recognize it in future problems.

Keep the tone encouraging and clear. Use code blocks for any code snippets.
"""


def _generate_one(problem: dict) -> tuple[int, str]:
    """Generate a learn session for one problem. Returns (id, text)."""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    test_examples = ""
    for tc in problem.get("test_cases", [])[:2]:
        inp = tc.get("input", "")
        exp = tc.get("expected", "")
        test_examples += f"- Input: `{inp}` → Expected: `{exp}`\n"

    user_message = f"""\
**Problem:** {problem['title']} ({problem['difficulty']}) — Category: {problem['category']}

**Full Description:**
{problem['description']}

**Python concepts already shown to the student:**
{problem['python_tips']}

**Reference solution (use this to base your walkthrough on, but explain it fully):**
```python
{problem['solution']}
```

**Example test cases to use for your trace:**
{test_examples}

Please deliver the full learning session now.
"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=MAX_TOKENS,
                temperature=0.4,
                system=_LEARN_SESSION_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return problem["id"], response.content[0].text.strip()
        except Exception as exc:
            err = str(exc)
            if "rate_limit" in err or "overloaded" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"  Rate limit on #{problem['id']}, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def main():
    # Parse args
    force = "--force" in sys.argv
    ids_arg = []
    if "--ids" in sys.argv:
        idx = sys.argv.index("--ids")
        for v in sys.argv[idx + 1:]:
            if v.startswith("--"):
                break
            try:
                ids_arg.append(int(v))
            except ValueError:
                pass

    # Import problems
    sys.path.insert(0, str(Path(__file__).parent))
    from problems.problems import PROBLEMS

    # Load existing sessions
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            sessions: dict[str, str] = json.load(f)
    else:
        sessions = {}

    # Determine which problems to generate
    targets = [p for p in PROBLEMS if (not ids_arg or p["id"] in ids_arg)]
    if not force:
        targets = [p for p in targets if str(p["id"]) not in sessions]

    if not targets:
        print("Nothing to generate (all sessions already exist). Use --force to regenerate.")
        return

    print(f"Generating {len(targets)} learn sessions with {MAX_WORKERS} workers...")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    failed = []
    done = 0

    for i, prob in enumerate(targets):
        try:
            pid, text = _generate_one(prob)
            sessions[str(pid)] = text
            done += 1
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print(f"  [{done}/{len(targets)}] OK #{pid} {prob['title']}")
        except Exception as exc:
            err_msg = str(exc).encode("ascii", "replace").decode("ascii")
            failed.append((prob["id"], prob["title"], err_msg))
            print(f"  FAIL #{prob['id']} {prob['title']}: {err_msg}")
        # Pace requests to stay within the 8K output tokens/min rate limit
        if i < len(targets) - 1:
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {done} generated, {len(failed)} failed.")
    if failed:
        print("Failed problems:")
        for pid, title, err in failed:
            print(f"  #{pid} {title}: {err}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
