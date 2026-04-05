"""
AI hint engine using the Anthropic (Claude) API.

- Loads the API key from .env (ANTHROPIC_API_KEY).
- Provides get_hint() for a single hint and get_hint_streamed() for streaming.
- If no key is present, returns a helpful message instead of crashing.
"""

import os
from pathlib import Path

# ─── Load .env ───────────────────────────────────────────────────────────────

def _load_env():
    """Load .env file from the project root (blind75-agent/)."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip()

_load_env()


def api_key_available() -> bool:
    """Return True if an Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an encouraging Python tutor helping a complete beginner learn data structures and algorithms through the Blind 75 problem set.

Your job is to give a HINT — not the solution. Your response must:
1. Be warm, supportive, and beginner-friendly.
2. Explain the key insight the student is missing WITHOUT writing the full solution.
3. Point to the relevant Python concept or data structure they should use.
4. If they have made progress, acknowledge what they got right first.
5. If their code has a specific bug, describe the bug in plain English and guide them toward fixing it.
6. Use simple analogies when helpful.
7. Keep your response concise — 3 to 6 sentences maximum.
8. Never paste a complete working solution.

Remember: the goal is to help them learn, not to do it for them.
"""


# ─── Hint generator ──────────────────────────────────────────────────────────

def get_hint(problem: dict, user_code: str, failed_tests: list[dict]) -> str:
    """
    Request a Socratic hint from Claude.

    Args:
        problem:      The full problem dict (title, description, python_tips…)
        user_code:    The student's current code
        failed_tests: List of test result dicts from test_runner.run_tests()

    Returns:
        A hint string.
    """
    if not api_key_available():
        return (
            "**AI hints are not enabled.**\n\n"
            "To enable them, copy `.env.example` to `.env` in the `blind75-agent/` "
            "folder and add your Anthropic API key:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n\n"
            "Get a key at https://console.anthropic.com/settings/keys"
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        return "Anthropic package is not installed. Run `pip install anthropic` and restart the app."

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Build a rich context message
    failed_summary = ""
    if failed_tests:
        lines = []
        for t in failed_tests[:3]:  # cap at 3 examples
            if t.get("error"):
                lines.append(f"- Error: {t['error'][:300]}")
            else:
                lines.append(
                    f"- Input: {t['input']}  |  Expected: {t['expected']}  |  Got: {t['actual']}"
                )
        failed_summary = "**Failed test cases:**\n" + "\n".join(lines)
    else:
        failed_summary = "The student has not run tests yet."

    user_message = f"""\
**Problem:** {problem['title']} ({problem['difficulty']})

**Description:**
{problem['description']}

**Python tips provided to the student:**
{problem['python_tips']}

**Student's current code:**
```python
{user_code}
```

{failed_summary}

Please give a helpful hint to guide the student toward the correct solution.
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0.5,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        return f"**Error contacting Claude:** {exc}"


# ─── Learn session ────────────────────────────────────────────────────────────

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


def get_learn_session_streamed(problem: dict):
    """
    Generator that streams a full teaching session for the given problem.
    Covers: key insight, step-by-step approach, example trace, Python walkthrough,
    complexity analysis, common mistakes, and the reusable pattern.
    """
    if not api_key_available():
        yield (
            "**AI Learn Session is not enabled.**\n\n"
            "Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.\n"
            "Get a key at https://console.anthropic.com/settings/keys"
        )
        return

    try:
        from anthropic import Anthropic
    except ImportError:
        yield "Anthropic package is not installed. Run `pip install anthropic` and restart the app."
        return

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    # Build a few example test cases as strings for the trace section
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

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            temperature=0.4,
            system=_LEARN_SESSION_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as exc:
        yield f"**Error contacting Claude:** {exc}"


def get_hint_streamed(problem: dict, user_code: str, failed_tests: list[dict]):
    """
    Generator version of get_hint() that yields text chunks as they stream in.
    Suitable for use with st.write_stream().
    """
    if not api_key_available():
        yield (
            "**AI hints are not enabled.**\n\n"
            "Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.\n"
            "Get a key at https://console.anthropic.com/settings/keys"
        )
        return

    try:
        from anthropic import Anthropic
    except ImportError:
        yield "Anthropic package is not installed. Run `pip install anthropic` and restart the app."
        return

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    failed_summary = ""
    if failed_tests:
        lines = []
        for t in failed_tests[:3]:
            if t.get("error"):
                lines.append(f"- Error: {t['error'][:300]}")
            else:
                lines.append(
                    f"- Input: {t['input']}  |  Expected: {t['expected']}  |  Got: {t['actual']}"
                )
        failed_summary = "**Failed test cases:**\n" + "\n".join(lines)
    else:
        failed_summary = "The student has not run tests yet."

    user_message = f"""\
**Problem:** {problem['title']} ({problem['difficulty']})

**Description:**
{problem['description']}

**Python tips provided to the student:**
{problem['python_tips']}

**Student's current code:**
```python
{user_code}
```

{failed_summary}

Please give a helpful hint to guide the student toward the correct solution.
"""

    try:
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0.5,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as exc:
        yield f"**Error contacting Claude:** {exc}"
