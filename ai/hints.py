"""
AI hint engine using the OpenAI API.

- Loads the API key from .env (OPENAI_API_KEY).
- Provides get_hint() for a streamed hint response.
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
                    os.environ.setdefault(key.strip(), val.strip())

_load_env()


def api_key_available() -> bool:
    """Return True if an OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


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
    Request a Socratic hint from OpenAI.

    Args:
        problem:      The full problem dict (title, description, python_tips…)
        user_code:    The student's current code
        failed_tests: List of test result dicts from test_runner.run_tests()

    Returns:
        A hint string (may be streamed by the caller).
    """
    if not api_key_available():
        return (
            "**AI hints are not enabled.**\n\n"
            "To enable them, copy `.env.example` to `.env` in the `blind75-agent/` "
            "folder and add your OpenAI API key:\n\n"
            "```\nOPENAI_API_KEY=sk-...\n```\n\n"
            "Get a key at https://platform.openai.com/api-keys"
        )

    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI package is not installed. Run `pip install openai` and restart the app."

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            max_tokens=400,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"**Error contacting OpenAI:** {exc}"


def get_hint_streamed(problem: dict, user_code: str, failed_tests: list[dict]):
    """
    Generator version of get_hint() that yields text chunks as they stream in.
    Suitable for use with st.write_stream().
    """
    if not api_key_available():
        yield (
            "**AI hints are not enabled.**\n\n"
            "Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
        return

    try:
        from openai import OpenAI
    except ImportError:
        yield "OpenAI package is not installed. Run `pip install openai` and restart the app."
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=400,
            temperature=0.5,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        yield f"**Error contacting OpenAI:** {exc}"
