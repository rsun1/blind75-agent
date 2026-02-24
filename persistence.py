"""
Persists user progress to a local progress.json file so data survives
browser refreshes and app restarts.

Saved fields:
  - solved          : list of problem IDs the user has passed all tests on
  - attempted       : list of problem IDs the user has run tests on
  - user_code_cache : dict of {problem_id: code_string}
  - show_solution   : list of problem IDs where the solution has been revealed
"""

import json
from pathlib import Path

PROGRESS_FILE = Path(__file__).parent / "progress.json"


def load_progress() -> dict:
    """
    Load saved progress from disk.
    Returns a dict with keys: solved, attempted, user_code_cache, show_solution.
    If the file doesn't exist or is corrupt, returns clean defaults.
    """
    defaults = {
        "solved":           [],
        "attempted":        [],
        "user_code_cache":  {},
        "show_solution":    [],
    }
    if not PROGRESS_FILE.exists():
        return defaults
    try:
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Validate and fill any missing keys
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
        # user_code_cache keys are stored as strings in JSON; convert to int
        data["user_code_cache"] = {
            int(k): v for k, v in data["user_code_cache"].items()
        }
        return data
    except (json.JSONDecodeError, Exception):
        return defaults


def save_progress(
    solved: set,
    attempted: set,
    user_code_cache: dict,
    show_solution: set,
) -> None:
    """
    Save the current progress state to disk atomically.
    """
    data = {
        "solved":           sorted(solved),
        "attempted":        sorted(attempted),
        "user_code_cache":  {str(k): v for k, v in user_code_cache.items()},
        "show_solution":    sorted(show_solution),
    }
    # Write to a temp file first, then rename — avoids corruption on crash
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(PROGRESS_FILE)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
