"""
Blind 75 Python Learning Agent
================================
A Streamlit web app to learn and practice the Blind 75 LeetCode problems.

Run with:
    streamlit run app.py
"""

import logging
import multiprocessing

# Suppress Streamlit's "missing ScriptRunContext" when test runner spawns a child process
# (the child re-imports this module on Windows and triggers the warning — safe to ignore)
class _SuppressStreamlitContextWarning(logging.Filter):
    def filter(self, record):
        msg = (record.getMessage() or "")
        if "ScriptRunContext" in msg or "bare mode" in msg:
            return False
        return True

_filter = _SuppressStreamlitContextWarning()
for _log_name in ("streamlit", "streamlit.logger", "streamlit.runtime.scriptrunner", "streamlit.runtime.scriptrunner_utils.script_run_context"):
    logging.getLogger(_log_name).addFilter(_filter)
logging.getLogger().addFilter(_filter)

import streamlit as st
from streamlit_ace import st_ace

from problems.problems import PROBLEMS, CATEGORIES, DIFFICULTY_ORDER, get_problem_by_id
from runner.test_runner import run_tests
from ai.hints import api_key_available, get_hint_streamed
from persistence import load_progress, save_progress

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Blind 75 — Python Learning Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    # Load persisted data from disk on the very first run of the session.
    # After that, session_state takes over until the next full reload.
    if "progress_loaded" not in st.session_state:
        saved = load_progress()
        st.session_state.solved           = set(saved["solved"])
        st.session_state.attempted        = set(saved["attempted"])
        st.session_state.user_code_cache  = saved["user_code_cache"]
        st.session_state.show_solution    = set(saved["show_solution"])
        st.session_state.progress_loaded  = True

    if "current_problem_id" not in st.session_state:
        st.session_state.current_problem_id = None
    if "last_test_results" not in st.session_state:
        st.session_state.last_test_results = {}
    if "hint_text" not in st.session_state:
        st.session_state.hint_text = {}
    if "editor_reset_counter" not in st.session_state:
        st.session_state.editor_reset_counter = {}  # pid -> int; bump to force editor recreate on Reset
    if "debug_counter" not in st.session_state:
        st.session_state.debug_counter = 0
    if "sidebar_show_filter" not in st.session_state:
        st.session_state.sidebar_show_filter = None  # "All problems" | "Not started" | "In progress" | "Solved"


def _save():
    """Persist the current progress state to disk."""
    try:
        save_progress(
            solved          = st.session_state.solved,
            attempted       = st.session_state.attempted,
            user_code_cache = st.session_state.user_code_cache,
            show_solution   = st.session_state.show_solution,
        )
    except Exception:
        pass  # Don't block or crash the app if save fails (e.g. disk full, permission)


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SHOW_FILTER_OPTIONS = ["All problems", "Not started", "In progress", "Solved"]

DIFFICULTY_COLORS = {
    "Easy":   "#00b8a3",
    "Medium": "#ffc01e",
    "Hard":   "#ff375f",
}

DIFFICULTY_BG = {
    "Easy":   "#e8faf8",
    "Medium": "#fff8e6",
    "Hard":   "#ffe8ec",
}


def _badge(difficulty: str) -> str:
    color = DIFFICULTY_COLORS.get(difficulty, "#888")
    bg    = DIFFICULTY_BG.get(difficulty, "#eee")
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color};'
        f'border-radius:12px;padding:2px 10px;font-size:0.78rem;font-weight:600;">'
        f'{difficulty}</span>'
    )


def _status_icon(problem_id: int) -> str:
    if problem_id in st.session_state.solved:
        return "✅"
    if problem_id in st.session_state.attempted:
        return "🔄"
    return "⬜"


def _difficulty_button_css() -> str:
    """Return a <style> block that colors the button immediately following a [data-difficulty] marker."""
    e, m, h = DIFFICULTY_COLORS["Easy"], DIFFICULTY_COLORS["Medium"], DIFFICULTY_COLORS["Hard"]
    eb, mb, hb = DIFFICULTY_BG["Easy"], DIFFICULTY_BG["Medium"], DIFFICULTY_BG["Hard"]
    return f"""
<style>
/* Difficulty-colored problem list buttons (marker + adjacent button) */
div.stMarkdown:has([data-difficulty="Easy"]) + div[data-testid*="stButton"] button {{
    background: {e} !important;
    color: white !important;
    border-color: {e} !important;
}}
div.stMarkdown:has([data-difficulty="Medium"]) + div[data-testid*="stButton"] button {{
    background: {m} !important;
    color: white !important;
    border-color: {m} !important;
}}
div.stMarkdown:has([data-difficulty="Hard"]) + div[data-testid*="stButton"] button {{
    background: {h} !important;
    color: white !important;
    border-color: {h} !important;
}}
div.stMarkdown:has([data-active="true"]) + div[data-testid*="stButton"] button {{
    font-weight: 700 !important;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.2) !important;
}}
</style>
"""


def _difficulty_marker_html(difficulty: str, is_active: bool) -> str:
    """Return hidden div to mark the next button's difficulty (and optional active state)."""
    active_attr = ' data-active="true"' if is_active else ""
    return (
        f'<div data-difficulty="{difficulty}"{active_attr} '
        f'style="height:0;overflow:hidden;margin:0;padding:0;"></div>'
    )


def _overall_progress() -> tuple[int, int]:
    return len(st.session_state.solved), len(PROBLEMS)


def _go_to_problem(problem_id: int):
    # Save any code currently in the editor before switching problems
    _save()
    st.session_state.current_problem_id = problem_id
    st.session_state.hint_text.pop(problem_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Blind 75 Agent")
        st.caption("Python learning companion")

        # Debug: if the app freezes, click this. If the number updates, the UI is fine and backend is stuck elsewhere.
        with st.expander("🔧 Responsiveness test", expanded=False):
            if st.button("Click to test"):
                st.session_state.debug_counter += 1
            st.caption(f"Counter: **{st.session_state.debug_counter}** — if this updates on click, the app framework is responsive.")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_problem_id = None

        st.divider()

        # Show save status
        from pathlib import Path
        progress_file = Path(__file__).parent / "progress.json"
        if progress_file.exists():
            import datetime
            mtime = datetime.datetime.fromtimestamp(progress_file.stat().st_mtime)
            st.caption(f"💾 Progress auto-saved · {mtime.strftime('%H:%M:%S')}")
        else:
            st.caption("💾 Progress will save on first action")

        # Overall progress bar
        solved, total = _overall_progress()
        st.markdown(f"**Progress: {solved} / {total}**")
        st.progress(solved / total if total else 0)

        st.divider()

        # Filter controls
        selected_category = st.selectbox(
            "Category",
            options=["All"] + CATEGORIES,
            index=0,
        )
        selected_difficulty = st.selectbox(
            "Difficulty",
            options=["All", "Easy", "Medium", "Hard"],
            index=0,
        )
        show_only = st.selectbox(
            "Show",
            options=SHOW_FILTER_OPTIONS,
            index=SHOW_FILTER_OPTIONS.index(st.session_state.sidebar_show_filter) if st.session_state.get("sidebar_show_filter") in SHOW_FILTER_OPTIONS else 0,
        )
        st.session_state.sidebar_show_filter = show_only  # keep in sync when user changes in sidebar

        st.divider()

        # Problem list
        filtered = [
            p for p in PROBLEMS
            if (selected_category == "All" or p["category"] == selected_category)
            and (selected_difficulty == "All" or p["difficulty"] == selected_difficulty)
            and (
                show_only == "All problems"
                or (show_only == "Not started"   and p["id"] not in st.session_state.attempted and p["id"] not in st.session_state.solved)
                or (show_only == "In progress"   and p["id"] in st.session_state.attempted and p["id"] not in st.session_state.solved)
                or (show_only == "Solved"         and p["id"] in st.session_state.solved)
            )
        ]

        if not filtered:
            st.info("No problems match the current filters.")
        else:
            st.markdown(_difficulty_button_css(), unsafe_allow_html=True)
            current_cat = None
            for p in filtered:
                if p["category"] != current_cat:
                    current_cat = p["category"]
                    st.markdown(f"**{current_cat}**")

                icon = _status_icon(p["id"])
                label = f"{icon} {p['id']}. {p['title']}"
                is_active = st.session_state.current_problem_id == p["id"]

                st.markdown(_difficulty_marker_html(p["difficulty"], is_active), unsafe_allow_html=True)
                if st.button(
                    label,
                    key=f"nav_{p['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    _go_to_problem(p["id"])
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Home screen
# ─────────────────────────────────────────────────────────────────────────────

def render_home():
    st.markdown("# 🧠 Blind 75 Python Learning Agent")
    st.markdown(
        "Welcome! This app helps you work through the famous **Blind 75** LeetCode problem list "
        "while learning Python from scratch. Each problem comes with:\n\n"
        "- 📖 **Learn tab** — problem description and Python concepts explained for beginners\n"
        "- 💻 **Practice tab** — an in-browser code editor with starter code\n"
        "- ✅ **Test runner** — run your code against real test cases instantly\n"
        "- 💡 **AI hints** — get a Socratic hint powered by OpenAI when you're stuck\n"
        "- 👁 **View solution** — reveal the reference solution after attempting"
    )

    st.divider()

    # Stats row — click a number to filter the sidebar and (except Total) open the first matching problem
    solved_count, total = _overall_progress()
    attempted     = len(st.session_state.attempted)
    in_progress_count = attempted - solved_count if attempted >= solved_count else 0
    not_started_count = total - attempted

    in_progress_ids = sorted([p["id"] for p in PROBLEMS if p["id"] in st.session_state.attempted and p["id"] not in st.session_state.solved])
    solved_ids     = sorted(st.session_state.solved)
    not_started_ids= sorted([p["id"] for p in PROBLEMS if p["id"] not in st.session_state.attempted and p["id"] not in st.session_state.solved])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Total Problems")
        if st.button(f"**{total}**", key="stat_total", use_container_width=True, help="Show all problems in sidebar"):
            st.session_state.sidebar_show_filter = "All problems"
            st.session_state.current_problem_id = None
            st.rerun()
    with col2:
        st.caption("Solved")
        if st.button(f"**{solved_count}**", key="stat_solved", use_container_width=True, disabled=(solved_count == 0), help="Go to first solved problem"):
            st.session_state.sidebar_show_filter = "Solved"
            st.session_state.current_problem_id = solved_ids[0] if solved_ids else None
            st.rerun()
    with col3:
        st.caption("In Progress")
        if st.button(f"**{in_progress_count}**", key="stat_in_progress", use_container_width=True, disabled=(in_progress_count == 0), help="Go to first in-progress problem"):
            st.session_state.sidebar_show_filter = "In progress"
            st.session_state.current_problem_id = in_progress_ids[0] if in_progress_ids else None
            st.rerun()
    with col4:
        st.caption("Not Started")
        if st.button(f"**{not_started_count}**", key="stat_not_started", use_container_width=True, disabled=(not_started_count == 0), help="Go to first not-started problem"):
            st.session_state.sidebar_show_filter = "Not started"
            st.session_state.current_problem_id = not_started_ids[0] if not_started_ids else None
            st.rerun()

    st.progress(solved_count / total if total else 0, text=f"{solved_count}/{total} solved")

    st.divider()

    # Category overview
    st.markdown("### Problems by Category")
    st.markdown(_difficulty_button_css(), unsafe_allow_html=True)
    for cat in CATEGORIES:
        cat_problems = [p for p in PROBLEMS if p["category"] == cat]
        cat_solved   = sum(1 for p in cat_problems if p["id"] in st.session_state.solved)
        cat_total    = len(cat_problems)

        easy   = sum(1 for p in cat_problems if p["difficulty"] == "Easy")
        medium = sum(1 for p in cat_problems if p["difficulty"] == "Medium")
        hard   = sum(1 for p in cat_problems if p["difficulty"] == "Hard")

        with st.expander(f"**{cat}** — {cat_solved}/{cat_total} solved", expanded=False):
            diff_str = "  ".join(filter(None, [
                f'<span style="color:{DIFFICULTY_COLORS["Easy"]}">● {easy} Easy</span>'   if easy   else "",
                f'<span style="color:{DIFFICULTY_COLORS["Medium"]}">● {medium} Medium</span>' if medium else "",
                f'<span style="color:{DIFFICULTY_COLORS["Hard"]}">● {hard} Hard</span>'    if hard   else "",
            ]))
            st.markdown(diff_str, unsafe_allow_html=True)
            st.progress(cat_solved / cat_total if cat_total else 0)

            for p in cat_problems:
                icon = _status_icon(p["id"])
                is_active = st.session_state.current_problem_id == p["id"]
                st.markdown(_difficulty_marker_html(p["difficulty"], is_active), unsafe_allow_html=True)
                if st.button(
                    f"{icon} {p['id']}. {p['title']}",
                    key=f"home_nav_{p['id']}",
                    use_container_width=False,
                ):
                    _go_to_problem(p["id"])
                    st.rerun()

    st.divider()

    # Quick start hint
    st.info(
        "**New here?** Start with problem **#1 Two Sum** — it's the classic first step. "
        "Use the sidebar on the left to browse and filter problems. "
        "Pick a problem to start learning!"
    )

    # AI key status
    if api_key_available():
        st.success("✅ OpenAI API key detected — AI hints are enabled.")
    else:
        st.warning(
            "⚠️ No OpenAI API key found. AI hints will be disabled. "
            "Copy `.env.example` → `.env` and add your key to enable them."
        )

    st.divider()
    with st.expander("⚠️ Danger zone", expanded=False):
        st.markdown("**Reset all progress** — clears solved status, attempted history, and all saved code.")
        if st.button("🗑 Reset everything", type="secondary"):
            st.session_state.solved          = set()
            st.session_state.attempted       = set()
            st.session_state.user_code_cache = {}
            st.session_state.show_solution   = set()
            st.session_state.last_test_results = {}
            _save()
            st.success("All progress has been reset.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Problem view
# ─────────────────────────────────────────────────────────────────────────────

def render_problem(problem: dict):
    pid = problem["id"]

    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(f"## {pid}. {problem['title']}")
    with col_badge:
        st.markdown(_badge(problem["difficulty"]), unsafe_allow_html=True)
        st.caption(problem["category"])

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 8])
    all_ids = [p["id"] for p in PROBLEMS]
    idx     = all_ids.index(pid)

    with nav_col1:
        if idx > 0 and st.button("◀ Prev"):
            _go_to_problem(all_ids[idx - 1])
            st.rerun()
    with nav_col2:
        if idx < len(all_ids) - 1 and st.button("Next ▶"):
            _go_to_problem(all_ids[idx + 1])
            st.rerun()

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_learn, tab_practice = st.tabs(["📖 Learn", "💻 Practice"])

    # ── Learn tab ────────────────────────────────────────────────────────────
    with tab_learn:
        st.markdown("### Problem Description")
        st.markdown(problem["description"])

        st.markdown("### Python Tips & Approach")
        st.markdown(problem["python_tips"])

        if st.button("💻 Go to Practice →", key=f"goto_practice_{pid}"):
            # Streamlit doesn't support switching tabs programmatically;
            # just inform the user.
            st.info("Click the **💻 Practice** tab above to start coding!")

    # ── Practice tab ─────────────────────────────────────────────────────────
    with tab_practice:
        _render_practice(problem)


def _render_practice(problem: dict):
    pid = problem["id"]

    # Show the question on the Practice tab so you don't have to switch to Learn
    with st.expander("📋 Problem description", expanded=True):
        st.markdown(problem["description"])

    # Load cached code or starter. Editor key includes reset counter so "Reset Code" recreates the widget with starter_code.
    default_code = st.session_state.user_code_cache.get(pid, problem["starter_code"])
    reset_count = st.session_state.editor_reset_counter.get(pid, 0)

    st.markdown("### Your Code")
    st.caption(
        "**Python 3** — Write your solution below. Click **Run Tests** to check it. "
        "Use the **Get AI Hint** button if you're stuck."
    )

    user_code = st_ace(
        value=default_code,
        language="python",  # Ace mode; caption below shows "Python 3" for consistency
        theme="tomorrow_night",
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
        min_lines=18,
        key=f"editor_{pid}_{reset_count}",
    )

    # Cache the current code
    if user_code:
        st.session_state.user_code_cache[pid] = user_code

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])

    with btn_col1:
        run_clicked = st.button("▶ Run Tests", type="primary", key=f"run_{pid}", use_container_width=True)

    with btn_col2:
        hint_clicked = st.button(
            "💡 Get AI Hint",
            key=f"hint_{pid}",
            use_container_width=True,
            disabled=not api_key_available(),
            help="Requires an OpenAI API key in .env" if not api_key_available() else "Get a hint from the AI tutor",
        )

    with btn_col3:
        reset_col, solution_col = st.columns(2)
        with reset_col:
            if st.button("↩ Reset Code", key=f"reset_{pid}", use_container_width=True):
                st.session_state.user_code_cache[pid] = problem["starter_code"]
                st.session_state.editor_reset_counter[pid] = reset_count + 1
                _save()
                st.rerun()
        with solution_col:
            if pid in st.session_state.show_solution:
                if st.button("🙈 Hide Solution", key=f"hide_sol_{pid}", use_container_width=True):
                    st.session_state.show_solution.discard(pid)
                    _save()
                    st.rerun()
            else:
                if st.button("👁 Show Solution", key=f"show_sol_{pid}", use_container_width=True):
                    st.session_state.show_solution.add(pid)
                    _save()
                    st.rerun()

    # ── Run Tests ─────────────────────────────────────────────────────────────
    if run_clicked:
        code_to_test = user_code or default_code
        st.session_state.attempted.add(pid)
        # Save the code the user actually tested so it survives a refresh
        st.session_state.user_code_cache[pid] = code_to_test

        with st.spinner("Running your code against test cases…"):
            results = run_tests(problem, code_to_test)
        st.session_state.last_test_results[pid] = results

        all_passed = all(r["passed"] for r in results if r["passed"] is not None)
        if all_passed and results:
            st.session_state.solved.add(pid)
        else:
            st.session_state.solved.discard(pid)

        _save()

    # ── Display test results ──────────────────────────────────────────────────
    if pid in st.session_state.last_test_results:
        results = st.session_state.last_test_results[pid]
        _render_test_results(results, pid)

    # ── AI Hint ───────────────────────────────────────────────────────────────
    if hint_clicked:
        code_for_hint = user_code or default_code
        failed_tests  = [
            r for r in st.session_state.last_test_results.get(pid, [])
            if not r.get("passed")
        ]
        hint_placeholder = st.empty()
        full_hint = ""
        with st.spinner("Thinking…"):
            for chunk in get_hint_streamed(problem, code_for_hint, failed_tests):
                full_hint += chunk
                hint_placeholder.markdown(f"**💡 AI Hint:**\n\n{full_hint}")
        st.session_state.hint_text[pid] = full_hint

    # Show cached hint if available
    elif pid in st.session_state.hint_text and st.session_state.hint_text[pid]:
        st.markdown(f"**💡 AI Hint:**\n\n{st.session_state.hint_text[pid]}")

    # ── Solution ──────────────────────────────────────────────────────────────
    if pid in st.session_state.show_solution:
        st.divider()
        st.markdown("### Reference Solution")
        st.warning(
            "Try to understand **why** the solution works, not just copy it. "
            "Trace through it with the test cases by hand!"
        )
        st.code(problem["solution"], language="python")


def _render_test_results(results: list[dict], pid: int):
    st.divider()
    st.markdown("### Test Results")

    passed_count = sum(1 for r in results if r.get("passed"))
    total_count  = len(results)
    all_passed   = passed_count == total_count and all(r.get("passed") is not None for r in results)

    if all_passed:
        st.success(f"🎉 All {total_count} test cases passed! Great job!")
    else:
        st.error(f"❌ {passed_count}/{total_count} test cases passed.")

    for i, result in enumerate(results, 1):
        passed = result.get("passed")
        icon   = "✅" if passed else ("⚠️" if passed is None else "❌")
        label  = f"{icon} Test case {i}"

        with st.expander(label, expanded=(not passed)):
            if result.get("error"):
                st.markdown("**Error:**")
                st.code(result["error"], language="text")
            else:
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Input**\n```\n{result.get('input', '—')}\n```")
                col2.markdown(f"**Expected**\n```\n{result.get('expected', '—')}\n```")

                actual_str = result.get("actual", "—")
                if passed:
                    col3.markdown(f"**Your output** ✅\n```\n{actual_str}\n```")
                else:
                    col3.markdown(f"**Your output** ❌\n```\n{actual_str}\n```")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    pid = st.session_state.current_problem_id
    if pid is None:
        render_home()
    else:
        problem = get_problem_by_id(pid)
        if problem is None:
            st.error(f"Problem #{pid} not found.")
        else:
            render_problem(problem)


if __name__ == "__main__":
    if multiprocessing.current_process().name == "MainProcess":
        main()
