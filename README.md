# Blind 75 Python Learning Agent
Built so that myself can practice and learn DSA.
A beginner-friendly web app to learn and practice the Blind 75 LeetCode problems in Python — powered by Claude AI.

## Features

- **108 problems** — all 75 classic Blind 75 problems (⭐) plus 33 bonus extras, organised by category (Arrays, Trees, DP, Graphs, etc.)
- **Blind 75 progress tracker** — dedicated progress bar and filter for the original 75 problems; 🏆 celebration when you complete them all
- **Learn tab** — problem description + Python concept explanations written for beginners
- **AI Learn Session** — full Claude-powered tutorial: key insight, step-by-step approach, example trace, annotated code walkthrough, and complexity analysis
- **Practice tab** — in-browser Python 3.10+ code editor with syntax highlighting and auto-indent
- **Test runner** — runs your code against real test cases instantly, with support for arrays, linked lists, binary trees, and class-based problems (MinStack, LRU Cache, Trie, etc.)
- **AI hints** — streamed Socratic hints from Claude when you're stuck (optional)
- **Progress tracking** — solved / in-progress / not-started status with filtering and search
- **Reference solutions** — reveal the solution after you've made a genuine attempt
- **Auto-save** — progress is saved to disk automatically between sessions

## Quick Start

### 1. Install Python 3.10+

Download from https://www.python.org/downloads/

### 2. Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### 3. (Optional but recommended) Enable AI Features

Copy the example env file:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Then open `.env` and replace `sk-ant-...` with your actual Anthropic API key.
Get one at: https://console.anthropic.com/settings/keys

Without a key the app still works — the AI Learn Session and AI Hints will just be disabled.

### 4. Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

## Project Structure

```
blind75-agent/
├── app.py                  # Main Streamlit app (UI, tabs, editor, navigation)
├── problems/
│   └── problems.py         # All 108 problems (descriptions, tips, test cases, solutions)
├── runner/
│   └── test_runner.py      # Sandboxed code executor (arrays, trees, linked lists, classes)
├── ai/
│   └── hints.py            # Claude-powered hint and learn session engine
├── persistence.py          # Auto-save progress to disk (progress.json)
├── requirements.txt
├── .env.example            # Copy to .env and add your Anthropic API key
└── README.md
```

## Tips for Beginners

- Start with **#1 Two Sum** and work through problems in order within each category.
- Read the **Learn tab** first — the Python Tips section explains every concept you need.
- Use **Start Learn Session** for a full AI-powered walkthrough before you code.
- When stuck, use **Get AI Hint** for a Socratic nudge (not the full answer!).
- Only reveal the **solution** after a genuine attempt — that's how learning sticks.
- Re-try problems you got wrong a few days later without looking at your previous code.
- Use the sidebar filters (category, difficulty, status, ⭐ Blind 75 only) to focus your practice.
