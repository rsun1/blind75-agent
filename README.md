# Blind 75 Python Learning Agent

A beginner-friendly web app to learn and practice the Blind 75 LeetCode problems in Python.

## Features

- **75 problems** organised by category (Arrays, Trees, DP, etc.)
- **Learn tab** — problem description + Python concept explanations written for beginners
- **Practice tab** — in-browser code editor with syntax highlighting
- **Test runner** — runs your code against real test cases instantly, no account needed
- **AI hints** — streamed Socratic hints from OpenAI when you're stuck (optional)
- **Progress tracking** — see which problems you've solved, attempted, or not started
- **Reference solutions** — reveal the solution after you've made an attempt

## Quick Start

### 1. Install Python (if you haven't already)

Download from https://www.python.org/downloads/ (Python 3.10+ recommended)

### 2. Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### 3. (Optional) Enable AI Hints

Copy the example env file:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Then open `.env` in any text editor and replace `sk-...` with your actual OpenAI API key.
Get one at: https://platform.openai.com/api-keys

If you skip this step the app still works — AI hints will just be disabled.

### 4. Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

## Project Structure

```
blind75-agent/
├── app.py                  # Main Streamlit app
├── problems/
│   └── problems.py         # All 75 problems (descriptions, tips, test cases, solutions)
├── runner/
│   └── test_runner.py      # Safely executes your code against test cases
├── ai/
│   └── hints.py            # OpenAI hint engine
├── requirements.txt
├── .env.example            # Copy to .env and add your OpenAI key
└── README.md
```

## Tips for Beginners

- Start with **#1 Two Sum** and work through the problems in order within each category.
- Read the **Learn tab** before jumping to code — the Python Tips section explains every concept you need.
- When stuck, use **Get AI Hint** for a nudge in the right direction (not the full answer!).
- Only reveal the **solution** after you've made a genuine attempt — that's how learning sticks.
- Re-try problems you got wrong a few days later without looking at your previous code.
