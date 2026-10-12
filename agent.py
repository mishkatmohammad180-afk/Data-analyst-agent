"""
Data Analyst Agent - Session 1: the core loop
Ask a plain-English question about a CSV. The model writes pandas code,
we run it, and if it errors we feed the error back (max 3 attempts).

Setup:
    pip install groq pandas
    export GROQ_API_KEY="your_key_here"

Usage:
    python agent.py data.csv "What is the average age by gender?"

WARNING: Session 1 runs model-written code with exec() and NO sandbox.
Only use your own small CSV files until Session 2 adds safety.
"""

import io
import re
import sys
import contextlib
import traceback

import pandas as pd
from groq import Groq

MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 3

client = Groq()  # reads GROQ_API_KEY from the environment


def describe_dataframe(df: pd.DataFrame) -> str:
    """Give the model the schema and a few rows, never the whole file."""
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Column info:\n{buf.getvalue()}\n"
        f"First 5 rows:\n{df.head().to_string()}"
    )


def build_system_prompt(schema: str) -> str:
    return f"""You are a careful data analyst. A pandas DataFrame named `df` is already loaded.

{schema}

Rules:
- Reply with ONE Python code block and nothing else.
- Use only pandas (as `pd`) and the existing `df`. Do not read or write files.
- Use print() to output every result you need. Only printed output is visible to you.
- Never modify `df` in place; work on copies.
- If the question cannot be answered from these columns, print exactly:
  CANNOT_ANSWER: <short reason>
"""


def extract_code(text: str) -> str:
    """Pull the code out of a ```python ... ``` block (or use the raw text)."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def run_code(code: str, df: pd.DataFrame):
    """Execute code, capture stdout. Returns (ok, output_or_error)."""
    namespace = {"df": df.copy(), "pd": pd}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace)
        return True, stdout.getvalue().strip()
    except Exception:
        return False, traceback.format_exc(limit=2)


def ask_model(messages):
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0
    )
    return response.choices[0].message.content


def answer_question(df: pd.DataFrame, question: str, verbose: bool = True):
    messages = [
        {"role": "system", "content": build_system_prompt(describe_dataframe(df))},
        {"role": "user", "content": question},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = ask_model(messages)
        code = extract_code(reply)
        ok, output = run_code(code, df)

        if verbose:
            print(f"\n--- Attempt {attempt} ---")
            print(f"Code:\n{code}\n")
            print(f"{'Output' if ok else 'Error'}:\n{output}")

        if ok and output:
            return {"ok": True, "code": code, "output": output, "attempts": attempt}

        # Feed the result back so the model can fix its own mistake
        problem = output if not ok else "Your code ran but printed nothing. Use print()."
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {"role": "user", "content": f"That failed:\n{problem}\nFix it and reply with one code block."}
        )

    return {"ok": False, "code": code, "output": output, "attempts": MAX_ATTEMPTS}


def main():
    if len(sys.argv) < 3:
        print('Usage: python agent.py data.csv "your question"')
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    result = answer_question(df, sys.argv[2])

    print("\n=== RESULT ===")
    if result["ok"]:
        print(result["output"])
        print(f"\n(solved in {result['attempts']} attempt(s))")
    else:
        print(f"Could not answer after {result['attempts']} attempts.")


if __name__ == "__main__":
    main()
