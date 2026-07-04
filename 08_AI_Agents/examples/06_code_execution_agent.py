# Author: Mohan Raju Amuri
"""
06_code_execution_agent.py — Agent that writes and runs Python code

What to remember:
- Code execution agents solve problems by generating Python → running it → reading output
- Two-step: LLM writes code → sandbox executes it → result feeds back to LLM
- Always sandbox execution: use restricted builtins, timeouts, no file system access
- Powerful for: data analysis, calculations, string manipulations, API calls

What NOT to do:
- Don't run LLM-generated code with full system access — critical security risk
- Don't trust LLM-generated code without sandboxing — it can delete files, make network calls
- Don't skip the output-back-to-LLM step — the agent needs to verify the result

Security rule: NEVER exec(llm_output) without a sandbox. Always restrict builtins.

Interview one-liner:
  "Code execution agents generate Python, run it in a sandbox, and feed the output back — no hallucinated math."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required (optional)
# ============================================================
# ollama pull llama3.2:3b && ollama serve
# Core code execution demo works without Ollama.
# ============================================================

import io
import sys
import math
import traceback
import contextlib
from datetime import datetime

LLM_MODEL = "llama3.2:3b"


# ── Sandboxed Python Executor ─────────────────────────────────────────────────
# The sandbox restricts what code can do:
#   - No file I/O, no os/subprocess, no network
#   - Only math, standard types, and safe builtins are available
#   - Output is captured, not printed to the real stdout

SAFE_BUILTINS = {
    # Math
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "divmod": divmod, "len": len,
    # Types
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "range": range,
    # Iteration
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed,
    # String
    "print": print, "repr": repr, "format": format,
}

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "math": math,
    "datetime": datetime,
}

def execute_python(code: str, timeout_seconds: int = 5) -> dict:
    """
    Execute Python code in a restricted sandbox.
    Returns: {success, output, error, return_value}
    """
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            local_vars = {}
            exec(compile(code, "<agent_code>", "exec"), SAFE_GLOBALS.copy(), local_vars)

        output = stdout_capture.getvalue().strip()
        # Capture the last assigned variable as the return value
        result_var = local_vars.get("result", local_vars.get("answer", None))

        return {
            "success": True,
            "output": output,
            "error": None,
            "local_vars": {k: v for k, v in local_vars.items()
                           if not k.startswith("_") and not callable(v)},
        }
    except Exception as e:
        return {
            "success": False,
            "output": stdout_capture.getvalue().strip(),
            "error": f"{type(e).__name__}: {e}",
            "local_vars": {},
        }


# ── Code Generation Prompt ────────────────────────────────────────────────────
CODE_PROMPT = """Write Python code to solve this problem.
Rules:
- Use only: math, datetime, basic Python types (list, dict, set, str, int, float)
- No imports (math and datetime are pre-imported)
- Print the final answer using print()
- Keep it concise

Problem: {problem}

Python code (only the code, no explanation, no markdown):"""

INTERPRET_PROMPT = """The code ran and produced this output:
{output}

Original question: {question}

Give a clear, concise answer based on the output:"""


# ── Code Execution Agent ──────────────────────────────────────────────────────
class CodeAgent:
    def __init__(self, llm=None):
        self.llm = llm
        self.history: list[dict] = []

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str | None:
        if self.llm is None:
            return None
        try:
            resp = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    def solve(self, question: str, mock_code: str = None) -> dict:
        """Generate code → execute → interpret result."""
        print(f"\nQuestion: {question}")
        print("─" * 50)

        # Step 1: Generate code
        code_raw = self._call_llm(CODE_PROMPT.format(problem=question))
        if code_raw is None:
            if mock_code:
                code_raw = mock_code
                print("  [using mock code]")
            else:
                return {"question": question, "code": "", "output": "", "answer": "Ollama required"}

        # Strip markdown code fences if present
        code = code_raw.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])
        code = code.strip()

        print(f"  Generated code:\n    " + code.replace("\n", "\n    "))

        # Step 2: Execute
        exec_result = execute_python(code)
        print(f"\n  Execution: {'✓ success' if exec_result['success'] else '✗ failed'}")
        if exec_result["output"]:
            print(f"  Output: {exec_result['output']}")
        if exec_result["error"]:
            print(f"  Error:  {exec_result['error']}")

        # Step 3: Retry once if execution failed (self-healing)
        if not exec_result["success"] and exec_result["error"]:
            fix_prompt = f"Fix this Python code that raised an error:\n\nCode:\n{code}\n\nError: {exec_result['error']}\n\nFixed code only:"
            fixed_code = self._call_llm(fix_prompt)
            if fixed_code:
                print(f"  Retrying with fixed code...")
                exec_result = execute_python(fixed_code.strip())
                code = fixed_code
                print(f"  Retry: {'✓ success' if exec_result['success'] else '✗ failed again'}")

        # Step 4: Interpret
        output_text = exec_result["output"] or str(exec_result.get("local_vars", {}))
        interpret = self._call_llm(
            INTERPRET_PROMPT.format(output=output_text, question=question),
            max_tokens=100,
        )
        answer = interpret or output_text

        print(f"\n  Answer: {answer}")
        return {"question": question, "code": code, "output": output_text, "answer": answer}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Code Execution Agent")
    print("=" * 60)

    # Init Ollama
    llm = None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        llm = client
        print(f"\nOllama connected — {LLM_MODEL}")
    except Exception:
        print("\nOllama not running — showing mock code execution")

    agent = CodeAgent(llm=llm)

    # Mock code for each problem (shown when Ollama not running)
    problems = [
        (
            "Calculate the compound interest on $10,000 at 5% annual rate over 10 years",
            "principal = 10000\nrate = 0.05\nyears = 10\namount = principal * (1 + rate) ** years\nprint(f'Final amount: ${amount:.2f}')\nprint(f'Interest earned: ${amount - principal:.2f}')"
        ),
        (
            "Find all prime numbers between 1 and 50",
            "def is_prime(n):\n    if n < 2: return False\n    return all(n % i != 0 for i in range(2, int(n**0.5)+1))\nprimes = [n for n in range(2, 51) if is_prime(n)]\nprint(f'Primes: {primes}')\nprint(f'Count: {len(primes)}')"
        ),
        (
            "What is the standard deviation of [10, 20, 30, 40, 50]?",
            "data = [10, 20, 30, 40, 50]\nmean = sum(data) / len(data)\nvariance = sum((x - mean)**2 for x in data) / len(data)\nstd = math.sqrt(variance)\nprint(f'Mean: {mean}')\nprint(f'Std Dev: {round(std, 4)}')"
        ),
    ]

    for question, mock_code in problems:
        print("\n" + "=" * 60)
        agent.solve(question, mock_code=mock_code)

    # Show the security model
    print("\n" + "=" * 60)
    print("Security: what the sandbox blocks")
    print("=" * 60)
    dangerous_snippets = [
        ("import os; os.remove('file.txt')", "import blocked — no os module"),
        ("open('/etc/passwd').read()",        "open() not in safe builtins"),
        ("__import__('subprocess').call(['rm', '-rf', '/'])", "__import__ blocked"),
    ]
    for code, expected_block in dangerous_snippets:
        result = execute_python(code)
        status = "BLOCKED ✓" if not result["success"] else "ESCAPED ✗"
        print(f"  {status}  {code[:50]}")
        if result["error"]:
            print(f"          Error: {result['error']}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Execute in a sandbox: restrict __builtins__, no os/subprocess")
    print("  - Two LLM calls: generate code → execute → interpret output")
    print("  - Self-healing: retry with error message in prompt if execution fails")
    print("  - Code agents beat pure LLM for math, data analysis, and transformations")
    print("  - Never eval() or exec() raw LLM output with full system permissions")
    print("=" * 60)
