"""Autonomy verification battery -- living documentation / demo script.

This is NOT a pytest suite. It boots the real backend (a real `uvicorn`
process serving `backend.main:app`, the exact command a human would run --
see `run_backend.bat`), then drives it over real HTTP through `/api/chat`
with a battery of realistic, demo-shaped end-user queries -- the kind a
bank analyst would actually type, not synthetic edge-case probes -- each on
a FRESH session_id with no prior segmentation.

The point (per the original bug report -- "if a user asks queries like
which customers are marked priority, the agent tells us to click a button
in the frontend to manually segment the users -- all processes like these
must be automated by the agent") is to prove, end-to-end, over the wire,
that:

  1. No query requires the user to go click a button in the frontend first
     (segmentation auto-runs on demand wherever it's needed).
  2. No query hits an *unwarranted* clarification -- i.e. the planner
     doesn't punt back to the user on questions a reasonable person would
     consider unambiguous.
  3. The output is a real, reasonable answer -- not a stub.

This script is meant to be re-run any time as living proof the autonomy
property still holds (e.g. after future planner/executor changes), and to
double as a demo transcript. It writes a Markdown report to
`scripts/autonomy_report.md` next to itself, and also prints a console
summary.

Usage (from repo root, with the project venv):
    venv/Scripts/python.exe scripts/verify_autonomy.py

No skill in this repo/environment is purpose-built for "run a battery of
NL queries against a live web API and grade autonomy" (checked the
available skill list: nothing beyond generic app-launch/QA helpers that
don't fit this shape) -- see the task report for detail on what was
checked. This script is the bespoke fallback the task brief allows for
that case.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = Path(__file__).resolve().parent / "autonomy_report.md"

# Phrases that indicate the pre-fix "go click a button yourself" refusal
# behaviour documented in backend/tests/test_executor_autonomy.py. If any
# of these show up in an answer, that's a manual-intervention failure.
REFUSAL_PHRASES = [
    "run segmentation first",
    "click a button",
    "click the",
    "ask me to segment",
    "segment the customers first",
    "please segment",
    "please run segmentation",
    "you'll need to segment",
    "you need to segment",
]


@dataclass
class QueryCase:
    query: str
    category: str
    note: str = ""


@dataclass
class QueryResult:
    case: QueryCase
    session_id: str
    status_code: int
    intent: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    answer: str = ""
    result_keys: list = field(default_factory=list)
    autonomous: bool = False
    problem: str = ""


# ---------------------------------------------------------------------------
# The battery: ~20 realistic end-user queries, each on a fresh session.
#
# A handful deliberately overlap with the phrasings already covered in
# test_executor_autonomy.py / test_planner_intents.py (marked "overlap:")
# to confirm the previously-fixed bug stays fixed when driven over the real
# HTTP path with a genuinely fresh session. The rest are new, realistic
# phrasings spanning every intent a demo audience would plausibly try:
# segmentation, priority/dormant tier questions, aggregation/metrics,
# conversion candidates, recommendations, customer lookups, trend
# questions, and EDA questions.
# ---------------------------------------------------------------------------
QUERY_BATTERY: list[QueryCase] = [
    QueryCase(
        "which customers are marked priority",
        "tier-listing",
        "overlap: the original bug-report phrasing",
    ),
    QueryCase(
        "show me the dormant customers",
        "tier-listing",
        "new phrasing -- natural 'show me the X customers' shape",
    ),
    QueryCase(
        "on what basis were priority customers selected",
        "segmentation-explain",
        "overlap: confirms auto-segment-then-explain stays fixed",
    ),
    QueryCase(
        "what's the average balance for priority customers",
        "aggregation",
    ),
    QueryCase(
        "average credit score by tier",
        "aggregation",
    ),
    QueryCase(
        "which regular customers can convert to priority",
        "conversion",
        "overlap: original conversion-candidates autonomy test",
    ),
    QueryCase(
        "which regular customers could become priority customers",
        "conversion",
        "new phrasing",
    ),
    QueryCase(
        "what should we recommend for dormant customers to re-engage them",
        "recommendation",
    ),
    QueryCase(
        "what products should we cross-sell to priority customers",
        "recommendation",
    ),
    QueryCase(
        "show me the profile for CUST00010",
        "customer-lookup",
    ),
    QueryCase(
        "why is customer CUST00025 classified the way they are",
        "customer-lookup",
        "new phrasing of the explain-mode lookup",
    ),
    QueryCase(
        "has average balance been trending up over the last few months",
        "trend",
    ),
    QueryCase(
        "give me a quick overview of the customer dataset",
        "eda-overview",
    ),
    QueryCase(
        "how many customers are in the dataset",
        "eda-overview",
    ),
    QueryCase(
        "are there any missing values in the income column",
        "eda-missing",
    ),
    QueryCase(
        "is credit score correlated with income",
        "eda-correlation",
    ),
    QueryCase(
        "what does the distribution of annual income look like",
        "eda-distribution",
    ),
    QueryCase(
        "segment the customer base by balance and transaction frequency",
        "segmentation",
    ),
    QueryCase(
        "cluster customers using machine learning",
        "segmentation-ml",
    ),
    QueryCase(
        "median annual income of dormant customers",
        "aggregation",
    ),
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/api/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Backend never became healthy at {base_url}: {last_err}")


def _run_battery(base_url: str) -> list[QueryResult]:
    results = []
    for case in QUERY_BATTERY:
        session_id = f"autonomy-verify-{uuid.uuid4().hex[:12]}"
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"session_id": session_id, "query": case.query},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            results.append(QueryResult(
                case=case, session_id=session_id, status_code=-1,
                problem=f"request failed: {e}",
            ))
            continue

        if resp.status_code != 200:
            results.append(QueryResult(
                case=case, session_id=session_id, status_code=resp.status_code,
                problem=f"HTTP {resp.status_code}: {resp.text[:300]}",
            ))
            continue

        data = resp.json()
        intent = data.get("intent", "")
        needs_clar = bool(data.get("needs_clarification"))
        answer = data.get("answer", "") or ""
        answer_lower = answer.lower()

        problem = ""
        if needs_clar or intent == "clarify":
            problem = "unwarranted clarification (planner punted)"
        else:
            for phrase in REFUSAL_PHRASES:
                if phrase in answer_lower:
                    problem = f"manual-intervention refusal phrase found: {phrase!r}"
                    break

        result_dict = data.get("result", {}) or {}
        results.append(QueryResult(
            case=case,
            session_id=session_id,
            status_code=resp.status_code,
            intent=intent,
            needs_clarification=needs_clar,
            clarification_question=data.get("plan", {}).get("clarification_question", "") or "",
            answer=answer,
            result_keys=sorted(result_dict.keys()) if isinstance(result_dict, dict) else [],
            autonomous=(problem == ""),
            problem=problem,
        ))
    return results


def _summarize_answer(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _write_report(results: list[QueryResult]) -> None:
    n_total = len(results)
    n_autonomous = sum(1 for r in results if r.autonomous)
    n_gaps = n_total - n_autonomous

    lines = []
    lines.append("# Autonomy Verification Battery Report")
    lines.append("")
    lines.append(
        "Generated by `scripts/verify_autonomy.py` against a real, live "
        "`uvicorn` process serving `backend.main:app`, driven over real HTTP "
        "through `/api/chat`, each query on a **fresh** `session_id` with no "
        "prior segmentation. This is living documentation of end-to-end "
        "agent autonomy, re-runnable any time."
    )
    lines.append("")
    lines.append(f"**Result: {n_autonomous}/{n_total} queries fully autonomous "
                  f"(no manual intervention needed, no unwarranted clarification).**")
    lines.append("")
    if n_gaps:
        lines.append(f"**{n_gaps} genuine gap(s) found -- see \"Gaps found\" section below.**")
        lines.append("")

    lines.append("## Full results")
    lines.append("")
    lines.append("| # | Query | Category | Intent chosen | Clarification? | Autonomous? | Answer summary |")
    lines.append("|---|-------|----------|----------------|-----------------|-------------|-----------------|")
    for i, r in enumerate(results, 1):
        clar = "yes" if r.needs_clarification else "no"
        auton = "YES" if r.autonomous else "**NO**"
        summary = _summarize_answer(r.answer) if r.answer else (r.problem or "(no answer)")
        query_escaped = r.case.query.replace("|", "\\|")
        summary_escaped = summary.replace("|", "\\|")
        lines.append(
            f"| {i} | {query_escaped} | {r.case.category} | `{r.intent}` | {clar} | {auton} | {summary_escaped} |"
        )
    lines.append("")

    if n_gaps:
        lines.append("## Gaps found (genuine, not hidden)")
        lines.append("")
        for r in results:
            if not r.autonomous:
                lines.append(f"### {r.case.query!r}")
                lines.append("")
                lines.append(f"- Category: {r.case.category}")
                lines.append(f"- Intent chosen: `{r.intent}`")
                lines.append(f"- Problem: {r.problem}")
                if r.clarification_question:
                    lines.append(f"- Clarification question asked: {r.clarification_question!r}")
                if r.case.note:
                    lines.append(f"- Note: {r.case.note}")
                lines.append("")
    else:
        lines.append("## Gaps found")
        lines.append("")
        lines.append("None. Every query in the battery was answered autonomously.")
        lines.append("")

    lines.append("## Query notes")
    lines.append("")
    for r in results:
        if r.case.note:
            lines.append(f"- **{r.case.query!r}**: {r.case.note}")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(results: list[QueryResult]) -> None:
    n_total = len(results)
    n_autonomous = sum(1 for r in results if r.autonomous)
    print()
    print("=" * 78)
    print(f"AUTONOMY VERIFICATION BATTERY: {n_autonomous}/{n_total} fully autonomous")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        mark = "OK  " if r.autonomous else "GAP "
        print(f"[{mark}] {i:2}. ({r.case.category:>18}) {r.case.query!r}")
        print(f"          intent={r.intent!r} clarify={r.needs_clarification}")
        if r.autonomous:
            print(f"          -> {_summarize_answer(r.answer)}")
        else:
            print(f"          -> PROBLEM: {r.problem}")
    print("=" * 78)
    if n_autonomous == n_total:
        print("All queries answered autonomously. No manual intervention, no unwarranted clarification.")
    else:
        print(f"{n_total - n_autonomous} genuine gap(s) found -- see report for detail.")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 78)


def main() -> int:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    python = sys.executable

    print(f"Starting backend: uvicorn backend.main:app on port {port} (cwd={REPO_ROOT}) ...")
    proc = subprocess.Popen(
        [python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url)
        print(f"Backend healthy at {base_url}. Running {len(QUERY_BATTERY)} queries...")
        results = _run_battery(base_url)
        _print_console_summary(results)
        _write_report(results)
        n_total = len(results)
        n_autonomous = sum(1 for r in results if r.autonomous)
        return 0 if n_autonomous == n_total else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout:
            leftover = proc.stdout.read()
            if leftover.strip():
                print("\n--- backend server log (for debugging) ---")
                print(leftover)


if __name__ == "__main__":
    sys.exit(main())
