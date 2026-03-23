"""
app.py — Streamlit UI for the Autonomous Coding Teammate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# If invoked via `python app.py`, relaunch through `streamlit run` so the
# ScriptRunContext exists and Streamlit widgets work correctly. When already
# under Streamlit (context exists), skip to avoid a double-launch.
if __name__ == "__main__" and get_script_run_ctx() is None:
    import sys
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

from orchestrator.controller import Controller, RunSummary
from orchestrator.memory import Memory, RunRecord

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Autonomous Coding Teammate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────

if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ─────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    sandbox_mode = st.selectbox(
        "Sandbox Mode",
        ["local", "docker"],
        help="'local' runs pytest in your current environment. "
             "'docker' isolates execution in a container (requires Docker).",
    )
    max_iters = st.slider(
        "Max Patch Iterations",
        min_value=1, max_value=8, value=4,
        help="How many times the agent will attempt to fix failing tests.",
    )
    low_confidence_gate = st.toggle(
        "Stop on Low-Confidence Patches",
        value=True,
        help="Halt the loop if the analyzer is uncertain, to avoid churn.",
    )
    runs_dir = st.text_input("Runs Directory", value="examples")

    st.divider()
    st.caption("Autonomous Coding Teammate · powered by GPT-4o")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _status_badge(success: bool) -> str:
    return "✅ Passed" if success else "❌ Failed"


def _confidence_color(confidence: str) -> str:
    return {"high": "green", "medium": "orange", "low": "red"}.get(confidence, "grey")


def _read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return "(could not read file)"


def _render_plan(summary: RunSummary) -> None:
    if not summary.plan or not summary.plan.tasks:
        st.info("No plan was produced for this run.")
        return

    st.markdown(f"**{len(summary.plan.tasks)} tasks · ~{summary.plan.total_hours:.1f} h estimated**")

    needs_clarification = summary.plan.needs_clarification()
    if needs_clarification:
        st.warning(
            f"⚠️ {len(needs_clarification)} task(s) flagged for clarification: "
            + ", ".join(t.id for t in needs_clarification)
        )

    for task in summary.plan.execution_order():
        with st.expander(f"[{task.id}] {task.title}  ({task.est_hours}h)", expanded=False):
            st.markdown(task.description)
            if task.depends_on:
                st.caption(f"Depends on: {', '.join(task.depends_on)}")
            if task.test_hint:
                st.caption(f"Test hint: {task.test_hint}")
            if task.clarification_needed:
                st.warning(f"Clarification needed: {task.clarification_needed}")


def _render_iterations(summary: RunSummary) -> None:
    if not summary.iterations:
        st.info("No iteration records.")
        return

    for rec in summary.iterations:
        icon = "✅" if rec.passed else "❌"
        label = f"{icon} Iteration {rec.iteration}"
        with st.expander(label, expanded=(rec.iteration == 1)):

            col1, col2 = st.columns(2)
            col1.metric("Return Code", rec.test_result.get("rc", "—"))
            col2.metric("Elapsed", f"{rec.test_result.get('elapsed_seconds', 0):.1f}s")

            if rec.test_result.get("stdout"):
                st.markdown("**stdout**")
                st.code(rec.test_result["stdout"], language="text")

            if rec.test_result.get("stderr"):
                st.markdown("**stderr**")
                st.code(rec.test_result["stderr"], language="text")

            if rec.analysis:
                st.divider()
                confidence = rec.analysis.confidence
                color = _confidence_color(confidence)
                st.markdown(
                    f"**Analyzer** &nbsp; "
                    f"<span style='color:{color};font-weight:600'>{confidence.upper()} confidence</span>",
                    unsafe_allow_html=True,
                )
                if rec.analysis.root_cause:
                    st.markdown(f"**Root cause:** {rec.analysis.root_cause}")
                if rec.analysis.explanation:
                    st.markdown(rec.analysis.explanation)
                if rec.patched_files:
                    st.markdown("**Patched files:**")
                    for pf in rec.patched_files:
                        st.code(pf, language="text")


def _render_generated_files(summary: RunSummary) -> None:
    run_dir = Path(summary.run_dir)
    if not run_dir.exists():
        st.info("Run directory not found on disk.")
        return

    py_files = sorted(run_dir.rglob("*.py"))
    other_files = [
        f for f in sorted(run_dir.rglob("*"))
        if f.is_file() and f.suffix != ".py"
    ]

    all_files = py_files + other_files
    if not all_files:
        st.info("No files found in run directory.")
        return

    for file_path in all_files:
        rel = file_path.relative_to(run_dir)
        lang = "python" if file_path.suffix == ".py" else "text"
        with st.expander(str(rel), expanded=False):
            st.code(_read_file_safe(file_path), language=lang)


def _render_run_summary(summary: RunSummary) -> None:
    """Render a full RunSummary in a tabbed layout."""

    # Header strip
    status_col, id_col, iter_col = st.columns([2, 3, 2])
    status_col.markdown(f"### {_status_badge(summary.success)}")
    id_col.markdown(f"**Run ID:** `{summary.run_id}`")
    iter_col.metric(
        "Iterations used",
        f"{summary.iterations_used} / {summary.max_iterations}",
    )

    if summary.error:
        st.error(f"**Error:** {summary.error}")

    tab_plan, tab_iters, tab_files, tab_logs = st.tabs(
        ["📋 Plan", "🔁 Iterations", "📂 Files", "📜 Logs"]
    )

    with tab_plan:
        _render_plan(summary)

    with tab_iters:
        _render_iterations(summary)

    with tab_files:
        _render_generated_files(summary)

    with tab_logs:
        log_path = Path(f"report/logs/{summary.run_id}.jsonl")
        if log_path.exists():
            st.download_button(
                "⬇️ Download log",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="application/x-ndjson",
            )
            entries = []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        import json
                        entries.append(json.loads(line))
                    except Exception:
                        pass

            level_filter = st.multiselect(
                "Filter by level",
                ["info", "warning", "error"],
                default=["info", "warning", "error"],
            )
            for entry in entries:
                if entry.get("level", "info") in level_filter:
                    level = entry.get("level", "info")
                    icon = {"info": "ℹ️", "warning": "⚠️", "error": "🔴"}.get(level, "•")
                    st.markdown(
                        f"`{entry.get('time','')[:19]}` {icon} "
                        f"**{entry.get('stage','')}** — {entry.get('detail','')}"
                    )
        else:
            st.info("No log file found for this run.")


# ─────────────────────────────────────────────────────────────
# Recent runs panel
# ─────────────────────────────────────────────────────────────

def _render_recent_runs() -> None:
    db_path = "memory.sqlite"
    if not os.path.exists(db_path):
        st.info("No run history yet.")
        return

    try:
        mem = Memory(db_path=db_path)
        recent: list[RunRecord] = mem.recent_runs(limit=10)
        mem.close()
    except Exception as exc:
        st.error(f"Could not load run history: {exc}")
        return

    if not recent:
        st.info("No runs recorded yet.")
        return

    for record in recent:
        badge = "✅" if record.success else "❌"
        elapsed = "—"
        if record.start_time and record.end_time:
            try:
                from datetime import datetime, timezone
                fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
                s = datetime.fromisoformat(record.start_time)
                e = datetime.fromisoformat(record.end_time)
                elapsed = f"{(e - s).total_seconds():.1f}s"
            except Exception:
                pass

        with st.expander(
            f"{badge} `{record.run_id}` · {record.requirement[:60]}…",
            expanded=False,
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Success", "Yes" if record.success else "No")
            col2.metric("Iterations", record.iterations)
            col3.metric("Elapsed", elapsed)
            st.caption(f"Started: {record.start_time[:19] if record.start_time else '—'}")
            st.markdown(f"**Requirement:** {record.requirement}")


# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────

st.title("🤖 Autonomous Coding Teammate")
st.caption("Describe what you want built. The agent plans, generates, tests, and patches automatically.")

# ── Input form ───────────────────────────────────────────────

with st.form("run_form"):
    requirement = st.text_area(
        "Requirement",
        value=(
            "Create a command-line script 'reverse_string.py' that reads a string "
            "from stdin and prints the reversed string. "
            "Include a pytest that confirms 'abcd' → 'dcba'."
        ),
        height=120,
        help="Describe exactly what to build. Be specific about filenames, behavior, and edge cases.",
    )
    submitted = st.form_submit_button("🚀 Run Agent", type="primary", use_container_width=True)

# ── Run ──────────────────────────────────────────────────────

if submitted:
    if not requirement.strip():
        st.error("Please enter a requirement before running.")
    else:
        progress_bar = st.progress(0, text="Initialising…")
        status_area = st.empty()

        stages = [
            (15, "Planning tasks…"),
            (35, "Generating code…"),
            (55, "Writing files…"),
            (75, "Running tests…"),
            (90, "Analysing results…"),
            (100, "Finalising…"),
        ]

        def _tick(i: int) -> None:
            if i < len(stages):
                pct, label = stages[i]
                progress_bar.progress(pct, text=label)
                status_area.caption(label)

        _tick(0)

        try:
            ctrl = Controller(
                mode=sandbox_mode,
                max_iters=max_iters,
                runs_dir=runs_dir,
                low_confidence_threshold=low_confidence_gate,
            )

            # Simulate stage ticks in a thread so the UI doesn't freeze
            import threading

            tick_idx = [1]

            def _advance() -> None:
                for i in range(1, len(stages)):
                    time.sleep(2.5)
                    _tick(i)
                    tick_idx[0] = i

            ticker = threading.Thread(target=_advance, daemon=True)
            ctx = get_script_run_ctx()
            if ctx is not None:
                add_script_run_ctx(ticker, ctx)
            ticker.start()

            summary: RunSummary = ctrl.run(requirement)
            ticker.join(timeout=1)

            progress_bar.progress(100, text="Done.")
            status_area.empty()

        except Exception as exc:
            progress_bar.empty()
            status_area.empty()
            st.error(f"Agent raised an unexpected error: {exc}")
            st.stop()

        st.session_state.last_summary = summary
        st.session_state.run_history.insert(0, summary)

        if summary.success:
            st.success(f"✅ Run `{summary.run_id}` completed successfully in {summary.iterations_used} iteration(s).")
        else:
            msg = f"❌ Run `{summary.run_id}` did not pass all tests after {summary.iterations_used} iteration(s)."
            if summary.error:
                msg += f"\n\n**Details:** {summary.error}"
            st.error(msg)

# ── Result display ───────────────────────────────────────────

if st.session_state.last_summary:
    st.divider()
    st.subheader("Latest Run")
    _render_run_summary(st.session_state.last_summary)

# ── History ──────────────────────────────────────────────────

st.divider()
st.subheader("📚 Recent Runs")
_render_recent_runs()