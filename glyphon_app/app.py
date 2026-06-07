from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz
import pandas as pd
import streamlit as st

from glyphon_app.api_client import ApiConfig, ApiError, MergeClient, PHASES, TableFixerClient
from glyphon_app.review_component import HAS_COMPONENT_V2, normalized_row_cells, render_review, review_payload
from glyphon_app.state import PageJob, append_log, first_next_phase, merge_fragments, phase_state, remaining_phases


st.set_page_config(page_title="Glyphon Review", layout="wide")


APP_CSS = """
<style>
:root {
  color-scheme: light dark;
}
.glyphon-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.glyphon-title h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0;
}
.phase-strip {
  display: grid;
  grid-template-columns: repeat(7, minmax(96px, 1fr));
  gap: 8px;
  margin: 12px 0;
}
.phase-chip {
  border: 1px solid color-mix(in srgb, CanvasText 18%, transparent);
  border-radius: 8px;
  padding: 9px 10px;
  background: color-mix(in srgb, Canvas 94%, CanvasText 6%);
  font-size: 13px;
}
.phase-chip strong {
  display: block;
  margin-bottom: 2px;
  font-size: 12px;
}
.phase-chip.done {
  border-color: #059669;
  box-shadow: inset 0 3px 0 #059669;
}
.phase-chip.stale {
  border-color: #d97706;
  box-shadow: inset 0 3px 0 #d97706;
}
.phase-chip.review {
  border-color: #2563eb;
  box-shadow: inset 0 3px 0 #2563eb;
}
.phase-chip.active {
  outline: 2px solid color-mix(in srgb, CanvasText 55%, transparent);
}
.queue-ok {
  color: #059669;
  font-weight: 650;
}
.queue-warn {
  color: #d97706;
  font-weight: 650;
}
.notice-bar {
  display: grid;
  gap: 10px;
  margin: 0 0 14px;
}
.notice-workspace {
  border: 1px solid color-mix(in srgb, CanvasText 14%, transparent);
  border-radius: 10px;
  background: color-mix(in srgb, Canvas 95%, CanvasText 5%);
  padding: 10px 12px;
}
.notice-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
}
.notice-head strong {
  font-size: 13px;
}
.notice-list {
  display: grid;
  gap: 6px;
}
.notice-item {
  display: grid;
  grid-template-columns: 64px 72px 78px 1fr;
  gap: 8px;
  align-items: start;
  font-size: 12px;
  padding: 7px 8px;
  border-radius: 8px;
  background: color-mix(in srgb, Canvas 90%, CanvasText 10%);
}
.notice-time,
.notice-phase,
.notice-status {
  color: color-mix(in srgb, CanvasText 72%, transparent);
}
.notice-status.done {
  color: #059669;
  font-weight: 650;
}
.notice-status.running,
.notice-status.review {
  color: #2563eb;
  font-weight: 650;
}
.notice-status.stale,
.notice-status.rejected {
  color: #d97706;
  font-weight: 650;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


def app_config() -> ApiConfig:
    with st.sidebar:
        st.header("API")
        table_url = st.text_input("Table fixer URL", value="http://localhost:8770")
        merge_url = st.text_input("Merge URL", value="http://localhost:8780")
        st.header("LLM")
        model = st.text_input("Model", value="gemma3:4b")
        base_url = st.text_input("Ollama URL", value="http://localhost:11434")
        token_budget = st.number_input("Context tokens", min_value=500, max_value=16000, value=3500, step=250)
        threshold = st.slider("Auto-apply confidence", 0.5, 1.0, 0.80, 0.01)
        structural = st.slider("Structural confidence", 0.8, 1.0, 0.95, 0.01)
    return ApiConfig(
        table_fixer_url=table_url,
        merge_url=merge_url,
        model=model,
        base_url=base_url,
        context_token_budget=int(token_budget),
        auto_apply_threshold=float(threshold),
        structural_auto_apply_threshold=float(structural),
    )


def reset_run_state(file_key: str) -> None:
    st.session_state.file_key = file_key
    st.session_state.jobs = []
    st.session_state.active_page = None
    st.session_state.active_phase = "source"
    st.session_state.merge_result = None
    st.session_state.show_rerun = False


def jobs() -> list[PageJob]:
    return st.session_state.setdefault("jobs", [])


def store_job(job: PageJob) -> None:
    for index, existing in enumerate(jobs()):
        if existing.page_number == job.page_number:
            jobs()[index] = job
            return
    jobs().append(job)


def run_initial(
    *,
    client: TableFixerClient,
    pdf_bytes: bytes,
    filename: str,
    pages: list[int],
    extraction_mode: str,
    run_style: str,
) -> None:
    st.session_state.jobs = []
    st.session_state.merge_result = None
    phases = PHASES if run_style == "Run all phases" else None
    execution_mode = "auto_apply" if phases else "auto_apply"
    progress = st.progress(0)
    for offset, page in enumerate(pages, start=1):
        response = client.create_workspace(
            pdf_bytes=pdf_bytes,
            filename=filename,
            page_numbers=[page],
            extraction_mode=extraction_mode,
            phases=phases,
            execution_mode=execution_mode,
        )
        job = PageJob(page_number=page, workspace_id=response["workspace_id"], response=response)
        append_log(job, "source", "done", "Workspace created")
        for run in response.get("phase_runs", []):
            append_log(
                job,
                run["phase"],
                "done",
                f"{run['applied_decision_count']} decisions applied; {run['warning_count_after_phase']} warnings remain",
            )
        store_job(job)
        progress.progress(offset / len(pages))
    st.session_state.active_page = pages[0] if pages else None
    st.session_state.active_phase = "warnings" if phases else "source"


def refresh_job(client: TableFixerClient, job: PageJob) -> PageJob:
    job.response = client.get_workspace(job.workspace_id)
    return job


def execute_next_phase(client: TableFixerClient, execution_mode: str) -> None:
    target = first_next_phase(jobs())
    if not target:
        return
    for job in jobs():
        if target not in job.next_phases:
            continue
        append_log(job, target, "running", f"Running {target} in {execution_mode}")
        job.response = client.execute(job.workspace_id, [target], execution_mode)
        run = (job.response.get("phase_runs") or [{}])[-1]
        append_log(
            job,
            target,
            "done" if execution_mode == "auto_apply" else "review",
            f"{run.get('decision_count', 0)} decisions returned",
        )
    st.session_state.active_phase = target


def execute_all_remaining(client: TableFixerClient) -> None:
    for job in jobs():
        phases = job.next_phases
        if not phases:
            continue
        append_log(job, "downstream", "running", f"Running {', '.join(phases)}")
        job.response = client.execute(job.workspace_id, phases, "auto_apply")
        for run in job.response.get("phase_runs", []):
            append_log(
                job,
                run["phase"],
                "done",
                f"{run['applied_decision_count']} decisions applied; {run['warning_count_after_phase']} warnings remain",
            )
    st.session_state.active_phase = "warnings"


def run_merge_if_ready(merge_client: MergeClient) -> None:
    if not jobs():
        return
    if any(job.next_phases or job.response.get("pending_review") for job in jobs()):
        return
    fragments = merge_fragments(jobs())
    if not fragments:
        return
    st.session_state.merge_result = merge_client.merge_decisions(fragments, llm=True)
    st.session_state.active_phase = "merge"


def queue_frame() -> pd.DataFrame:
    records = []
    for job in jobs():
        records.extend(job.logs)
        accepted = ", ".join(job.status.get("accepted_phases", [])) or "source"
        records.append(
            {
                "page": job.page_number,
                "phase": "workspace",
                "status": "stale" if job.invalidated_phases else "ready",
                "message": f"Accepted: {accepted}; next: {', '.join(job.next_phases) or 'none'}",
            }
        )
    return pd.DataFrame(records)


def render_notifications_bar() -> None:
    if not jobs():
        return
    blocks: list[str] = ['<div class="notice-bar">']
    for job in sorted(jobs(), key=lambda item: item.page_number):
        entries = list(reversed(job.logs[-5:]))
        latest = entries[0]["timestamp"] if entries else "--:--:--"
        blocks.append(
            "<section class=\"notice-workspace\">"
            f"<div class=\"notice-head\"><strong>Workspace {job.workspace_id}</strong>"
            f"<span>Page {job.page_number} | Last update {latest}</span></div>"
        )
        if entries:
            blocks.append('<div class="notice-list">')
            for entry in entries:
                status_class = str(entry["status"]).replace(" ", "-").lower()
                blocks.append(
                    "<div class=\"notice-item\">"
                    f"<span class=\"notice-time\">{entry['timestamp']}</span>"
                    f"<span class=\"notice-phase\">{entry['phase']}</span>"
                    f"<span class=\"notice-status {status_class}\">{entry['status']}</span>"
                    f"<span>{entry['message']}</span>"
                    "</div>"
                )
            blocks.append("</div>")
        else:
            blocks.append("<div class=\"notice-list\"><div class=\"notice-item\"><span class=\"notice-time\">--:--:--</span><span class=\"notice-phase\">workspace</span><span class=\"notice-status\">ready</span><span>No activity yet.</span></div></div>")
        blocks.append("</section>")
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_phase_strip(job: PageJob) -> None:
    phases = ["source", *PHASES, "merge"]
    labels = {
        "source": "Source",
        "reconciliation": "Reconcile",
        "metadata": "Metadata",
        "headers": "Headers",
        "columns": "Columns",
        "warnings": "Warnings",
        "merge": "Merge",
    }
    html = ['<div class="phase-strip">']
    for phase in phases:
        if phase == "source":
            state = "done"
        elif phase == "merge":
            state = "done" if st.session_state.get("merge_result") else "pending"
        else:
            state = phase_state(job, phase)
        active = " active" if st.session_state.get("active_phase") == phase else ""
        html.append(
            f'<div class="phase-chip {state}{active}"><strong>{labels[phase]}</strong>{state}</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    cols = st.columns(len(phases))
    for col, phase in zip(cols, phases):
        if col.button(labels[phase], key=f"phase_{phase}_{job.workspace_id}", use_container_width=True):
            st.session_state.active_phase = phase
            if phase in job.invalidated_phases:
                st.session_state.show_rerun = True
            st.rerun()


def selected_job() -> PageJob | None:
    if not jobs():
        return None
    active_page = st.session_state.get("active_page") or jobs()[0].page_number
    for job in jobs():
        if job.page_number == active_page:
            return job
    return jobs()[0]


def phase_result(client: TableFixerClient, job: PageJob, phase: str) -> dict[str, Any]:
    pending = job.response.get("pending_review")
    if pending and pending.get("phase") == phase:
        return pending["proposed_result"]
    if phase == "merge":
        return {}
    if phase == job.response.get("result", {}).get("phase"):
        return job.response["result"]
    return client.get_snapshot(job.workspace_id, phase)


def render_pending_controls(client: TableFixerClient, job: PageJob) -> None:
    pending = job.response.get("pending_review")
    if not pending:
        return
    phase = pending["phase"]
    st.info(f"`{phase}` is waiting for review on page {job.page_number}.")
    accept, reject = st.columns(2)
    if accept.button("Accept Preview", type="primary", use_container_width=True):
        job.response = client.review(job.workspace_id, phase, "accept")
        append_log(job, phase, "done", "Preview accepted")
        st.rerun()
    if reject.button("Reject Preview", use_container_width=True):
        job.response = client.review(job.workspace_id, phase, "reject")
        append_log(job, phase, "rejected", "Preview rejected")
        st.rerun()


def render_merge() -> None:
    merge_result = st.session_state.get("merge_result")
    if not merge_result:
        st.info("Merge has not run yet. Complete all page-level warning phases first.")
        return
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Merge Groups")
        st.dataframe(pd.DataFrame(merge_result.get("merge_groups", [])), use_container_width=True)
    with right:
        st.subheader("Boundary Decisions")
        st.dataframe(pd.DataFrame(merge_result.get("decisions", [])), use_container_width=True)


def editable_cells(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = table.get("metadata_rows", []) + table.get("header_rows", []) + table.get("data_rows", [])
    return [cell for row in rows for cell in normalized_row_cells(row)]


def rerun_dialog(client: TableFixerClient, merge_client: MergeClient) -> None:
    def body() -> None:
        pending = remaining_phases(jobs())
        st.write("Downstream phases are stale or not yet complete. Choose how to continue.")
        st.caption(f"Remaining phase sequence: {', '.join(pending) or 'none'}")
        phase_col, all_col = st.columns(2)
        if phase_col.button("Run phase by phase", type="primary", use_container_width=True):
            execute_next_phase(client, "preview")
            st.session_state.show_rerun = False
            st.rerun()
        if all_col.button("Run all remaining phases", use_container_width=True):
            execute_all_remaining(client)
            run_merge_if_ready(merge_client)
            st.session_state.show_rerun = False
            st.rerun()
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_rerun = False
            st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Rerun Downstream Phases")
        def modal() -> None:
            body()

        modal()
    else:
        with st.container():
            body()


config = app_config()
client = TableFixerClient(config)
merge_client = MergeClient(config)

st.markdown(
    '<div class="glyphon-title"><h1>Glyphon Table Review</h1><span>API-backed extraction, review, rerun, and merge</span></div>',
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to start.")
    st.stop()

pdf_bytes = uploaded.getvalue()
file_key = sha256(pdf_bytes).hexdigest()
if st.session_state.get("file_key") != file_key:
    reset_run_state(file_key)

total_pages = page_count(pdf_bytes)
with st.sidebar:
    st.header("Extraction")
    if hasattr(st, "segmented_control"):
        extraction_mode = st.segmented_control("Mode", ["auto", "text", "ocr"], default="auto")
    else:
        extraction_mode = st.radio("Mode", ["auto", "text", "ocr"], horizontal=True)
    selected_pages = st.multiselect("Pages", list(range(1, total_pages + 1)), default=list(range(1, total_pages + 1)))
    run_style = st.radio("Initial run", ["Run phase by phase", "Run all phases"], index=0)
    if st.button("Create Run", type="primary", use_container_width=True):
        try:
            with st.spinner("Creating page workspaces..."):
                run_initial(
                    client=client,
                    pdf_bytes=pdf_bytes,
                    filename=uploaded.name,
                    pages=selected_pages or list(range(1, total_pages + 1)),
                    extraction_mode=extraction_mode or "auto",
                    run_style=run_style,
                )
                if run_style == "Run all phases":
                    run_merge_if_ready(merge_client)
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

if not jobs():
    st.info("Choose extraction settings, then create a run.")
    st.stop()

render_notifications_bar()

top = st.columns([1.1, 1, 1])
with top[0]:
    page_options = [job.page_number for job in jobs()]
    active_page = st.selectbox(
        "Page workspace",
        page_options,
        index=page_options.index(st.session_state.get("active_page") or page_options[0]),
    )
    st.session_state.active_page = active_page
with top[1]:
    if st.button("Next Phase", type="primary", use_container_width=True):
        st.session_state.show_rerun = True
        st.rerun()
with top[2]:
    if st.button("Run Merge", use_container_width=True):
        try:
            run_merge_if_ready(merge_client)
            st.session_state.active_phase = "merge"
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

job = selected_job()
assert job is not None
render_phase_strip(job)

if st.session_state.get("show_rerun"):
    rerun_dialog(client, merge_client)

with st.expander("Queue And Logs", expanded=True):
    frame = queue_frame()
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True, height=220)

phase = st.session_state.get("active_phase", "source")
render_pending_controls(client, job)

undo_col, redo_col, refresh_col = st.columns([1, 1, 2])
if undo_col.button("Undo Manual Edit", use_container_width=True):
    try:
        job.response = client.undo(job.workspace_id)
        append_log(job, phase, "stale", "Manual edit undone; downstream invalidated")
        st.rerun()
    except ApiError as exc:
        st.warning(str(exc))
if redo_col.button("Redo Manual Edit", use_container_width=True):
    try:
        job.response = client.redo(job.workspace_id)
        append_log(job, phase, "stale", "Manual edit redone; downstream invalidated")
        st.rerun()
    except ApiError as exc:
        st.warning(str(exc))
if refresh_col.button("Refresh Workspace", use_container_width=True):
    job.response = client.get_workspace(job.workspace_id)
    st.rerun()

if phase == "merge":
    render_merge()
    st.stop()

try:
    result = phase_result(client, job, phase)
except ApiError as exc:
    st.warning(str(exc))
    result = job.response.get("result", {})

tables = result.get("tables", [])
if not tables:
    st.info("No tables are available for this phase.")
    st.stop()

table_id = st.selectbox("Table", [table["table_id"] for table in tables])
table = next(table for table in tables if table["table_id"] == table_id)
image_url = client.page_image_url(job.workspace_id, int(table.get("page_number", job.page_number)), zoom=2.0)
event = render_review(
    review_payload(result=result, table_id=table_id, image_url=image_url),
    key=f"review_{job.workspace_id}_{phase}_{table_id}",
)
edit_event = getattr(event, "edit_cell_text", None)

if isinstance(edit_event, dict) and "cell_id" in edit_event:
    accepted = set(job.status.get("accepted_phases", []))
    editable_phase = phase == "source" or phase in accepted
    if not editable_phase:
        st.warning("Accept or reject the pending preview before making manual edits.")
    else:
        try:
            job.response = client.manual_actions(
                job.workspace_id,
                base_phase=phase,
                actions=[{"type": "edit_cell_text", **edit_event}],
                note="Edited from Glyphon Streamlit review component.",
            )
            append_log(job, phase, "stale", "Manual cell edit applied; downstream invalidated")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

if not HAS_COMPONENT_V2:
    st.info("This Streamlit version does not support component callbacks. PDF and table selection still render, and manual edits are available below.")
    cells = editable_cells(table)
    if cells:
        labels = {
            cell["cell_id"]: f'{cell["header"]} | {cell["row_id"]} | {cell.get("text", "")[:60]}'
            for cell in cells
        }
        selected_cell_id = st.selectbox(
            "Cell to edit",
            list(labels),
            format_func=lambda cell_id: labels[cell_id],
            key=f"fallback_cell_{job.workspace_id}_{phase}_{table_id}",
        )
        selected_cell = next(cell for cell in cells if cell["cell_id"] == selected_cell_id)
        fallback_text = st.text_area(
            "Edited text",
            value=selected_cell.get("text", ""),
            key=f"fallback_text_{job.workspace_id}_{phase}_{selected_cell_id}",
            height=100,
        )
        if st.button("Apply Cell Edit", type="primary", use_container_width=True):
            accepted = set(job.status.get("accepted_phases", []))
            editable_phase = phase == "source" or phase in accepted
            if not editable_phase:
                st.warning("Accept or reject the pending preview before making manual edits.")
            else:
                try:
                    job.response = client.manual_actions(
                        job.workspace_id,
                        base_phase=phase,
                        actions=[{"type": "edit_cell_text", "cell_id": selected_cell_id, "text": fallback_text}],
                        note="Edited from Glyphon Streamlit fallback editor.",
                    )
                    append_log(job, phase, "stale", "Manual cell edit applied; downstream invalidated")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))

with st.expander("Phase Details", expanded=False):
    metrics = st.columns(4)
    metrics[0].metric("Tables", len(result.get("tables", [])))
    metrics[1].metric("Warnings", len(result.get("warnings", [])))
    metrics[2].metric("Decisions", len(result.get("decisions", [])))
    metrics[3].metric("Invalidated", len(job.invalidated_phases))
    st.dataframe(pd.DataFrame(result.get("decisions", [])), use_container_width=True, height=260)
