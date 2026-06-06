from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import fitz

from parser import parse_pdf_pages
from table_fixer.export import excel_export, json_export, rows_by_role, table_records
from table_fixer.models import PipelineSnapshot
from table_fixer.ollama_client import OllamaClientError, OllamaLLMClient
from table_fixer.pipeline import PhaseRun, TableFixerPipeline, snapshot_from_parser
from table_fixer.workspace import (
    create_workspace,
    persist_event,
    persist_parser_results,
    persist_phase_run,
    persist_snapshot,
)


st.set_page_config(page_title="Glyphon LLM Table Fixer", layout="wide")
st.title("Glyphon LLM Table Fixer")

PHASES = ["reconciliation", "metadata", "headers", "columns", "warnings"]
PHASE_LABELS = {
    "reconciliation": "Phase 1 - Reconcile Tables",
    "metadata": "Phase 2 - Metadata",
    "headers": "Phase 3 - Headers",
    "columns": "Phase 4 - Columns",
    "warnings": "Phase 5 - Warnings",
}


@st.cache_data(show_spinner=False)
def parse_upload(pdf_bytes: bytes, page_numbers: tuple[int, ...]):
    return parse_pdf_pages(pdf_bytes, page_numbers=list(page_numbers))


@st.cache_data(show_spinner=False)
def get_pdf_page_numbers(pdf_bytes: bytes) -> list[int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_numbers = list(range(1, len(doc) + 1))
    doc.close()
    return page_numbers


def initialize(file_key: str, source: PipelineSnapshot) -> None:
    if st.session_state.get("fixer_file_key") == file_key:
        return
    st.session_state.fixer_file_key = file_key
    st.session_state.phase_snapshots = {"source": source}
    st.session_state.pending_runs = {}
    persist_snapshot(Path(st.session_state.table_fixer_workspace), "source", source)


def previous_phase(phase: str) -> str:
    index = PHASES.index(phase)
    return "source" if index == 0 else PHASES[index - 1]


def invalidate_from(phase: str) -> None:
    index = PHASES.index(phase)
    for later in PHASES[index:]:
        st.session_state.phase_snapshots.pop(later, None)
        st.session_state.pending_runs.pop(later, None)


def current_final_snapshot() -> PipelineSnapshot:
    for phase in reversed(PHASES):
        if phase in st.session_state.phase_snapshots:
            return st.session_state.phase_snapshots[phase]
    return st.session_state.phase_snapshots["source"]


def token_frame(snapshot: PipelineSnapshot) -> pd.DataFrame:
    return pd.DataFrame([usage.__dict__ for usage in snapshot.prompt_usage])


def show_tables(snapshot: PipelineSnapshot, role: str = "data") -> None:
    if role != "data":
        records = rows_by_role(snapshot, role)
        st.dataframe(pd.DataFrame(records), use_container_width=True, height=260)
        return
    for table_id in snapshot.tables:
        with st.expander(table_id, expanded=True):
            st.dataframe(pd.DataFrame(table_records(snapshot, table_id)), use_container_width=True)


def show_decisions(run: PhaseRun) -> None:
    st.dataframe(pd.DataFrame([record.__dict__ for record in run.decisions]), use_container_width=True, height=280)
    usage = pd.DataFrame([response.usage.__dict__ for response in run.responses])
    if not usage.empty:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Prompts", len(usage))
        metric_columns[1].metric("Estimated input tokens", int(usage["input_tokens"].sum()))
        metric_columns[2].metric("Estimated output tokens", int(usage["output_tokens"].sum()))
        metric_columns[3].metric("Duration ms", round(float(usage["duration_ms"].sum()), 1))
        st.dataframe(usage, use_container_width=True, height=220)
    with st.expander("Prompt contexts and raw responses", expanded=False):
        for response in run.responses:
            st.markdown(f"**{response.usage.purpose}** (`{response.prompt_id}`)")
            st.json(response.context)
            st.code(response.raw_response or "", language="json")


def build_pipeline(
    client: OllamaLLMClient,
    *,
    auto_apply_threshold: float,
    structural_auto_apply_threshold: float,
    context_token_budget: int,
) -> TableFixerPipeline:
    try:
        return TableFixerPipeline(
            client,
            auto_apply_threshold=auto_apply_threshold,
            structural_auto_apply_threshold=structural_auto_apply_threshold,
            context_token_budget=context_token_budget,
        )
    except TypeError as exc:
        if "structural_auto_apply_threshold" not in str(exc):
            raise
        pipeline = TableFixerPipeline(
            client,
            auto_apply_threshold=auto_apply_threshold,
            context_token_budget=context_token_budget,
        )
        pipeline.structural_auto_apply_threshold = structural_auto_apply_threshold
        return pipeline


def execute_phase(phase: str, pipeline: TableFixerPipeline, auto_apply: bool) -> None:
    source = st.session_state.phase_snapshots[previous_phase(phase)]
    invalidate_from(phase)
    method = getattr(pipeline, f"run_{phase}")
    try:
        run = method(source, auto_apply=auto_apply)
    except OllamaClientError as exc:
        st.error(str(exc))
        return
    if auto_apply:
        st.session_state.phase_snapshots[phase] = run.proposed_snapshot
        persist_phase_run(Path(st.session_state.table_fixer_workspace), phase, run, "auto_applied")
    else:
        st.session_state.pending_runs[phase] = run
        persist_phase_run(Path(st.session_state.table_fixer_workspace), phase, run, "preview")
    st.rerun()


def phase_controls(phase: str, pipeline: TableFixerPipeline) -> PipelineSnapshot | None:
    dependency = previous_phase(phase)
    if dependency not in st.session_state.phase_snapshots:
        st.info(f"Accept {PHASE_LABELS.get(dependency, 'Source')} before running this phase.")
        return None
    run_preview, run_auto = st.columns(2)
    if run_preview.button("Run preview", key=f"preview_{phase}", use_container_width=True):
        execute_phase(phase, pipeline, False)
    if run_auto.button("Run + auto-apply", key=f"auto_{phase}", use_container_width=True):
        execute_phase(phase, pipeline, True)
    pending: PhaseRun | None = st.session_state.pending_runs.get(phase)
    if pending:
        st.subheader("Proposed decisions")
        show_decisions(pending)
        st.subheader("Proposed result")
        show_tables(pending.proposed_snapshot)
        accept, reject = st.columns(2)
        if accept.button("Accept phase", key=f"accept_{phase}", type="primary", use_container_width=True):
            st.session_state.phase_snapshots[phase] = pending.proposed_snapshot
            st.session_state.pending_runs.pop(phase, None)
            persist_phase_run(Path(st.session_state.table_fixer_workspace), phase, pending, "accepted")
            st.rerun()
        if reject.button("Reject phase", key=f"reject_{phase}", use_container_width=True):
            st.session_state.pending_runs.pop(phase, None)
            persist_event(Path(st.session_state.table_fixer_workspace), phase, "rejected")
            st.rerun()
    accepted = st.session_state.phase_snapshots.get(phase)
    if accepted:
        st.success(f"{PHASE_LABELS[phase]} accepted.")
    return accepted


uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
model = st.sidebar.text_input("Ollama model", value="gemma3:4b")
base_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
token_budget = st.sidebar.number_input("Max context tokens", min_value=500, max_value=16000, value=3500, step=250)
threshold = st.sidebar.slider("Auto-apply confidence", min_value=0.5, max_value=1.0, value=0.80, step=0.01)
structural_threshold = st.sidebar.slider(
    "Structural auto-apply confidence",
    min_value=0.8,
    max_value=1.0,
    value=0.95,
    step=0.01,
)
st.sidebar.caption("Token counts use tiktoken estimates; Ollama native counts are preserved when available.")

if not uploaded:
    st.info("Upload a PDF to start the five-phase table-fixing workflow.")
    st.stop()

pdf_bytes = uploaded.read()
file_key = sha256(pdf_bytes).hexdigest()
all_page_numbers = get_pdf_page_numbers(pdf_bytes)
extraction_request = st.session_state.get("table_fixer_extraction_request")

if (
    not extraction_request
    or extraction_request.get("file_key") != file_key
    or not extraction_request.get("workspace")
):
    st.session_state.table_fixer_extraction_request = {
        "file_key": file_key,
        "selected_pages": [],
        "submitted": False,
    }
    extraction_request = st.session_state.table_fixer_extraction_request

st.sidebar.subheader("Extraction scope")
selected_pages = st.sidebar.multiselect(
    "Pages to extract",
    options=all_page_numbers,
    default=extraction_request.get("selected_pages", []),
    help="Leave this empty to extract all pages in the PDF.",
    key=f"table_fixer_page_scope_{file_key}",
)
st.sidebar.caption("Leave the page selector empty to run extraction on every page.")

run_extraction = st.sidebar.button("Run extraction", type="primary", use_container_width=True)
if run_extraction:
    requested_pages = selected_pages or all_page_numbers
    workspace = create_workspace(uploaded.name, file_key, requested_pages)
    st.session_state.table_fixer_extraction_request = {
        "file_key": file_key,
        "selected_pages": selected_pages,
        "submitted": True,
        "workspace": str(workspace),
    }
    st.session_state.table_fixer_workspace = str(workspace)
    st.session_state.fixer_file_key = ""
    st.rerun()

if not st.session_state.table_fixer_extraction_request.get("submitted"):
    st.info("Choose pages in the sidebar, or leave the selection empty for all pages, then click `Run extraction`.")
    st.caption(f"This PDF has {len(all_page_numbers)} page(s).")
    st.stop()

selected_pages = st.session_state.table_fixer_extraction_request.get("selected_pages", [])
st.session_state.table_fixer_workspace = st.session_state.table_fixer_extraction_request["workspace"]
active_page_numbers = tuple(selected_pages or all_page_numbers)
with st.spinner("Running parser..."):
    page_results = parse_upload(pdf_bytes, active_page_numbers)
persist_parser_results(Path(st.session_state.table_fixer_workspace), page_results)
source_snapshot = snapshot_from_parser(page_results)
initialize(file_key, source_snapshot)

client = OllamaLLMClient(model=model, base_url=base_url)
pipeline = build_pipeline(
    client,
    auto_apply_threshold=threshold,
    structural_auto_apply_threshold=structural_threshold,
    context_token_budget=int(token_budget),
)
st.sidebar.caption(f"Workspace: {st.session_state.table_fixer_workspace}")

tabs = st.tabs([
    "Source",
    PHASE_LABELS["reconciliation"],
    PHASE_LABELS["metadata"],
    PHASE_LABELS["headers"],
    PHASE_LABELS["columns"],
    PHASE_LABELS["warnings"],
    "Final & Audit",
])

with tabs[0]:
    source = st.session_state.phase_snapshots["source"]
    show_tables(source)
    st.subheader("Cells with bbox, warnings, and lineage")
    st.dataframe(pd.DataFrame([cell.__dict__ for cell in source.cells.values()]), use_container_width=True, height=360)
    st.subheader("Original parser diagnostics")
    st.dataframe(pd.DataFrame([issue.__dict__ for issue in source.issues.values()]), use_container_width=True, height=300)

with tabs[1]:
    accepted = phase_controls("reconciliation", pipeline)
    if accepted:
        st.subheader("Reconciled source tables")
        show_tables(accepted)

with tabs[2]:
    accepted = phase_controls("metadata", pipeline)
    if accepted:
        st.subheader("Preserved metadata rows")
        show_tables(accepted, "metadata")
        st.subheader("Clean tables")
        show_tables(accepted)

with tabs[3]:
    accepted = phase_controls("headers", pipeline)
    if accepted:
        st.subheader("Preserved header rows")
        show_tables(accepted, "header")
        st.subheader("Logical tables")
        show_tables(accepted)

with tabs[4]:
    accepted = phase_controls("columns", pipeline)
    if accepted:
        st.subheader("Column repair results")
        show_tables(accepted)
        st.subheader("Rebased issues")
        st.dataframe(pd.DataFrame([issue.__dict__ for issue in accepted.issues.values()]), use_container_width=True)

with tabs[5]:
    accepted = phase_controls("warnings", pipeline)
    if accepted:
        st.subheader("Remaining warnings")
        st.dataframe(
            pd.DataFrame([issue.__dict__ for issue in accepted.active_warnings()]),
            use_container_width=True,
        )

with tabs[6]:
    final = current_final_snapshot()
    show_tables(final)
    st.subheader("Decision audit")
    st.dataframe(pd.DataFrame([record.__dict__ for record in final.decisions]), use_container_width=True, height=300)
    st.subheader("Prompt token ledger")
    usage = token_frame(final)
    if not usage.empty:
        st.dataframe(usage, use_container_width=True, height=300)
        st.json({
            "estimated_input_tokens": int(usage["input_tokens"].sum()),
            "estimated_output_tokens": int(usage["output_tokens"].sum()),
            "native_prompt_tokens": int(usage["native_prompt_tokens"].fillna(0).sum()),
            "native_output_tokens": int(usage["native_output_tokens"].fillna(0).sum()),
        })
    with st.expander("Accepted prompt audit", expanded=False):
        st.json(final.prompt_audits)
    phase_snapshots = st.session_state.phase_snapshots
    st.download_button(
        "Download full audit JSON",
        data=json_export(final, phase_snapshots),
        file_name="glyphon_table_fixer_audit.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download final Excel",
        data=excel_export(final),
        file_name="glyphon_table_fixer.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
