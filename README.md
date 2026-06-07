# Glyphon

Glyphon is a document-table extraction and repair workspace built around three connected parts:

1. A geometry-driven parser that turns PDF pages into rows, cells, diagnostics, and candidate table fragments.
2. A multi-phase table-fixer pipeline that uses deterministic validation plus tightly scoped LLM decisions to repair extraction mistakes.
3. Review surfaces and APIs for preview, acceptance, manual correction, audit export, and cross-page merge decisions.

This repository contains both the current service-oriented workflow and an older direct Streamlit correction UI.

## Repository Map

### Core extraction

- `text_pipeline.py`: low-level bbox utilities, row clustering, column detection, assignment scoring.
- `scanned_parser.py`: geometry-based table extraction from positioned text items, with issue generation.
- `parser.py`: OCR-backed PDF parser using RapidOCR and PyMuPDF.
- `text_parser.py`: native-text PDF parser for searchable PDFs, reusing the same downstream extraction contract.

### Current table-fixer system

- `table_fixer/api.py`: FastAPI service for workspace creation, phase execution, preview review, manual actions, undo, and redo.
- `table_fixer/pipeline.py`: five-phase repair pipeline.
- `table_fixer/context.py`: compact prompt context builders and row/table profiling.
- `table_fixer/repairs.py`: deterministic validation and snapshot mutation logic.
- `table_fixer/models.py`: normalized in-memory state model for tables, rows, cells, issues, decisions, and usage.
- `table_fixer/workspace.py`: workspace/audit persistence.
- `table_fixer/export.py`: JSON and Excel export helpers.
- `table_fixer/app.py`: Streamlit app for driving the pipeline directly.

### Review UI

- `glyphon_app/app.py`: Streamlit review orchestrator for page-level jobs and merge flow.
- `glyphon_app/api_client.py`: clients for the table-fixer API and merge API.
- `glyphon_app/review_component.py`: PDF/table synchronized review component.
- `glyphon_app/state.py`: client-side review/job state helpers.

### Cross-page merge service

- `post_separation_merge/api.py`: FastAPI service that decides whether page-end/page-start fragments belong to the same logical table.
- `post_separation_merge/ollama_client.py`: minimal structured Ollama client for merge decisions.

### Legacy direct editor

- `app.py`: older Streamlit table editor using the parser plus direct dataframe edit operations in `correction.py`.
- `correction.py`: row/column/cell edit primitives and export helpers.

### Tests

- `test_text_parser.py`
- `test_extraction_diagnostics.py`
- `table_fixer/test_pipeline.py`
- `table_fixer/test_api.py`
- `post_separation_merge/test_api.py`

## What The System Does

At a high level:

1. Parse PDF pages with either OCR, native text extraction, or mixed auto mode.
2. Build a normalized `PipelineSnapshot` from parser output.
3. Run repair phases in order:
   - `reconciliation`
   - `metadata`
   - `headers`
   - `columns`
   - `warnings`
4. Persist every accepted snapshot, preview snapshot, prompt audit, token count, decision record, and table CSV into a timestamped workspace.
5. Optionally review preview decisions, apply manual edits, and merge fragments across page breaks.

The current design is deliberately conservative:

- deterministic rules handle obvious cases first
- LLM prompts are phase-specific and context-limited
- LLM output is structured JSON only
- every proposed change is validated before application
- invalid decisions never partially mutate a table

## Main Concepts

### Parser contract

Both `parser.py` and `text_parser.py` return the same `PageExtractionResult` shape:

- `rows`
- `cells`
- `assignments`
- `issues`
- `row_table_indexes`
- `row_layout_region_indexes`
- page dimensions and geometry metadata

That common contract is what lets the rest of the system switch between OCR and native text extraction without changing the repair pipeline.

### Snapshot model

`table_fixer.models.PipelineSnapshot` is the central state object. It stores:

- logical tables and column names
- row role classification (`data`, `metadata`, `header`)
- cell text, bbox, source indexes, and warning links
- issues/warnings
- decisions and prompt usage
- lineage for rows, columns, and cells
- invalidated downstream phases

Every phase reads one snapshot and produces a new snapshot.

### Workspaces

Each run creates a folder under `table_fixer/workspaces/` containing:

- `manifest.json`
- `source.pdf`
- raw parser outputs
- source and phase snapshots
- decisions and prompt usage
- prompt audits and LLM trace
- role-specific exports
- per-table CSVs

This makes runs auditable and resumable.

## Five-Phase Table Fixer

### 1. Reconciliation

Detects when the parser accidentally split one same-page table into adjacent fragments. It uses deterministic evidence first:

- same column count
- column geometry alignment
- column type similarity
- occupancy similarity
- vertical continuity
- titled-table rejection signals

Only ambiguous cases go to the LLM.

### 2. Metadata

Classifies a contiguous top prefix of rows as metadata. The code bounds the possible prefix before prompting and rejects non-prefix decisions.

### 3. Headers

Identifies header rows and short continuation groups. Accepted header groups can split one extracted segment into multiple logical tables with inherited lineage.

### 4. Columns

Repairs structural column problems:

- merged columns detected by repeated geometry evidence
- unnamed placeholder columns such as `col_N`

Column splits are only accepted when the header itself justifies the split and the resulting regex or fallback rule validates across the full column.

### 5. Warnings

Repairs remaining row-local warning cases such as displaced or merged entries. The prompt is row-scoped, and code only applies a change when the repair text preserves the original tokens exactly once.

## APIs And UIs

### Table fixer API

Start it with:

```bash
uvicorn table_fixer.api:app --host 0.0.0.0 --port 8770
```

Key endpoints:

- `GET /health`
- `GET /openapi.yaml`
- `POST /table-fixer/workspaces`
- `GET /table-fixer/workspaces/{workspace_id}`
- `GET /table-fixer/workspaces/{workspace_id}/snapshots/{phase}`
- `GET /table-fixer/workspaces/{workspace_id}/pages/{page_number}/image`
- `POST /table-fixer/workspaces/{workspace_id}/execute`
- `POST /table-fixer/workspaces/{workspace_id}/reviews/{phase}`
- `POST /table-fixer/workspaces/{workspace_id}/manual-actions`
- `POST /table-fixer/workspaces/{workspace_id}/history/undo`
- `POST /table-fixer/workspaces/{workspace_id}/history/redo`

### Merge API

Start it with:

```bash
uvicorn post_separation_merge.api:app --host 0.0.0.0 --port 8780
```

Endpoints:

- `GET /health`
- `POST /merge-decisions`

### Streamlit apps

Current pipeline app:

```bash
streamlit run table_fixer/app.py
```

Review app:

```bash
streamlit run glyphon_app/app.py
```

Legacy direct editor:

```bash
streamlit run app.py
```

## Setup

### Python dependencies

Repo-level dependencies:

```bash
pip install -r requirements.txt
```

Table-fixer service/UI dependencies:

```bash
pip install -r table_fixer/requirements.txt
```

Merge service dependencies:

```bash
pip install -r post_separation_merge/requirements.txt
```

### Ollama

The LLM-backed paths assume a local Ollama server:

```bash
ollama serve
ollama pull gemma3:4b
```

Defaults used in the repo:

- model: `gemma3:4b`
- Ollama base URL: `http://localhost:11434`

## Typical Flows

### Service-first flow

1. Start Ollama.
2. Start `table_fixer.api`.
3. Start `post_separation_merge.api`.
4. Run `glyphon_app/app.py`.
5. Upload a PDF and create per-page workspaces.
6. Run phases, review previews if needed, apply manual actions, then merge fragments across pages.

### Direct pipeline flow

1. Start Ollama.
2. Run `streamlit run table_fixer/app.py`.
3. Upload a PDF and execute the repair phases directly inside the app.

### Legacy editing flow

1. Run `streamlit run app.py`.
2. Parse a PDF.
3. Apply direct dataframe edits such as cell moves, row merges, and column splits.

## Testing

The test suite is `unittest`-based. Example commands:

```bash
python3 -m unittest test_text_parser.py
python3 -m unittest test_extraction_diagnostics.py
python3 -m unittest table_fixer.test_pipeline
python3 -m unittest table_fixer.test_api
python3 -m unittest post_separation_merge.test_api
```

## Design Priorities

This codebase consistently favors:

- explicit auditability over hidden mutation
- normalized intermediate state over dataframe-only workflows
- small structured prompts over free-form prompting
- deterministic gating before applying LLM output
- preservation of source geometry and ancestry for review/debugging

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [table_fixer/README.md](./table_fixer/README.md)
