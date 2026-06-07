# Glyphon LLM Table Fixer

Standalone five-phase LLM decision and deterministic repair workflow.

```bash
pip install -r table_fixer/requirements.txt
ollama serve
ollama pull gemma3:4b
streamlit run table_fixer/app.py
```

The app runs the existing parser, preserves parser diagnostics, and sends only compact phase-specific context to Ollama. Prompt contexts, raw responses, estimated `tiktoken` counts, native Ollama counts, decisions, lineage, and phase snapshots are available in the UI and audit export.

The phase order is table reconciliation, metadata identification, header identification, structural-column
repair, then remaining warning repair. Reconciliation runs first so an accidental parser split cannot hide
a header fragment or cause a continuation segment to be stripped as metadata.

Each click of `Run extraction` creates a timestamped directory under `table_fixer/workspaces/`.
It contains raw parser diagnostics and CSVs, normalized source data, phase previews and accepted
snapshots, LLM responses, decisions, token usage, preserved metadata/headers, and final table CSVs.

Structural auto-apply defaults to a conservative `0.95` confidence threshold. Cell moves and other
warning repairs are applied only after binary LLM decisions pass deterministic validation; occupied-target
moves are rejected, and invalid metadata/header decisions cannot consume or partially mutate a table.

Actionable cell-level diagnostics are grouped into one small row-local repair task. OCR-confidence diagnostics
never reach the LLM and remain available as parser diagnostics. Each repair prompt shows the complete row as
simple header-entry pairs, identifies the affected cells and merge/displacement problems, and lists the columns
that may change. The LLM returns a `final_values` header-to-text map. Code applies it atomically only when every
original repair-zone token is preserved exactly once and no new value was invented.

Merged-column prompts handle one column at a time and decide eligibility from the header alone. Values may
only help construct and validate a regex after the header clearly supports a split. Code rejects proposed
fields that do not exactly reconstruct the current header, preventing repeated value patterns from splitting
normal headers such as `Tenant Name`. A conservative whitespace fallback is allowed only for an already
header-approved split.

Phase 4 also treats every remaining `col_N` header as a structural defect. One focused prompt decides whether
the whole placeholder column should move into one named column, split into 2-3 named columns, or be renamed as
a genuine field. Code applies the answer only when every value follows the same rule, destinations are empty,
and no data or preserved header/metadata text can be lost. Completely empty placeholder columns are removed
without an LLM call.
