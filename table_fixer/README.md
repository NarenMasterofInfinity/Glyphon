# Glyphon LLM Table Fixer

Standalone five-phase LLM decision and deterministic repair workflow.

```bash
pip install -r table_fixer/requirements.txt
ollama serve
ollama pull gemma3:4b
streamlit run table_fixer/app.py
```

The app runs the existing parser, preserves parser diagnostics, and sends only compact phase-specific context to Ollama. Prompt contexts, raw responses, estimated `tiktoken` counts, native Ollama counts, decisions, lineage, and phase snapshots are available in the UI and audit export.

The phase order is table reconciliation, metadata identification, header identification, merged-column
repair, then remaining warning repair. Reconciliation runs first so an accidental parser split cannot
hide a header fragment or cause a continuation segment to be stripped as metadata.

Each click of `Run extraction` creates a timestamped directory under `table_fixer/workspaces/`.
It contains raw parser diagnostics and CSVs, normalized source data, phase previews and accepted
snapshots, LLM responses, decisions, token usage, preserved metadata/headers, and final table CSVs.

Structural auto-apply defaults to a conservative `0.95` confidence threshold. Cell moves and other
warning repairs are applied only after binary LLM decisions pass deterministic validation; occupied-target
moves are rejected, and invalid metadata/header decisions cannot consume or partially mutate a table.

Cell-level diagnostics are grouped into one small row-local repair task. Each prompt contains only the
problem description, relevant headers, current text split into `available_parts`, and nearby examples that
are explicitly context-only. The LLM returns a `final_values` header-to-text map. Code applies it atomically
only when every original token is preserved exactly once and no new value was invented.

Merged-column prompts handle one column at a time. If the LLM identifies a split but repeatedly produces
an invalid regex, code can apply a conservative whitespace split only when every column value has the same
2-4 part structure.
