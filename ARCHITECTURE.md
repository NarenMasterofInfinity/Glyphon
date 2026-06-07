# Architecture

This document describes the implemented architecture of Glyphon as it exists in this repository. It is organized around the data model, execution flow, service boundaries, and validation rules that keep the system conservative.

## 1. System Overview

Glyphon is not one application. It is a set of cooperating layers:

1. Extraction layer: turns PDFs into positioned text items, rows, cells, assignments, and diagnostics.
2. Repair layer: turns raw parser output into a normalized snapshot and iteratively repairs it through ordered phases.
3. API/workspace layer: persists runs, exposes reviewable state, and enforces phase ordering.
4. Review layer: allows humans to inspect page images, warnings, snapshots, previews, and manual edits.
5. Merge layer: groups page-level table fragments into cross-page logical tables.

The current center of gravity is the `table_fixer` package. Most other surfaces either feed it or consume its outputs.

## 2. Top-Level Components

### 2.1 Extraction

Files:

- `text_pipeline.py`
- `scanned_parser.py`
- `parser.py`
- `text_parser.py`

Responsibilities:

- represent positioned text as `BBoxItem`
- cluster items into rows
- infer columns and boundary candidates
- assign items into cells
- emit diagnostics for ambiguity, collisions, low OCR confidence, and layout incompatibility

Important architectural point: both OCR-based and native-text parsing terminate in the same downstream contract. That contract is what decouples the rest of the repo from the extraction source.

### 2.2 Table-fixer core

Files:

- `table_fixer/models.py`
- `table_fixer/context.py`
- `table_fixer/pipeline.py`
- `table_fixer/repairs.py`
- `table_fixer/export.py`
- `table_fixer/workspace.py`
- `table_fixer/ollama_client.py`

Responsibilities:

- normalize parser output into stable logical identifiers
- generate compact prompt contexts
- call Ollama with structured schemas
- validate decisions
- apply mutations to a cloned snapshot
- persist a full audit trail

### 2.3 Table-fixer API

File:

- `table_fixer/api.py`

Responsibilities:

- create workspaces
- dispatch extraction mode (`text`, `ocr`, `auto`)
- run ordered phases
- manage preview/accept/reject
- expose accepted snapshots and page images
- apply manual actions
- support undo/redo for manual actions

### 2.4 Review application

Files:

- `glyphon_app/app.py`
- `glyphon_app/api_client.py`
- `glyphon_app/review_component.py`
- `glyphon_app/state.py`

Responsibilities:

- orchestrate per-page jobs against the APIs
- show synchronized PDF/table review
- track page-level progress across phases
- gather cross-page fragments for merge decisions

### 2.5 Post-separation merge service

Files:

- `post_separation_merge/api.py`
- `post_separation_merge/ollama_client.py`

Responsibilities:

- consider only page-end to next-page-start fragment pairs
- merge deterministically on exact normalized header equality
- otherwise ask the LLM for a constrained merge decision
- build merge groups from accepted pairwise decisions

### 2.6 Legacy editor

Files:

- `app.py`
- `correction.py`

This predates the normalized `PipelineSnapshot` workflow. It operates directly on pandas dataframes and remains useful as a manual correction surface, but it is architecturally separate from the newer API-first pipeline.

## 3. Data Model

The core model is `PipelineSnapshot` in `table_fixer/models.py`.

### 3.1 Why snapshot state exists

The parser’s native output is page-oriented. The repair pipeline needs state that is:

- stable across phase mutations
- easy to serialize
- easy to diff and audit
- expressive enough for lineage and warning rebasing

`PipelineSnapshot` provides that intermediate form.

### 3.2 Main entities

#### `LogicalTable`

Represents one logical table instance in the current snapshot.

Fields include:

- `table_id`
- `page_number`
- `source_table_index`
- `column_ids`
- `column_names`
- `row_ids`
- `metadata_row_ids`
- `header_row_ids`
- `ancestor_table_ids`

#### `RowState`

Represents a row and its role in the repaired table model.

Key fields:

- `row_id`
- `table_id`
- `source_row_number`
- `role`
- `profile`
- `ancestor_row_ids`

#### `CellState`

Represents one cell value plus its geometric provenance.

Key fields:

- `cell_id`
- `row_id`
- `column_id`
- `text`
- `bbox`
- `source_item_indexes`
- `warning_ids`
- `assignment_score`
- `alternatives`
- `ancestor_cell_ids`

#### `IssueState`

Represents parser or repair issues. Warnings remain active until resolved, reviewed, dismissed, or superseded.

#### `DecisionRecord`

Stores one validated phase decision with:

- target
- action
- confidence
- payload
- validity
- validation errors
- applied flag
- prompt linkage

#### `PromptUsage`

Stores both estimated token counts and native Ollama counters when available.

### 3.3 Snapshot lineage

The snapshot also carries:

- `row_lineage`
- `column_lineage`
- `cell_lineage`

These mappings matter when:

- tables are reconciled
- header rows split one extracted segment into multiple tables
- columns are renamed or split
- warnings are rebased after mutations

The repo treats lineage as a first-class audit feature, not an optional extra.

## 4. Extraction Architecture

### 4.1 `BBoxItem` as the common primitive

`text_pipeline.BBoxItem` is the canonical positioned-text unit. Both OCR output and native PDF words are converted into this type.

That choice keeps the geometry logic shared.

### 4.2 Row clustering

`cluster_rows()`:

- optionally compensates for slant by rotating points
- groups items by projected y-position with an adaptive tolerance
- sorts items within rows by x-position

This is the first place where the parser stops thinking in raw OCR/native-word order and starts thinking in table structure.

### 4.3 Column inference

The extraction stack uses two related concepts:

- column centers
- column boundary candidates

Boundary candidates are more diagnostic-rich. They can be accepted or rejected based on repeated row support or one very strong gap.

This matters because a single suspicious gap should not create a structural column unless the evidence is strong enough.

### 4.4 Layout segmentation

`scanned_parser.py` segments rows into layout regions using vertical gaps and compatibility checks. It can keep incompatible regions in the same source table while still surfacing warnings such as `incompatible_layout_regions`.

This prevents overly eager hard splits while still preserving evidence.

### 4.5 Assignment and diagnostics

When items do not fit neatly into one inferred cell, the extractor emits diagnostics rather than guessing silently. Examples covered by tests:

- `weak_column_boundary`
- `possible_cell_collision`
- `item_crosses_boundary`
- `possible_merged_cell`
- `possible_merged_column`
- `ambiguous_row_assignment`
- `low_ocr_confidence`

These issues become the input surface for later repair phases.

### 4.6 OCR and native text parsers

### OCR path

`parser.py`:

- rasterizes each PDF page via PyMuPDF
- runs RapidOCR
- converts OCR quads into `BBoxItem`
- passes them into `extract_table_scanned()`

### Native text path

`text_parser.py`:

- reads `page.get_text("words", sort=True)` from PyMuPDF
- converts words into `BBoxItem`
- passes them into the same `extract_table_scanned()`

### Auto mode

`table_fixer/api.py` chooses page-by-page:

- native text for pages that contain words
- OCR for pages without native text

This mixed-mode design is more granular than document-level routing.

## 5. Snapshot Construction

`table_fixer.pipeline.snapshot_from_parser()` converts page-oriented extraction results into the normalized graph.

Important implementation details:

- table IDs are generated as `p{page}_t{table}`
- row IDs are derived from table ID plus source row number
- column IDs are generated per table
- cell IDs are `row_id::column_id`
- parser issues are converted into `IssueState`
- affected cell IDs are mapped from parser evidence where possible
- row profiles are computed immediately with `ensure_profiles()`

This is the point where a loose parser output becomes strict, addressable state.

## 6. Row Profiling And Prompt Context Compression

`table_fixer/context.py` provides two distinct capabilities:

1. feature extraction for rows and tables
2. compact prompt context generation under a token budget

### 6.1 Row profiles

Profiles include:

- fill ratio
- text/numeric ratio
- average text length
- occupancy pattern
- following-row similarity
- bbox width ratio
- alignment consistency
- header candidacy

These values support both deterministic heuristics and prompt context shaping.

### 6.2 Context builders

The pipeline never dumps full tables blindly into prompts. Context builders produce compact, purpose-specific inputs such as:

- adjacent table evidence for reconciliation
- bounded top rows for metadata
- candidate row groups for headers
- one-column evidence for structural column repair
- one-row warning repair tasks
- placeholder-column summaries

`fit_context_budget()` trims designated expandable sections to stay within the configured token budget.

Architecturally, that means context reduction is part of the product logic, not just a prompt implementation detail.

## 7. The Five-Phase Pipeline

The pipeline is implemented in `table_fixer/pipeline.py`.

The fixed order is:

1. `reconciliation`
2. `metadata`
3. `headers`
4. `columns`
5. `warnings`

The ordering matters because later phases depend on the structural shape established by earlier phases.

### 7.1 Reconciliation

Goal:

- merge same-page adjacent fragments when they are really one table

Decision strategy:

- deterministic keep-separate if titles differ or column counts differ
- deterministic merge for very high compatibility
- deterministic keep-separate for very low compatibility
- LLM only for the middle band

Application:

- remap right-table column IDs to the left-table schema
- move rows under the surviving table
- remap cells and issue references
- preserve lineage

This phase changes table identity and must therefore run before metadata/header work.

### 7.2 Metadata

Goal:

- peel off a contiguous metadata prefix from the top of a table

Validation rules:

- rows must be an exact top prefix
- allowed prefix is bounded before prompting
- removing metadata must leave at least one data row

Application:

- move rows from `row_ids` to `metadata_row_ids`
- mark row role as `metadata`
- rebase issues to current data rows

### 7.3 Headers

Goal:

- identify header rows and short continuation groups

Decision strategy:

- prompt candidate rows individually
- gather accepted groups
- normalize overlapping/adjacent groups

Validation rules:

- groups must be contiguous
- groups may contain at most two rows
- each header split must leave data rows below it

Application:

- one physical segment may split into multiple logical tables
- new tables inherit metadata/header/data rows appropriately
- column names are composed from accepted header row text
- row/table lineage is updated

This phase is where the system moves from parser columns like `col_1` toward semantic headers.

### 7.4 Columns

Goal:

- repair structural column defects

Two main cases:

1. `possible_merged_column` warnings
2. remaining placeholder columns such as `col_N`

For merged columns, the LLM must decide whether the current header itself justifies a split. Values are used only for regex construction and validation after the header supports the split.

Validation stack:

- header split validation
- regex validation over actual column values
- optional regex repair attempts
- optional whitespace fallback when already header-justified

For placeholder columns, the LLM can choose:

- move to one named column
- split into multiple named columns
- rename as a genuine field
- unresolved

Every action is validated as a whole-column rule. Partial, lossy, or destination-colliding actions are rejected.

### 7.5 Warnings

Goal:

- resolve remaining row-local warnings conservatively

Prompt structure:

- one row at a time
- complete row shown as header-entry pairs
- only listed repair columns may change

Decision contract:

- the model returns a `final_values` map for repair headers

Validation rules:

- original repair-zone text must be reused exactly once
- no invention
- no loss
- no duplication
- occupied-target moves are rejected

This makes warning repair more like constrained transformation than open text editing.

## 8. LLM Integration Model

Glyphon uses Ollama through structured JSON calls.

### 8.1 Table-fixer client

`table_fixer/ollama_client.py`:

- sends system prompt plus compact JSON context
- requests a JSON object matching a provided schema
- records native and estimated token usage
- records every attempt in `prompt_attempts`

If a call fails or returns invalid JSON, the failed attempt is still preserved in the trace.

### 8.2 Prompt design principles in code

The repo’s prompts are narrow and phase-specific. The code enforces several principles:

- JSON object only
- no markdown
- no schema echoing
- minimal context
- deterministic post-validation

The system never trusts a decision just because the model returned a syntactically valid object.

## 9. Persistence And Auditability

`table_fixer/workspace.py` defines the workspace persistence contract.

Each run creates a timestamped workspace with:

- `manifest.json`
- `source.pdf`
- `parser/` raw outputs
- `source/` normalized snapshot
- `{phase}_preview/` or `{phase}_auto_applied/`
- event directories such as rejections
- `llm_trace.json`

Each snapshot directory includes:

- `snapshot.json`
- `issues.json`
- `decisions.json`
- `prompt_usage.json`
- `prompt_audits.json`
- `metadata_rows.json`
- `header_rows.json`
- `tables/*.csv`

This layout is intentionally redundant. It favors easy offline debugging and audit consumption over storage efficiency.

## 10. API State Machine

The state machine lives in `table_fixer/api.py`.

### 10.1 API state file

Each workspace has `api_state.json` with:

- accepted labels per phase
- pending preview
- invalidated phases
- manual sequence
- manual history
- redo stack

### 10.2 Accepted vs pending state

There are always two conceptual views:

- accepted state: the current committed snapshot lineage
- pending preview: a proposed next-phase snapshot awaiting accept/reject

The API forbids running further phases while a preview is pending.

### 10.3 Phase ordering

The API only accepts requests for the next contiguous phase sequence from the current accepted state.

That prevents:

- skipping ahead
- replaying stale downstream phases after edits
- branching into inconsistent repair histories

## 11. Manual Actions, Undo, And Redo

Manual edits are applied through the API, not by mutating the accepted snapshot in place.

Supported action types include:

- `edit_cell_text`
- `set_warning_status`
- `move_cell`
- `split_cell_text`
- `merge_adjacent_cells`
- `set_decision_enabled`

Implementation pattern:

1. load the accepted snapshot for the chosen base phase
2. clone it
3. apply action list
4. persist a new manual snapshot label
5. invalidate all downstream accepted phases
6. append an entry to manual history

Undo/redo restores saved accepted-label maps from history entries. This is label-state restoration, not reverse replay of individual operations.

That is simpler and more robust than trying to synthesize inverse operations for every edit type.

## 12. Review Application Architecture

`glyphon_app/app.py` is a thin orchestration client over the APIs.

### 12.1 Per-page job model

The app tracks one `PageJob` per page. Each job stores:

- page number
- workspace ID
- latest workspace response
- local activity log

This is why the review flow can run extraction/repair per page and then later merge across pages.

### 12.2 Review component

`glyphon_app/review_component.py` renders:

- page image
- warning overlays
- table cells
- selection details

The component keeps PDF geometry and logical cell data synchronized, which is essential for meaningful manual review.

### 12.3 Merge orchestration

The review app converts accepted page tables into simplified fragments:

- page number
- source table index
- start/end-of-page status
- header
- sample rows

Those fragments are sent to the merge API.

## 13. Cross-Page Merge Architecture

The merge service is intentionally much smaller than the table-fixer.

### 13.1 Scope restriction

It only considers pairs where:

- left fragment is the page end
- right fragment is the next page start
- pages are consecutive

That keeps the decision surface narrow.

### 13.2 Deterministic first

If normalized headers match exactly, it merges with full confidence without calling the LLM.

### 13.3 Constrained LLM mode

If headers differ but column counts match, the LLM may decide to merge and choose a `proposed_header`.

Critical validation rule:

- each proposed header cell must be copied from the corresponding left or right input header cell

So the model may select between observed alternatives, but it may not invent a new header.

## 14. Legacy Streamlit Editor

The root `app.py` plus `correction.py` represent an older architecture:

- parser output becomes a dataframe
- users preview operations as `PendingChange`
- operations mutate rows/columns/cells directly
- exports are dataframe-oriented

This surface is still useful, but it does not share the normalized snapshot state machine of `table_fixer`.

In practical terms, there are two editing paradigms in this repo:

- dataframe editing
- snapshot/phase editing

The newer one is the more extensible and auditable design.

## 15. Test Coverage And What It Protects

Tests are concentrated on the highest-risk logic.

### Extraction tests

- native-text parser contract preservation
- geometry diagnostics
- merged-column evidence promotion
- ambiguous-row and OCR confidence reporting

### Table-fixer tests

- prompt/usage recording
- workspace persistence
- metadata context shaping
- deterministic reconciliation
- API invalidation rules
- mixed extraction-mode dispatch

### Merge tests

- deterministic header match
- LLM header selection
- disabled-LLM behavior
- pair-scope restrictions
- invented-header rejection

The tests reinforce the repo’s main architectural claim: LLM output is always bounded by deterministic contracts.

## 16. Architectural Tradeoffs

### Strengths

- strong audit trail
- narrow, validated LLM surface area
- extraction/parser abstraction boundary is clean
- phase ordering is explicit and enforced
- manual edits integrate with the same workspace history

### Costs

- snapshot persistence is verbose
- logic is spread across several medium-sized modules
- some older surfaces duplicate capabilities from the newer workflow
- manual and automated editing models coexist rather than fully converging

Those tradeoffs are deliberate. This codebase chooses inspectability and safety over minimalism.

## 17. Extension Points

The most natural places to extend the system are:

- new extraction backends that still emit `PageExtractionResult`
- new warning types plus corresponding context/repair validators
- richer export targets
- more explicit downstream merge application after merge-group decisions
- stronger end-to-end orchestration around multi-page documents

The least safe place to extend casually is inside phase mutation logic in `table_fixer/repairs.py`, because that code carries the invariants that make previews, lineage, and auditability trustworthy.
