# Table Separation

`table_seperation` performs deterministic 2D spatial graph segmentation and
conservative component merging. It does not depend on CSV row order.

Use it directly with normalized cells:

```python
from table_seperation import segment_tables

tables = segment_tables(cells)
```

Use the existing Glyphon parser contract without modifying the parser:

```python
from table_seperation import segment_parser_pages
from text_parser import parse_pdf_pages

pages = parse_pdf_pages(pdf_bytes)
tables = segment_parser_pages(pages)
```

Pass an optional callable as `llm_judge=`. It is invoked only for deterministic
`UNCERTAIN` boundaries. Low-confidence responses are conservatively merged.

## Streamlit UI

Run the page-selection and splitting UI from the repository root:

```bash
streamlit run table_seperation/app.py
```

The UI uses the existing native-text or OCR parser, previews detected tables on
the selected PDF page, exposes segmentation thresholds, and exports JSON/CSV.

## Synthetic Scenarios

`sample_scenarios.py` contains realistic cell text and bounding boxes for
side-by-side, vertical, offset, and diagonal table layouts. Run all scenarios:

```bash
python3 -m table_seperation.sample_scenarios
```
