from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from typing import Any

import pandas as pd


SYSTEM_COLUMNS = ["page_number", "table_index", "row_number"]


@dataclass
class PendingChange:
    action: str
    description: str
    preview_df: pd.DataFrame
    metadata: dict[str, Any]


def data_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in SYSTEM_COLUMNS]


def clone_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def normalize_row_numbers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "page_number" not in normalized.columns or "row_number" not in normalized.columns:
        return normalized

    for _, indexes in normalized.groupby("page_number", sort=False).groups.items():
        normalized.loc[indexes, "row_number"] = range(1, len(indexes) + 1)
    return normalized


def merge_columns(df: pd.DataFrame, columns: list[str], output_name: str) -> pd.DataFrame:
    if len(columns) < 2:
        raise ValueError("Select at least two columns to merge.")

    merged = df.copy()
    output_name = output_name.strip() or columns[0]
    first_position = merged.columns.get_loc(columns[0])
    merged_values = (
        merged[columns]
        .fillna("")
        .astype(str)
        .apply(lambda row: " ".join(value for value in row if value.strip()).strip(), axis=1)
    )
    merged = merged.drop(columns=columns)
    merged.insert(first_position, output_name, merged_values)
    return merged


def split_column_by_delimiter(
    df: pd.DataFrame,
    column: str,
    delimiter: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    if not delimiter:
        raise ValueError("Delimiter cannot be empty.")

    split_df = df.copy()
    position = split_df.columns.get_loc(column)
    parts = split_df[column].fillna("").astype(str).str.split(delimiter, n=1, expand=True)
    left_values = parts[0].str.strip()
    right_values = parts[1].str.strip() if parts.shape[1] > 1 else ""

    split_df = split_df.drop(columns=[column])
    split_df.insert(position, left_name.strip() or f"{column}_left", left_values)
    split_df.insert(position + 1, right_name.strip() or f"{column}_right", right_values)
    return split_df


def insert_column(df: pd.DataFrame, after_column: str | None, column_name: str) -> pd.DataFrame:
    inserted = df.copy()
    column_name = column_name.strip() or f"col_{len(data_columns(inserted)) + 1}"
    position = len(inserted.columns)
    if after_column and after_column in inserted.columns:
        position = inserted.columns.get_loc(after_column) + 1
    inserted.insert(position, column_name, "")
    return inserted


def delete_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        raise ValueError("Select at least one column to remove.")
    return df.drop(columns=columns)


def rename_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("New column name cannot be empty.")
    if new_name in df.columns and new_name != column:
        raise ValueError(f"Column {new_name} already exists.")
    return df.rename(columns={column: new_name})


def edit_cell(
    df: pd.DataFrame,
    page_number: int,
    row_number: int,
    column: str,
    value: str,
) -> pd.DataFrame:
    edited = df.copy()
    mask = (edited["page_number"] == page_number) & (edited["row_number"] == row_number)
    indexes = edited.index[mask].tolist()
    if not indexes:
        raise ValueError("Selected cell could not be found.")
    edited.at[indexes[0], column] = value
    return edited


def move_cell(
    df: pd.DataFrame,
    source_page: int,
    source_row: int,
    source_column: str,
    target_page: int,
    target_row: int,
    target_column: str,
) -> pd.DataFrame:
    moved = df.copy()
    source_mask = (moved["page_number"] == source_page) & (moved["row_number"] == source_row)
    target_mask = (moved["page_number"] == target_page) & (moved["row_number"] == target_row)
    source_indexes = moved.index[source_mask].tolist()
    target_indexes = moved.index[target_mask].tolist()
    if not source_indexes or not target_indexes:
        raise ValueError("Source or target cell could not be found.")

    source_index = source_indexes[0]
    target_index = target_indexes[0]
    source_value = "" if pd.isna(moved.at[source_index, source_column]) else str(moved.at[source_index, source_column]).strip()
    target_value = "" if pd.isna(moved.at[target_index, target_column]) else str(moved.at[target_index, target_column]).strip()

    if source_value:
        moved.at[target_index, target_column] = f"{target_value} {source_value}".strip() if target_value else source_value
    moved.at[source_index, source_column] = ""
    return moved


def merge_rows(df: pd.DataFrame, indexes: list[int]) -> pd.DataFrame:
    if len(indexes) < 2:
        raise ValueError("Select at least two rows to merge.")

    merged = df.copy()
    ordered = sorted(indexes)
    base_index = ordered[0]
    columns = data_columns(merged)

    for column in columns:
        values = []
        for index in ordered:
            value = merged.at[index, column]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                values.append(text)
        merged.at[base_index, column] = " ".join(values).strip()

    merged = merged.drop(index=ordered[1:]).reset_index(drop=True)
    return normalize_row_numbers(merged)


def split_row_by_delimiter(
    df: pd.DataFrame,
    index: int,
    delimiter: str,
) -> pd.DataFrame:
    if not delimiter:
        raise ValueError("Delimiter cannot be empty.")

    split_df = df.copy()
    original = split_df.loc[index].copy()
    new_row = original.copy()
    for column in data_columns(split_df):
        left, separator, right = str(original[column]).partition(delimiter)
        split_df.at[index, column] = left.strip()
        new_row[column] = right.strip() if separator else ""

    upper = split_df.iloc[: index + 1]
    lower = split_df.iloc[index + 1 :]
    split_df = pd.concat([upper, pd.DataFrame([new_row]), lower], ignore_index=True)
    return normalize_row_numbers(split_df)


def insert_row(df: pd.DataFrame, after_index: int | None, page_number: int) -> pd.DataFrame:
    inserted = df.copy()
    row = {column: "" for column in inserted.columns}
    row["page_number"] = page_number
    if "table_index" in row:
        nearby_index = after_index if after_index is not None and after_index in inserted.index else None
        row["table_index"] = int(inserted.at[nearby_index, "table_index"]) if nearby_index is not None else 1
    row["row_number"] = 0
    position = len(inserted) if after_index is None else after_index + 1
    inserted = pd.concat(
        [inserted.iloc[:position], pd.DataFrame([row]), inserted.iloc[position:]],
        ignore_index=True,
    )
    return normalize_row_numbers(inserted)


def delete_rows(df: pd.DataFrame, indexes: list[int]) -> pd.DataFrame:
    if not indexes:
        raise ValueError("Select at least one row to remove.")
    deleted = df.drop(index=indexes).reset_index(drop=True)
    return normalize_row_numbers(deleted)


def _excel_safe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list, tuple)) else value
            for key, value in record.items()
        }
        for record in records
    ]


def dataframe_to_excel(
    df: pd.DataFrame,
    issues: list[dict[str, Any]] | None = None,
    assignments: list[dict[str, Any]] | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, sheet_name="corrected_table", index=False)
        pd.DataFrame(_excel_safe_records(issues or [])).to_excel(writer, sheet_name="extraction_issues", index=False)
        pd.DataFrame(_excel_safe_records(assignments or [])).to_excel(writer, sheet_name="assignment_candidates", index=False)
    return buffer.getvalue()


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def operation_record(action: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "metadata": metadata,
    }


def pending_to_metadata(pending: PendingChange) -> dict[str, Any]:
    return {
        "action": pending.action,
        "description": pending.description,
        "metadata": pending.metadata,
    }
