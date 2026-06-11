"""Realistic positioned-cell fixtures for manually exercising table separation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cell(text: str, x: float, y: float, width: float = 42, height: float = 11, page: int = 1) -> dict[str, Any]:
    return {
        "text": text,
        "page": page,
        "x1": x,
        "y1": y,
        "x2": x + width,
        "y2": y + height,
    }


def table(
    x: float,
    y: float,
    headers: list[str],
    rows: list[list[str]],
    *,
    column_gap: float = 62,
    row_gap: float = 17,
    page: int = 1,
) -> list[dict[str, Any]]:
    values = [headers, *rows]
    return [
        cell(text, x + column * column_gap, y + row * row_gap, page=page)
        for row, row_values in enumerate(values)
        for column, text in enumerate(row_values)
    ]


def scenarios() -> dict[str, dict[str, Any]]:
    side_by_side = table(
        20,
        25,
        ["Item", "Qty", "Amount"],
        [["Pens", "2", "10.00"], ["Paper", "5", "25.00"], ["Folders", "3", "18.00"]],
    ) + table(
        260,
        25,
        ["Department", "Employees"],
        [["Operations", "14"], ["Finance", "8"], ["Legal", "5"]],
        column_gap=90,
    )

    close_staggered_side_by_side = table(
        20,
        25,
        ["Product", "Units"],
        [["Alpha", "12"], ["Beta", "8"], ["Gamma", "15"], ["Delta", "4"]],
        column_gap=72,
    ) + table(
        175,
        50,
        ["Region", "Revenue"],
        [["North", "1250"], ["South", "980"], ["West", "1430"]],
        column_gap=82,
    )

    top_and_below = table(
        20,
        25,
        ["Invoice", "Customer", "Total"],
        [["INV-01", "Acme", "120"], ["INV-02", "Orbit", "90"]],
        column_gap=78,
    ) + table(
        20,
        135,
        ["Product", "Category", "Price"],
        [["Monitor", "Hardware", "220"], ["Support", "Service", "80"]],
        column_gap=78,
    )

    below_shifted_right = table(
        20,
        25,
        ["Account", "Owner", "Balance"],
        [["1001", "Asha", "500"], ["1002", "Ben", "720"]],
        column_gap=80,
    ) + table(
        65,
        130,
        ["Office", "Headcount"],
        [["Pune", "24"], ["Mumbai", "31"]],
        column_gap=92,
    )

    below_shifted_left = table(
        110,
        25,
        ["Account", "Owner", "Balance"],
        [["1001", "Asha", "500"], ["1002", "Ben", "720"]],
        column_gap=80,
    ) + table(
        25,
        130,
        ["Office", "Headcount"],
        [["Pune", "24"], ["Mumbai", "31"]],
        column_gap=92,
    )

    diagonal_partial_overlap = table(
        20,
        25,
        ["Order", "Units", "Value"],
        [["A-1", "2", "40"], ["A-2", "4", "80"]],
        column_gap=72,
    ) + table(
        135,
        115,
        ["Team", "Score"],
        [["Red", "18"], ["Blue", "21"]],
        column_gap=82,
    )

    one_over_two = table(
        20,
        25,
        ["ID", "Customer", "Region", "Owner", "Status", "Amount"],
        [
            ["A-100", "Acme", "North", "Asha", "Open", "1200"],
            ["A-101", "Orbit", "West", "Ben", "Closed", "950"],
            ["A-102", "Nova", "South", "Chen", "Open", "1430"],
        ],
        column_gap=70,
    ) + table(
        20,
        125,
        ["Office", "Employees"],
        [["Pune", "24"], ["Delhi", "19"], ["Mumbai", "31"]],
        column_gap=92,
    ) + table(
        245,
        125,
        ["Category", "Units", "Revenue"],
        [["Hardware", "18", "5400"], ["Service", "12", "3100"], ["Software", "25", "8200"]],
        column_gap=78,
    )

    return {
        "side_by_side": {"expected_tables": 2, "cells": side_by_side},
        "close_staggered_side_by_side": {"expected_tables": 2, "cells": close_staggered_side_by_side},
        "top_and_below": {"expected_tables": 2, "cells": top_and_below},
        "below_shifted_right": {"expected_tables": 2, "cells": below_shifted_right},
        "below_shifted_left": {"expected_tables": 2, "cells": below_shifted_left},
        "diagonal_partial_overlap": {"expected_tables": 2, "cells": diagonal_partial_overlap},
        "one_six_column_table_over_two_side_by_side_tables": {
            "expected_tables": 3,
            "cells": one_over_two,
        },
    }


def write_json(path: str | Path) -> None:
    Path(path).write_text(json.dumps(scenarios(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    from table_seperation import segment_tables

    for name, scenario in scenarios().items():
        result = segment_tables(scenario["cells"])
        status = "PASS" if len(result) == scenario["expected_tables"] else "FAIL"
        print(f"{status} {name}: expected={scenario['expected_tables']} actual={len(result)}")
