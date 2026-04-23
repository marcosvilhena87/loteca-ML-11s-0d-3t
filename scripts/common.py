from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

OUTCOMES: Tuple[str, str, str] = ("1", "X", "2")
TIE_BREAK_ORDER: Dict[str, int] = {"1": 0, "2": 1, "X": 2}


@dataclass(frozen=True)
class MatchRanking:
    top1: str
    top2: str
    top3: str


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def parse_decimal(value: str) -> float:
    return float((value or "0").replace(".", "").replace(",", "."))


def format_decimal(value: float, ndigits: int = 6) -> str:
    return f"{value:.{ndigits}f}".replace(".", ",")


def read_csv_semicolon(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def write_csv_semicolon(path: str | Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str | Path, payload: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def rank_outcomes(probabilities: Dict[str, float]) -> MatchRanking:
    ranked = sorted(OUTCOMES, key=lambda key: (-probabilities[key], TIE_BREAK_ORDER[key]))
    return MatchRanking(top1=ranked[0], top2=ranked[1], top3=ranked[2])


def probabilities_from_row(row: Dict[str, str]) -> Dict[str, float]:
    return {
        "1": parse_decimal(row["p(1)"]),
        "X": parse_decimal(row["p(x)"]),
        "2": parse_decimal(row["p(2)"]),
    }


def result_from_row(row: Dict[str, str]) -> str:
    for outcome in OUTCOMES:
        if row.get(outcome, "0") == "1":
            return outcome
    return ""
